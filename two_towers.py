import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import argparse
import sys
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Any, List, Optional
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    """Centralized configuration for the project."""
    # System
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() 
                          else "cuda" if torch.cuda.is_available() 
                          else "cpu")
    DATA_PATH = "Data/ceo_types_v0.2.csv"
    OUTPUT_PATH = "/Users/smaric/Papers/Complementarities/Data/BLM Replication Final/Two Towers Implementation/Output"
    
    # Hyperparameters
    EPOCHS = 40
    LEARNING_RATE = 0.0004
    LATENT_DIM = 60
    
    # Embedding Dimensions
    EMBEDDING_DIM_SMALL = 2
    EMBEDDING_DIM_MEDIUM = 8
    EMBEDDING_DIM_LARGE = 48

    # Feature Definitions
    ID_COLS = ['gvkey', 'match_exec_id']
    
    # CEO Features
    CEO_NUMERIC_COLS = ['Age'] # 'tenure' will be derived
    CEO_CAT_COLS = ['Gender', 'maxedu', 'ivy', 'm', 'Output', 'Throghput', 'Peripheral']
    CEO_RAW_COLS = CEO_NUMERIC_COLS + CEO_CAT_COLS + ['ceo_year', 'dep_baby_ceo']
    
    # Firm Features
    FIRM_NUMERIC_COLS = ['ind_firms_60w', 'non_competition_score', 'boardindpw', 
                         'boardsizew', 'busyw', 'pct_blockw', 'logatw', 'exp_roa',
                         'rdintw', 'capintw', 'leverage', 'divyieldw']
    FIRM_CAT_COLS = ['compindustry', 'ba_state', 'rd_control', 'dpayer']
    FIRM_RAW_COLS = FIRM_NUMERIC_COLS + FIRM_CAT_COLS + ['fiscalyear']

    # Target & Weights
    TARGET_COL = 'match_means'
    WEIGHT_COL = 'sd_match_means'
    
    @property
    def all_required_cols(self) -> List[str]:
        """Returns a list of all unique columns required from the CSV."""
        return list(set(
            self.ID_COLS + self.CEO_RAW_COLS + self.FIRM_RAW_COLS + 
            [self.TARGET_COL, self.WEIGHT_COL]
        ))

# ==========================================
# 2. DATA PROCESSOR
# ==========================================
class DataProcessor:
    """
    Handles loading, cleaning, feature engineering, encoding, and scaling.
    Encapsulates the state (scalers/encoders) for consistency.
    """
    def __init__(self, config: Config):
        self.cfg = config
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Initialize scalers
        self.scalers['firm'] = StandardScaler()
        self.scalers['ceo'] = StandardScaler()
        
        # Define final feature sets (post-engineering)
        self.final_firm_numeric = self.cfg.FIRM_NUMERIC_COLS
        self.final_ceo_numeric = self.cfg.CEO_NUMERIC_COLS + ['tenure']
        
        self.processed_df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Loads raw data from CSV."""
        print(f"Loading data from {self.cfg.DATA_PATH}...")
        try:
            df = pd.read_csv(self.cfg.DATA_PATH, on_bad_lines='skip')
        except FileNotFoundError:
            print(f"Error: File not found at {self.cfg.DATA_PATH}")
            return pd.DataFrame()
        
        # Check and select columns
        missing = [c for c in self.cfg.all_required_cols if c not in df.columns]
        if missing:
            print(f"Error: Missing essential columns: {missing}")
            return pd.DataFrame()

        return df[self.cfg.all_required_cols].copy()

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 1: Cleaning and Feature Engineering (Stateless)."""
        if df.empty:
            return df
            
        # 1. Cleaning
        initial_len = len(df)
        df = df.dropna().copy()
        print(f"Dropped {initial_len - len(df)} rows with NaNs. Final count: {len(df)}")
        
        # 2. Feature Engineering
        df['tenure'] = (df['fiscalyear'] - df['ceo_year']).clip(lower=0)
        
        # 3. Weight Calculation
        epsilon = 1e-6
        df['weights'] = 1 / (df[self.cfg.WEIGHT_COL]**2 + epsilon)
        
        return df

    def fit(self, df: pd.DataFrame):
        """Step 2: Fit scalers and encoders on training data."""
        print("Fitting scalers and encoders on training data...")
        
        # Categorical Encoding
        for col in self.cfg.FIRM_CAT_COLS:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(df[col].astype(str))
            
        for col in self.cfg.CEO_CAT_COLS:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(df[col].astype(str))
        
        # Scaling Numeric Features
        self.scalers['firm'].fit(df[self.final_firm_numeric])
        self.scalers['ceo'].fit(df[self.final_ceo_numeric])
        
    def transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Step 3: Apply transformations and return tensors."""
        if df.empty:
            return {}
            
        df = df.copy()
        
        # Categorical Encoding (Handle unknown labels safely-ish)
        def safe_transform(encoder, series):
            # Map unknown classes to the first class (0) - simple fallback
            known_classes = set(encoder.classes_)
            return series.apply(lambda x: encoder.transform([x])[0] if x in known_classes else 0)

        for col in self.cfg.FIRM_CAT_COLS:
            df[f'{col}_code'] = safe_transform(self.encoders[col], df[col].astype(str))
            
        for col in self.cfg.CEO_CAT_COLS:
            df[f'{col}_code'] = safe_transform(self.encoders[col], df[col].astype(str))
        
        # Scaling Numeric Features
        df[self.final_firm_numeric] = self.scalers['firm'].transform(df[self.final_firm_numeric])
        df[self.final_ceo_numeric] = self.scalers['ceo'].transform(df[self.final_ceo_numeric])
        
        # Store processed DF for visualization/debugging
        self.processed_df = df
        
        return self._to_tensors(df)

    def _to_tensors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Converts DataFrame columns to PyTorch tensors and packages metadata."""
        # Metadata for Model Initialization
        metadata = {
            'n_firm_numeric': len(self.final_firm_numeric),
            'firm_cat_counts': [len(self.encoders[col].classes_) for col in self.cfg.FIRM_CAT_COLS],
            
            'n_ceo_numeric': len(self.final_ceo_numeric),
            'ceo_cat_counts': [len(self.encoders[col].classes_) for col in self.cfg.CEO_CAT_COLS],
        }
        
        # Stack categorical features
        firm_cat_data = np.stack([df[f'{col}_code'].values for col in self.cfg.FIRM_CAT_COLS], axis=1)
        ceo_cat_data = np.stack([df[f'{col}_code'].values for col in self.cfg.CEO_CAT_COLS], axis=1)
        
        tensors = {
            # Firm
            'firm_numeric': torch.tensor(df[self.final_firm_numeric].values, dtype=torch.float32),
            'firm_cat': torch.tensor(firm_cat_data, dtype=torch.long),
            
            # CEO
            'ceo_numeric': torch.tensor(df[self.final_ceo_numeric].values, dtype=torch.float32),
            'ceo_cat': torch.tensor(ceo_cat_data, dtype=torch.long),
            
            # Target & Weights
            'target': torch.tensor(df[self.cfg.TARGET_COL].values, dtype=torch.float32).view(-1, 1),
            'weights': torch.tensor(df['weights'].values, dtype=torch.float32).view(-1, 1),
        }
        
        return {**tensors, **metadata}
        
    def get_feature_names(self) -> List[str]:
        """Returns a flattened list of feature names in the order used by the model wrapper."""
        return (
            self.final_firm_numeric + 
            self.cfg.FIRM_CAT_COLS + 
            self.final_ceo_numeric + 
            self.cfg.CEO_CAT_COLS
        )
    
    def get_flat_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms a DataFrame and returns a flattened numpy array of features.
        
        Layout: [Firm Numeric] [Firm Cat] [CEO Numeric] [CEO Cat]
        
        This is useful for explainability tools (SHAP, PDP) that expect a 2D array.
        
        Args:
            df: DataFrame with required columns (will be transformed using fitted scalers/encoders)
            
        Returns:
            2D numpy array of shape (n_samples, n_features)
        """
        data_dict = self.transform(df)
        return np.hstack([
            data_dict['firm_numeric'].numpy(),
            data_dict['firm_cat'].numpy(),
            data_dict['ceo_numeric'].numpy(),
            data_dict['ceo_cat'].numpy()
        ])

class CEOFirmDataset(Dataset):
    """
    PyTorch Dataset for mini-batch training.
    """
    def __init__(self, data_dict: Dict[str, Any]):
        self.data = data_dict
        self.length = len(data_dict['target'])
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        return {
            'firm_numeric': self.data['firm_numeric'][idx],
            'firm_cat': self.data['firm_cat'][idx],
            'ceo_numeric': self.data['ceo_numeric'][idx],
            'ceo_cat': self.data['ceo_cat'][idx],
            'target': self.data['target'][idx],
            'weights': self.data['weights'][idx]
        }

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
class CEOFirmMatcher(nn.Module):
    """
    Two-Tower Neural Network.
    Encodes CEO and Firm features separately and computes similarity.
    """
    def __init__(self, metadata: Dict[str, int], config: Config):
        super(CEOFirmMatcher, self).__init__()
        
        # --- Embeddings ---
        # Firm Embeddings
        self.firm_embeddings = nn.ModuleList([
            nn.Embedding(n_classes, config.EMBEDDING_DIM_LARGE)
            for n_classes in metadata['firm_cat_counts']
        ])
        
        # CEO Embeddings
        self.ceo_embeddings = nn.ModuleList([
            nn.Embedding(n_classes, config.EMBEDDING_DIM_MEDIUM)
            for n_classes in metadata['ceo_cat_counts']
        ])
        
        # --- TOWER A: FIRM ENCODER ---
        firm_input_dim = metadata['n_firm_numeric'] + len(metadata['firm_cat_counts']) * config.EMBEDDING_DIM_LARGE
        self.firm_tower = nn.Sequential(
            nn.Linear(firm_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, config.LATENT_DIM)
        )
        
        # --- TOWER B: CEO ENCODER ---
        ceo_input_dim = metadata['n_ceo_numeric'] + len(metadata['ceo_cat_counts']) * config.EMBEDDING_DIM_MEDIUM
                         
        self.ceo_tower = nn.Sequential(
            nn.Linear(ceo_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, config.LATENT_DIM)
        )
        
        # Learnable temperature for cosine similarity scaling
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, f_numeric, f_cat, c_numeric, c_cat):
        # Firm Tower
        f_embs = [emb(f_cat[:, i]) for i, emb in enumerate(self.firm_embeddings)]
        f_combined = torch.cat([f_numeric] + f_embs, dim=1)
        u_firm = self.firm_tower(f_combined)
        
        # CEO Tower
        c_embs = [emb(c_cat[:, i]) for i, emb in enumerate(self.ceo_embeddings)]
        c_combined = torch.cat([c_numeric] + c_embs, dim=1)
        v_ceo = self.ceo_tower(c_combined)
        
        # L2 Normalization
        u_firm = u_firm / u_firm.norm(dim=1, keepdim=True)
        v_ceo = v_ceo / v_ceo.norm(dim=1, keepdim=True)
        
        # Match Score (Scaled Dot Product)
        # We use a learnable scale because match_means are not bounded to [-1, 1]
        # Actually, match_means are N(0,1), so they can be > 1. 
        # Standard cosine sim is [-1, 1]. We need to scale it.
        logit_scale = self.logit_scale.exp()
        match_score = (u_firm * v_ceo).sum(dim=1, keepdim=True) * logit_scale
        
        return match_score

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_model(train_loader: DataLoader, val_loader: DataLoader, 
                metadata: Dict[str, int], config: Config) -> Optional[CEOFirmMatcher]:
    
    # Initialize Model
    model = CEOFirmMatcher(metadata, config).to(config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    print(f"Starting training on {config.DEVICE} for {config.EPOCHS} epochs...")
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            preds = model(
                batch['firm_numeric'], batch['firm_cat'],
                batch['ceo_numeric'], batch['ceo_cat']
            )
            
            # Weighted MSE Loss
            loss = (batch['weights'] * (preds - batch['target'])**2).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Avg Train Loss = {avg_loss:.4f}")
            
    return model

# ==========================================
# 5. EXPLAINABILITY (PDP & SHAP)
# ==========================================

class ModelWrapper:
    """
    Wraps the PyTorch model to expose a sklearn-like API (predict taking a single numpy array).
    Needed for SHAP and general interpretation tools.
    
    Input Layout (Flattened):
    [Firm Numeric...] [Firm Cat] [CEO Numeric...] [CEO Gender] [CEO MaxEdu] [CEO Ivy]
    """
    def __init__(self, model: CEOFirmMatcher, processor: DataProcessor):
        self.model = model
        self.processor = processor
        self.device = processor.cfg.DEVICE
        
        # Calculate indices for slicing the flattened input
        self.n_firm_num = len(processor.final_firm_numeric)
        self.n_firm_cat = len(processor.cfg.FIRM_CAT_COLS)
        self.n_ceo_num = len(processor.final_ceo_numeric)
        self.n_ceo_cat = len(processor.cfg.CEO_CAT_COLS)
        
        # Slice Ranges
        self.idx_firm_num_end = self.n_firm_num
        self.idx_firm_cat_end = self.idx_firm_num_end + self.n_firm_cat
        self.idx_ceo_num_end = self.idx_firm_cat_end + self.n_ceo_num
        self.idx_ceo_cat_end = self.idx_ceo_num_end + self.n_ceo_cat
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Slice and Type Cast
        # 1. Firm Numeric
        f_num = X_tensor[:, :self.idx_firm_num_end]
        
        # 2. Firm Cat (Long)
        f_cat = X_tensor[:, self.idx_firm_num_end:self.idx_firm_cat_end].long()
        
        # 3. CEO Numeric
        c_num = X_tensor[:, self.idx_firm_cat_end:self.idx_ceo_num_end]
        
        # 4. CEO Cat (Long)
        c_cat = X_tensor[:, self.idx_ceo_num_end:].long()
        
        with torch.no_grad():
            preds = self.model(f_num, f_cat, c_num, c_cat)
            
        return preds.cpu().numpy().flatten()

def explain_model_pdp(wrapper: ModelWrapper, df: pd.DataFrame, features_to_plot: List[str]):
    """Generates Partial Dependence Plots for specified features.
    
    Args:
        wrapper: ModelWrapper instance with trained model
        df: DataFrame to use for computing PDPs  
        features_to_plot: List of feature names to generate plots for
    """
    print("\nGenerating Partial Dependence Plots (PDP)...")
    
    # Use helper to get flattened feature matrix
    X_flat = wrapper.processor.get_flat_features(df)
    
    feature_names = wrapper.processor.get_feature_names()
    
    # Find indices of features to plot
    indices_to_plot = []
    valid_names = []
    for name in features_to_plot:
        if name in feature_names:
            indices_to_plot.append(feature_names.index(name))
            valid_names.append(name)
        else:
            print(f"Warning: Feature '{name}' not found in model inputs.")
            
    if not indices_to_plot:
        return

    # Manual PDP Loop (Robust for mixed types)
    # Since sklearn PDP can be finicky with mixed types flattened into float matrix, 
    # we'll do a robust 1D sweep.
    
    fig, axes = plt.subplots(1, len(indices_to_plot), figsize=(5 * len(indices_to_plot), 4))
    if len(indices_to_plot) == 1:
        axes = [axes]
        
    for ax, idx, name in zip(axes, indices_to_plot, valid_names):
        # Get range
        vals = X_flat[:, idx]
        # Grid: 50 points
        grid = np.linspace(vals.min(), vals.max(), 50)
        
        pdp_y = []
        # For each grid point, substitute column and predict average
        # Create a batch of size N for each grid point is expensive (N*50 inferences).
        # We can subsample N for speed if needed (e.g. N=1000)
        sample_indices = np.random.choice(X_flat.shape[0], min(1000, X_flat.shape[0]), replace=False)
        X_sample = X_flat[sample_indices].copy()
        
        for val in grid:
            X_temp = X_sample.copy()
            X_temp[:, idx] = val
            preds = wrapper.predict(X_temp)
            pdp_y.append(np.mean(preds))
            
        ax.plot(grid, pdp_y, color='blue')
        ax.set_title(f"PDP: {name}")
        ax.set_xlabel("Standardized Value / Code")
        ax.set_ylabel("Avg Match Score")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    output_dir = wrapper.processor.cfg.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "pdp_plots.svg")
    plt.savefig(save_path)
    print(f"Saved PDP plots to {save_path}")
    plt.close()

def explain_model_shap(wrapper: ModelWrapper, df: pd.DataFrame):
    """Generates SHAP summary plot for feature importance analysis.
    
    Args:
        wrapper: ModelWrapper instance with trained model
        df: DataFrame to use for SHAP analysis
    """
    print("\nCalculating SHAP values (this may take a moment)...")
    
    # Use helper to get flattened feature matrix
    X_flat = wrapper.processor.get_flat_features(df)
    feature_names = wrapper.processor.get_feature_names()
    
    # 2. Background Data (Summary) for KernelExplainer
    # Using K-means to summarize data to 50 points for speed
    X_summary = shap.kmeans(X_flat, 25)
    
    # 3. Explainer
    explainer = shap.KernelExplainer(wrapper.predict, X_summary)
    
    # 4. Calculate SHAP values on a subset of test data
    # Calculating on full dataset is too slow for KernelExplainer
    subset_size = min(200, X_flat.shape[0])
    X_subset = X_flat[:subset_size]
    
    shap_values = explainer.shap_values(X_subset)
    
    # 5. Plot
    plt.figure()
    shap.summary_plot(shap_values, X_subset, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    output_dir = wrapper.processor.cfg.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "shap_summary.svg")
    plt.savefig(save_path)
    print(f"Saved SHAP summary to {save_path}")
    plt.close()

# ==========================================
# 6. VISUALIZATION (Interaction Heatmap)
# ==========================================
def plot_interaction_heatmap(model: CEOFirmMatcher, processor: DataProcessor, 
                             x_feature: str, y_feature: str, filename: str):
    if model is None or processor.processed_df is None:
        return

    print(f"\nGenerating interaction heatmap ({x_feature} vs {y_feature})...")
    
    # Prepare data
    data_dict = processor._to_tensors(processor.processed_df)
    tensor_keys = [k for k in data_dict.keys() if isinstance(data_dict[k], torch.Tensor)]
    device_data = {k: data_dict[k].to(processor.cfg.DEVICE) for k in tensor_keys}
    
    # Helper to find feature index and type
    def get_feature_info(name):
        if name in processor.final_firm_numeric:
            idx = list(processor.scalers['firm'].feature_names_in_).index(name)
            return 'firm_numeric', idx
        elif name in processor.final_ceo_numeric:
            idx = list(processor.scalers['ceo'].feature_names_in_).index(name)
            return 'ceo_numeric', idx
        elif name in processor.cfg.FIRM_CAT_COLS:
            idx = processor.cfg.FIRM_CAT_COLS.index(name)
            return 'firm_cat', idx
        elif name in processor.cfg.CEO_CAT_COLS:
            idx = processor.cfg.CEO_CAT_COLS.index(name)
            return 'ceo_cat', idx
        else:
            raise ValueError(f"Feature {name} not supported for heatmap.")

    try:
        x_type, x_idx = get_feature_info(x_feature)
        y_type, y_idx = get_feature_info(y_feature)
    except ValueError as e:
        print(f"Visualization Error: {e}")
        return

    # Define Grid Range
    def get_range(feat_type, feat_idx):
        if feat_type in ['firm_cat', 'ceo_cat']:
             # Get number of classes for this categorical feature
             if feat_type == 'firm_cat':
                 col_name = processor.cfg.FIRM_CAT_COLS[feat_idx]
             else:
                 col_name = processor.cfg.CEO_CAT_COLS[feat_idx]
             
             n_classes = len(processor.encoders[col_name].classes_)
             # If too many classes, just show first 10 or so? 
             # For interaction plots, usually we want all or a specific subset.
             # Assuming small cardinality for now as per original code (ivy, gender, etc)
             return np.arange(n_classes)
        else:
            return np.linspace(device_data[feat_type][:, feat_idx].min().item(), 
                               device_data[feat_type][:, feat_idx].max().item(), 50)

    x_vals = get_range(x_type, x_idx)
    y_vals = get_range(y_type, y_idx)
    
    heatmap = np.zeros((len(y_vals), len(x_vals)))
    
    # Baselines
    avg_f_numeric = torch.mean(device_data['firm_numeric'], dim=0, keepdim=True)
    avg_c_numeric = torch.mean(device_data['ceo_numeric'], dim=0, keepdim=True)
    
    def calculate_mode(tensor):
        """Helper to calculate mode, handling MPS limitations."""
        if tensor.device.type == 'mps':
            return torch.mode(tensor.cpu(), dim=0)[0].to(tensor.device).view(1, -1)
        return torch.mode(tensor, dim=0)[0].view(1, -1)

    mode_firm_cat = calculate_mode(device_data['firm_cat'])
    mode_ceo_cat = calculate_mode(device_data['ceo_cat'])
    
    model.eval()
    with torch.no_grad():
        for i, y_val in enumerate(y_vals):
            for j, x_val in enumerate(x_vals):
                # Reset inputs to baseline
                f_in = avg_f_numeric.clone()
                c_in = avg_c_numeric.clone()
                f_cat_in = mode_firm_cat.clone()
                c_cat_in = mode_ceo_cat.clone()
                
                def update_input(ftype, fidx, val):
                    if ftype == 'firm_numeric':
                        f_in[:, fidx] = float(val)
                    elif ftype == 'ceo_numeric':
                        c_in[:, fidx] = float(val)
                    elif ftype == 'firm_cat':
                        f_cat_in[:, fidx] = int(val)
                    elif ftype == 'ceo_cat':
                        c_cat_in[:, fidx] = int(val)

                update_input(x_type, x_idx, x_val)
                update_input(y_type, y_idx, y_val)
                
                score = model(f_in, f_cat_in, c_in, c_cat_in)
                heatmap[i, j] = score.item()

    # Determine if axes are categorical
    x_is_cat = x_type in ['firm_cat', 'ceo_cat']
    y_is_cat = y_type in ['firm_cat', 'ceo_cat']
    
    # Plotting - use different rendering based on axis types
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Choose interpolation based on whether we have categorical axes
    if x_is_cat or y_is_cat:
        # No interpolation for categorical - show discrete blocks
        im = ax.imshow(heatmap, aspect='auto', cmap='RdBu_r', origin='lower', 
                       interpolation='nearest')
    else:
        # Smooth interpolation for continuous-continuous
        im = ax.imshow(heatmap, aspect='auto', cmap='RdBu_r', origin='lower',
                       interpolation='bicubic',
                       extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()])
    
    plt.colorbar(im, label='Predicted Match Quality')
    
    # Handle axis labels and ticks
    if x_is_cat:
        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_xticklabels([int(v) for v in x_vals])
        ax.set_xlabel(f'{x_feature} (Category)')
    else:
        if not (x_is_cat or y_is_cat):  # Only set these if using extent
            pass  # extent handles it
        else:
            ax.set_xticks(np.linspace(0, len(x_vals)-1, 5))
            ax.set_xticklabels([f'{v:.1f}' for v in np.linspace(x_vals.min(), x_vals.max(), 5)])
        ax.set_xlabel(f'{x_feature} (Standardized)')
    
    if y_is_cat:
        ax.set_yticks(np.arange(len(y_vals)))
        ax.set_yticklabels([int(v) for v in y_vals])
        ax.set_ylabel(f'{y_feature} (Category)')
    else:
        if not (x_is_cat or y_is_cat):  # Only set these if using extent
            pass  # extent handles it
        else:
            ax.set_yticks(np.linspace(0, len(y_vals)-1, 5))
            ax.set_yticklabels([f'{v:.1f}' for v in np.linspace(y_vals.min(), y_vals.max(), 5)])
        ax.set_ylabel(f'{y_feature} (Standardized)')
    
    ax.set_title(f'Interaction: {x_feature} vs {y_feature}')
    
    output_dir = processor.cfg.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    print(f"Saved heatmap to {path}")
    plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Two Towers Model")
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data for verification')
    args = parser.parse_args()

    config = Config()
    print(f"Running Two Towers Model on {config.DEVICE}")
    
    # 1. Data Processing
    processor = DataProcessor(config)
    
    if args.synthetic:
        print("Using SYNTHETIC data...")
        from synthetic_data import generate_synthetic_data
        raw_df = generate_synthetic_data(1000)
    else:
        raw_df = processor.load_data()
        
    if not raw_df.empty:
        # Step 1: Prepare Features (Stateless)
        df_clean = processor.prepare_features(raw_df)
        
        # Step 2: Split Data
        train_df, val_df = train_test_split(df_clean, test_size=0.2, random_state=42)
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
        
        # Step 3: Fit Scalers on Train
        processor.fit(train_df)
        
        # Step 4: Transform
        train_data = processor.transform(train_df)
        val_data = processor.transform(val_df)
        
        # Step 5: Create Datasets & Loaders
        train_dataset = CEOFirmDataset(train_data)
        val_dataset = CEOFirmDataset(val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        # Metadata from train_data (assuming consistent schema)
        # We need to extract metadata from the dictionary returned by transform
        # But transform returns {**tensors, **metadata}.
        # So we can just use train_data as metadata source.
        
        # 2. Training
        trained_model = train_model(train_loader, val_loader, train_data, config)
        
        if trained_model:
            # 4. Explainability
            # For explainability, we can use the validation set or the full set (transformed)
            # Let's use validation set for speed and correctness (unseen data)
            wrapper = ModelWrapper(trained_model, processor)
            
            # PDP for key features
            # We need to pass a DataFrame to explain_model_pdp, which then calls _to_tensors.
            # But _to_tensors relies on fitted scalers, which is fine (processor is stateful now).
            features_to_plot = config.FIRM_NUMERIC_COLS + config.CEO_NUMERIC_COLS + ['tenure']
            explain_model_pdp(wrapper, val_df, features_to_plot)
            
            # SHAP
            # explain_model_shap(wrapper, val_df)

            # 3. Interaction Plots
            # We need to ensure processor.processed_df is set or passed correctly.
            # processor.transform sets self.processed_df.
            # Let's ensure it's set to something useful for the heatmap (e.g. val_df)
            processor.transform(val_df) 
            
            plot_interaction_heatmap(trained_model, processor, 'logatw', 'Age', 'heatmap_size_age.svg')
            plot_interaction_heatmap(trained_model, processor, 'logatw', 'Output', 'heatmap_size_skill.svg')
            plot_interaction_heatmap(trained_model, processor, 'exp_roa', 'tenure', 'heatmap_perf_exp.svg')
            
            # New Heatmaps
            plot_interaction_heatmap(trained_model, processor, 'rdintw', 'Output', 'heatmap_rd_skill.svg')
            plot_interaction_heatmap(trained_model, processor, 'rdintw', 'Age', 'heatmap_rd_age.svg')
            plot_interaction_heatmap(trained_model, processor, 'logatw', 'ivy', 'heatmap_size_ivy.svg')
            plot_interaction_heatmap(trained_model, processor, 'tenure', 'boardindpw', 'heatmap_tenure_boardind.svg')
            
            plot_interaction_heatmap(trained_model, processor, 'maxedu', 'rdintw', 'heatmap_maxedu_rd.svg')
            plot_interaction_heatmap(trained_model, processor, 'maxedu', 'capintw', 'heatmap_maxedu_capx.svg')
            plot_interaction_heatmap(trained_model, processor, 'logatw', 'm', 'heatmap_size_mover.svg')
            plot_interaction_heatmap(trained_model, processor, 'leverage', 'Age', 'heatmap_leverage_age.svg')
