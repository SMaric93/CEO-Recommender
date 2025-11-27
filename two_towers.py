import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Tuple, Any, List, Optional

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
    
    # Hyperparameters
    EPOCHS = 20
    LEARNING_RATE = 0.001
    LATENT_DIM = 32
    
    # Embedding Dimensions
    EMBEDDING_DIM_SMALL = 2
    EMBEDDING_DIM_MEDIUM = 4
    EMBEDDING_DIM_LARGE = 8
    
    # Feature Definitions
    ID_COLS = ['gvkey', 'match_exec_id']
    
    # CEO Features
    CEO_NUMERIC_COLS = ['Age', 'Output', 'Throghput', 'Peripheral'] # 'tenure' will be derived
    CEO_CAT_COLS = ['Gender', 'maxedu', 'ivy']
    CEO_RAW_COLS = CEO_NUMERIC_COLS + CEO_CAT_COLS + ['ceo_year', 'year_born', 'dep_baby_ceo', 'DOB']
    
    # Firm Features
    FIRM_NUMERIC_COLS = ['ind_firms_60', 'non_competition_score', 'boardindpw', 
                         'boardsizew', 'busyw', 'pct_blockw', 'logat', 'exp_roa']
    FIRM_CAT_COLS = ['compindustry']
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

    def preprocess(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Main pipeline: Clean -> Engineer -> Encode -> Scale -> Tensorize."""
        if df.empty:
            return {}
            
        # 1. Cleaning
        initial_len = len(df)
        df = df.dropna()
        print(f"Dropped {initial_len - len(df)} rows with NaNs. Final count: {len(df)}")
        
        # 2. Feature Engineering
        df['tenure'] = (df['fiscalyear'] - df['ceo_year']).clip(lower=0)
        
        # 3. Categorical Encoding
        # Firm
        self.encoders['compindustry'] = LabelEncoder()
        df['compindustry_code'] = self.encoders['compindustry'].fit_transform(df['compindustry'].astype(str))
        
        # CEO
        self.encoders['Gender'] = LabelEncoder()
        df['gender_code'] = self.encoders['Gender'].fit_transform(df['Gender'].astype(str))
        
        # Ensure numerical categories are integers
        df['maxedu'] = df['maxedu'].astype(int)
        df['ivy'] = df['ivy'].astype(int)
        
        # 4. Scaling Numeric Features
        df[self.final_firm_numeric] = self.scalers['firm'].fit_transform(df[self.final_firm_numeric])
        df[self.final_ceo_numeric] = self.scalers['ceo'].fit_transform(df[self.final_ceo_numeric])
        
        # 5. Weight Calculation
        epsilon = 1e-6
        df['weights'] = 1 / (df[self.cfg.WEIGHT_COL]**2 + epsilon)
        
        # Store processed DF for visualization/debugging
        self.processed_df = df
        
        return self._to_tensors(df)

    def _to_tensors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Converts DataFrame columns to PyTorch tensors and packages metadata."""
        # Metadata for Model Initialization
        metadata = {
            'n_firm_numeric': len(self.final_firm_numeric),
            'n_firm_compindustry': len(self.encoders['compindustry'].classes_),
            
            'n_ceo_numeric': len(self.final_ceo_numeric),
            'n_ceo_gender': len(self.encoders['Gender'].classes_),
            'n_ceo_maxedu': int(df['maxedu'].max()) + 1,
            'n_ceo_ivy': int(df['ivy'].max()) + 1,
        }
        
        tensors = {
            # Firm
            'firm_numeric': torch.tensor(df[self.final_firm_numeric].values, dtype=torch.float32),
            'firm_compindustry': torch.tensor(df['compindustry_code'].values, dtype=torch.long),
            
            # CEO
            'ceo_numeric': torch.tensor(df[self.final_ceo_numeric].values, dtype=torch.float32),
            'ceo_gender': torch.tensor(df['gender_code'].values, dtype=torch.long),
            'ceo_maxedu': torch.tensor(df['maxedu'].values, dtype=torch.long),
            'ceo_ivy': torch.tensor(df['ivy'].values, dtype=torch.long),
            
            # Target & Weights
            'target': torch.tensor(df[self.cfg.TARGET_COL].values, dtype=torch.float32).view(-1, 1),
            'weights': torch.tensor(df['weights'].values, dtype=torch.float32).view(-1, 1),
        }
        
        return {**tensors, **metadata}
        
    def get_feature_names(self) -> List[str]:
        """Returns a flattened list of feature names in the order used by the model wrapper."""
        return (
            self.final_firm_numeric + 
            ['compindustry'] + 
            self.final_ceo_numeric + 
            ['Gender', 'maxedu', 'ivy']
        )

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
        self.firm_compindustry_emb = nn.Embedding(metadata['n_firm_compindustry'], config.EMBEDDING_DIM_LARGE)
        
        self.ceo_gender_emb = nn.Embedding(metadata['n_ceo_gender'], config.EMBEDDING_DIM_MEDIUM)
        self.ceo_maxedu_emb = nn.Embedding(metadata['n_ceo_maxedu'], config.EMBEDDING_DIM_MEDIUM)
        self.ceo_ivy_emb = nn.Embedding(metadata['n_ceo_ivy'], config.EMBEDDING_DIM_SMALL)
        
        # --- TOWER A: FIRM ENCODER ---
        firm_input_dim = metadata['n_firm_numeric'] + config.EMBEDDING_DIM_LARGE
        self.firm_tower = nn.Sequential(
            nn.Linear(firm_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config.LATENT_DIM)
        )
        
        # --- TOWER B: CEO ENCODER ---
        ceo_input_dim = (metadata['n_ceo_numeric'] + 
                         config.EMBEDDING_DIM_MEDIUM +  # Gender
                         config.EMBEDDING_DIM_MEDIUM +  # MaxEdu
                         config.EMBEDDING_DIM_SMALL)    # Ivy
                         
        self.ceo_tower = nn.Sequential(
            nn.Linear(ceo_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config.LATENT_DIM)
        )
        
    def forward(self, f_numeric, f_compindustry, c_numeric, c_gender, c_maxedu, c_ivy):
        # Firm Tower
        f_emb = self.firm_compindustry_emb(f_compindustry)
        f_combined = torch.cat([f_numeric, f_emb], dim=1)
        u_firm = self.firm_tower(f_combined)
        
        # CEO Tower
        c_g_emb = self.ceo_gender_emb(c_gender)
        c_e_emb = self.ceo_maxedu_emb(c_maxedu)
        c_i_emb = self.ceo_ivy_emb(c_ivy)
        c_combined = torch.cat([c_numeric, c_g_emb, c_e_emb, c_i_emb], dim=1)
        v_ceo = self.ceo_tower(c_combined)
        
        # Match Score (Dot Product)
        match_score = (u_firm * v_ceo).sum(dim=1, keepdim=True)
        return match_score

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_model(data: Dict[str, Any], config: Config) -> Optional[CEOFirmMatcher]:
    if not data:
        return None

    # Initialize Model
    model = CEOFirmMatcher(data, config).to(config.DEVICE)
    
    # Move Tensors to Device
    tensor_keys = [k for k in data.keys() if isinstance(data[k], torch.Tensor)]
    model_data = {k: data[k].to(config.DEVICE) for k in tensor_keys}
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    print(f"Starting training on {config.DEVICE} for {config.EPOCHS} epochs...")
    
    for epoch in range(config.EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        preds = model(
            model_data['firm_numeric'], model_data['firm_compindustry'],
            model_data['ceo_numeric'], model_data['ceo_gender'], 
            model_data['ceo_maxedu'], model_data['ceo_ivy']
        )
        
        # Weighted MSE Loss
        loss = (model_data['weights'] * (preds - model_data['target'])**2).mean()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Weighted MSE Loss = {loss.item():.4f}")
            
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
        self.n_ceo_num = len(processor.final_ceo_numeric)
        
        # Slice Ranges
        self.idx_firm_num_end = self.n_firm_num
        self.idx_firm_cat_end = self.idx_firm_num_end + 1
        self.idx_ceo_num_end = self.idx_firm_cat_end + self.n_ceo_num
        self.idx_gender_end = self.idx_ceo_num_end + 1
        self.idx_maxedu_end = self.idx_gender_end + 1
        # Ivy is the rest
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Slice and Type Cast
        # 1. Firm Numeric
        f_num = X_tensor[:, :self.idx_firm_num_end]
        
        # 2. Firm Cat (Long)
        f_cat = X_tensor[:, self.idx_firm_num_end:self.idx_firm_cat_end].long().squeeze(1)
        
        # 3. CEO Numeric
        c_num = X_tensor[:, self.idx_firm_cat_end:self.idx_ceo_num_end]
        
        # 4. CEO Gender (Long)
        c_gen = X_tensor[:, self.idx_ceo_num_end:self.idx_gender_end].long().squeeze(1)
        
        # 5. CEO MaxEdu (Long)
        c_edu = X_tensor[:, self.idx_gender_end:self.idx_maxedu_end].long().squeeze(1)
        
        # 6. CEO Ivy (Long)
        c_ivy = X_tensor[:, self.idx_maxedu_end:].long().squeeze(1)
        
        with torch.no_grad():
            preds = self.model(f_num, f_cat, c_num, c_gen, c_edu, c_ivy)
            
        return preds.cpu().numpy().flatten()

def explain_model_pdp(wrapper: ModelWrapper, df: pd.DataFrame, features_to_plot: List[str]):
    """Generates Partial Dependence Plots for specified features."""
    print("\nGenerating Partial Dependence Plots (PDP)...")
    
    # Prepare Data Matrix
    # We need to construct the flattened matrix that ModelWrapper expects from the processed DF
    # Fortunately, processor._to_tensors gives us the components, we just need to concat them numpy-style
    
    # We can use the raw df, process it again to get arrays (cleaner)
    data_dict = wrapper.processor._to_tensors(df)
    
    # Concatenate in order: FirmNum, FirmCat, CEONum, CEOGen, CEOEdu, CEOIvy
    # Note: Tensors are on CPU by default in _to_tensors
    X_flat = np.hstack([
        data_dict['firm_numeric'].numpy(),
        data_dict['firm_compindustry'].numpy().reshape(-1, 1),
        data_dict['ceo_numeric'].numpy(),
        data_dict['ceo_gender'].numpy().reshape(-1, 1),
        data_dict['ceo_maxedu'].numpy().reshape(-1, 1),
        data_dict['ceo_ivy'].numpy().reshape(-1, 1)
    ])
    
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
    plt.show()

def explain_model_shap(wrapper: ModelWrapper, df: pd.DataFrame):
    """Generates SHAP summary plot."""
    print("\nCalculating SHAP values (this may take a moment)...")
    
    # 1. Prepare Data
    data_dict = wrapper.processor._to_tensors(df)
    X_flat = np.hstack([
        data_dict['firm_numeric'].numpy(),
        data_dict['firm_compindustry'].numpy().reshape(-1, 1),
        data_dict['ceo_numeric'].numpy(),
        data_dict['ceo_gender'].numpy().reshape(-1, 1),
        data_dict['ceo_maxedu'].numpy().reshape(-1, 1),
        data_dict['ceo_ivy'].numpy().reshape(-1, 1)
    ])
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
    plt.show()

# ==========================================
# 6. VISUALIZATION (Interaction Heatmap)
# ==========================================
def plot_interaction_heatmap(model: CEOFirmMatcher, processor: DataProcessor):
    if model is None or processor.processed_df is None:
        return

    print("\nGenerating interaction heatmap (Firm Log(Assets) vs CEO Age)...")
    
    # Prepare data for evaluation (reuse the processor to get tensors)
    # Note: We generate fresh tensors from the dataframe to ensure we have valid references
    # even if the 'data' dict from training was garbage collected or modified.
    data_dict = processor._to_tensors(processor.processed_df)
    tensor_keys = [k for k in data_dict.keys() if isinstance(data_dict[k], torch.Tensor)]
    device_data = {k: data_dict[k].to(processor.cfg.DEVICE) for k in tensor_keys}
    
    # Get Indices for features to vary
    try:
        logat_idx = list(processor.scalers['firm'].feature_names_in_).index('logat')
        age_idx = list(processor.scalers['ceo'].feature_names_in_).index('Age')
    except (ValueError, KeyError) as e:
        print(f"Visualization Error: Feature not found in scaler - {e}")
        return

    # Define Grid Range based on actual data distribution
    # Using min/max of the scaled data
    logat_vals = np.linspace(device_data['firm_numeric'][:, logat_idx].min().item(), 
                             device_data['firm_numeric'][:, logat_idx].max().item(), 50)
    age_vals = np.linspace(device_data['ceo_numeric'][:, age_idx].min().item(), 
                           device_data['ceo_numeric'][:, age_idx].max().item(), 50)
    
    heatmap = np.zeros((len(age_vals), len(logat_vals)))
    
    # Calculate Baselines (Averages/Modes) to hold other features constant
    avg_f_numeric = torch.mean(device_data['firm_numeric'], dim=0, keepdim=True)
    avg_c_numeric = torch.mean(device_data['ceo_numeric'], dim=0, keepdim=True)
    
    mode_compindustry = torch.mode(device_data['firm_compindustry'])[0].view(1)
    mode_gender = torch.mode(device_data['ceo_gender'])[0].view(1)
    mode_maxedu = torch.mode(device_data['ceo_maxedu'])[0].view(1)
    mode_ivy = torch.mode(device_data['ceo_ivy'])[0].view(1)
    
    model.eval()
    with torch.no_grad():
        for i, age_val in enumerate(age_vals):
            for j, logat_val in enumerate(logat_vals):
                # Construct Firm Input
                f_in = avg_f_numeric.clone()
                f_in[:, logat_idx] = float(logat_val)
                
                # Construct CEO Input
                c_in = avg_c_numeric.clone()
                c_in[:, age_idx] = float(age_val)
                
                score = model(
                    f_in, mode_compindustry,
                    c_in, mode_gender, mode_maxedu, mode_ivy
                )
                heatmap[i, j] = score.item()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap, aspect='auto', cmap='RdBu_r', origin='lower',
               extent=[logat_vals.min(), logat_vals.max(), age_vals.min(), age_vals.max()])
    plt.colorbar(label='Predicted Match Quality')
    plt.xlabel('Firm Log(Assets) (Standardized)')
    plt.ylabel('CEO Age (Standardized)')
    plt.title('Interaction: Firm Log(Assets) vs CEO Age')
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    config = Config()
    print(f"Running Two Towers Model on {config.DEVICE}")
    
    # 1. Data Processing
    processor = DataProcessor(config)
    raw_df = processor.load_data()
    processed_data = processor.preprocess(raw_df)
    
    # 2. Training
    if processed_data:
        trained_model = train_model(processed_data, config)
        
        if trained_model:
            # 3. Interaction Plot
            plot_interaction_heatmap(trained_model, processor)
            
            # 4. Explainability
            wrapper = ModelWrapper(trained_model, processor)
            
            # PDP for key features
            explain_model_pdp(wrapper, processor.processed_df, 
                              ['logat', 'Age', 'tenure', 'exp_roa', 'non_competition_score'])
            
            # SHAP
            explain_model_shap(wrapper, processor.processed_df)
