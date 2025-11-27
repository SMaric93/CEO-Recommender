import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ==========================================
# 0. DEVICE CONFIGURATION
# ==========================================
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
def load_and_preprocess_data(filepath="Data/ceo_types_v0.2.csv"):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

    # --- Define Required Columns ---
    # Identifiers
    id_cols = ['gvkey', 'match_exec_id']
    
    # CEO Variables
    ceo_raw_cols = ['maxedu', 'ivy', 'Age', 'DOB', 'ceo_year', 'Gender', 
                    'Output', 'Throghput', 'Peripheral', 'year_born', 'dep_baby_ceo']
    
    # Firm Variables
    firm_raw_cols = ['compindustry', 'ind_firms_60', 'non_competition_score', 'boardindpw', 
                     'boardsizew', 'busyw', 'pct_blockw', 'logat', 'exp_roa', 'fiscalyear'] # fiscalyear needed for tenure
    
    # Target & Weight
    target_weight_cols = ['match_means', 'sd_match_means']
    
    all_required_cols = id_cols + ceo_raw_cols + firm_raw_cols + target_weight_cols
    
    # --- Check for Missing Columns and Filter DataFrame ---
    missing = [c for c in all_required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing essential columns in CSV: {missing}. Please ensure all required columns are present.")
        return None

    df = df[all_required_cols].copy()
    
    # --- Cleaning: Drop rows with any NaN in required columns ---
    initial_len = len(df)
    df = df.dropna()
    print(f"Dropped {initial_len - len(df)} rows with missing values. Final count: {len(df)}")
    
    # --- Feature Engineering ---
    # Calculate CEO Tenure
    df['tenure'] = df['fiscalyear'] - df['ceo_year']
    df['tenure'] = df['tenure'].clip(lower=0) # Clip negative tenure to 0 for robustness
    
    # --- Categorical Encoding ---
    # Firm Industry
    compindustry_encoder = LabelEncoder()
    df['compindustry_code'] = compindustry_encoder.fit_transform(df['compindustry'].astype(str))
    
    # CEO Gender
    gender_encoder = LabelEncoder()
    df['gender_code'] = gender_encoder.fit_transform(df['Gender'].astype(str))
    
    # Ensure maxedu and ivy are treated as integers for embedding
    df['maxedu'] = df['maxedu'].astype(int)
    df['ivy'] = df['ivy'].astype(int)

    # --- Define Dense Feature Sets ---
    firm_dense_cols = [
        'ind_firms_60', 'non_competition_score', 'boardindpw', 'boardsizew', 
        'busyw', 'pct_blockw', 'logat', 'exp_roa'
    ]
    ceo_dense_cols = [
        'Age', 'Output', 'Throghput', 'Peripheral', 'tenure'
    ]
    
    # --- Standardize Dense Features ---
    scaler_firm = StandardScaler()
    df[firm_dense_cols] = scaler_firm.fit_transform(df[firm_dense_cols])
    
    scaler_ceo = StandardScaler()
    df[ceo_dense_cols] = scaler_ceo.fit_transform(df[ceo_dense_cols])
    
    # --- Calculate Sample Weights ---
    # Weights are inverse of variance (1/sd^2). Add epsilon for stability.
    epsilon = 1e-6
    df['weights'] = 1 / (df['sd_match_means']**2 + epsilon)
    # Normalize weights to sum to 1 to avoid changing overall loss scale too much, if desired.
    # df['weights'] = df['weights'] / df['weights'].sum() 
    
    # --- Convert to Tensors ---
    data = {
        # Identifiers (for potential future use, not directly in model input)
        'gvkey': torch.tensor(df['gvkey'].values, dtype=torch.long),
        'match_exec_id': torch.tensor(df['match_exec_id'].values, dtype=torch.long),

        # Firm Features
        'firm_dense': torch.tensor(df[firm_dense_cols].values, dtype=torch.float32),
        'firm_compindustry': torch.tensor(df['compindustry_code'].values, dtype=torch.long),
        
        # CEO Features
        'ceo_dense': torch.tensor(df[ceo_dense_cols].values, dtype=torch.float32),
        'ceo_gender': torch.tensor(df['gender_code'].values, dtype=torch.long),
        'ceo_maxedu': torch.tensor(df['maxedu'].values, dtype=torch.long),
        'ceo_ivy': torch.tensor(df['ivy'].values, dtype=torch.long),
        
        # Target & Weights
        'target': torch.tensor(df['match_means'].values, dtype=torch.float32).view(-1, 1),
        'weights': torch.tensor(df['weights'].values, dtype=torch.float32).view(-1, 1),
        
        # Metadata for model init (Embedding sizes)
        'n_firm_dense': len(firm_dense_cols),
        'n_firm_compindustry': len(compindustry_encoder.classes_),
        'n_ceo_dense': len(ceo_dense_cols),
        'n_ceo_gender': len(gender_encoder.classes_),
        'n_ceo_maxedu': int(df['maxedu'].max()) + 1, # Max value + 1 for embedding size
        'n_ceo_ivy': int(df['ivy'].max()) + 1,
        
        # For Visualization later (unscaled or raw helpers)
        'raw_df': df, 
        'firm_scaler': scaler_firm,
        'ceo_scaler': scaler_ceo
    }
    
    print("Data processing complete.")
    return data

# ==========================================
# 2. THE TWO-TOWER MODEL ARCHITECTURE
# ==========================================
class CEOFirmMatcher(nn.Module):
    def __init__(self, 
                 n_firm_dense, n_firm_compindustry,
                 n_ceo_dense, n_ceo_gender, n_ceo_maxedu, n_ceo_ivy,
                 latent_dim=32):
        super(CEOFirmMatcher, self).__init__()
        
        # --- Embeddings ---
        self.firm_compindustry_emb = nn.Embedding(n_firm_compindustry, 8) # Embedding size 8
        
        self.ceo_gender_emb = nn.Embedding(n_ceo_gender, 4) # Embedding size 4
        self.ceo_maxedu_emb = nn.Embedding(n_ceo_maxedu, 4) # Embedding size 4
        self.ceo_ivy_emb = nn.Embedding(n_ceo_ivy, 2)       # Embedding size 2
        
        # --- TOWER A: FIRM ENCODER ---
        firm_input_dim = n_firm_dense + 8 # Dense features + compindustry embedding
        self.firm_tower = nn.Sequential(
            nn.Linear(firm_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        
        # --- TOWER B: CEO ENCODER ---
        ceo_input_dim = n_ceo_dense + 4 + 4 + 2 # Dense features + gender + maxedu + ivy embeddings
        self.ceo_tower = nn.Sequential(
            nn.Linear(ceo_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        
    def forward(self, f_dense, f_compindustry, c_dense, c_gender, c_maxedu, c_ivy):
        # Firm Tower
        f_compindustry_vec = self.firm_compindustry_emb(f_compindustry)
        f_combined = torch.cat([f_dense, f_compindustry_vec], dim=1)
        u_firm = self.firm_tower(f_combined)
        
        # CEO Tower
        c_gender_vec = self.ceo_gender_emb(c_gender)
        c_maxedu_vec = self.ceo_maxedu_emb(c_maxedu)
        c_ivy_vec = self.ceo_ivy_emb(c_ivy)
        c_combined = torch.cat([c_dense, c_gender_vec, c_maxedu_vec, c_ivy_vec], dim=1)
        v_ceo = self.ceo_tower(c_combined)
        
        # Match Score (Dot Product)
        match_score = (u_firm * v_ceo).sum(dim=1, keepdim=True)
        return match_score

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_model():
    data = load_and_preprocess_data()
    if data is None:
        return None, None

    # Init Model with dynamic dimensions from loaded data
    model = CEOFirmMatcher(
        n_firm_dense=data['n_firm_dense'], 
        n_firm_compindustry=data['n_firm_compindustry'],
        n_ceo_dense=data['n_ceo_dense'], 
        n_ceo_gender=data['n_ceo_gender'],
        n_ceo_maxedu=data['n_ceo_maxedu'], 
        n_ceo_ivy=data['n_ceo_ivy']
    ).to(DEVICE)
    
    # Move tensors to device
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(DEVICE)
            
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training on {DEVICE}...")
    
    epochs = 20
    for epoch in range(epochs):
        model.train() # Set model to training mode
        optimizer.zero_grad()
        
        preds = model(
            data['firm_dense'], data['firm_compindustry'],
            data['ceo_dense'], data['ceo_gender'], data['ceo_maxedu'], data['ceo_ivy']
        )
        
        # Weighted MSE Loss
        # criterion = (weights * (predictions - target)**2).mean()
        loss = (data['weights'] * (preds - data['target'])**2).mean()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Weighted MSE Loss = {loss.item():.4f}")
            
    return model, data

# ==========================================
# 4. VISUALIZATION
# ==========================================
def plot_interaction_heatmap(model, data):
    if model is None or data is None:
        return

    # We'll visualize: Firm Log(Assets) (logat) vs CEO Age
    
    print("Generating interaction heatmap (Firm Log(Assets) vs CEO Age)...")
    
    # Create Grid for logat and Age
    # Use actual min/max of scaled data for better visualization range
    logat_min, logat_max = data['firm_dense'][:, data['firm_scaler'].feature_names_in_ == 'logat'].min().item(), data['firm_dense'][:, data['firm_scaler'].feature_names_in_ == 'logat'].max().item()
    age_min, age_max = data['ceo_dense'][:, data['ceo_scaler'].feature_names_in_ == 'Age'].min().item(), data['ceo_dense'][:, data['ceo_scaler'].feature_names_in_ == 'Age'].max().item()
    
    logat_vals = np.linspace(logat_min, logat_max, 50)
    age_vals = np.linspace(age_min, age_max, 50)
    
    heatmap = np.zeros((len(age_vals), len(logat_vals)))
    
    # Averages/Modes for other features to hold them constant
    avg_f_dense = torch.mean(data['firm_dense'], dim=0, keepdim=True).to(DEVICE)
    avg_c_dense = torch.mean(data['ceo_dense'], dim=0, keepdim=True).to(DEVICE)
    
    mode_compindustry = torch.mode(data['firm_compindustry'])[0].view(1).to(DEVICE)
    mode_gender = torch.mode(data['ceo_gender'])[0].view(1).to(DEVICE)
    mode_maxedu = torch.mode(data['ceo_maxedu'])[0].view(1).to(DEVICE)
    mode_ivy = torch.mode(data['ceo_ivy'])[0].view(1).to(DEVICE)
    
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        for i, age_val in enumerate(age_vals):
            for j, logat_val in enumerate(logat_vals):
                # Construct Firm Input: modify logat (index based on firm_dense_cols)
                f_dense_input = avg_f_dense.clone()
                logat_idx = data['firm_scaler'].feature_names_in_ == 'logat'
                f_dense_input[:, logat_idx] = float(logat_val)
                
                # Construct CEO Input: modify Age (index based on ceo_dense_cols)
                c_dense_input = avg_c_dense.clone()
                age_idx = data['ceo_scaler'].feature_names_in_ == 'Age'
                c_dense_input[:, age_idx] = float(age_val)
                
                # Predict
                score = model(
                    f_dense_input, mode_compindustry,
                    c_dense_input, mode_gender, mode_maxedu, mode_ivy
                )
                heatmap[i, j] = score.item()
                
    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap, aspect='auto', cmap='RdBu_r', origin='lower',
               extent=[logat_vals.min(), logat_vals.max(), age_vals.min(), age_vals.max()])
    plt.colorbar(label='Predicted Match Quality')
    plt.xlabel('Firm Log(Assets) (Standardized)')
    plt.ylabel('CEO Age (Standardized)')
    plt.title('Interaction: Firm Log(Assets) vs CEO Age')
    plt.show()

if __name__ == "__main__":
    trained_model, processed_data = train_model()
    plot_interaction_heatmap(trained_model, processed_data)
