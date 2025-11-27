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

    # --- Select Relevant Columns ---
    # Firm Features: 
    #   Dense: logat (Size), rdint (R&D), leverage, roa
    #   Categorical: ind (Industry)
    # CEO Features:
    #   Dense: Age
    #   Derived: Tenure (fiscalyear - ceo_year)
    #   Categorical: Gender, maxedu, ivy
    # Target: match_means (Posterior Match Quality)
    
    required_cols = [
        'logat', 'rdint', 'leverage', 'roa', 'ind',         # Firm
        'Age', 'ceo_year', 'fiscalyear', 'Gender', 'maxedu', 'ivy', # CEO
        'match_means'                                       # Target
    ]
    
    # Check for missing columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns in CSV: {missing}")
        # Drop missing from required to attempt loading partially or raise error
        # For now, we strictly require them.
        return None

    df = df[required_cols].copy()
    
    # --- Cleaning ---
    # Drop rows with NaNs in critical columns
    initial_len = len(df)
    df = df.dropna()
    print(f"Dropped {initial_len - len(df)} rows with missing values. Final count: {len(df)}")
    
    # --- Feature Engineering ---
    # Tenure
    df['tenure'] = df['fiscalyear'] - df['ceo_year']
    # Clip negative tenure (data errors) to 0
    df['tenure'] = df['tenure'].clip(lower=0)
    
    # --- Encoders (Categorical) ---
    # Industry
    ind_encoder = LabelEncoder()
    df['ind_code'] = ind_encoder.fit_transform(df['ind'].astype(str))
    
    # Gender
    gender_encoder = LabelEncoder()
    df['gender_code'] = gender_encoder.fit_transform(df['Gender'].astype(str))
    
    # MaxEdu & Ivy (ensure they are safe integers for Embedding)
    # Assuming maxedu is 1-based or similar, we map to 0-based if needed, 
    # but if it's already small ints, just ensuring int type is enough.
    # We'll assume maxedu and ivy are already integer codes or simple floats.
    df['maxedu'] = df['maxedu'].astype(int)
    df['ivy'] = df['ivy'].astype(int)
    
    # --- Scalers (Dense) ---
    firm_dense_cols = ['logat', 'rdint', 'leverage', 'roa']
    ceo_dense_cols = ['Age', 'tenure']
    
    scaler_firm = StandardScaler()
    df[firm_dense_cols] = scaler_firm.fit_transform(df[firm_dense_cols])
    
    scaler_ceo = StandardScaler()
    df[ceo_dense_cols] = scaler_ceo.fit_transform(df[ceo_dense_cols])
    
    # --- To Tensors ---
    # We create a dictionary to hold the dataset tensors and metadata
    data = {
        # Features
        'firm_dense': torch.tensor(df[firm_dense_cols].values, dtype=torch.float32),
        'firm_ind': torch.tensor(df['ind_code'].values, dtype=torch.long),
        
        'ceo_dense': torch.tensor(df[ceo_dense_cols].values, dtype=torch.float32),
        'ceo_gender': torch.tensor(df['gender_code'].values, dtype=torch.long),
        'ceo_edu': torch.tensor(df['maxedu'].values, dtype=torch.long),
        'ceo_ivy': torch.tensor(df['ivy'].values, dtype=torch.long),
        
        # Target
        'target': torch.tensor(df['match_means'].values, dtype=torch.float32).view(-1, 1),
        
        # Metadata (Dimensions for Model)
        'n_firm_ind': len(ind_encoder.classes_),
        'n_ceo_gender': len(gender_encoder.classes_),
        'n_ceo_edu': int(df['maxedu'].max()) + 1,
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
                 n_firm_dense=4, n_firm_ind=10,
                 n_ceo_dense=2, n_ceo_gender=2, n_ceo_edu=10, n_ceo_ivy=2,
                 latent_dim=32):
        super(CEOFirmMatcher, self).__init__()
        
        # --- Embeddings ---
        self.firm_ind_emb = nn.Embedding(n_firm_ind, 8)
        
        self.ceo_gender_emb = nn.Embedding(n_ceo_gender, 4)
        self.ceo_edu_emb = nn.Embedding(n_ceo_edu, 4)
        self.ceo_ivy_emb = nn.Embedding(n_ceo_ivy, 2)
        
        # --- TOWER A: FIRM ENCODER ---
        # Input = Dense(4) + Ind_Emb(8) = 12
        self.firm_tower = nn.Sequential(
            nn.Linear(n_firm_dense + 8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        
        # --- TOWER B: CEO ENCODER ---
        # Input = Dense(2) + Gender(4) + Edu(4) + Ivy(2) = 12
        self.ceo_tower = nn.Sequential(
            nn.Linear(n_ceo_dense + 4 + 4 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        
    def forward(self, f_dense, f_ind, c_dense, c_gender, c_edu, c_ivy):
        # Firm Tower
        f_ind_vec = self.firm_ind_emb(f_ind)
        f_combined = torch.cat([f_dense, f_ind_vec], dim=1)
        u_firm = self.firm_tower(f_combined)
        
        # CEO Tower
        c_g_vec = self.ceo_gender_emb(c_gender)
        c_e_vec = self.ceo_edu_emb(c_edu)
        c_i_vec = self.ceo_ivy_emb(c_ivy)
        c_combined = torch.cat([c_dense, c_g_vec, c_e_vec, c_i_vec], dim=1)
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

    # Init Model
    model = CEOFirmMatcher(
        n_firm_ind=data['n_firm_ind'],
        n_ceo_gender=data['n_ceo_gender'],
        n_ceo_edu=data['n_ceo_edu'],
        n_ceo_ivy=data['n_ceo_ivy']
    ).to(DEVICE)
    
    # Move tensors to device
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(DEVICE)
            
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training on {DEVICE}...")
    
    epochs = 20
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        preds = model(
            data['firm_dense'], data['firm_ind'],
            data['ceo_dense'], data['ceo_gender'], data['ceo_edu'], data['ceo_ivy']
        )
        
        loss = criterion(preds, data['target'])
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: MSE Loss = {loss.item():.4f}")
            
    return model, data

# ==========================================
# 4. VISUALIZATION
# ==========================================
def plot_interaction_heatmap(model, data):
    if model is None or data is None:
        return

    # We'll visualize: Firm R&D (rdint) vs CEO Education (maxedu)
    # Assuming interaction: High R&D firms might match better with higher edu?
    
    print("Generating interaction heatmap (Firm R&D vs CEO Edu)...")
    
    # Create Grid
    rd_vals = np.linspace(-2, 2, 50) # Standardized range (approx -2 to +2 sigma)
    edu_levels = np.unique(data['raw_df']['maxedu']) # Use actual education levels found
    edu_levels = np.sort(edu_levels)
    
    heatmap = np.zeros((len(edu_levels), len(rd_vals)))
    
    # Averages for other features (to hold them constant)
    # We need these as 1-element tensors on DEVICE
    avg_f_dense = torch.mean(data['firm_dense'], dim=0, keepdim=True) # [1, 4]
    avg_c_dense = torch.mean(data['ceo_dense'], dim=0, keepdim=True) # [1, 2]
    
    # Most common/mode for categoricals
    mode_ind = torch.mode(data['firm_ind'])[0].view(1)
    mode_gender = torch.mode(data['ceo_gender'])[0].view(1)
    mode_ivy = torch.mode(data['ceo_ivy'])[0].view(1)
    
    model.eval()
    with torch.no_grad():
        for i, edu in enumerate(edu_levels):
            for j, rd in enumerate(rd_vals):
                # Construct Firm Input: modify R&D (index 1 in dense: logat, rdint, lev, roa)
                f_dense_input = avg_f_dense.clone()
                f_dense_input[0, 1] = float(rd) 
                
                # Construct CEO Input: modify Edu
                c_edu_input = torch.tensor([edu], dtype=torch.long).to(DEVICE)
                
                # Predict
                score = model(
                    f_dense_input, mode_ind,
                    avg_c_dense, mode_gender, c_edu_input, mode_ivy
                )
                heatmap[i, j] = score.item()
                
    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap, aspect='auto', cmap='RdBu_r', origin='lower',
               extent=[rd_vals.min(), rd_vals.max(), edu_levels.min(), edu_levels.max()])
    plt.colorbar(label='Predicted Match Quality')
    plt.xlabel('Firm R&D Intensity (Standardized)')
    plt.ylabel('CEO Education Level')
    plt.title('Interaction: Firm R&D vs CEO Education')
    plt.show()

if __name__ == "__main__":
    trained_model, processed_data = train_model()
    plot_interaction_heatmap(trained_model, processed_data)

