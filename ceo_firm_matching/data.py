"""
CEO-Firm Matching: Data Processing Module

Handles data loading, feature engineering, encoding, scaling, and PyTorch Dataset creation.
"""
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional

from .config import Config


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
    """PyTorch Dataset for mini-batch training."""
    
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
