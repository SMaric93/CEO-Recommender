"""
CEO-Firm Matching: Structural Distillation Data Processing Module

Handles data loading, feature engineering, encoding, scaling, and PyTorch Dataset creation
specifically for the Structural Distillation Network with probability targets.
"""
import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple, Optional

from .structural_config import StructuralConfig
from .synthetic import generate_synthetic_data


class StructuralDataProcessor:
    """
    Handles feature scaling and prepares probability targets for Structural Distillation.
    Ensures targets are valid probability distributions (sum=1).
    
    Designed to follow the same patterns as DataProcessor but with probability targets
    instead of match_means regression targets.
    """
    def __init__(self, config: StructuralConfig):
        self.cfg = config
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {
            'firm': StandardScaler(),
            'ceo': StandardScaler()
        }
        
        # Define final feature sets (post-engineering)
        # Note: tenure is derived, Age is in CEO_NUMERIC_COLS
        self.final_ceo_numeric = self.cfg.CEO_NUMERIC_COLS  # ['Age', 'tenure']
        self.final_firm_numeric = self.cfg.FIRM_NUMERIC_COLS

    def load_and_prep(self) -> Tuple[Dataset, Dataset, pd.DataFrame]:
        """Full pipeline: Load -> Clean -> Engineer -> Split -> Tensorize"""
        if not os.path.exists(self.cfg.DATA_PATH):
            print(f"Warning: Data not found at {self.cfg.DATA_PATH}. Generating SYNTHETIC data.")
            df = self._generate_synthetic()
        else:
            df = pd.read_csv(self.cfg.DATA_PATH)

        # 1. Feature Engineering (Tenure)
        if 'tenure' not in df.columns and 'fiscalyear' in df.columns and 'ceo_year' in df.columns:
            df['tenure'] = (df['fiscalyear'] - df['ceo_year']).clip(lower=0)

        # 2. Cleaning - check for required columns
        req_cols = (
            self.cfg.CEO_CAT_COLS + 
            self.cfg.FIRM_NUMERIC_COLS + 
            self.cfg.FIRM_CAT_COLS + 
            self.cfg.CEO_PROB_COLS + 
            self.cfg.FIRM_PROB_COLS
        )
        # Age and tenure handled separately since tenure is derived
        if 'Age' in df.columns:
            req_cols = req_cols + ['Age']
        
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            print(f"Warning: Missing columns: {missing}. Generating synthetic data instead.")
            df = self._generate_synthetic()
            if 'tenure' not in df.columns:
                df['tenure'] = (df['fiscalyear'] - df['ceo_year']).clip(lower=0)
        
        df = df.dropna(subset=[c for c in req_cols if c in df.columns]).reset_index(drop=True)

        # 3. Normalize Posteriors (Critical for KL Div stability)
        c_probs = df[self.cfg.CEO_PROB_COLS].values
        df[self.cfg.CEO_PROB_COLS] = c_probs / (c_probs.sum(axis=1, keepdims=True) + 1e-9)
        
        f_probs = df[self.cfg.FIRM_PROB_COLS].values
        df[self.cfg.FIRM_PROB_COLS] = f_probs / (f_probs.sum(axis=1, keepdims=True) + 1e-9)

        # 4. Split
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        # 5. Fit Transformers (Train Only)
        self._fit_transformers(train_df)

        # 6. Transform
        train_data = self._transform(train_df)
        val_data = self._transform(val_df)

        return DistillationDataset(train_data), DistillationDataset(val_data), val_df

    def _fit_transformers(self, df: pd.DataFrame):
        """Fit encoders and scalers on training data only."""
        # Categorical Encoding
        for col in self.cfg.FIRM_CAT_COLS + self.cfg.CEO_CAT_COLS:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(df[col].astype(str))
        
        # Numeric Scaling
        self.scalers['firm'].fit(df[self.final_firm_numeric])
        self.scalers['ceo'].fit(df[self.final_ceo_numeric])

    def _transform(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Apply transformations and return tensor dictionary."""
        df = df.copy()
        
        # Safe Encode function
        def encode(enc: LabelEncoder, series: pd.Series) -> np.ndarray:
            known_classes = set(enc.classes_)
            return series.astype(str).apply(
                lambda x: enc.transform([x])[0] if x in known_classes else 0
            ).values

        # Encode categoricals
        f_cat = np.stack([
            encode(self.encoders[c], df[c]) 
            for c in self.cfg.FIRM_CAT_COLS
        ], axis=1)
        
        c_cat = np.stack([
            encode(self.encoders[c], df[c]) 
            for c in self.cfg.CEO_CAT_COLS
        ], axis=1)
        
        # Scale numerics
        f_num = self.scalers['firm'].transform(df[self.final_firm_numeric])
        c_num = self.scalers['ceo'].transform(df[self.final_ceo_numeric])

        return {
            'firm_num': torch.tensor(f_num, dtype=torch.float32),
            'firm_cat': torch.tensor(f_cat, dtype=torch.long),
            'ceo_num': torch.tensor(c_num, dtype=torch.float32),
            'ceo_cat': torch.tensor(c_cat, dtype=torch.long),
            'target_ceo': torch.tensor(df[self.cfg.CEO_PROB_COLS].values, dtype=torch.float32),
            'target_firm': torch.tensor(df[self.cfg.FIRM_PROB_COLS].values, dtype=torch.float32)
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata needed for model initialization."""
        return {
            'n_firm_num': len(self.final_firm_numeric),
            'n_ceo_num': len(self.final_ceo_numeric),
            'firm_cat_cards': [len(self.encoders[c].classes_) for c in self.cfg.FIRM_CAT_COLS],
            'ceo_cat_cards': [len(self.encoders[c].classes_) for c in self.cfg.CEO_CAT_COLS]
        }
    
    def get_feature_names(self) -> list:
        """Returns a flattened list of feature names for interpretation."""
        return (
            self.final_firm_numeric + 
            list(self.cfg.FIRM_CAT_COLS) + 
            self.final_ceo_numeric + 
            list(self.cfg.CEO_CAT_COLS)
        )

    def _generate_synthetic(self, n: int = 2000) -> pd.DataFrame:
        """
        Generates mock data consistent with the structural distillation schema.
        Uses the dedicated structural synthetic data generator.
        """
        from .synthetic import generate_structural_synthetic_data
        return generate_structural_synthetic_data(n_samples=n)


class DistillationDataset(Dataset):
    """PyTorch Dataset for Structural Distillation mini-batch training."""
    
    def __init__(self, data_dict: Dict[str, torch.Tensor]):
        self.data = data_dict
        self.length = len(data_dict['firm_num'])
        
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.data.items()}
