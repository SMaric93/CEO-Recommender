"""
CEO-Firm Matching: Configuration Module

Centralized configuration for the Two Tower Recommender System.
"""
import torch
from typing import List


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
    CEO_NUMERIC_COLS = ['Age']  # 'tenure' will be derived
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
