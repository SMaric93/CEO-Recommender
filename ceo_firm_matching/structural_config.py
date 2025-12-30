"""
CEO-Firm Matching: Structural Distillation Configuration Module

Centralized configuration for the Structural Distillation Network.
Extends the base Config with BLM-specific structural priors and probability targets.
"""
import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class StructuralConfig:
    """
    Configuration for Structural Distillation Network.
    
    Crucially, this holds the Structural 'Truth' (The Interaction Matrix)
    derived from the BLM estimation in the paper.
    """
    # System
    DEVICE: torch.device = field(default_factory=lambda: torch.device(
        "mps" if torch.backends.mps.is_available() 
        else "cuda" if torch.cuda.is_available() 
        else "cpu"
    ))
    DATA_PATH: str = "Data/blm_posteriors.csv"
    OUTPUT_PATH: str = "./Output/Structural_Distillation"
    
    # Model Hyperparameters
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 256
    DROPOUT: float = 0.2
    LATENT_DIM: int = 128
    EMBEDDING_DIM: int = 8
    
    # --- STRUCTURAL PRIORS (Matrix A) ---
    # The 5x5 Interaction Matrix estimated by the BLM model.
    # Rows = CEO Types, Cols = Firm Classes.
    # REPLACE WITH TABLE 3 ESTIMATES FROM PAPER.
    BLM_INTERACTION_MATRIX: List[List[float]] = field(default_factory=lambda: [
        [-0.5, -0.3,  0.0,  0.1,  0.2],  # Type 1
        [-0.2, -0.1,  0.1,  0.3,  0.4],  # Type 2
        [ 0.0,  0.2,  0.4,  0.6,  0.7],  # Type 3
        [ 0.1,  0.4,  0.7,  0.9,  1.1],  # Type 4
        [ 0.3,  0.6,  0.9,  1.2,  1.5]   # Type 5 (Star)
    ])

    # --- TARGET DEFINITIONS ---
    # Posterior Probabilities from BLM (Soft Labels)
    CEO_PROB_COLS: List[str] = field(default_factory=lambda: [f'prob_ceo_{i+1}' for i in range(5)])
    FIRM_PROB_COLS: List[str] = field(default_factory=lambda: [f'prob_firm_{i+1}' for i in range(5)])

    # --- FEATURE DEFINITIONS ---
    # Aligned with base Config for consistency
    CEO_NUMERIC_COLS: List[str] = field(default_factory=lambda: ['Age', 'tenure'])
    CEO_CAT_COLS: List[str] = field(default_factory=lambda: [
        'Gender', 'maxedu', 'ivy', 'm', 'Output', 'Throghput', 'Peripheral'
    ])
    
    FIRM_NUMERIC_COLS: List[str] = field(default_factory=lambda: [
        'ind_firms_60w', 'non_competition_score', 'boardindpw', 'boardsizew', 
        'busyw', 'pct_blockw', 'logatw', 'exp_roa', 'rdintw', 'capintw', 
        'leverage', 'divyieldw'
    ])
    FIRM_CAT_COLS: List[str] = field(default_factory=lambda: [
        'compindustry', 'ba_state', 'rd_control', 'dpayer'
    ])

    @property
    def all_cols(self) -> List[str]:
        """Helper to grab all necessary columns (excluding derived ones like tenure)."""
        # Note: 'tenure' is derived from fiscalyear - ceo_year
        base_cols = (
            self.CEO_CAT_COLS + 
            self.FIRM_NUMERIC_COLS + 
            self.FIRM_CAT_COLS + 
            self.CEO_PROB_COLS + 
            self.FIRM_PROB_COLS + 
            ['Age', 'fiscalyear', 'ceo_year']  # Age + columns for tenure calculation
        )
        return list(set(base_cols))  # Remove duplicates
