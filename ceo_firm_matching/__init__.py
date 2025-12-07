"""
CEO-Firm Matching Package

Two Tower Recommender System for CEO-Firm Matching.
"""
from .config import Config
from .data import DataProcessor, CEOFirmDataset
from .model import CEOFirmMatcher
from .training import train_model
from .explain import ModelWrapper, explain_model_pdp, explain_model_shap
from .visualization import plot_interaction_heatmap
from .synthetic import generate_synthetic_data

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "Config",
    # Data
    "DataProcessor",
    "CEOFirmDataset",
    # Model
    "CEOFirmMatcher",
    # Training
    "train_model",
    # Explainability
    "ModelWrapper",
    "explain_model_pdp",
    "explain_model_shap",
    # Visualization
    "plot_interaction_heatmap",
    # Synthetic Data
    "generate_synthetic_data",
]
