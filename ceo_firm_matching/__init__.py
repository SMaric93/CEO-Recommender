"""
CEO-Firm Matching Package

Two Tower Recommender System for CEO-Firm Matching.
Includes Structural Distillation Network with BLM priors.
"""
from .config import Config
from .data import DataProcessor, CEOFirmDataset
from .model import CEOFirmMatcher
from .training import train_model
from .explain import ModelWrapper, explain_model_pdp, explain_model_shap
from .visualization import plot_interaction_heatmap
from .synthetic import generate_synthetic_data, generate_structural_synthetic_data

# Structural Distillation Network components
from .structural_config import StructuralConfig
from .structural_data import StructuralDataProcessor, DistillationDataset
from .structural_model import StructuralDistillationNet
from .structural_training import train_structural_model
from .structural_explain import IlluminationEngine
from .structural_visualization import (
    plot_type_probability_distributions,
    plot_expected_match_distribution,
    plot_type_confusion_matrix,
    plot_feature_type_correlation,
    create_structural_report,
)

__version__ = "0.2.0"

__all__ = [
    # === Two Tower Model ===
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
    "generate_structural_synthetic_data",
    
    # === Structural Distillation Network ===
    # Configuration
    "StructuralConfig",
    # Data
    "StructuralDataProcessor",
    "DistillationDataset",
    # Model
    "StructuralDistillationNet",
    # Training
    "train_structural_model",
    # Explainability
    "IlluminationEngine",
    # Visualization
    "plot_type_probability_distributions",
    "plot_expected_match_distribution",
    "plot_type_confusion_matrix",
    "plot_feature_type_correlation",
    "create_structural_report",
]

