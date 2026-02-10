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

# WRDS Utilities (new in v0.3.0)
from . import wrds
from . import analysis

__version__ = "0.4.0"

# === Extension Modules (v0.4.0) ===
from . import enriched_features       # Extension 1: Enriched CEO Tower
from . import contrastive             # Extension 2: Contrastive Learning
from . import multitask_model         # Extension 4: Multi-Task Learning
from . import temporal_model          # Extension 5: Time-Varying Embeddings
from . import industry_model          # Extension 6: Industry-Specific Matching
from . import network_features        # Extension 8: Board Interlock Network
from . import analytical_extensions   # Extensions 3, 7, 9, 10

__all__ = [
    # === Two Tower Model ===
    "Config",
    "DataProcessor",
    "CEOFirmDataset",
    "CEOFirmMatcher",
    "train_model",
    "ModelWrapper",
    "explain_model_pdp",
    "explain_model_shap",
    "plot_interaction_heatmap",
    "generate_synthetic_data",
    "generate_structural_synthetic_data",

    # === Structural Distillation Network ===
    "StructuralConfig",
    "StructuralDataProcessor",
    "DistillationDataset",
    "StructuralDistillationNet",
    "train_structural_model",
    "IlluminationEngine",
    "plot_type_probability_distributions",
    "plot_expected_match_distribution",
    "plot_type_confusion_matrix",
    "plot_feature_type_correlation",
    "create_structural_report",

    # === Extensions (v0.4.0) ===
    "enriched_features",
    "contrastive",
    "multitask_model",
    "temporal_model",
    "industry_model",
    "network_features",
    "analytical_extensions",
]

