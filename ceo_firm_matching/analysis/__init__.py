# Analysis Utilities Subpackage
"""
Centralized analysis functions for CEO-Firm matching.

Modules:
- diagnostics: Basic diagnostic utilities
- ols: OLS regression analysis with robustness
- ml: Machine learning (RF, GBM, SHAP)
- two_towers: Neural network analysis
- visualization: Publication-ready figures
"""
from .diagnostics import (
    run_descriptive_analysis,
    run_regression_analysis,
    run_random_forest,
    stratified_analysis,
)
from .ols import OLSAnalyzer, OLSSpec, OLSResult
from .ml import MLAnalyzer, MLResult
from .two_towers import TwoTowersAnalyzer, TwoTowersResult
from .visualization import (
    plot_coefficient_forest,
    plot_feature_importance,
    plot_model_comparison,
    plot_interaction_heatmap,
    plot_embedding_scatter,
    plot_shap_summary,
    create_dashboard,
    apply_publication_style,
)

__all__ = [
    # Diagnostics
    "run_descriptive_analysis",
    "run_regression_analysis",
    "run_random_forest",
    "stratified_analysis",
    # OLS
    "OLSAnalyzer",
    "OLSSpec",
    "OLSResult",
    # ML
    "MLAnalyzer",
    "MLResult",
    # Two-Towers
    "TwoTowersAnalyzer",
    "TwoTowersResult",
    # Visualization
    "plot_coefficient_forest",
    "plot_feature_importance",
    "plot_model_comparison",
    "plot_interaction_heatmap",
    "plot_embedding_scatter",
    "plot_shap_summary",
    "create_dashboard",
    "apply_publication_style",
]
