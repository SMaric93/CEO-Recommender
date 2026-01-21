"""
Visualization Module for Analysis Framework.

Publication-ready visualizations:
- Coefficient plots with confidence intervals
- SHAP summary beeswarm plots
- Feature importance bar charts
- Model comparison bar charts
- Two-Towers interaction heatmaps
- Embedding scatter plots
"""
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# Publication style settings
PUBLICATION_STYLE = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}


def apply_publication_style():
    """Apply publication-ready matplotlib style."""
    if MATPLOTLIB_AVAILABLE:
        plt.rcParams.update(PUBLICATION_STYLE)


def plot_coefficient_forest(
    results: Dict[str, Any],
    variables: List[str],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 10),
    title: str = "Coefficient Estimates by Specification"
) -> plt.Figure:
    """
    Forest plot of coefficients with confidence intervals.
    
    Args:
        results: Dict of OLSResult objects
        variables: Variables to plot
        output_path: Path to save figure
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required")
    
    apply_publication_style()
    
    n_vars = len(variables)
    n_specs = len(results)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_specs))
    y_positions = np.arange(n_vars)
    height = 0.8 / n_specs
    
    for i, (spec_name, result) in enumerate(results.items()):
        coefs = []
        ci_lows = []
        ci_highs = []
        
        for var in variables:
            if var in result.coefficients:
                coefs.append(result.coefficients[var])
                ci_lows.append(result.conf_int_low.get(var, 0))
                ci_highs.append(result.conf_int_high.get(var, 0))
            else:
                coefs.append(np.nan)
                ci_lows.append(np.nan)
                ci_highs.append(np.nan)
        
        coefs = np.array(coefs)
        errors = np.array([[c - l, h - c] for c, l, h in zip(coefs, ci_lows, ci_highs)]).T
        
        y = y_positions + i * height - 0.4 + height/2
        
        ax.errorbar(
            coefs, y,
            xerr=errors,
            fmt='o',
            color=colors[i],
            label=spec_name,
            markersize=6,
            capsize=3,
            capthick=1.5,
            linewidth=1.5
        )
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(variables)
    ax.set_xlabel('Coefficient Estimate')
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"   ✓ Saved: {output_path}")
    
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Feature Importance"
) -> plt.Figure:
    """
    Horizontal bar chart of feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        output_path: Path to save figure
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required")
    
    apply_publication_style()
    
    df = importance_df.head(top_n).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(df)))
    
    ax.barh(df['feature'], df['importance'], color=colors)
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"   ✓ Saved: {output_path}")
    
    return fig


def plot_model_comparison(
    results: Dict[str, Any],
    metric: str = 'test_r2',
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Model Performance Comparison"
) -> plt.Figure:
    """
    Bar chart comparing model performance.
    
    Args:
        results: Dict of MLResult objects
        metric: Metric to compare ('test_r2', 'cv_r2_mean', 'rmse')
        output_path: Path to save figure
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required")
    
    apply_publication_style()
    
    models = list(results.keys())
    values = [getattr(results[m], metric, 0) for m in models]
    
    # CV error bars if available
    errors = None
    if metric == 'cv_r2_mean':
        errors = [getattr(results[m], 'cv_r2_std', 0) for m in models]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    x = np.arange(len(models))
    
    bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.5)
    
    if errors:
        ax.errorbar(x, values, yerr=errors, fmt='none', color='black', capsize=5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.3f}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"   ✓ Saved: {output_path}")
    
    return fig


def plot_interaction_heatmap(
    heatmap_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "CEO-Firm Interaction Effects",
    cmap: str = 'RdBu_r'
) -> plt.Figure:
    """
    Heatmap of Two-Towers interaction effects.
    
    Args:
        heatmap_df: DataFrame with heatmap values
        output_path: Path to save figure
        figsize: Figure size
        title: Plot title
        cmap: Colormap name
        
    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required")
    
    apply_publication_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(heatmap_df.values, cmap=cmap, aspect='auto')
    
    # Labels
    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_xticklabels([c.split('=')[1] for c in heatmap_df.columns], rotation=45, ha='right')
    ax.set_yticklabels([c.split('=')[1] for c in heatmap_df.index])
    
    # Axis labels from column names
    firm_feat = heatmap_df.columns[0].split('=')[0]
    ceo_feat = heatmap_df.index[0].split('=')[0]
    ax.set_xlabel(firm_feat)
    ax.set_ylabel(ceo_feat)
    
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Match Quality')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"   ✓ Saved: {output_path}")
    
    return fig


def plot_embedding_scatter(
    embeddings_2d: np.ndarray,
    labels: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "CEO/Firm Embeddings (t-SNE)"
) -> plt.Figure:
    """
    Scatter plot of reduced embeddings.
    
    Args:
        embeddings_2d: 2D embeddings from t-SNE/PCA
        labels: Optional labels for coloring
        output_path: Path to save figure
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required")
    
    apply_publication_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        plt.colorbar(scatter, ax=ax, label='Match Quality')
    else:
        ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.6,
            s=20
        )
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"   ✓ Saved: {output_path}")
    
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: List[str],
    X_test: pd.DataFrame,
    output_path: Optional[str] = None,
    max_display: int = 15
) -> plt.Figure:
    """
    SHAP summary beeswarm plot.
    
    Args:
        shap_values: SHAP values array
        feature_names: Feature names
        X_test: Test features DataFrame
        output_path: Path to save figure
        max_display: Max features to display
        
    Returns:
        matplotlib Figure
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap required")
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required")
    
    apply_publication_style()
    
    fig = plt.figure(figsize=(10, 8))
    
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"   ✓ Saved: {output_path}")
    
    return fig


def create_dashboard(
    ols_results: Dict[str, Any],
    ml_results: Dict[str, Any],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create multi-panel summary dashboard.
    
    Args:
        ols_results: OLS analysis results
        ml_results: ML analysis results
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required")
    
    apply_publication_style()
    
    fig = plt.figure(figsize=figsize)
    
    # Panel 1: OLS R² comparison
    ax1 = fig.add_subplot(2, 2, 1)
    if ols_results:
        specs = list(ols_results.keys())
        r2s = [ols_results[s].r_squared for s in specs]
        ax1.bar(range(len(specs)), r2s, color='steelblue')
        ax1.set_xticks(range(len(specs)))
        ax1.set_xticklabels(specs, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('R²')
        ax1.set_title('OLS Specifications')
        ax1.grid(axis='y', alpha=0.3)
    
    # Panel 2: ML R² comparison
    ax2 = fig.add_subplot(2, 2, 2)
    if ml_results:
        models = list(ml_results.keys())
        test_r2s = [ml_results[m].test_r2 for m in models]
        cv_r2s = [ml_results[m].cv_r2_mean for m in models]
        x = np.arange(len(models))
        width = 0.35
        ax2.bar(x - width/2, test_r2s, width, label='Test R²', color='coral')
        ax2.bar(x + width/2, cv_r2s, width, label='CV R²', color='teal')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_ylabel('R²')
        ax2.set_title('ML Models')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    # Panel 3: Top feature importance (first ML model)
    ax3 = fig.add_subplot(2, 2, 3)
    if ml_results:
        first_result = list(ml_results.values())[0]
        if hasattr(first_result, 'feature_importance') and not first_result.feature_importance.empty:
            imp = first_result.feature_importance.head(10).sort_values('importance', ascending=True)
            ax3.barh(imp['feature'], imp['importance'], color='forestgreen')
            ax3.set_xlabel('Importance')
            ax3.set_title(f'Top Features ({first_result.model_name})')
            ax3.grid(axis='x', alpha=0.3)
    
    # Panel 4: Summary statistics text
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_text = "Analysis Summary\n" + "="*40 + "\n\n"
    
    if ols_results:
        best_ols = max(ols_results.items(), key=lambda x: x[1].r_squared)
        summary_text += f"Best OLS: {best_ols[0]}\n"
        summary_text += f"  R² = {best_ols[1].r_squared:.4f}\n"
        summary_text += f"  N = {best_ols[1].n_obs:,}\n\n"
    
    if ml_results:
        best_ml = max(ml_results.items(), key=lambda x: x[1].test_r2)
        summary_text += f"Best ML: {best_ml[0]}\n"
        summary_text += f"  Test R² = {best_ml[1].test_r2:.4f}\n"
        summary_text += f"  CV R² = {best_ml[1].cv_r2_mean:.4f}±{best_ml[1].cv_r2_std:.3f}\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"   ✓ Saved: {output_path}")
    
    return fig
