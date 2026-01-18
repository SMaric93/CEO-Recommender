"""
Visualization and explainability tools for the Two Towers model.

This module contains:
- explain_model_pdp: Partial Dependence Plots
- explain_model_shap: SHAP feature importance
- plot_interaction_heatmap: 2D interaction heatmaps
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
from typing import List

from ceo_firm_matching.config import Config
from data_processing import DataProcessor
from model import CEOFirmMatcher, ModelWrapper

__all__ = ['explain_model_pdp', 'explain_model_shap', 'plot_interaction_heatmap']


def explain_model_pdp(wrapper: ModelWrapper, df: pd.DataFrame, features_to_plot: List[str]):
    """Generates Partial Dependence Plots for specified features.
    
    Args:
        wrapper: ModelWrapper instance with trained model
        df: DataFrame to use for computing PDPs  
        features_to_plot: List of feature names to generate plots for
    """
    print("\nGenerating Partial Dependence Plots (PDP)...")
    
    # Use helper to get flattened feature matrix
    X_flat = wrapper.processor.get_flat_features(df)
    
    feature_names = wrapper.processor.get_feature_names()
    
    # Find indices of features to plot
    indices_to_plot = []
    valid_names = []
    for name in features_to_plot:
        if name in feature_names:
            indices_to_plot.append(feature_names.index(name))
            valid_names.append(name)
        else:
            print(f"Warning: Feature '{name}' not found in model inputs.")
            
    if not indices_to_plot:
        return

    # Determine which features are categorical
    all_cat_cols = set(wrapper.processor.cfg.FIRM_CAT_COLS + wrapper.processor.cfg.CEO_CAT_COLS)
    
    # Manual PDP Loop
    fig, axes = plt.subplots(1, len(indices_to_plot), figsize=(5 * len(indices_to_plot), 4))
    if len(indices_to_plot) == 1:
        axes = [axes]
        
    for ax, idx, name in zip(axes, indices_to_plot, valid_names):
        vals = X_flat[:, idx]
        is_categorical = name in all_cat_cols
        
        # Use discrete unique values for categorical, continuous grid otherwise
        if is_categorical:
            grid = np.sort(np.unique(vals.astype(int)))
        else:
            grid = np.linspace(vals.min(), vals.max(), 50)
        
        pdp_y = []
        sample_indices = np.random.choice(X_flat.shape[0], min(1000, X_flat.shape[0]), replace=False)
        X_sample = X_flat[sample_indices].copy()
        
        for val in grid:
            X_temp = X_sample.copy()
            X_temp[:, idx] = val
            preds = wrapper.predict(X_temp)
            pdp_y.append(np.mean(preds))
        
        # Bar plot for categorical, line plot for continuous
        if is_categorical:
            ax.bar(grid, pdp_y, color='steelblue', edgecolor='black', alpha=0.8)
            ax.set_xticks(grid)
            ax.set_xlabel("Category Code")
        else:
            ax.plot(grid, pdp_y, color='blue')
            ax.set_xlabel("Standardized Value")
            
        ax.set_title(f"PDP: {name}")
        ax.set_ylabel("Avg Match Score")
        ax.grid(True, alpha=0.3, axis='y')
        
    plt.tight_layout()
    output_dir = wrapper.processor.cfg.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "pdp_plots.svg")
    plt.savefig(save_path)
    print(f"Saved PDP plots to {save_path}")
    plt.close()


def explain_model_shap(wrapper: ModelWrapper, df: pd.DataFrame):
    """Generates SHAP summary plot for feature importance analysis.
    
    Args:
        wrapper: ModelWrapper instance with trained model
        df: DataFrame to use for SHAP analysis
    """
    print("\nCalculating SHAP values (this may take a moment)...")
    
    # Use helper to get flattened feature matrix
    X_flat = wrapper.processor.get_flat_features(df)
    feature_names = wrapper.processor.get_feature_names()
    
    # Background Data (Summary) for KernelExplainer
    X_summary = shap.kmeans(X_flat, 25)
    
    # Explainer
    explainer = shap.KernelExplainer(wrapper.predict, X_summary)
    
    # Calculate SHAP values on a subset
    subset_size = min(200, X_flat.shape[0])
    X_subset = X_flat[:subset_size]
    
    shap_values = explainer.shap_values(X_subset)
    
    # Plot
    plt.figure()
    shap.summary_plot(shap_values, X_subset, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    output_dir = wrapper.processor.cfg.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "shap_summary.svg")
    plt.savefig(save_path)
    print(f"Saved SHAP summary to {save_path}")
    plt.close()


def plot_interaction_heatmap(model: CEOFirmMatcher, processor: DataProcessor, 
                             x_feature: str, y_feature: str, filename: str):
    """
    Generate a 2D heatmap showing interaction effects between two features.
    
    Args:
        model: Trained CEOFirmMatcher
        processor: Fitted DataProcessor
        x_feature: Feature name for x-axis
        y_feature: Feature name for y-axis
        filename: Output filename (saved to OUTPUT_PATH)
    """
    if model is None or processor.processed_df is None:
        return

    print(f"\nGenerating interaction heatmap ({x_feature} vs {y_feature})...")
    
    # Prepare data
    data_dict = processor._to_tensors(processor.processed_df)
    tensor_keys = [k for k in data_dict.keys() if isinstance(data_dict[k], torch.Tensor)]
    device_data = {k: data_dict[k].to(processor.cfg.DEVICE) for k in tensor_keys}
    
    # Helper to find feature index and type
    def get_feature_info(name):
        if name in processor.final_firm_numeric:
            idx = list(processor.scalers['firm'].feature_names_in_).index(name)
            return 'firm_numeric', idx
        elif name in processor.final_ceo_numeric:
            idx = list(processor.scalers['ceo'].feature_names_in_).index(name)
            return 'ceo_numeric', idx
        elif name in processor.cfg.FIRM_CAT_COLS:
            idx = processor.cfg.FIRM_CAT_COLS.index(name)
            return 'firm_cat', idx
        elif name in processor.cfg.CEO_CAT_COLS:
            idx = processor.cfg.CEO_CAT_COLS.index(name)
            return 'ceo_cat', idx
        else:
            raise ValueError(f"Feature {name} not supported for heatmap.")

    try:
        x_type, x_idx = get_feature_info(x_feature)
        y_type, y_idx = get_feature_info(y_feature)
    except ValueError as e:
        print(f"Visualization Error: {e}")
        return

    # Define Grid Range
    def get_range(feat_type, feat_idx):
        if feat_type in ['firm_cat', 'ceo_cat']:
             if feat_type == 'firm_cat':
                 col_name = processor.cfg.FIRM_CAT_COLS[feat_idx]
             else:
                 col_name = processor.cfg.CEO_CAT_COLS[feat_idx]
             n_classes = len(processor.encoders[col_name].classes_)
             return np.arange(n_classes)
        else:
            return np.linspace(device_data[feat_type][:, feat_idx].min().item(), 
                               device_data[feat_type][:, feat_idx].max().item(), 50)

    x_vals = get_range(x_type, x_idx)
    y_vals = get_range(y_type, y_idx)
    
    heatmap = np.zeros((len(y_vals), len(x_vals)))
    
    # Baselines
    avg_f_numeric = torch.mean(device_data['firm_numeric'], dim=0, keepdim=True)
    avg_c_numeric = torch.mean(device_data['ceo_numeric'], dim=0, keepdim=True)
    
    def calculate_mode(tensor):
        """Helper to calculate mode, handling MPS limitations."""
        if tensor.device.type == 'mps':
            return torch.mode(tensor.cpu(), dim=0)[0].to(tensor.device).view(1, -1)
        return torch.mode(tensor, dim=0)[0].view(1, -1)

    mode_firm_cat = calculate_mode(device_data['firm_cat'])
    mode_ceo_cat = calculate_mode(device_data['ceo_cat'])
    
    model.eval()
    with torch.no_grad():
        for i, y_val in enumerate(y_vals):
            for j, x_val in enumerate(x_vals):
                # Reset inputs to baseline
                f_in = avg_f_numeric.clone()
                c_in = avg_c_numeric.clone()
                f_cat_in = mode_firm_cat.clone()
                c_cat_in = mode_ceo_cat.clone()
                
                def update_input(ftype, fidx, val):
                    if ftype == 'firm_numeric':
                        f_in[:, fidx] = float(val)
                    elif ftype == 'ceo_numeric':
                        c_in[:, fidx] = float(val)
                    elif ftype == 'firm_cat':
                        f_cat_in[:, fidx] = int(val)
                    elif ftype == 'ceo_cat':
                        c_cat_in[:, fidx] = int(val)

                update_input(x_type, x_idx, x_val)
                update_input(y_type, y_idx, y_val)
                
                score = model(f_in, f_cat_in, c_in, c_cat_in)
                heatmap[i, j] = score.item()

    # Determine if axes are categorical
    x_is_cat = x_type in ['firm_cat', 'ceo_cat']
    y_is_cat = y_type in ['firm_cat', 'ceo_cat']
    
    # Plotting - use different rendering based on axis types
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Choose interpolation based on whether we have categorical axes
    if x_is_cat or y_is_cat:
        # No interpolation for categorical - show discrete blocks
        im = ax.imshow(heatmap, aspect='auto', cmap='RdBu_r', origin='lower', 
                       interpolation='nearest')
    else:
        # Smooth interpolation for continuous-continuous
        im = ax.imshow(heatmap, aspect='auto', cmap='RdBu_r', origin='lower',
                       interpolation='bicubic',
                       extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()])
    
    plt.colorbar(im, label='Predicted Match Quality')
    
    # Handle axis labels and ticks
    if x_is_cat:
        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_xticklabels([int(v) for v in x_vals])
        ax.set_xlabel(f'{x_feature} (Category)')
    else:
        if not (x_is_cat or y_is_cat):  # Only set these if using extent
            pass  # extent handles it
        else:
            ax.set_xticks(np.linspace(0, len(x_vals)-1, 5))
            ax.set_xticklabels([f'{v:.1f}' for v in np.linspace(x_vals.min(), x_vals.max(), 5)])
        ax.set_xlabel(f'{x_feature} (Standardized)')
    
    if y_is_cat:
        ax.set_yticks(np.arange(len(y_vals)))
        ax.set_yticklabels([int(v) for v in y_vals])
        ax.set_ylabel(f'{y_feature} (Category)')
    else:
        if not (x_is_cat or y_is_cat):  # Only set these if using extent
            pass  # extent handles it
        else:
            ax.set_yticks(np.linspace(0, len(y_vals)-1, 5))
            ax.set_yticklabels([f'{v:.1f}' for v in np.linspace(y_vals.min(), y_vals.max(), 5)])
        ax.set_ylabel(f'{y_feature} (Standardized)')
    
    ax.set_title(f'Interaction: {x_feature} vs {y_feature}')
    
    output_dir = processor.cfg.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    print(f"Saved heatmap to {path}")
    plt.close()

