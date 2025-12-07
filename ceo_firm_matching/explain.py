"""
CEO-Firm Matching: Explainability Module

Model interpretation tools including SHAP and Partial Dependence Plots.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os
from typing import List

from .config import Config
from .model import CEOFirmMatcher
from .data import DataProcessor


class ModelWrapper:
    """
    Wraps the PyTorch model to expose a sklearn-like API (predict taking a single numpy array).
    Needed for SHAP and general interpretation tools.
    
    Input Layout (Flattened):
    [Firm Numeric...] [Firm Cat] [CEO Numeric...] [CEO Cat...]
    """
    def __init__(self, model: CEOFirmMatcher, processor: DataProcessor):
        self.model = model
        self.processor = processor
        self.device = processor.cfg.DEVICE
        
        # Calculate indices for slicing the flattened input
        self.n_firm_num = len(processor.final_firm_numeric)
        self.n_firm_cat = len(processor.cfg.FIRM_CAT_COLS)
        self.n_ceo_num = len(processor.final_ceo_numeric)
        self.n_ceo_cat = len(processor.cfg.CEO_CAT_COLS)
        
        # Slice Ranges
        self.idx_firm_num_end = self.n_firm_num
        self.idx_firm_cat_end = self.idx_firm_num_end + self.n_firm_cat
        self.idx_ceo_num_end = self.idx_firm_cat_end + self.n_ceo_num
        self.idx_ceo_cat_end = self.idx_ceo_num_end + self.n_ceo_cat
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Slice and Type Cast
        # 1. Firm Numeric
        f_num = X_tensor[:, :self.idx_firm_num_end]
        
        # 2. Firm Cat (Long)
        f_cat = X_tensor[:, self.idx_firm_num_end:self.idx_firm_cat_end].long()
        
        # 3. CEO Numeric
        c_num = X_tensor[:, self.idx_firm_cat_end:self.idx_ceo_num_end]
        
        # 4. CEO Cat (Long)
        c_cat = X_tensor[:, self.idx_ceo_num_end:].long()
        
        with torch.no_grad():
            preds = self.model(f_num, f_cat, c_num, c_cat)
            
        return preds.cpu().numpy().flatten()


def explain_model_pdp(wrapper: ModelWrapper, df: pd.DataFrame, features_to_plot: List[str]):
    """Generates Partial Dependence Plots for specified features."""
    print("\nGenerating Partial Dependence Plots (PDP)...")
    
    # Prepare Data Matrix
    data_dict = wrapper.processor.transform(df)
    
    # Concatenate in order: FirmNum, FirmCat, CEONum, CEOCat
    X_flat = np.hstack([
        data_dict['firm_numeric'].numpy(),
        data_dict['firm_cat'].numpy(),
        data_dict['ceo_numeric'].numpy(),
        data_dict['ceo_cat'].numpy()
    ])
    
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

    # Manual PDP Loop (Robust for mixed types)
    fig, axes = plt.subplots(1, len(indices_to_plot), figsize=(5 * len(indices_to_plot), 4))
    if len(indices_to_plot) == 1:
        axes = [axes]
        
    for ax, idx, name in zip(axes, indices_to_plot, valid_names):
        # Get range
        vals = X_flat[:, idx]
        # Grid: 50 points
        grid = np.linspace(vals.min(), vals.max(), 50)
        
        pdp_y = []
        # Subsample for speed
        sample_indices = np.random.choice(X_flat.shape[0], min(1000, X_flat.shape[0]), replace=False)
        X_sample = X_flat[sample_indices].copy()
        
        for val in grid:
            X_temp = X_sample.copy()
            X_temp[:, idx] = val
            preds = wrapper.predict(X_temp)
            pdp_y.append(np.mean(preds))
            
        ax.plot(grid, pdp_y, color='blue')
        ax.set_title(f"PDP: {name}")
        ax.set_xlabel("Standardized Value / Code")
        ax.set_ylabel("Avg Match Score")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    output_dir = wrapper.processor.cfg.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "pdp_plots.svg")
    plt.savefig(save_path)
    print(f"Saved PDP plots to {save_path}")
    plt.close()


def explain_model_shap(wrapper: ModelWrapper, df: pd.DataFrame):
    """Generates SHAP summary plot."""
    print("\nCalculating SHAP values (this may take a moment)...")
    
    # 1. Prepare Data
    data_dict = wrapper.processor._to_tensors(df)
    X_flat = np.hstack([
        data_dict['firm_numeric'].numpy(),
        data_dict['firm_cat'].numpy(),
        data_dict['ceo_numeric'].numpy(),
        data_dict['ceo_cat'].numpy()
    ])
    feature_names = wrapper.processor.get_feature_names()
    
    # 2. Background Data (Summary) for KernelExplainer
    X_summary = shap.kmeans(X_flat, 25)
    
    # 3. Explainer
    explainer = shap.KernelExplainer(wrapper.predict, X_summary)
    
    # 4. Calculate SHAP values on a subset
    subset_size = min(200, X_flat.shape[0])
    X_subset = X_flat[:subset_size]
    
    shap_values = explainer.shap_values(X_subset)
    
    # 5. Plot
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
