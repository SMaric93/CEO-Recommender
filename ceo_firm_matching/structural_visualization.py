"""
CEO-Firm Matching: Structural Distillation Visualization Module

Visualization tools specific to the Structural Distillation Network.
Includes type probability distributions, match value surfaces, and comparison plots.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional, Tuple, List
from torch.utils.data import DataLoader

from .structural_config import StructuralConfig
from .structural_model import StructuralDistillationNet
from .structural_data import StructuralDataProcessor


def plot_type_probability_distributions(
    model: StructuralDistillationNet,
    processor: StructuralDataProcessor,
    loader: DataLoader,
    save: bool = True
) -> Optional[str]:
    """
    Plot the distribution of predicted CEO and Firm type probabilities.
    
    Creates violin plots showing how the model distributes entities across types.
    
    Args:
        model: Trained StructuralDistillationNet
        processor: Fitted StructuralDataProcessor
        loader: DataLoader with data to analyze
        save: If True, save plot to OUTPUT_PATH
        
    Returns:
        Path to saved figure, or None if not saved
    """
    cfg = processor.cfg
    model.eval()
    
    ceo_probs_list = []
    firm_probs_list = []
    
    with torch.no_grad():
        for batch in loader:
            f_num = batch['firm_num'].to(cfg.DEVICE)
            f_cat = batch['firm_cat'].to(cfg.DEVICE)
            c_num = batch['ceo_num'].to(cfg.DEVICE)
            c_cat = batch['ceo_cat'].to(cfg.DEVICE)
            
            ceo_probs, firm_probs = model.get_type_probabilities(
                f_num, f_cat, c_num, c_cat
            )
            
            ceo_probs_list.append(ceo_probs.cpu().numpy())
            firm_probs_list.append(firm_probs.cpu().numpy())
    
    ceo_probs = np.concatenate(ceo_probs_list, axis=0)
    firm_probs = np.concatenate(firm_probs_list, axis=0)
    
    # Prepare data for plotting
    ceo_data = []
    for i in range(5):
        for p in ceo_probs[:, i]:
            ceo_data.append({'Type': f'CEO Type {i+1}', 'Probability': p})
    
    firm_data = []
    for i in range(5):
        for p in firm_probs[:, i]:
            firm_data.append({'Type': f'Firm Class {i+1}', 'Probability': p})
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # CEO Types
    ceo_df = pd.DataFrame(ceo_data)
    sns.violinplot(data=ceo_df, x='Type', y='Probability', ax=axes[0], palette='Reds')
    axes[0].set_title('CEO Type Probability Distribution', fontsize=12)
    axes[0].set_xlabel('CEO Type')
    axes[0].set_ylabel('Probability')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Firm Classes
    firm_df = pd.DataFrame(firm_data)
    sns.violinplot(data=firm_df, x='Type', y='Probability', ax=axes[1], palette='Blues')
    axes[1].set_title('Firm Class Probability Distribution', fontsize=12)
    axes[1].set_xlabel('Firm Class')
    axes[1].set_ylabel('Probability')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    save_path = None
    if save:
        output_dir = cfg.OUTPUT_PATH
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "type_distributions.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved type distributions to {save_path}")
    
    plt.close()
    return save_path


def plot_expected_match_distribution(
    model: StructuralDistillationNet,
    processor: StructuralDataProcessor,
    loader: DataLoader,
    save: bool = True
) -> Optional[str]:
    """
    Plot the distribution of expected match values.
    
    Args:
        model: Trained StructuralDistillationNet
        processor: Fitted StructuralDataProcessor
        loader: DataLoader with data to analyze
        save: If True, save plot to OUTPUT_PATH
        
    Returns:
        Path to saved figure, or None if not saved
    """
    cfg = processor.cfg
    model.eval()
    
    match_values = []
    
    with torch.no_grad():
        for batch in loader:
            f_num = batch['firm_num'].to(cfg.DEVICE)
            f_cat = batch['firm_cat'].to(cfg.DEVICE)
            c_num = batch['ceo_num'].to(cfg.DEVICE)
            c_cat = batch['ceo_cat'].to(cfg.DEVICE)
            
            _, _, expected_match = model(f_num, f_cat, c_num, c_cat)
            match_values.append(expected_match.cpu().numpy())
    
    match_values = np.concatenate(match_values, axis=0).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(match_values, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    axes[0].axvline(np.mean(match_values), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(match_values):.3f}')
    axes[0].axvline(np.median(match_values), color='orange', linestyle='--', 
                    label=f'Median: {np.median(match_values):.3f}')
    axes[0].set_title('Expected Match Value Distribution', fontsize=12)
    axes[0].set_xlabel('Expected Match Value')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot(match_values, vert=True)
    axes[1].set_title('Expected Match Value Box Plot', fontsize=12)
    axes[1].set_ylabel('Expected Match Value')
    axes[1].set_xticklabels(['All Matches'])
    
    plt.tight_layout()
    
    save_path = None
    if save:
        output_dir = cfg.OUTPUT_PATH
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "match_distribution.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved match distribution to {save_path}")
    
    plt.close()
    return save_path


def plot_type_confusion_matrix(
    model: StructuralDistillationNet,
    processor: StructuralDataProcessor,
    loader: DataLoader,
    entity: str = 'ceo',
    save: bool = True
) -> Optional[str]:
    """
    Plot confusion matrix comparing predicted vs target type probabilities.
    
    Shows how well the model replicates BLM posterior probabilities.
    
    Args:
        model: Trained StructuralDistillationNet
        processor: Fitted StructuralDataProcessor
        loader: DataLoader with data to analyze
        entity: 'ceo' or 'firm'
        save: If True, save plot to OUTPUT_PATH
        
    Returns:
        Path to saved figure, or None if not saved
    """
    cfg = processor.cfg
    model.eval()
    
    pred_types = []
    target_types = []
    
    with torch.no_grad():
        for batch in loader:
            f_num = batch['firm_num'].to(cfg.DEVICE)
            f_cat = batch['firm_cat'].to(cfg.DEVICE)
            c_num = batch['ceo_num'].to(cfg.DEVICE)
            c_cat = batch['ceo_cat'].to(cfg.DEVICE)
            
            if entity == 'ceo':
                target = batch['target_ceo']
                ceo_probs, _ = model.get_type_probabilities(f_num, f_cat, c_num, c_cat)
                pred = ceo_probs.cpu()
            else:
                target = batch['target_firm']
                _, firm_probs = model.get_type_probabilities(f_num, f_cat, c_num, c_cat)
                pred = firm_probs.cpu()
            
            # Get argmax for confusion matrix
            pred_types.extend(pred.argmax(dim=1).numpy())
            target_types.extend(target.argmax(dim=1).numpy())
    
    pred_types = np.array(pred_types)
    target_types = np.array(target_types)
    
    # Create confusion matrix
    confusion = np.zeros((5, 5), dtype=int)
    for p, t in zip(pred_types, target_types):
        confusion[t, p] += 1
    
    # Normalize by row (target)
    confusion_norm = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-9)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        confusion_norm, 
        annot=True, 
        fmt='.2f',
        cmap='Blues',
        xticklabels=[f'Pred {i+1}' for i in range(5)],
        yticklabels=[f'True {i+1}' for i in range(5)],
        ax=ax
    )
    
    entity_name = 'CEO Type' if entity == 'ceo' else 'Firm Class'
    ax.set_title(f'{entity_name} Prediction Accuracy\n(Normalized by True Label)', fontsize=12)
    ax.set_xlabel(f'Predicted {entity_name}')
    ax.set_ylabel(f'True {entity_name} (BLM Posterior)')
    
    plt.tight_layout()
    
    save_path = None
    if save:
        output_dir = cfg.OUTPUT_PATH
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"confusion_{entity}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.close()
    return save_path


def plot_feature_type_correlation(
    model: StructuralDistillationNet,
    processor: StructuralDataProcessor,
    loader: DataLoader,
    feature_name: str,
    entity: str = 'ceo',
    save: bool = True
) -> Optional[str]:
    """
    Plot how a specific feature correlates with type probabilities.
    
    Args:
        model: Trained StructuralDistillationNet
        processor: Fitted StructuralDataProcessor
        loader: DataLoader with data to analyze
        feature_name: Name of the feature to analyze
        entity: 'ceo' or 'firm'
        save: If True, save plot to OUTPUT_PATH
        
    Returns:
        Path to saved figure, or None if not saved
    """
    cfg = processor.cfg
    model.eval()
    
    feature_values = []
    type_probs = []
    
    # Find feature index
    feature_names = processor.get_feature_names()
    if feature_name not in feature_names:
        print(f"Warning: Feature '{feature_name}' not found in model inputs.")
        return None
    
    feature_idx = feature_names.index(feature_name)
    n_firm_num = len(processor.final_firm_numeric)
    n_firm_cat = len(processor.cfg.FIRM_CAT_COLS)
    
    with torch.no_grad():
        for batch in loader:
            f_num = batch['firm_num'].to(cfg.DEVICE)
            f_cat = batch['firm_cat'].to(cfg.DEVICE)
            c_num = batch['ceo_num'].to(cfg.DEVICE)
            c_cat = batch['ceo_cat'].to(cfg.DEVICE)
            
            # Get feature values based on index
            if feature_idx < n_firm_num:
                feat_vals = f_num[:, feature_idx].cpu().numpy()
            elif feature_idx < n_firm_num + n_firm_cat:
                feat_vals = f_cat[:, feature_idx - n_firm_num].cpu().numpy().astype(float)
            elif feature_idx < n_firm_num + n_firm_cat + len(processor.final_ceo_numeric):
                ceo_idx = feature_idx - n_firm_num - n_firm_cat
                feat_vals = c_num[:, ceo_idx].cpu().numpy()
            else:
                ceo_cat_idx = feature_idx - n_firm_num - n_firm_cat - len(processor.final_ceo_numeric)
                feat_vals = c_cat[:, ceo_cat_idx].cpu().numpy().astype(float)
            
            feature_values.extend(feat_vals)
            
            if entity == 'ceo':
                ceo_probs, _ = model.get_type_probabilities(f_num, f_cat, c_num, c_cat)
                type_probs.append(ceo_probs.cpu().numpy())
            else:
                _, firm_probs = model.get_type_probabilities(f_num, f_cat, c_num, c_cat)
                type_probs.append(firm_probs.cpu().numpy())
    
    feature_values = np.array(feature_values)
    type_probs = np.concatenate(type_probs, axis=0)
    
    # Create scatter plots
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    
    for i, ax in enumerate(axes):
        ax.scatter(feature_values, type_probs[:, i], alpha=0.3, s=10)
        # Add trend line
        z = np.polyfit(feature_values, type_probs[:, i], 1)
        p = np.poly1d(z)
        x_line = np.linspace(feature_values.min(), feature_values.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2)
        
        entity_name = 'CEO Type' if entity == 'ceo' else 'Firm Class'
        ax.set_title(f'{entity_name} {i+1}')
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Probability')
    
    plt.suptitle(f'Feature "{feature_name}" vs {entity.upper()} Type Probabilities', fontsize=14)
    plt.tight_layout()
    
    save_path = None
    if save:
        output_dir = cfg.OUTPUT_PATH
        os.makedirs(output_dir, exist_ok=True)
        safe_name = feature_name.replace('/', '_')
        save_path = os.path.join(output_dir, f"feature_type_{safe_name}_{entity}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature-type correlation to {save_path}")
    
    plt.close()
    return save_path


def create_structural_report(
    model: StructuralDistillationNet,
    processor: StructuralDataProcessor,
    loader: DataLoader
) -> List[str]:
    """
    Generate a complete visualization report for the Structural Distillation Network.
    
    Creates all standard visualizations and saves them to OUTPUT_PATH.
    
    Args:
        model: Trained StructuralDistillationNet
        processor: Fitted StructuralDataProcessor
        loader: DataLoader with data to analyze
        
    Returns:
        List of paths to saved figures
    """
    from .structural_explain import IlluminationEngine
    
    print("\n" + "=" * 60)
    print("GENERATING STRUCTURAL DISTILLATION REPORT")
    print("=" * 60)
    
    output_paths = []
    
    # 1. Interaction Matrix
    print("\n1. Plotting interaction matrix...")
    illuminator = IlluminationEngine(model, processor)
    path = illuminator.plot_interaction_matrix()
    if path:
        output_paths.append(path)
    
    # 2. Gradient Sensitivity
    print("2. Computing gradient sensitivity...")
    driver_df = illuminator.compute_global_sensitivity(loader)
    path = illuminator.plot_drivers(driver_df)
    if path:
        output_paths.append(path)
    
    # 3. Type Distributions
    print("3. Plotting type probability distributions...")
    path = plot_type_probability_distributions(model, processor, loader)
    if path:
        output_paths.append(path)
    
    # 4. Match Value Distribution
    print("4. Plotting match value distribution...")
    path = plot_expected_match_distribution(model, processor, loader)
    if path:
        output_paths.append(path)
    
    # 5. Confusion Matrices
    print("5. Plotting confusion matrices...")
    path = plot_type_confusion_matrix(model, processor, loader, entity='ceo')
    if path:
        output_paths.append(path)
    path = plot_type_confusion_matrix(model, processor, loader, entity='firm')
    if path:
        output_paths.append(path)
    
    print(f"\nReport complete. Generated {len(output_paths)} visualizations.")
    print(f"Output directory: {processor.cfg.OUTPUT_PATH}")
    
    return output_paths
