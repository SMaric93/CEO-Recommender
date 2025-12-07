"""
CEO-Firm Matching: Visualization Module

Interaction heatmap plotting for model interpretation.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from .config import Config
from .model import CEOFirmMatcher
from .data import DataProcessor


def plot_interaction_heatmap(model: CEOFirmMatcher, processor: DataProcessor, 
                             x_feature: str, y_feature: str, filename: str):
    """
    Generate an interaction heatmap between two features.
    
    Args:
        model: Trained CEOFirmMatcher model
        processor: DataProcessor with processed_df set
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
             # Get number of classes for this categorical feature
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
                    nonlocal f_in, c_in, f_cat_in, c_cat_in
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

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap, aspect='auto', cmap='RdBu_r', origin='lower', interpolation='bicubic',
               extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()])
    plt.colorbar(label='Predicted Match Quality')
    plt.xlabel(f'{x_feature} (Standardized)')
    plt.ylabel(f'{y_feature} (Standardized)')
    plt.title(f'Interaction: {x_feature} vs {y_feature}')
    
    output_dir = processor.cfg.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    print(f"Saved heatmap to {path}")
    plt.close()
