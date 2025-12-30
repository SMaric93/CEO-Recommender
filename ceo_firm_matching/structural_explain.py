"""
CEO-Firm Matching: Illumination Engine

Gradient-based sensitivity analysis for interpreting the Structural Distillation Network.
Reverse-engineers the drivers of match value via gradient sensitivity.
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.utils.data import DataLoader
from typing import Optional

from .structural_config import StructuralConfig
from .structural_model import StructuralDistillationNet
from .structural_data import StructuralDataProcessor


class IlluminationEngine:
    """
    Reverse-engineers drivers of match value via Gradient Sensitivity Analysis.
    
    This provides interpretability for the Structural Distillation Network by
    computing d(MatchValue)/d(Feature) across the dataset.
    
    Attributes:
        model: Trained StructuralDistillationNet
        processor: StructuralDataProcessor with fitted transformers
    """
    
    def __init__(self, model: StructuralDistillationNet, processor: StructuralDataProcessor):
        """
        Initialize the Illumination Engine.
        
        Args:
            model: Trained StructuralDistillationNet model
            processor: StructuralDataProcessor with fitted scalers/encoders
        """
        self.model = model
        self.processor = processor
        self.model.eval()

    def compute_global_sensitivity(self, loader: DataLoader) -> pd.DataFrame:
        """
        Calculates d(Match)/d(Feature) across the whole dataset.
        
        Accumulates gradients across all batches to find robust global drivers.
        This answers: "Which features most influence expected match value?"
        
        Args:
            loader: DataLoader with validation or test data
            
        Returns:
            DataFrame with columns: Feature, Sensitivity, Magnitude, Type
        """
        print("\n--- Running Gradient Sensitivity Analysis ---")
        
        cfg = self.processor.cfg
        
        # Initialize gradient accumulators
        grads = {
            f: [] for f in 
            list(self.processor.final_firm_numeric) + list(self.processor.final_ceo_numeric)
        }
        
        for batch in loader:
            # Prepare inputs WITH gradient tracking
            f_num = batch['firm_num'].to(cfg.DEVICE).requires_grad_(True)
            c_num = batch['ceo_num'].to(cfg.DEVICE).requires_grad_(True)
            f_cat = batch['firm_cat'].to(cfg.DEVICE)
            c_cat = batch['ceo_cat'].to(cfg.DEVICE)
            
            # Forward pass
            _, _, match_val = self.model(f_num, f_cat, c_num, c_cat)
            
            # Backward pass (gradient of sum w.r.t. inputs)
            match_val.sum().backward()
            
            # Collect gradients for firm numeric features
            for i, name in enumerate(self.processor.final_firm_numeric):
                if f_num.grad is not None:
                    grads[name].append(f_num.grad[:, i].detach().cpu().numpy())
            
            # Collect gradients for CEO numeric features  
            for i, name in enumerate(self.processor.final_ceo_numeric):
                if c_num.grad is not None:
                    grads[name].append(c_num.grad[:, i].detach().cpu().numpy())
        
        # Aggregate results
        results = []
        for feat, val_list in grads.items():
            if len(val_list) > 0:
                all_grads = np.concatenate(val_list)
                results.append({
                    'Feature': feat,
                    'Sensitivity': np.mean(all_grads),
                    'Magnitude': np.mean(np.abs(all_grads)),
                    'Std': np.std(all_grads),
                    'Type': 'Firm' if feat in self.processor.final_firm_numeric else 'CEO'
                })
        
        df = pd.DataFrame(results).sort_values('Magnitude', ascending=False)
        
        print(f"  Analyzed {len(df)} features")
        print(f"  Top 3 by Magnitude: {df.head(3)['Feature'].tolist()}")
        
        return df

    def plot_drivers(self, df: pd.DataFrame, save: bool = True) -> Optional[str]:
        """
        Creates a bar plot showing structural drivers of match value.
        
        Args:
            df: DataFrame from compute_global_sensitivity()
            save: If True, save plot to OUTPUT_PATH
            
        Returns:
            Path to saved figure, or None if not saved
        """
        plt.figure(figsize=(12, 8))
        
        # Color by type
        palette = {'Firm': '#3498db', 'CEO': '#e74c3c'}
        
        sns.barplot(
            data=df, 
            y='Feature', 
            x='Sensitivity', 
            hue='Type',
            palette=palette,
            dodge=False
        )
        
        plt.title("Structural Drivers of Match Value\n(Gradient Sensitivity Analysis)", fontsize=14)
        plt.axvline(0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel("d(Match Value) / d(Feature)", fontsize=11)
        plt.ylabel("Feature", fontsize=11)
        plt.legend(title='Entity Type', loc='lower right')
        plt.tight_layout()
        
        save_path = None
        if save:
            output_dir = self.processor.cfg.OUTPUT_PATH
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "match_drivers.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved driver analysis to {save_path}")
        
        plt.close()
        return save_path

    def analyze_type_distributions(self, loader: DataLoader) -> pd.DataFrame:
        """
        Analyzes the distribution of predicted CEO and Firm types.
        
        Returns:
            DataFrame with type distribution statistics
        """
        print("\n--- Analyzing Type Distributions ---")
        
        cfg = self.processor.cfg
        ceo_probs_all = []
        firm_probs_all = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                f_num = batch['firm_num'].to(cfg.DEVICE)
                f_cat = batch['firm_cat'].to(cfg.DEVICE)
                c_num = batch['ceo_num'].to(cfg.DEVICE)
                c_cat = batch['ceo_cat'].to(cfg.DEVICE)
                
                ceo_probs, firm_probs = self.model.get_type_probabilities(
                    f_num, f_cat, c_num, c_cat
                )
                
                ceo_probs_all.append(ceo_probs.cpu().numpy())
                firm_probs_all.append(firm_probs.cpu().numpy())
        
        ceo_probs = np.concatenate(ceo_probs_all, axis=0)
        firm_probs = np.concatenate(firm_probs_all, axis=0)
        
        # Create summary statistics
        results = []
        for i in range(5):
            results.append({
                'Type': f'CEO Type {i+1}',
                'Mean Probability': ceo_probs[:, i].mean(),
                'Std Probability': ceo_probs[:, i].std(),
                'Median Probability': np.median(ceo_probs[:, i])
            })
        for i in range(5):
            results.append({
                'Type': f'Firm Class {i+1}',
                'Mean Probability': firm_probs[:, i].mean(),
                'Std Probability': firm_probs[:, i].std(),
                'Median Probability': np.median(firm_probs[:, i])
            })
        
        return pd.DataFrame(results)

    def plot_interaction_matrix(self, save: bool = True) -> Optional[str]:
        """
        Visualizes the frozen BLM interaction matrix.
        
        Args:
            save: If True, save plot to OUTPUT_PATH
            
        Returns:
            Path to saved figure, or None if not saved
        """
        A = self.model.A.cpu().numpy()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            A, 
            annot=True, 
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            xticklabels=[f'Firm {i+1}' for i in range(5)],
            yticklabels=[f'CEO {i+1}' for i in range(5)]
        )
        plt.title("BLM Interaction Matrix (Frozen)\n$A_{ij}$ = Match Value for CEO Type $i$ × Firm Class $j$", fontsize=12)
        plt.xlabel("Firm Class", fontsize=11)
        plt.ylabel("CEO Type", fontsize=11)
        plt.tight_layout()
        
        save_path = None
        if save:
            output_dir = self.processor.cfg.OUTPUT_PATH
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "interaction_matrix.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved interaction matrix to {save_path}")
        
        plt.close()
        return save_path
