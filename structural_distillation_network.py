"""
Structural Distillation Network

A neural network model for CEO-Firm matching using structural distillation
with BLM (Bonhomme-Lamadon-Manresa) econometric priors.

This module provides the main entry point for running the Structural Distillation
Network. It integrates with the ceo_firm_matching package structure.

Architecture: Observables -> Latent Types -> Frozen Interaction -> Match Value

The model:
1. Learns to predict CEO and Firm type probabilities from observable features
2. Uses a frozen interaction matrix (from BLM estimation) to compute expected match values
3. Is trained via KL divergence to match BLM posterior probabilities

Usage:
    python structural_distillation_network.py [--synthetic]
    
    Options:
        --synthetic: Use synthetic data instead of loading from CSV
"""

import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader

from ceo_firm_matching import (
    StructuralConfig,
    StructuralDataProcessor,
    StructuralDistillationNet,
    train_structural_model,
    IlluminationEngine,
)

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [STRUCTURAL-DL] - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function for Structural Distillation Network."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Structural Distillation Network')
    parser.add_argument('--synthetic', action='store_true', 
                        help='Use synthetic data for testing')
    args = parser.parse_args()
    
    # Initialize configuration
    cfg = StructuralConfig()
    
    # Create output directory
    os.makedirs(cfg.OUTPUT_PATH, exist_ok=True)
    
    print("=" * 60)
    print("STRUCTURAL DISTILLATION NETWORK")
    print("CEO-Firm Matching with BLM Priors")
    print("=" * 60)
    print(f"\nDevice: {cfg.DEVICE}")
    print(f"Data Path: {cfg.DATA_PATH}")
    print(f"Output Path: {cfg.OUTPUT_PATH}")
    
    # 1. Data Pipeline
    print("\n--- Step 1: Data Pipeline ---")
    processor = StructuralDataProcessor(cfg)
    
    if args.synthetic:
        print("Using synthetic data...")
        cfg.DATA_PATH = "SYNTHETIC"  # Force synthetic generation
    
    train_ds, val_ds, val_df = processor.load_and_prep()
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True,
        drop_last=True  # For BatchNorm stability
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False
    )
    
    print(f"  Training samples: {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    
    # 2. Model Initialization
    print("\n--- Step 2: Model Initialization ---")
    metadata = processor.get_metadata()
    
    print(f"  Firm features: {metadata['n_firm_num']} numeric, {len(metadata['firm_cat_cards'])} categorical")
    print(f"  CEO features: {metadata['n_ceo_num']} numeric, {len(metadata['ceo_cat_cards'])} categorical")
    
    # 3. Training
    print("\n--- Step 3: Training ---")
    model = train_structural_model(train_loader, val_loader, metadata, cfg)
    
    if model is None:
        print("Training failed!")
        return 1
    
    # 4. Illumination (Explainability)
    print("\n--- Step 4: Illumination Analysis ---")
    illuminator = IlluminationEngine(model, processor)
    
    # Plot interaction matrix
    illuminator.plot_interaction_matrix()
    
    # Compute sensitivity analysis
    driver_df = illuminator.compute_global_sensitivity(val_loader)
    illuminator.plot_drivers(driver_df)
    
    # Analyze type distributions
    type_df = illuminator.analyze_type_distributions(val_loader)
    
    # 5. Summary Report
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n--- TOP MATCH VALUE DRIVERS ---")
    top_drivers = driver_df.head(5)[['Feature', 'Sensitivity', 'Magnitude', 'Type']]
    print(top_drivers.to_string(index=False))
    
    print("\n--- TYPE DISTRIBUTION SUMMARY ---")
    print(type_df.to_string(index=False))
    
    print(f"\n--- OUTPUT FILES ---")
    print(f"  {os.path.join(cfg.OUTPUT_PATH, 'match_drivers.png')}")
    print(f"  {os.path.join(cfg.OUTPUT_PATH, 'interaction_matrix.png')}")
    
    # Save results to CSV
    driver_path = os.path.join(cfg.OUTPUT_PATH, "sensitivity_analysis.csv")
    driver_df.to_csv(driver_path, index=False)
    print(f"  {driver_path}")
    
    type_path = os.path.join(cfg.OUTPUT_PATH, "type_distributions.csv")
    type_df.to_csv(type_path, index=False)
    print(f"  {type_path}")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    main()
