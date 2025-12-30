"""
CEO-Firm Matching: Structural Distillation CLI

Command line interface for the Structural Distillation Network.
"""
import argparse
import os
from torch.utils.data import DataLoader

from .structural_config import StructuralConfig
from .structural_data import StructuralDataProcessor
from .structural_training import train_structural_model
from .structural_explain import IlluminationEngine


def main():
    """Main entry point for the Structural Distillation Network pipeline."""
    parser = argparse.ArgumentParser(
        description="Train Structural Distillation Network with BLM Priors"
    )
    parser.add_argument(
        '--synthetic', 
        action='store_true', 
        help='Use synthetic data for verification'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for training (default: 256)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to BLM posteriors CSV file'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Output directory for results'
    )
    args = parser.parse_args()

    # Initialize configuration
    config = StructuralConfig()
    
    # Override config with CLI arguments
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA_PATH = args.data_path
    if args.output_path:
        config.OUTPUT_PATH = args.output_path
    
    # Create output directory
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    
    print("=" * 60)
    print("STRUCTURAL DISTILLATION NETWORK")
    print("CEO-Firm Matching with BLM Priors")
    print("=" * 60)
    print(f"\nDevice: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    
    # 1. Data Pipeline
    print("\n--- Step 1: Data Pipeline ---")
    processor = StructuralDataProcessor(config)
    
    if args.synthetic:
        print("Using SYNTHETIC data...")
        # Force synthetic by setting non-existent path
        config.DATA_PATH = "SYNTHETIC_MODE"
    
    train_ds, val_ds, val_df = processor.load_and_prep()
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False
    )
    
    print(f"  Training samples: {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    
    # 2. Model Initialization & Training
    print("\n--- Step 2: Training ---")
    metadata = processor.get_metadata()
    model = train_structural_model(train_loader, val_loader, metadata, config)
    
    if model is None:
        print("Training failed!")
        return 1
    
    # 3. Illumination Analysis
    print("\n--- Step 3: Illumination Analysis ---")
    illuminator = IlluminationEngine(model, processor)
    
    # Visualize interaction matrix
    illuminator.plot_interaction_matrix()
    
    # Compute and plot sensitivity analysis
    driver_df = illuminator.compute_global_sensitivity(val_loader)
    illuminator.plot_drivers(driver_df)
    
    # Analyze type distributions
    type_df = illuminator.analyze_type_distributions(val_loader)
    
    # 4. Save Results
    print("\n--- Step 4: Saving Results ---")
    
    driver_path = os.path.join(config.OUTPUT_PATH, "sensitivity_analysis.csv")
    driver_df.to_csv(driver_path, index=False)
    print(f"  Saved: {driver_path}")
    
    type_path = os.path.join(config.OUTPUT_PATH, "type_distributions.csv")
    type_df.to_csv(type_path, index=False)
    print(f"  Saved: {type_path}")
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n--- TOP MATCH VALUE DRIVERS ---")
    print(driver_df.head(5)[['Feature', 'Sensitivity', 'Magnitude', 'Type']].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
