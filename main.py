#!/usr/bin/env python3
"""
Main entry point for the Two Towers CEO-Firm matching model.

Usage:
    python main.py                  # Train with real data
    python main.py --synthetic      # Train with synthetic data (for testing)
"""
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ceo_firm_matching import (
    Config,
    DataProcessor,
    CEOFirmDataset,
    ModelWrapper,
    train_model,
    explain_model_pdp,
    plot_interaction_heatmap,
    generate_synthetic_data,
)


def main():
    """Main training and analysis pipeline."""
    parser = argparse.ArgumentParser(description="Train Two Towers Model")
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data for verification')
    args = parser.parse_args()

    config = Config()
    print(f"Running Two Towers Model on {config.DEVICE}")
    
    # 1. Data Processing
    processor = DataProcessor(config)
    
    if args.synthetic:
        print("Using SYNTHETIC data...")
        raw_df = generate_synthetic_data(1000)
    else:
        raw_df = processor.load_data()
        
    if raw_df.empty:
        print("No data loaded. Exiting.")
        return
    
    # Step 1: Prepare Features (Stateless)
    df_clean = processor.prepare_features(raw_df)
    
    # Step 2: Split Data
    train_df, val_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # Step 3: Fit Scalers on Train
    processor.fit(train_df)
    
    # Step 4: Transform
    train_data = processor.transform(train_df)
    val_data = processor.transform(val_df)
    
    # Step 5: Create Datasets & Loaders
    train_dataset = CEOFirmDataset(train_data)
    val_dataset = CEOFirmDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 2. Training
    trained_model = train_model(train_loader, val_loader, train_data, config)
    
    if trained_model:
        # 3. Explainability
        wrapper = ModelWrapper(trained_model, processor)
        
        # PDP for key features
        features_to_plot = config.FIRM_NUMERIC_COLS + config.CEO_NUMERIC_COLS + ['tenure']
        explain_model_pdp(wrapper, val_df, features_to_plot)
        
        # SHAP (commented out - slow)
        # explain_model_shap(wrapper, val_df)

        # 4. Interaction Plots
        processor.transform(val_df)  # Ensure processed_df is set
        
        plot_interaction_heatmap(trained_model, processor, 'logatw', 'Age', 'heatmap_size_age.svg')
        plot_interaction_heatmap(trained_model, processor, 'logatw', 'Output', 'heatmap_size_skill.svg')
        plot_interaction_heatmap(trained_model, processor, 'exp_roa', 'tenure', 'heatmap_perf_exp.svg')
        plot_interaction_heatmap(trained_model, processor, 'rdintw', 'Output', 'heatmap_rd_skill.svg')
        plot_interaction_heatmap(trained_model, processor, 'rdintw', 'Age', 'heatmap_rd_age.svg')
        plot_interaction_heatmap(trained_model, processor, 'logatw', 'ivy', 'heatmap_size_ivy.svg')
        plot_interaction_heatmap(trained_model, processor, 'tenure', 'boardindpw', 'heatmap_tenure_boardind.svg')
        plot_interaction_heatmap(trained_model, processor, 'maxedu', 'rdintw', 'heatmap_maxedu_rd.svg')
        plot_interaction_heatmap(trained_model, processor, 'maxedu', 'capintw', 'heatmap_maxedu_capx.svg')
        plot_interaction_heatmap(trained_model, processor, 'logatw', 'm', 'heatmap_size_mover.svg')
        plot_interaction_heatmap(trained_model, processor, 'leverage', 'Age', 'heatmap_leverage_age.svg')


if __name__ == "__main__":
    main()
