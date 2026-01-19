#!/usr/bin/env python3
"""
Exploratory Analysis: Finding Interesting CEO-Firm Match Patterns

This script trains the Two-Tower model and analyzes:
1. Best/Worst matches in the dataset
2. CEO archetypes that match well with different firm types
3. Embedding space visualization
4. Complementarity patterns (what features predict good matches)
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import seaborn as sns

from ceo_firm_matching import (
    Config,
    DataProcessor,
    CEOFirmDataset,
    CEOFirmMatcher,
    train_model,
)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def main():
    config = Config()
    print(f"Device: {config.DEVICE}")
    
    # 1. Load and prepare data
    print("\n=== LOADING DATA ===")
    processor = DataProcessor(config)
    raw_df = processor.load_data()
    
    if raw_df.empty:
        print("No data found!")
        return
    
    df_clean = processor.prepare_features(raw_df)
    
    # 2. Quick data exploration
    print("\n=== DATA EXPLORATION ===")
    print(f"Total CEO-Firm-Year observations: {len(df_clean)}")
    print(f"Unique Firms (gvkey): {df_clean['gvkey'].nunique()}")
    print(f"Unique CEOs (match_exec_id): {df_clean['match_exec_id'].nunique()}")
    print(f"\nMatch Quality Distribution:")
    print(df_clean['match_means'].describe())
    
    # 3. Find best and worst matches in raw data
    print("\n=== TOP 10 BEST MATCHES (Highest match_means) ===")
    best_matches = df_clean.nlargest(10, 'match_means')[
        ['gvkey', 'match_exec_id', 'Age', 'Gender', 'logatw', 'exp_roa', 
         'rdintw', 'maxedu', 'Output', 'match_means']
    ]
    print(best_matches.to_string(index=False))
    
    print("\n=== TOP 10 WORST MATCHES (Lowest match_means) ===")
    worst_matches = df_clean.nsmallest(10, 'match_means')[
        ['gvkey', 'match_exec_id', 'Age', 'Gender', 'logatw', 'exp_roa',
         'rdintw', 'maxedu', 'Output', 'match_means']
    ]
    print(worst_matches.to_string(index=False))
    
    # 4. Train the model
    print("\n=== TRAINING MODEL ===")
    train_df, val_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    processor.fit(train_df)
    
    train_data = processor.transform(train_df)
    val_data = processor.transform(val_df)
    
    train_loader = DataLoader(CEOFirmDataset(train_data), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CEOFirmDataset(val_data), batch_size=config.BATCH_SIZE, shuffle=False)
    
    model = train_model(train_loader, val_loader, train_data, config)
    
    if model is None:
        print("Training failed!")
        return
    
    # 5. Extract embeddings from full dataset
    print("\n=== EXTRACTING EMBEDDINGS ===")
    full_data = processor.transform(df_clean)
    
    model.eval()
    with torch.no_grad():
        f_numeric = full_data['firm_numeric'].to(config.DEVICE)
        f_cat = full_data['firm_cat'].to(config.DEVICE)
        c_numeric = full_data['ceo_numeric'].to(config.DEVICE)
        c_cat = full_data['ceo_cat'].to(config.DEVICE)
        
        # Get embeddings before final projection
        f_embs = [emb(f_cat[:, i]) for i, emb in enumerate(model.firm_embeddings)]
        f_combined = torch.cat([f_numeric] + f_embs, dim=1)
        firm_embeddings = model.firm_tower(f_combined)
        
        c_embs = [emb(c_cat[:, i]) for i, emb in enumerate(model.ceo_embeddings)]
        c_combined = torch.cat([c_numeric] + c_embs, dim=1)
        ceo_embeddings = model.ceo_tower(c_combined)
        
        # Normalize
        firm_embeddings = firm_embeddings / firm_embeddings.norm(dim=1, keepdim=True)
        ceo_embeddings = ceo_embeddings / ceo_embeddings.norm(dim=1, keepdim=True)
        
        # Predictions
        logit_scale = model.logit_scale.exp()
        predictions = (firm_embeddings * ceo_embeddings).sum(dim=1) * logit_scale
        
    firm_emb_np = firm_embeddings.cpu().numpy()
    ceo_emb_np = ceo_embeddings.cpu().numpy()
    preds_np = predictions.cpu().numpy()
    
    df_clean = df_clean.copy()
    df_clean['predicted_match'] = preds_np
    
    print(f"Correlation between predicted and actual: {df_clean['match_means'].corr(df_clean['predicted_match']):.4f}")
    
    # 6. Analyze complementarity patterns
    print("\n=== COMPLEMENTARITY ANALYSIS ===")
    
    # Bin firms by size
    df_clean['size_bin'] = pd.qcut(df_clean['logatw'], 4, labels=['Small', 'Mid-Small', 'Mid-Large', 'Large'])
    
    # Bin CEOs by age
    df_clean['age_bin'] = pd.cut(df_clean['Age'], bins=[0, 45, 55, 65, 100], labels=['Young (<45)', 'Mid (45-55)', 'Senior (55-65)', 'Veteran (65+)'])
    
    print("\n--- Average Match Quality by Firm Size & CEO Age ---")
    pivot = df_clean.pivot_table(
        values='match_means',
        index='age_bin',
        columns='size_bin',
        aggfunc='mean',
        observed=False
    )
    print(pivot.round(3))
    
    # 7. Skill complementarity
    print("\n--- Match Quality by R&D Intensity & CEO Output Skill ---")
    # Use cut with explicit bins since rdintw has many zeros
    rd_q = df_clean['rdintw'].quantile([0, 0.5, 0.9, 1.0]).values
    df_clean['rd_bin'] = pd.cut(df_clean['rdintw'], bins=[-0.001, 0.001, rd_q[2], rd_q[3]+0.01], 
                                 labels=['No R&D', 'Some R&D', 'High R&D'])
    pivot_rd = df_clean.pivot_table(
        values='match_means',
        index='Output',
        columns='rd_bin',
        aggfunc='mean',
        observed=False
    )
    print(pivot_rd.round(3))
    
    print("\n--- Match Quality by Firm Performance & CEO Education ---")
    df_clean['perf_bin'] = pd.qcut(df_clean['exp_roa'], 3, labels=['Low ROA', 'Mid ROA', 'High ROA'])
    pivot_edu = df_clean.pivot_table(
        values='match_means',
        index='maxedu',
        columns='perf_bin',
        aggfunc='mean',
        observed=False
    )
    print(pivot_edu.round(3))
    
    # 9. Create visualization
    print("\n=== CREATING VISUALIZATIONS ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Predicted vs Actual
    ax1 = axes[0, 0]
    ax1.scatter(df_clean['match_means'], df_clean['predicted_match'], alpha=0.3, s=10)
    ax1.plot([-0.6, 0.9], [-0.6, 0.9], 'r--', lw=2)
    ax1.set_xlabel('Actual Match Quality')
    ax1.set_ylabel('Predicted Match Quality')
    ax1.set_title(f"Model Fit (Corr: {df_clean['match_means'].corr(df_clean['predicted_match']):.3f})")
    
    # Plot 2: Size-Age Complementarity Heatmap
    ax2 = axes[0, 1]
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax2)
    ax2.set_title('Match Quality: Firm Size × CEO Age')
    
    # Plot 3: Embedding Space (PCA of firm embeddings colored by size)
    ax3 = axes[1, 0]
    pca = PCA(n_components=2)
    firm_pca = pca.fit_transform(firm_emb_np)
    size_numeric = df_clean['logatw'].values
    scatter = ax3.scatter(firm_pca[:, 0], firm_pca[:, 1], c=size_numeric, cmap='viridis', alpha=0.5, s=10)
    plt.colorbar(scatter, ax=ax3, label='Log Assets')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_title('Firm Embedding Space (colored by size)')
    
    # Plot 4: CEO Embedding Space (colored by age)
    ax4 = axes[1, 1]
    ceo_pca = pca.fit_transform(ceo_emb_np)
    age_numeric = df_clean['Age'].values
    scatter2 = ax4.scatter(ceo_pca[:, 0], ceo_pca[:, 1], c=age_numeric, cmap='plasma', alpha=0.5, s=10)
    plt.colorbar(scatter2, ax=ax4, label='CEO Age')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_title('CEO Embedding Space (colored by age)')
    
    plt.tight_layout()
    plt.savefig('Output/match_analysis.png', dpi=150)
    print("Saved: Output/match_analysis.png")
    
    # 10. Find most "surprising" matches
    print("\n=== MOST SURPRISING MATCHES ===")
    df_clean['prediction_error'] = df_clean['match_means'] - df_clean['predicted_match']
    
    print("\n--- Unexpectedly GOOD Matches (high actual, low predicted) ---")
    surprising_good = df_clean.nlargest(5, 'prediction_error')[
        ['gvkey', 'match_exec_id', 'Age', 'logatw', 'rdintw', 'match_means', 'predicted_match']
    ]
    print(surprising_good.to_string(index=False))
    
    print("\n--- Unexpectedly BAD Matches (low actual, high predicted) ---")
    surprising_bad = df_clean.nsmallest(5, 'prediction_error')[
        ['gvkey', 'match_exec_id', 'Age', 'logatw', 'rdintw', 'match_means', 'predicted_match']
    ]
    print(surprising_bad.to_string(index=False))
    
    # 11. Summary statistics by key dimensions
    print("\n=== KEY INSIGHTS ===")
    
    # Gender differences
    gender_stats = df_clean.groupby('Gender')['match_means'].agg(['mean', 'std', 'count'])
    print("\n--- Match Quality by Gender ---")
    print(gender_stats.round(3))
    
    # Ivy league effect
    ivy_stats = df_clean.groupby('ivy')['match_means'].agg(['mean', 'std', 'count'])
    print("\n--- Match Quality by Ivy League Education ---")
    print(ivy_stats.round(3))
    
    # Mover effect (CEO mobility)
    if 'm' in df_clean.columns:
        mover_stats = df_clean.groupby('m')['match_means'].agg(['mean', 'std', 'count'])
        print("\n--- Match Quality by CEO Mobility (m) ---")
        print(mover_stats.round(3))
    
    print("\n=== ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()
