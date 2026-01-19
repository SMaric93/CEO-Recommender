#!/usr/bin/env python3
"""
DEEP DIVE: Advanced CEO-Firm Match Analysis

This script goes beyond basics to explore:
1. Industry-specific matching patterns
2. State-level CEO labor markets
3. Counterfactual analysis: What if CEOs were swapped?
4. Embedding clustering: CEO and Firm archetypes
5. Optimal matching: Who should lead which firms?
6. Tenure dynamics: How do matches evolve over time?
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import seaborn as sns
from scipy import stats

from ceo_firm_matching import (
    Config,
    DataProcessor,
    CEOFirmDataset,
    CEOFirmMatcher,
    train_model,
)

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

def main():
    config = Config()
    print(f"🚀 Deep Dive Analysis on {config.DEVICE}")
    
    # 1. Load and prepare data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    processor = DataProcessor(config)
    raw_df = processor.load_data()
    df_clean = processor.prepare_features(raw_df)
    
    print(f"Observations: {len(df_clean):,}")
    print(f"Unique Firms: {df_clean['gvkey'].nunique():,}")
    print(f"Unique CEOs: {df_clean['match_exec_id'].nunique():,}")
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    train_df, val_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    processor.fit(train_df)
    
    train_data = processor.transform(train_df)
    val_data = processor.transform(val_df)
    
    train_loader = DataLoader(CEOFirmDataset(train_data), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CEOFirmDataset(val_data), batch_size=config.BATCH_SIZE, shuffle=False)
    
    model = train_model(train_loader, val_loader, train_data, config)
    
    # Extract embeddings
    print("\n" + "="*60)
    print("EXTRACTING EMBEDDINGS")
    print("="*60)
    full_data = processor.transform(df_clean)
    
    model.eval()
    with torch.no_grad():
        f_numeric = full_data['firm_numeric'].to(config.DEVICE)
        f_cat = full_data['firm_cat'].to(config.DEVICE)
        c_numeric = full_data['ceo_numeric'].to(config.DEVICE)
        c_cat = full_data['ceo_cat'].to(config.DEVICE)
        
        f_embs = [emb(f_cat[:, i]) for i, emb in enumerate(model.firm_embeddings)]
        f_combined = torch.cat([f_numeric] + f_embs, dim=1)
        firm_embeddings = model.firm_tower(f_combined)
        
        c_embs = [emb(c_cat[:, i]) for i, emb in enumerate(model.ceo_embeddings)]
        c_combined = torch.cat([c_numeric] + c_embs, dim=1)
        ceo_embeddings = model.ceo_tower(c_combined)
        
        firm_embeddings = firm_embeddings / firm_embeddings.norm(dim=1, keepdim=True)
        ceo_embeddings = ceo_embeddings / ceo_embeddings.norm(dim=1, keepdim=True)
        
        logit_scale = model.logit_scale.exp()
        predictions = (firm_embeddings * ceo_embeddings).sum(dim=1) * logit_scale
        
    firm_emb_np = firm_embeddings.cpu().numpy()
    ceo_emb_np = ceo_embeddings.cpu().numpy()
    
    df_clean = df_clean.copy()
    df_clean['predicted_match'] = predictions.cpu().numpy()
    
    print(f"Model R²: {df_clean['match_means'].corr(df_clean['predicted_match'])**2:.3f}")
    
    # ================================================================
    # ANALYSIS 1: INDUSTRY PATTERNS
    # ================================================================
    print("\n" + "="*60)
    print("📊 ANALYSIS 1: INDUSTRY-SPECIFIC PATTERNS")
    print("="*60)
    
    industry_stats = df_clean.groupby('compindustry').agg({
        'match_means': ['mean', 'std', 'count'],
        'logatw': 'mean',
        'rdintw': 'mean',
        'Age': 'mean'
    }).round(3)
    industry_stats.columns = ['Match Mean', 'Match Std', 'N', 'Avg Size', 'Avg R&D', 'Avg CEO Age']
    industry_stats = industry_stats.sort_values('Match Mean', ascending=False)
    
    print("\n--- Match Quality by Industry ---")
    print(industry_stats.head(15).to_string())
    
    # Top and bottom industries
    print(f"\n🏆 Best Industry: {industry_stats.index[0]} (Match: {industry_stats.iloc[0]['Match Mean']:.3f})")
    print(f"📉 Worst Industry: {industry_stats.index[-1]} (Match: {industry_stats.iloc[-1]['Match Mean']:.3f})")
    
    # ================================================================
    # ANALYSIS 2: STATE-LEVEL CEO MARKETS
    # ================================================================
    print("\n" + "="*60)
    print("🗺️  ANALYSIS 2: STATE-LEVEL CEO MARKETS")
    print("="*60)
    
    state_stats = df_clean.groupby('ba_state').agg({
        'match_means': ['mean', 'count'],
        'Age': 'mean',
        'ivy': 'mean',
        'logatw': 'mean'
    }).round(3)
    state_stats.columns = ['Match Mean', 'N', 'Avg Age', 'Ivy %', 'Avg Size']
    state_stats = state_stats[state_stats['N'] >= 50].sort_values('Match Mean', ascending=False)
    
    print("\n--- Top 10 States by Match Quality ---")
    print(state_stats.head(10).to_string())
    
    print("\n--- Bottom 5 States by Match Quality ---")
    print(state_stats.tail(5).to_string())
    
    # ================================================================
    # ANALYSIS 3: CEO ARCHETYPES (CLUSTERING)
    # ================================================================
    print("\n" + "="*60)
    print("👔 ANALYSIS 3: CEO ARCHETYPES")
    print("="*60)
    
    # Cluster CEOs based on embeddings
    n_clusters = 5
    kmeans_ceo = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clean['ceo_cluster'] = kmeans_ceo.fit_predict(ceo_emb_np)
    
    ceo_clusters = df_clean.groupby('ceo_cluster').agg({
        'Age': 'mean',
        'ivy': 'mean',
        'maxedu': 'mean',
        'Output': 'mean',
        'Throghput': 'mean',
        'Peripheral': 'mean',
        'match_means': 'mean',
        'match_exec_id': 'count'
    }).round(3)
    ceo_clusters.columns = ['Avg Age', 'Ivy %', 'Education', 'Output', 'Throughput', 'Peripheral', 'Match', 'Count']
    
    print("\n--- CEO Archetypes (5 Clusters) ---")
    print(ceo_clusters.to_string())
    
    # Name the archetypes
    print("\n🏷️  Archetype Interpretations:")
    for i in range(n_clusters):
        row = ceo_clusters.loc[i]
        age_label = "Young" if row['Avg Age'] < 50 else ("Senior" if row['Avg Age'] > 58 else "Mid-career")
        edu_label = "Elite" if row['Ivy %'] > 0.3 else ("Educated" if row['Education'] > 3 else "Practical")
        skill_label = "Output-focused" if row['Output'] > 0.6 else ("Throughput" if row['Throughput'] > 0.5 else "Generalist")
        print(f"  Cluster {i}: {age_label} {edu_label} {skill_label} (Match: {row['Match']:.3f})")
    
    # ================================================================
    # ANALYSIS 4: FIRM ARCHETYPES
    # ================================================================
    print("\n" + "="*60)
    print("🏢 ANALYSIS 4: FIRM ARCHETYPES")
    print("="*60)
    
    kmeans_firm = KMeans(n_clusters=5, random_state=42, n_init=10)
    df_clean['firm_cluster'] = kmeans_firm.fit_predict(firm_emb_np)
    
    firm_clusters = df_clean.groupby('firm_cluster').agg({
        'logatw': 'mean',
        'exp_roa': 'mean',
        'rdintw': 'mean',
        'leverage': 'mean',
        'boardindpw': 'mean',
        'match_means': 'mean',
        'gvkey': 'count'
    }).round(3)
    firm_clusters.columns = ['Size', 'ROA', 'R&D', 'Leverage', 'Board Indep', 'Match', 'Count']
    
    print("\n--- Firm Archetypes (5 Clusters) ---")
    print(firm_clusters.to_string())
    
    # ================================================================
    # ANALYSIS 5: CROSS-TABULATION - WHO MATCHES WITH WHOM?
    # ================================================================
    print("\n" + "="*60)
    print("🔀 ANALYSIS 5: CEO-FIRM CLUSTER MATCHING")
    print("="*60)
    
    cross_match = df_clean.pivot_table(
        values='match_means',
        index='ceo_cluster',
        columns='firm_cluster',
        aggfunc='mean',
        observed=False
    ).round(3)
    
    print("\n--- Match Quality: CEO Cluster × Firm Cluster ---")
    print(cross_match.to_string())
    
    # Find optimal pairings
    print("\n🎯 Optimal CEO-Firm Pairings:")
    for ceo_c in range(n_clusters):
        best_firm = cross_match.loc[ceo_c].idxmax()
        match_val = cross_match.loc[ceo_c, best_firm]
        print(f"  CEO Cluster {ceo_c} → Firm Cluster {best_firm} (Match: {match_val:.3f})")
    
    # ================================================================
    # ANALYSIS 6: TENURE DYNAMICS
    # ================================================================
    print("\n" + "="*60)
    print("⏱️  ANALYSIS 6: TENURE DYNAMICS")
    print("="*60)
    
    df_clean['tenure_bin'] = pd.cut(df_clean['tenure'], 
                                     bins=[-1, 2, 5, 10, 20, 100],
                                     labels=['New (0-2)', 'Settling (3-5)', 'Established (6-10)', 
                                            'Veteran (11-20)', 'Legacy (20+)'])
    
    tenure_stats = df_clean.groupby('tenure_bin', observed=False).agg({
        'match_means': ['mean', 'std'],
        'predicted_match': 'mean',
        'match_exec_id': 'count'
    }).round(3)
    tenure_stats.columns = ['Actual Match', 'Std', 'Predicted Match', 'N']
    
    print("\n--- Match Quality Over CEO Tenure ---")
    print(tenure_stats.to_string())
    
    # ================================================================
    # ANALYSIS 7: COUNTERFACTUAL - WHAT IF WE SWAPPED CEOs?
    # ================================================================
    print("\n" + "="*60)
    print("🔄 ANALYSIS 7: COUNTERFACTUAL MATCHING")
    print("="*60)
    
    # Get unique firms and CEOs (most recent observation)
    latest = df_clean.sort_values('fiscalyear').groupby('gvkey').last().reset_index()
    latest_ceos = df_clean.sort_values('fiscalyear').groupby('match_exec_id').last().reset_index()
    
    print(f"Unique firm-years for counterfactual: {len(latest)}")
    
    # Sample for computational feasibility
    sample_n = min(200, len(latest))
    sample_firms = latest.sample(sample_n, random_state=42)
    sample_ceos = latest_ceos.sample(sample_n, random_state=42)
    
    # Get embeddings for sample
    firm_idx = sample_firms.index.tolist()
    ceo_idx = sample_ceos.index.tolist()
    
    # For each firm, find the BEST matching CEO from our sample
    sample_firm_emb = torch.tensor(firm_emb_np[firm_idx]).to(config.DEVICE)
    sample_ceo_emb = torch.tensor(ceo_emb_np[ceo_idx]).to(config.DEVICE)
    
    # Compute all pairwise similarities
    with torch.no_grad():
        # (n_firms, n_ceos)
        similarity_matrix = torch.mm(sample_firm_emb, sample_ceo_emb.t()) * logit_scale
    
    sim_np = similarity_matrix.cpu().numpy()
    
    # Find optimal assignment for each firm
    optimal_matches = []
    for i, (firm_row_idx, firm_row) in enumerate(sample_firms.iterrows()):
        best_ceo_pos = sim_np[i].argmax()
        best_ceo_row = sample_ceos.iloc[best_ceo_pos]
        
        actual_match = firm_row['predicted_match']
        counterfactual_match = sim_np[i, best_ceo_pos]
        
        optimal_matches.append({
            'firm_gvkey': firm_row['gvkey'],
            'firm_size': firm_row['logatw'],
            'firm_rd': firm_row['rdintw'],
            'actual_ceo_age': firm_row['Age'],
            'optimal_ceo_age': best_ceo_row['Age'],
            'actual_ceo_edu': firm_row['maxedu'],
            'optimal_ceo_edu': best_ceo_row['maxedu'],
            'actual_match': actual_match,
            'optimal_match': counterfactual_match,
            'improvement': counterfactual_match - actual_match
        })
    
    counterfactual_df = pd.DataFrame(optimal_matches)
    
    print("\n--- Counterfactual Matching Results ---")
    print(f"Average actual match: {counterfactual_df['actual_match'].mean():.3f}")
    print(f"Average optimal match: {counterfactual_df['optimal_match'].mean():.3f}")
    print(f"Average improvement: {counterfactual_df['improvement'].mean():.3f}")
    
    # Firms with biggest potential improvement
    print("\n🚀 Top 5 Firms with Most Improvement Potential:")
    top_improve = counterfactual_df.nlargest(5, 'improvement')
    for _, row in top_improve.iterrows():
        print(f"  Firm {int(row['firm_gvkey'])}: {row['actual_match']:.2f} → {row['optimal_match']:.2f} (+{row['improvement']:.2f})")
        print(f"    Current CEO: Age {row['actual_ceo_age']:.0f}, Edu {row['actual_ceo_edu']:.0f}")
        print(f"    Optimal CEO: Age {row['optimal_ceo_age']:.0f}, Edu {row['optimal_ceo_edu']:.0f}")
    
    # ================================================================
    # ANALYSIS 8: MISALLOCATION ANALYSIS
    # ================================================================
    print("\n" + "="*60)
    print("⚠️  ANALYSIS 8: MISALLOCATION ANALYSIS")
    print("="*60)
    
    # Firms currently with poor matches that could do better
    df_clean['match_gap'] = df_clean['predicted_match'] - df_clean['match_means']
    
    print("\n--- Match Gap Distribution ---")
    print(df_clean['match_gap'].describe().round(3))
    
    # Over-matched (doing better than predicted)
    over_matched = df_clean[df_clean['match_gap'] < -0.3]
    print(f"\n😊 Over-matched (actual >> predicted): {len(over_matched)} observations")
    if len(over_matched) > 0:
        print("  Characteristics:")
        print(f"    Avg Age: {over_matched['Age'].mean():.1f}")
        print(f"    Avg Firm Size: {over_matched['logatw'].mean():.2f}")
        print(f"    Ivy %: {over_matched['ivy'].mean():.1%}")
    
    # Under-matched (doing worse than predicted)
    under_matched = df_clean[df_clean['match_gap'] > 0.3]
    print(f"\n😞 Under-matched (actual << predicted): {len(under_matched)} observations")
    if len(under_matched) > 0:
        print("  Characteristics:")
        print(f"    Avg Age: {under_matched['Age'].mean():.1f}")
        print(f"    Avg Firm Size: {under_matched['logatw'].mean():.2f}")
        print(f"    Ivy %: {under_matched['ivy'].mean():.1%}")
    
    # ================================================================
    # ANALYSIS 9: VISUALIZATION
    # ================================================================
    print("\n" + "="*60)
    print("📈 CREATING VISUALIZATIONS")
    print("="*60)
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. CEO-Firm Cluster Heatmap
    ax1 = fig.add_subplot(2, 3, 1)
    sns.heatmap(cross_match, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax1)
    ax1.set_title('CEO × Firm Cluster Match Quality', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Firm Cluster')
    ax1.set_ylabel('CEO Cluster')
    
    # 2. Tenure Dynamics
    ax2 = fig.add_subplot(2, 3, 2)
    tenure_plot = df_clean.groupby('tenure_bin', observed=False)['match_means'].mean()
    bars = ax2.bar(range(len(tenure_plot)), tenure_plot.values, color='steelblue', edgecolor='black')
    ax2.set_xticks(range(len(tenure_plot)))
    ax2.set_xticklabels(tenure_plot.index, rotation=45, ha='right')
    ax2.set_ylabel('Average Match Quality')
    ax2.set_title('Match Quality vs CEO Tenure', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 3. t-SNE of CEO embeddings with clusters
    ax3 = fig.add_subplot(2, 3, 3)
    subsample_idx = np.random.choice(len(ceo_emb_np), min(2000, len(ceo_emb_np)), replace=False)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    ceo_tsne = tsne.fit_transform(ceo_emb_np[subsample_idx])
    scatter = ax3.scatter(ceo_tsne[:, 0], ceo_tsne[:, 1], 
                          c=df_clean['ceo_cluster'].values[subsample_idx], 
                          cmap='tab10', alpha=0.6, s=15)
    ax3.set_title('CEO Embedding Space (t-SNE)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    
    # 4. Counterfactual improvement distribution
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(counterfactual_df['improvement'], bins=30, color='green', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', lw=2)
    ax4.axvline(x=counterfactual_df['improvement'].mean(), color='blue', linestyle='-', lw=2, 
                label=f"Mean: {counterfactual_df['improvement'].mean():.2f}")
    ax4.set_xlabel('Match Improvement (Optimal - Actual)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Counterfactual Match Improvement', fontsize=12, fontweight='bold')
    ax4.legend()
    
    # 5. Industry rankings
    ax5 = fig.add_subplot(2, 3, 5)
    top_industries = industry_stats.head(10)
    colors = ['green' if x > 0 else 'red' for x in top_industries['Match Mean']]
    ax5.barh(range(len(top_industries)), top_industries['Match Mean'], color=colors, edgecolor='black')
    ax5.set_yticks(range(len(top_industries)))
    ax5.set_yticklabels(top_industries.index)
    ax5.axvline(x=0, color='black', linestyle='-', lw=1)
    ax5.set_xlabel('Average Match Quality')
    ax5.set_title('Top Industries by Match Quality', fontsize=12, fontweight='bold')
    ax5.invert_yaxis()
    
    # 6. Age-Size Interaction 3D-like contour
    ax6 = fig.add_subplot(2, 3, 6)
    size_bins = pd.qcut(df_clean['logatw'], 10, labels=False, duplicates='drop')
    age_bins = pd.cut(df_clean['Age'], 10, labels=False)
    
    heatmap_data = df_clean.groupby([age_bins, size_bins])['match_means'].mean().unstack()
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, ax=ax6, cbar_kws={'label': 'Match Quality'})
    ax6.set_xlabel('Firm Size Decile')
    ax6.set_ylabel('CEO Age Decile')
    ax6.set_title('Size × Age Interaction (Fine-grained)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Output/deep_dive_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: Output/deep_dive_analysis.png")
    
    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n" + "="*60)
    print("📋 EXECUTIVE SUMMARY")
    print("="*60)
    
    print("""
    KEY FINDINGS:
    
    1. 🎯 MODEL FIT: R² = {:.1%} - The model captures meaningful patterns
    
    2. 👔 CEO ARCHETYPES: 5 distinct types emerged from embeddings
       - Different archetypes have dramatically different match outcomes
       
    3. 🏢 FIRM ARCHETYPES: 5 firm types with distinct optimal CEO profiles
    
    4. 🔄 COUNTERFACTUAL: Average potential improvement of {:.2f}
       - Many CEO-Firm matches are suboptimal
       
    5. ⏱️ TENURE: Match quality changes over CEO tenure
       - {} tenure yields best matches
       
    6. 🗺️ GEOGRAPHY: {} has best CEO-Firm matches
    
    7. 📊 INDUSTRY: {} leads, {} lags
    """.format(
        df_clean['match_means'].corr(df_clean['predicted_match'])**2,
        counterfactual_df['improvement'].mean(),
        tenure_stats['Actual Match'].idxmax(),
        state_stats.index[0] if len(state_stats) > 0 else "N/A",
        industry_stats.index[0],
        industry_stats.index[-1]
    ))
    
    print("\n🎉 DEEP DIVE COMPLETE!")

if __name__ == "__main__":
    main()
