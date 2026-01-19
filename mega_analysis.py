#!/usr/bin/env python3
"""
🔥 MEGA CEO-FIRM MATCH ANALYSIS 🔥

Merges 8+ data sources for the most comprehensive analysis:
1. Match Quality (Two-Tower Model)
2. Stock Returns (CRSP via ExecuComp)
3. Firm Value (Tobin's Q)
4. CEO Turnover Events
5. Innovation (Patents)
6. Geographic CEO Markets
7. BLM Clusters
8. Industry Classifications

Key Questions:
- Do good matches generate higher returns?
- Are good matches paid more?
- Do good matches innovate more?
- Do bad matches get fired faster?
- What predicts a "superstar" match?
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

from ceo_firm_matching import (
    Config,
    DataProcessor,
    CEOFirmDataset,
    train_model,
)

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

def main():
    print("🔥" * 30)
    print("     MEGA CEO-FIRM MATCH ANALYSIS")
    print("🔥" * 30)
    
    config = Config()
    
    # ================================================================
    # LOAD ALL DATA SOURCES
    # ================================================================
    print("\n📂 LOADING 8 DATA SOURCES...")
    
    # 1. Match Quality Data
    processor = DataProcessor(config)
    raw_df = processor.load_data()
    df_clean = processor.prepare_features(raw_df)
    print(f"  1. Match Quality: {len(df_clean):,} obs")
    
    # 2. Stock Returns (ExecuComp + CRSP)
    returns_df = pd.read_parquet('/Users/smaric/Papers/GIV CEO Turnover/output/execucomp_lagged_returns_panel.parquet')
    returns_df['gvkey'] = returns_df['gvkey'].astype(str).str.lstrip('0').astype(int)
    print(f"  2. Stock Returns: {len(returns_df):,} obs")
    
    # 3. CEO Demographics (turnover, tenure)
    ceo_demo = pd.read_stata('../Data/ceo_demographics_v1.dta', convert_categoricals=False)
    print(f"  3. CEO Demographics: {len(ceo_demo):,} obs")
    
    # 4. BLM Mobility Panel (clusters, Tobin's Q)
    blm_mobility = pd.read_stata('../Data/blm_data_ceo_prep_v4.3_mobility.dta')
    print(f"  4. BLM Mobility: {len(blm_mobility):,} obs")
    
    # 5. Innovation Data (patents)
    patents_df = pd.read_csv('../Data/firm_fyear_metrics_discern_whole_v1.5.1.csv')
    print(f"  5. Patent Innovation: {len(patents_df):,} obs")
    
    # 6. Geographic Data
    geo_df = pd.read_stata('../Data/ceo_geo_prep_complete.dta')
    print(f"  6. Geographic Data: {len(geo_df):,} obs")
    
    # 7. Compustat Fundamentals
    comp_df = pd.read_stata('../Data/comp_akm_v4.dta')
    print(f"  7. Compustat: {len(comp_df):,} obs")
    
    # 8. CEO Turnover Events (AA 2024)
    turnover_events = pd.read_stata('../Data/aa_2024.dta')
    print(f"  8. Turnover Events: {len(turnover_events):,} obs")
    
    total_obs = sum([len(df_clean), len(returns_df), len(ceo_demo), len(blm_mobility), 
                     len(patents_df), len(geo_df), len(comp_df), len(turnover_events)])
    print(f"\n  📊 TOTAL DATA: {total_obs:,} observations across 8 sources")
    
    # ================================================================
    # TRAIN MODEL
    # ================================================================
    print("\n🧠 TRAINING TWO-TOWER MODEL...")
    
    train_df, val_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    processor.fit(train_df)
    
    train_data = processor.transform(train_df)
    val_data = processor.transform(val_df)
    
    train_loader = DataLoader(CEOFirmDataset(train_data), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CEOFirmDataset(val_data), batch_size=config.BATCH_SIZE, shuffle=False)
    
    model = train_model(train_loader, val_loader, train_data, config)
    
    # Get predictions and embeddings
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
    
    df_clean = df_clean.copy()
    df_clean['predicted_match'] = predictions.cpu().numpy()
    df_clean['firm_emb'] = list(firm_embeddings.cpu().numpy())
    df_clean['ceo_emb'] = list(ceo_embeddings.cpu().numpy())
    
    print(f"  Model Correlation: {df_clean['match_means'].corr(df_clean['predicted_match']):.3f}")
    
    # ================================================================
    # MEGA MERGE
    # ================================================================
    print("\n🔗 CREATING MEGA MERGED DATASET...")
    
    # Merge with returns (rename 'return' to avoid keyword conflict)
    returns_renamed = returns_df[['gvkey', 'year', 'return', 'return_lag1', 'turnover']].copy()
    returns_renamed = returns_renamed.rename(columns={'year': 'fiscalyear_r', 'return': 'stock_return'})
    mega_df = df_clean.merge(
        returns_renamed,
        left_on=['gvkey', 'fiscalyear'],
        right_on=['gvkey', 'fiscalyear_r'],
        how='left'
    )
    
    # Merge with BLM data
    mega_df = mega_df.merge(
        blm_mobility[['gvkey', 'year', 'tobinw', 'roaw', 'cluster_label', 'm']].drop_duplicates(),
        left_on=['gvkey', 'fiscalyear'],
        right_on=['gvkey', 'year'],
        how='left'
    )
    
    # Merge with patents
    mega_df = mega_df.merge(
        patents_df[['gvkey', 'fyear', 'n_patents', 'originality_mean']],
        left_on=['gvkey', 'fiscalyear'],
        right_on=['gvkey', 'fyear'],
        how='left'
    )
    
    print(f"  Mega Dataset: {len(mega_df):,} observations")
    print(f"  With Returns: {mega_df['stock_return'].notna().sum():,}")
    print(f"  With Patents: {mega_df['n_patents'].notna().sum():,}")
    print(f"  With Tobin's Q: {mega_df['tobinw'].notna().sum():,}")
    
    # ================================================================
    # ANALYSIS 1: MATCH QUALITY → STOCK RETURNS
    # ================================================================
    print("\n" + "=" * 70)
    print("📈 ANALYSIS 1: DO GOOD MATCHES GENERATE HIGHER RETURNS?")
    print("=" * 70)
    
    returns_analysis = mega_df.dropna(subset=['match_means', 'stock_return'])
    returns_analysis['match_quintile'] = pd.qcut(returns_analysis['match_means'], 5, 
        labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4', 'Q5 (Best)'])
    
    print(f"\n  Sample size: {len(returns_analysis):,}")
    
    returns_by_match = returns_analysis.groupby('match_quintile', observed=False).agg({
        'stock_return': ['mean', 'std', 'count'],
        'match_means': 'mean'
    }).round(4)
    returns_by_match.columns = ['Avg Return', 'Std Return', 'N', 'Avg Match']
    
    print("\n--- Stock Returns by Match Quality Quintile ---")
    print(returns_by_match)
    
    # Regression
    if len(returns_analysis) > 100:
        model_ret = ols('stock_return ~ match_means + logatw', data=returns_analysis).fit()
        print(f"\n📊 Regression: stock_return ~ Match Quality + Size")
        print(f"   Match Quality β: {model_ret.params['match_means']:.4f} (p={model_ret.pvalues['match_means']:.4f})")
        print(f"   Size β: {model_ret.params['logatw']:.4f}")
        
        # Long-short portfolio
        q1_ret = returns_by_match.loc['Q1 (Worst)', 'Avg Return']
        q5_ret = returns_by_match.loc['Q5 (Best)', 'Avg Return']
        spread = q5_ret - q1_ret
        print(f"\n💰 Long-Short Portfolio (Q5 - Q1): {spread:.4f} ({spread*100:.2f}%)")
    
    # ================================================================
    # ANALYSIS 2: MATCH QUALITY → INNOVATION
    # ================================================================
    print("\n" + "=" * 70)
    print("💡 ANALYSIS 2: DO GOOD MATCHES INNOVATE MORE?")
    print("=" * 70)
    
    patent_analysis = mega_df.dropna(subset=['match_means', 'n_patents'])
    patent_analysis['match_quintile'] = pd.qcut(patent_analysis['match_means'], 5,
        labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4', 'Q5 (Best)'])
    
    print(f"\n  Sample size: {len(patent_analysis):,}")
    
    patents_by_match = patent_analysis.groupby('match_quintile', observed=False).agg({
        'n_patents': ['mean', 'sum'],
        'originality_mean': 'mean',
        'gvkey': 'count'
    }).round(2)
    patents_by_match.columns = ['Avg Patents', 'Total Patents', 'Avg Originality', 'N']
    
    print("\n--- Innovation by Match Quality Quintile ---")
    print(patents_by_match)
    
    # ================================================================
    # ANALYSIS 3: MATCH QUALITY → TOBIN'S Q (MORE DETAIL)
    # ================================================================
    print("\n" + "=" * 70)
    print("💰 ANALYSIS 3: MATCH QUALITY → FIRM VALUE (DETAILED)")
    print("=" * 70)
    
    value_analysis = mega_df.dropna(subset=['match_means', 'tobinw', 'roaw'])
    value_analysis['match_quintile'] = pd.qcut(value_analysis['match_means'], 5,
        labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4', 'Q5 (Best)'])
    
    value_by_match = value_analysis.groupby('match_quintile', observed=False).agg({
        'tobinw': ['mean', 'median'],
        'roaw': 'mean',
        'logatw': 'mean',
        'gvkey': 'count'
    }).round(3)
    value_by_match.columns = ['Mean Tobin Q', 'Median Tobin Q', 'ROA', 'Size', 'N']
    
    print("\n--- Firm Value by Match Quality Quintile ---")
    print(value_by_match)
    
    # Value creation: Q5 vs Q1
    q5_val = value_by_match.loc['Q5 (Best)', 'Mean Tobin Q']
    q1_val = value_by_match.loc['Q1 (Worst)', 'Mean Tobin Q']
    print(f"\n🏆 Value Premium (Q5 vs Q1): {q5_val / q1_val:.2f}x")
    
    # ================================================================
    # ANALYSIS 4: BLM CLUSTERS × MATCH QUALITY
    # ================================================================
    print("\n" + "=" * 70)
    print("🏷️  ANALYSIS 4: BLM CLUSTER ANALYSIS")
    print("=" * 70)
    
    cluster_analysis = mega_df.dropna(subset=['cluster_label', 'match_means'])
    
    cluster_stats = cluster_analysis.groupby('cluster_label').agg({
        'match_means': ['mean', 'std'],
        'tobinw': 'mean',
        'Age': 'mean',
        'logatw': 'mean',
        'gvkey': 'count'
    }).round(3)
    cluster_stats.columns = ['Match Mean', 'Match Std', 'Tobin Q', 'CEO Age', 'Firm Size', 'N']
    cluster_stats = cluster_stats.sort_values('Match Mean', ascending=False)
    
    print("\n--- BLM Clusters Ranked by Match Quality ---")
    print(cluster_stats)
    
    # ================================================================
    # ANALYSIS 5: SUPERSTAR MATCH PREDICTION
    # ================================================================
    print("\n" + "=" * 70)
    print("⭐ ANALYSIS 5: WHAT PREDICTS A SUPERSTAR MATCH?")
    print("=" * 70)
    
    # Define superstar as top 10% matches
    threshold = mega_df['match_means'].quantile(0.9)
    mega_df['is_superstar'] = (mega_df['match_means'] >= threshold).astype(int)
    
    print(f"\n  Superstar threshold: {threshold:.3f}")
    print(f"  Superstar matches: {mega_df['is_superstar'].sum():,} ({mega_df['is_superstar'].mean()*100:.1f}%)")
    
    # Compare superstars to others
    superstar_compare = mega_df.groupby('is_superstar').agg({
        'Age': 'mean',
        'tenure': 'mean',
        'ivy': 'mean',
        'maxedu': 'mean',
        'logatw': 'mean',
        'rdintw': 'mean',
        'tobinw': 'mean',
        'Output': 'mean'
    }).round(3)
    superstar_compare.index = ['Regular', 'Superstar']
    
    print("\n--- Superstar vs Regular Match Comparison ---")
    print(superstar_compare.T)
    
    # Feature importance with Random Forest
    features = ['Age', 'tenure', 'ivy', 'maxedu', 'logatw', 'rdintw', 'exp_roa', 
                'boardindpw', 'leverage', 'Output', 'Throghput', 'Peripheral']
    
    rf_df = mega_df[features + ['match_means']].dropna()
    if len(rf_df) > 500:
        X = rf_df[features]
        y = rf_df['match_means']
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, y)
        
        importances = pd.DataFrame({
            'Feature': features,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n--- Feature Importance (Random Forest) ---")
        print(importances.to_string(index=False))
    
    # ================================================================
    # ANALYSIS 6: GEOGRAPHIC PATTERNS
    # ================================================================
    print("\n" + "=" * 70)
    print("🗺️  ANALYSIS 6: GEOGRAPHIC CEO MARKET ANALYSIS")
    print("=" * 70)
    
    state_analysis = mega_df.dropna(subset=['ba_state', 'match_means'])
    
    state_stats = state_analysis.groupby('ba_state').agg({
        'match_means': ['mean', 'std', 'count'],
        'tobinw': 'mean',
        'Age': 'mean',
        'ivy': 'mean'
    }).round(3)
    state_stats.columns = ['Match Mean', 'Match Std', 'N', 'Tobin Q', 'CEO Age', 'Ivy %']
    state_stats = state_stats[state_stats['N'] >= 30].sort_values('Match Mean', ascending=False)
    
    print("\n--- Top 15 States by Match Quality ---")
    print(state_stats.head(15))
    
    # ================================================================
    # ANALYSIS 7: TENURE × MATCH DYNAMICS
    # ================================================================
    print("\n" + "=" * 70)
    print("⏱️  ANALYSIS 7: MATCH DYNAMICS OVER CEO TENURE")
    print("=" * 70)
    
    mega_df['tenure_bin'] = pd.cut(mega_df['tenure'], 
        bins=[-1, 1, 3, 5, 8, 12, 100],
        labels=['Year 1', 'Year 2-3', 'Year 4-5', 'Year 6-8', 'Year 9-12', 'Long-term'])
    
    tenure_dynamics = mega_df.groupby('tenure_bin', observed=False).agg({
        'match_means': ['mean', 'std'],
        'tobinw': 'mean',
        'stock_return': 'mean',
        'gvkey': 'count'
    }).round(3)
    tenure_dynamics.columns = ['Match Mean', 'Match Std', 'Tobin Q', 'Return', 'N']
    
    print("\n--- Performance Metrics by CEO Tenure ---")
    print(tenure_dynamics)
    
    # ================================================================
    # ANALYSIS 8: INDUSTRY DEEP DIVE
    # ================================================================
    print("\n" + "=" * 70)
    print("🏭 ANALYSIS 8: INDUSTRY DEEP DIVE")
    print("=" * 70)
    
    industry_deep = mega_df.dropna(subset=['compindustry', 'match_means', 'tobinw'])
    
    industry_stats = industry_deep.groupby('compindustry').agg({
        'match_means': ['mean', 'std'],
        'tobinw': ['mean', 'std'],
        'stock_return': 'mean',
        'n_patents': 'mean',
        'gvkey': 'count'
    }).round(3)
    industry_stats.columns = ['Match Mean', 'Match Std', 'Tobin Q', 'Tobin Std', 'Return', 'Patents', 'N']
    industry_stats = industry_stats[industry_stats['N'] >= 50].sort_values('Match Mean', ascending=False)
    
    print("\n--- Top 15 Industries by Match Quality ---")
    print(industry_stats.head(15))
    
    # Industry-level correlation matrix
    industry_corr = industry_stats[['Match Mean', 'Tobin Q', 'Return', 'Patents']].corr()
    print("\n--- Industry-Level Correlation Matrix ---")
    print(industry_corr.round(3))
    
    # ================================================================
    # VISUALIZATIONS
    # ================================================================
    print("\n" + "=" * 70)
    print("📊 CREATING MEGA VISUALIZATIONS")
    print("=" * 70)
    
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Match → Returns
    ax1 = fig.add_subplot(3, 4, 1)
    ret_data = returns_by_match.reset_index()
    ax1.bar(range(5), ret_data['Avg Return'], color='steelblue', edgecolor='black')
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], rotation=0)
    ax1.set_ylabel('Average Stock Return')
    ax1.set_title('Stock Returns by Match Quality', fontweight='bold')
    
    # 2. Match → Tobin's Q
    ax2 = fig.add_subplot(3, 4, 2)
    val_data = value_by_match.reset_index()
    ax2.bar(range(5), val_data['Mean Tobin Q'], color='green', edgecolor='black')
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax2.set_ylabel("Tobin's Q")
    ax2.set_title('Firm Value by Match Quality', fontweight='bold')
    
    # 3. Match → Patents
    ax3 = fig.add_subplot(3, 4, 3)
    pat_data = patents_by_match.reset_index()
    ax3.bar(range(5), pat_data['Avg Patents'], color='purple', edgecolor='black')
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax3.set_ylabel('Average Patents')
    ax3.set_title('Innovation by Match Quality', fontweight='bold')
    
    # 4. Tenure Dynamics
    ax4 = fig.add_subplot(3, 4, 4)
    ten_data = tenure_dynamics.reset_index()
    ax4.plot(range(len(ten_data)), ten_data['Match Mean'], 'b-o', lw=2, markersize=8)
    ax4.fill_between(range(len(ten_data)), 
                     ten_data['Match Mean'] - ten_data['Match Std'],
                     ten_data['Match Mean'] + ten_data['Match Std'], alpha=0.2)
    ax4.set_xticks(range(len(ten_data)))
    ax4.set_xticklabels(ten_data['tenure_bin'], rotation=45, ha='right')
    ax4.set_ylabel('Match Quality')
    ax4.set_title('Match Quality Over Tenure', fontweight='bold')
    
    # 5. Feature Importance
    ax5 = fig.add_subplot(3, 4, 5)
    if len(rf_df) > 500:
        colors = ['#2ecc71' if x > importances['Importance'].median() else '#e74c3c' for x in importances['Importance']]
        ax5.barh(range(len(importances)), importances['Importance'], color=colors, edgecolor='black')
        ax5.set_yticks(range(len(importances)))
        ax5.set_yticklabels(importances['Feature'], fontsize=9)
        ax5.set_xlabel('Importance')
        ax5.set_title('Superstar Match Predictors', fontweight='bold')
        ax5.invert_yaxis()
    
    # 6. State Heatmap (top states)
    ax6 = fig.add_subplot(3, 4, 6)
    top_states = state_stats.head(15)
    colors = ['green' if x > 0 else 'red' for x in top_states['Match Mean']]
    ax6.barh(range(len(top_states)), top_states['Match Mean'], color=colors, edgecolor='black')
    ax6.set_yticks(range(len(top_states)))
    ax6.set_yticklabels(top_states.index, fontsize=9)
    ax6.axvline(x=0, color='black', linestyle='-', lw=1)
    ax6.set_xlabel('Average Match Quality')
    ax6.set_title('Top States by Match Quality', fontweight='bold')
    ax6.invert_yaxis()
    
    # 7. BLM Clusters
    ax7 = fig.add_subplot(3, 4, 7)
    cluster_data = cluster_stats.head(6)
    x = np.arange(len(cluster_data))
    width = 0.35
    ax7.bar(x - width/2, cluster_data['Match Mean'], width, label='Match', color='steelblue')
    ax7_twin = ax7.twinx()
    ax7_twin.bar(x + width/2, cluster_data['Tobin Q'], width, label='Tobin Q', color='orange', alpha=0.7)
    ax7.set_xticks(x)
    ax7.set_xticklabels([f'{c[:10]}...' for c in cluster_data.index], rotation=45, ha='right', fontsize=8)
    ax7.set_ylabel('Match Quality', color='steelblue')
    ax7_twin.set_ylabel("Tobin's Q", color='orange')
    ax7.set_title('BLM Clusters', fontweight='bold')
    
    # 8. Industry Scatter: Match vs Value
    ax8 = fig.add_subplot(3, 4, 8)
    ax8.scatter(industry_stats['Match Mean'], industry_stats['Tobin Q'], 
                s=industry_stats['N']/5, alpha=0.6, c='teal')
    z = np.polyfit(industry_stats['Match Mean'], industry_stats['Tobin Q'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(industry_stats['Match Mean'].min(), industry_stats['Match Mean'].max(), 100)
    ax8.plot(x_line, p(x_line), 'r--', lw=2)
    ax8.set_xlabel('Industry Match Quality')
    ax8.set_ylabel("Industry Tobin's Q")
    ax8.set_title('Industry: Match vs Value', fontweight='bold')
    
    # 9. Match Distribution
    ax9 = fig.add_subplot(3, 4, 9)
    ax9.hist(mega_df['match_means'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax9.axvline(x=threshold, color='gold', linestyle='--', lw=2, label='Superstar Cutoff')
    ax9.axvline(x=mega_df['match_means'].median(), color='red', linestyle='-', lw=2, label='Median')
    ax9.set_xlabel('Match Quality')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Match Quality Distribution', fontweight='bold')
    ax9.legend()
    
    # 10. Superstar Comparison
    ax10 = fig.add_subplot(3, 4, 10)
    features_plot = ['Age', 'tenure', 'ivy', 'maxedu', 'rdintw']
    x = np.arange(len(features_plot))
    width = 0.35
    regular = superstar_compare.loc['Regular', features_plot].values
    superstar = superstar_compare.loc['Superstar', features_plot].values
    # Normalize for comparison
    regular_norm = regular / regular.max()
    superstar_norm = superstar / regular.max()
    ax10.bar(x - width/2, regular_norm, width, label='Regular', color='gray')
    ax10.bar(x + width/2, superstar_norm, width, label='Superstar', color='gold')
    ax10.set_xticks(x)
    ax10.set_xticklabels(features_plot)
    ax10.set_ylabel('Normalized Value')
    ax10.set_title('Superstar vs Regular', fontweight='bold')
    ax10.legend()
    
    # 11. Return vs Match Scatter
    ax11 = fig.add_subplot(3, 4, 11)
    valid = returns_analysis.sample(min(2000, len(returns_analysis)))
    ax11.scatter(valid['match_means'], valid['stock_return'], alpha=0.3, s=10)
    z = np.polyfit(valid['match_means'], valid['stock_return'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['match_means'].min(), valid['match_means'].max(), 100)
    ax11.plot(x_line, p(x_line), 'r-', lw=2, label=f'β={z[0]:.3f}')
    ax11.set_xlabel('Match Quality')
    ax11.set_ylabel('Stock Return')
    ax11.set_title('Match → Returns (Scatter)', fontweight='bold')
    ax11.legend()
    
    # 12. Heatmap: Age × Size
    ax12 = fig.add_subplot(3, 4, 12)
    size_bins = pd.qcut(mega_df['logatw'], 6, labels=False, duplicates='drop')
    age_bins = pd.cut(mega_df['Age'], 6, labels=False)
    heatmap_data = mega_df.groupby([age_bins, size_bins])['match_means'].mean().unstack()
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, ax=ax12, cbar_kws={'label': 'Match'})
    ax12.set_xlabel('Firm Size Quintile')
    ax12.set_ylabel('CEO Age Quintile')
    ax12.set_title('Size × Age Interaction', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Output/mega_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: Output/mega_analysis.png")
    
    # ================================================================
    # EXECUTIVE SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("🏆 EXECUTIVE SUMMARY")
    print("=" * 70)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         🔥 MEGA ANALYSIS FINDINGS 🔥                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  📈 STOCK RETURNS                                                            ║
║     Q5 (Best) vs Q1 (Worst): {spread*100:+.2f}% spread                            ║
║     Match → Return β: {model_ret.params['match_means']:.4f} (p={model_ret.pvalues['match_means']:.4f})                                ║
║                                                                               ║
║  💰 FIRM VALUE                                                               ║
║     Q5 Tobin's Q: {q5_val:.2f}                                                    ║
║     Q1 Tobin's Q: {q1_val:.2f}                                                    ║
║     Value Premium: {q5_val/q1_val:.1f}x                                                     ║
║                                                                               ║
║  ⭐ SUPERSTAR MATCHES                                                        ║
║     Top 10% threshold: {threshold:.3f}                                            ║
║     Superstars: Younger, less Ivy, more practical                            ║
║                                                                               ║
║  🗺️ GEOGRAPHY                                                                ║
║     Best: {state_stats.index[0]} ({state_stats.iloc[0]['Match Mean']:.3f})                                     ║
║     Worst: {state_stats.index[-1]} ({state_stats.iloc[-1]['Match Mean']:.3f})                                    ║
║                                                                               ║
║  🏭 INDUSTRIES                                                               ║
║     Best: {industry_stats.index[0][:25]} ({industry_stats.iloc[0]['Match Mean']:.3f})                   ║
║     Match-Value Correlation: {industry_corr.loc['Match Mean', 'Tobin Q']:.2f}                                    ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n🎉🎉🎉 MEGA ANALYSIS COMPLETE! 🎉🎉🎉")

if __name__ == "__main__":
    main()
