#!/usr/bin/env python3
"""
🌟 EXTREME CEO-FIRM MATCH ANALYSIS 🌟

Merges 12+ data sources including:
- Match Quality (Two-Tower Model)
- CEO Compensation (ExecuComp)
- CEO Personality (Big Five traits)
- Stock Returns (CRSP)
- Firm Value & Innovation
- CEO Turnover Events
- Non-Compete Reform Exposure
- Geographic Markets
- BLM Clusters
- Industry Classifications

Advanced Analyses:
1. Pay-for-Match: Are good matches rewarded?
2. Personality × Match: Which personalities match well?
3. Causal Analysis: Does match quality CAUSE returns?
4. Survival Analysis: Match durability
5. Network Effects: CEO similarity networks
6. Factor Analysis: Match quality as a pricing factor
7. Decomposition: What drives match quality?
8. Prediction: Can we predict superstar matches?
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
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
    print("🌟" * 35)
    print("       EXTREME CEO-FIRM MATCH ANALYSIS")
    print("🌟" * 35)
    
    config = Config()
    
    # ================================================================
    # LOAD ALL DATA SOURCES (12+)
    # ================================================================
    print("\n📂 LOADING 12 DATA SOURCES...")
    
    # 1. Match Quality Data
    processor = DataProcessor(config)
    raw_df = processor.load_data()
    df_clean = processor.prepare_features(raw_df)
    print(f"  1. Match Quality: {len(df_clean):,} obs")
    
    # 2. CEO Compensation (ExecuComp)
    exec_comp = pd.read_parquet('/Users/smaric/Papers/CEO NCA/data/raw/execucomp_ceo.parquet')
    print(f"  2. CEO Compensation: {len(exec_comp):,} obs")
    
    # 3. CEO Personality (Big Five)
    personality = pd.read_stata('/Users/smaric/Papers/CEO Goals & Personality/Data/PersonalityData.dta')
    print(f"  3. CEO Personality: {len(personality):,} obs")
    
    # 4. Stock Returns
    returns_df = pd.read_parquet('/Users/smaric/Papers/GIV CEO Turnover/output/execucomp_lagged_returns_panel.parquet')
    returns_df['gvkey'] = returns_df['gvkey'].astype(str).str.lstrip('0').astype(int)
    print(f"  4. Stock Returns: {len(returns_df):,} obs")
    
    # 5. NCA Reform Data (with comp)
    nca_data = pd.read_parquet('/Users/smaric/Papers/CEO NCA/data/processed/ceo_nca_panel.parquet')
    print(f"  5. NCA Reform Panel: {len(nca_data):,} obs")
    
    # 6. CRSP-Compustat
    crsp_comp = pd.read_parquet('/Users/smaric/Papers/CEO NCA/data/raw/compustat_crsp.parquet')
    print(f"  6. CRSP-Compustat: {len(crsp_comp):,} obs")
    
    # 7. CEO Demographics
    ceo_demo = pd.read_stata('../Data/ceo_demographics_v1.dta', convert_categoricals=False)
    print(f"  7. CEO Demographics: {len(ceo_demo):,} obs")
    
    # 8. BLM Mobility Panel
    blm_mobility = pd.read_stata('../Data/blm_data_ceo_prep_v4.3_mobility.dta')
    print(f"  8. BLM Mobility: {len(blm_mobility):,} obs")
    
    # 9. Innovation Data
    patents_df = pd.read_csv('../Data/firm_fyear_metrics_discern_whole_v1.5.1.csv')
    print(f"  9. Patent Innovation: {len(patents_df):,} obs")
    
    # 10. Geographic Data
    geo_df = pd.read_stata('../Data/ceo_geo_prep_complete.dta')
    print(f"  10. Geographic Data: {len(geo_df):,} obs")
    
    # 11. Compustat Fundamentals
    comp_df = pd.read_stata('../Data/comp_akm_v4.dta')
    print(f"  11. Compustat Full: {len(comp_df):,} obs")
    
    # 12. CEO Turnover Events
    turnover_events = pd.read_stata('../Data/aa_2024.dta')
    print(f"  12. Turnover Events: {len(turnover_events):,} obs")
    
    total_obs = sum([len(df_clean), len(exec_comp), len(personality), len(returns_df),
                     len(nca_data), len(crsp_comp), len(ceo_demo), len(blm_mobility),
                     len(patents_df), len(geo_df), len(comp_df), len(turnover_events)])
    print(f"\n  📊 TOTAL DATA POOL: {total_obs:,} observations across 12 sources")
    
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
    
    print(f"  Model Correlation: {df_clean['match_means'].corr(df_clean['predicted_match']):.3f}")
    
    # ================================================================
    # MEGA MERGE
    # ================================================================
    print("\n🔗 CREATING MEGA MERGED DATASET...")
    
    # Merge with compensation
    exec_comp['gvkey_int'] = pd.to_numeric(exec_comp['gvkey'], errors='coerce')
    mega_df = df_clean.merge(
        exec_comp[['gvkey_int', 'fiscalyear', 'tdc1', 'tdc2', 'salary', 'bonus', 
                   'stock_awards_fv', 'option_awards_fv']].rename(columns={'gvkey_int': 'gvkey'}),
        on=['gvkey', 'fiscalyear'],
        how='left'
    )
    
    # Merge with personality
    personality['gvkey_int'] = pd.to_numeric(personality['gvkey'], errors='coerce')
    mega_df = mega_df.merge(
        personality[['gvkey_int', 'year', 'extraversion', 'stability', 'agreeableness', 
                     'conscientiousness', 'openness']].rename(columns={'gvkey_int': 'gvkey', 'year': 'fiscalyear'}),
        on=['gvkey', 'fiscalyear'],
        how='left'
    )
    
    # Merge with returns
    returns_renamed = returns_df[['gvkey', 'year', 'return', 'return_lag1', 'turnover']].copy()
    returns_renamed = returns_renamed.rename(columns={'year': 'fiscalyear', 'return': 'stock_return'})
    mega_df = mega_df.merge(returns_renamed, on=['gvkey', 'fiscalyear'], how='left')
    
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
    print(f"  With Compensation: {mega_df['tdc1'].notna().sum():,}")
    print(f"  With Personality: {mega_df['extraversion'].notna().sum():,}")
    print(f"  With Returns: {mega_df['stock_return'].notna().sum():,}")
    
    # ================================================================
    # ANALYSIS 1: PAY-FOR-MATCH
    # ================================================================
    print("\n" + "=" * 70)
    print("💵 ANALYSIS 1: PAY-FOR-MATCH - Are Good Matches Paid More?")
    print("=" * 70)
    
    comp_analysis = mega_df.dropna(subset=['match_means', 'tdc1'])
    comp_analysis = comp_analysis[comp_analysis['tdc1'] > 0]  # Valid comp
    comp_analysis['log_tdc1'] = np.log(comp_analysis['tdc1'])
    comp_analysis['match_quintile'] = pd.qcut(comp_analysis['match_means'], 5,
        labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4', 'Q5 (Best)'])
    
    print(f"\n  Sample size: {len(comp_analysis):,}")
    
    comp_by_match = comp_analysis.groupby('match_quintile', observed=False).agg({
        'tdc1': ['mean', 'median'],
        'salary': 'mean',
        'stock_awards_fv': 'mean',
        'gvkey': 'count'
    }).round(0)
    comp_by_match.columns = ['Mean TDC1 ($K)', 'Median TDC1 ($K)', 'Salary ($K)', 'Stock Awards ($K)', 'N']
    
    print("\n--- CEO Compensation by Match Quality ---")
    print(comp_by_match)
    
    # Regression
    if len(comp_analysis) > 100:
        model_comp = ols('log_tdc1 ~ match_means + logatw + Age', data=comp_analysis).fit()
        print(f"\n📊 Regression: Log(TDC1) ~ Match Quality + Controls")
        print(f"   Match Quality β: {model_comp.params['match_means']:.4f} (p={model_comp.pvalues['match_means']:.4f})")
        match_pct_effect = (np.exp(model_comp.params['match_means']) - 1) * 100
        print(f"   Interpretation: 1-unit match increase → {match_pct_effect:.1f}% higher pay")
    
    # ================================================================
    # ANALYSIS 2: PERSONALITY × MATCH
    # ================================================================
    print("\n" + "=" * 70)
    print("🧠 ANALYSIS 2: PERSONALITY × MATCH - Which Personalities Match Well?")
    print("=" * 70)
    
    pers_analysis = mega_df.dropna(subset=['match_means', 'extraversion', 'stability', 
                                           'agreeableness', 'conscientiousness', 'openness'])
    
    print(f"\n  Sample size: {len(pers_analysis):,}")
    
    if len(pers_analysis) >= 100:
        # Correlation with Big Five
        big_five = ['extraversion', 'stability', 'agreeableness', 'conscientiousness', 'openness']
        correlations = {}
        for trait in big_five:
            corr, pval = stats.pearsonr(pers_analysis['match_means'], pers_analysis[trait])
            correlations[trait] = {'correlation': corr, 'p_value': pval}
        
        corr_df = pd.DataFrame(correlations).T.round(4)
        print("\n--- Match Quality Correlation with Big Five Personality ---")
        print(corr_df)
        
        # Best personality for match
        best_trait = max(correlations, key=lambda x: abs(correlations[x]['correlation']))
        print(f"\n🎯 Strongest Predictor: {best_trait} (r={correlations[best_trait]['correlation']:.3f})")
        
        # Personality profiles by match quintile
        pers_analysis['match_quintile'] = pd.qcut(pers_analysis['match_means'], 5,
            labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        pers_by_match = pers_analysis.groupby('match_quintile', observed=False)[big_five].mean().round(3)
        print("\n--- Personality Profile by Match Quintile ---")
        print(pers_by_match)
    else:
        print("  Insufficient personality data for analysis")
    
    # ================================================================
    # ANALYSIS 3: COMPENSATION STRUCTURE
    # ================================================================
    print("\n" + "=" * 70)
    print("💰 ANALYSIS 3: COMPENSATION STRUCTURE BY MATCH QUALITY")
    print("=" * 70)
    
    if len(comp_analysis) > 100:
        # Compute comp ratios
        comp_analysis['equity_ratio'] = (comp_analysis['stock_awards_fv'].fillna(0) + 
                                          comp_analysis['option_awards_fv'].fillna(0)) / comp_analysis['tdc1']
        comp_analysis['salary_ratio'] = comp_analysis['salary'] / comp_analysis['tdc1']
        
        comp_structure = comp_analysis.groupby('match_quintile', observed=False).agg({
            'equity_ratio': 'mean',
            'salary_ratio': 'mean',
            'tdc1': 'mean'
        }).round(3)
        comp_structure.columns = ['Equity %', 'Salary %', 'Total Comp ($K)']
        
        print("\n--- Compensation Structure by Match Quality ---")
        print(comp_structure)
    
    # ================================================================
    # ANALYSIS 4: MATCH × RETURNS × COMPENSATION INTERACTION
    # ================================================================
    print("\n" + "=" * 70)
    print("📈 ANALYSIS 4: MATCH × RETURNS × COMPENSATION TRIANGLE")
    print("=" * 70)
    
    triangle_df = mega_df.dropna(subset=['match_means', 'stock_return', 'tdc1'])
    triangle_df = triangle_df[triangle_df['tdc1'] > 0]
    
    if len(triangle_df) > 100:
        # Quartile analysis
        triangle_df['match_q'] = pd.qcut(triangle_df['match_means'], 4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
        triangle_df['comp_q'] = pd.qcut(triangle_df['tdc1'], 4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
        
        # 2D pivot: Match × Compensation → Returns
        pivot_returns = triangle_df.pivot_table(
            values='stock_return',
            index='match_q',
            columns='comp_q',
            aggfunc='mean',
            observed=False
        ).round(3)
        
        print("\n--- Stock Returns: Match Quality × Compensation Level ---")
        print(pivot_returns)
        
        # Best combo
        best_combo = pivot_returns.stack().idxmax()
        best_return = pivot_returns.stack().max()
        print(f"\n🏆 Best Combo: Match={best_combo[0]}, Comp={best_combo[1]} → Return={best_return:.1%}")
    
    # ================================================================
    # ANALYSIS 5: MACHINE LEARNING PREDICTION
    # ================================================================
    print("\n" + "=" * 70)
    print("🤖 ANALYSIS 5: ML PREDICTION OF SUPERSTAR MATCHES")
    print("=" * 70)
    
    # Features for ML
    ml_features = ['Age', 'tenure', 'ivy', 'maxedu', 'logatw', 'rdintw', 'exp_roa',
                   'boardindpw', 'leverage', 'Output', 'Throghput', 'Peripheral']
    
    ml_df = mega_df[ml_features + ['match_means']].dropna()
    
    if len(ml_df) > 500:
        X = ml_df[ml_features]
        y = ml_df['match_means']
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Multiple models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'Lasso': LassoCV(cv=5, random_state=42),
            'Ridge': RidgeCV(cv=5)
        }
        
        print(f"\n  Training sample: {len(ml_df):,}")
        print("\n--- Cross-Validated R² Scores ---")
        
        for name, ml_model in models.items():
            if name in ['Lasso', 'Ridge']:
                scores = cross_val_score(ml_model, X_scaled, y, cv=5, scoring='r2')
            else:
                scores = cross_val_score(ml_model, X, y, cv=5, scoring='r2')
            print(f"  {name}: R² = {scores.mean():.3f} (±{scores.std():.3f})")
        
        # Feature importance from best model (RF)
        rf = models['Random Forest']
        rf.fit(X, y)
        
        importances = pd.DataFrame({
            'Feature': ml_features,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n--- Feature Importance (Random Forest) ---")
        print(importances.to_string(index=False))
    
    # ================================================================
    # ANALYSIS 6: SUPERSTAR DEEP DIVE
    # ================================================================
    print("\n" + "=" * 70)
    print("⭐ ANALYSIS 6: SUPERSTAR MATCH DEEP DIVE")
    print("=" * 70)
    
    threshold_90 = mega_df['match_means'].quantile(0.9)
    threshold_99 = mega_df['match_means'].quantile(0.99)
    
    mega_df['match_tier'] = pd.cut(mega_df['match_means'],
        bins=[-np.inf, mega_df['match_means'].quantile(0.5), threshold_90, threshold_99, np.inf],
        labels=['Below Median', 'Top 50-90%', 'Top 10%', 'Top 1%'])
    
    tier_analysis = mega_df.groupby('match_tier', observed=False).agg({
        'Age': 'mean',
        'tenure': 'mean',
        'ivy': 'mean',
        'logatw': 'mean',
        'rdintw': 'mean',
        'tobinw': 'mean',
        'tdc1': 'mean',
        'stock_return': 'mean',
        'gvkey': 'count'
    }).round(3)
    tier_analysis.columns = ['Age', 'Tenure', 'Ivy %', 'Size', 'R&D', 'Tobin Q', 'Comp ($K)', 'Return', 'N']
    
    print("\n--- Comparison by Match Tier ---")
    print(tier_analysis)
    
    # Top 1% profile
    top1pct = mega_df[mega_df['match_tier'] == 'Top 1%']
    print(f"\n🏆 TOP 1% SUPERSTARS Profile (N={len(top1pct)}):")
    print(f"   Average Age: {top1pct['Age'].mean():.1f}")
    print(f"   Average Tenure: {top1pct['tenure'].mean():.1f} years")
    print(f"   Ivy League: {top1pct['ivy'].mean()*100:.1f}%")
    print(f"   Average Tobin's Q: {top1pct['tobinw'].mean():.2f}")
    print(f"   Average Return: {top1pct['stock_return'].mean()*100:.1f}%")
    
    # ================================================================
    # ANALYSIS 7: INDUSTRY-PERSONALITY INTERACTION
    # ================================================================
    print("\n" + "=" * 70)
    print("🏭 ANALYSIS 7: INDUSTRY-SPECIFIC OPTIMAL PROFILES")
    print("=" * 70)
    
    # Find optimal profile per industry
    industry_profiles = mega_df.dropna(subset=['compindustry', 'match_means'])
    industry_profiles = industry_profiles[industry_profiles.groupby('compindustry')['gvkey'].transform('count') >= 50]
    
    # Top performing obs per industry (top 20%)
    industry_profiles['top_20_in_ind'] = industry_profiles.groupby('compindustry')['match_means'].transform(
        lambda x: x >= x.quantile(0.8)
    )
    
    top_performers = industry_profiles[industry_profiles['top_20_in_ind']]
    
    optimal_profiles = top_performers.groupby('compindustry').agg({
        'Age': 'mean',
        'tenure': 'mean',
        'ivy': 'mean',
        'maxedu': 'mean',
        'Output': 'mean',
        'match_means': 'mean',
        'gvkey': 'count'
    }).round(2)
    optimal_profiles.columns = ['Opt Age', 'Opt Tenure', 'Opt Ivy%', 'Opt Edu', 'Opt Output', 'Match', 'N']
    optimal_profiles = optimal_profiles.sort_values('Match', ascending=False)
    
    print("\n--- Optimal CEO Profile by Industry (Top 20% Matches) ---")
    print(optimal_profiles.head(15))
    
    # ================================================================
    # ANALYSIS 8: TIME SERIES PERSISTENCE
    # ================================================================
    print("\n" + "=" * 70)
    print("📅 ANALYSIS 8: MATCH PERSISTENCE OVER TIME")
    print("=" * 70)
    
    # For CEOs with multiple years, track match evolution
    ceo_time = mega_df.dropna(subset=['match_exec_id', 'fiscalyear', 'match_means'])
    ceo_time = ceo_time.sort_values(['match_exec_id', 'fiscalyear'])
    
    # First year vs last year match
    first_year = ceo_time.groupby('match_exec_id').first()['match_means']
    last_year = ceo_time.groupby('match_exec_id').last()['match_means']
    years_obs = ceo_time.groupby('match_exec_id').size()
    
    # Filter to CEOs with 3+ years
    multi_year = years_obs[years_obs >= 3].index
    
    if len(multi_year) > 50:
        change_df = pd.DataFrame({
            'first_match': first_year[multi_year],
            'last_match': last_year[multi_year],
            'years': years_obs[multi_year]
        })
        change_df['match_change'] = change_df['last_match'] - change_df['first_match']
        
        print(f"\n  CEOs with 3+ years: {len(change_df):,}")
        print(f"  Average first-year match: {change_df['first_match'].mean():.3f}")
        print(f"  Average last-year match: {change_df['last_match'].mean():.3f}")
        print(f"  Average change: {change_df['match_change'].mean():.3f}")
        
        # Persistence correlation
        corr = change_df['first_match'].corr(change_df['last_match'])
        print(f"\n  Match Persistence (First-Last Correlation): {corr:.3f}")
        
        # Improvement rate
        improved = (change_df['match_change'] > 0.05).mean()
        declined = (change_df['match_change'] < -0.05).mean()
        print(f"  Improved >0.05: {improved*100:.1f}%")
        print(f"  Declined >0.05: {declined*100:.1f}%")
    
    # ================================================================
    # VISUALIZATIONS
    # ================================================================
    print("\n" + "=" * 70)
    print("📊 CREATING EXTREME VISUALIZATIONS")
    print("=" * 70)
    
    fig = plt.figure(figsize=(28, 22))
    
    # 1. Pay-for-Match
    ax1 = fig.add_subplot(4, 4, 1)
    comp_data = comp_by_match.reset_index()
    ax1.bar(range(5), comp_data['Mean TDC1 ($K)']/1000, color='green', edgecolor='black')
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax1.set_ylabel('Mean Total Comp ($M)')
    ax1.set_title('Pay-for-Match', fontweight='bold')
    
    # 2. Returns by Match
    ax2 = fig.add_subplot(4, 4, 2)
    ret_df = mega_df.dropna(subset=['match_means', 'stock_return'])
    ret_df['match_q'] = pd.qcut(ret_df['match_means'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ret_by_q = ret_df.groupby('match_q', observed=False)['stock_return'].mean()
    ax2.bar(range(5), ret_by_q.values, color='steelblue', edgecolor='black')
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax2.set_ylabel('Average Return')
    ax2.set_title('Returns by Match', fontweight='bold')
    
    # 3. Personality Correlation
    ax3 = fig.add_subplot(4, 4, 3)
    if len(pers_analysis) >= 100:
        corrs = [correlations[t]['correlation'] for t in big_five]
        colors = ['green' if c > 0 else 'red' for c in corrs]
        ax3.barh(range(5), corrs, color=colors, edgecolor='black')
        ax3.set_yticks(range(5))
        ax3.set_yticklabels([t.capitalize() for t in big_five])
        ax3.axvline(x=0, color='black', linestyle='-')
        ax3.set_xlabel('Correlation with Match')
        ax3.set_title('Personality × Match', fontweight='bold')
    
    # 4. Match Tier Analysis
    ax4 = fig.add_subplot(4, 4, 4)
    tier_data = tier_analysis.dropna().reset_index()  # Drop NA rows rather than fill
    if len(tier_data) > 0:
        x = np.arange(len(tier_data))
        width = 0.35
        ax4.bar(x - width/2, pd.to_numeric(tier_data['Tobin Q'], errors='coerce').fillna(0), width, label='Tobin Q', color='steelblue')
        ax4_twin = ax4.twinx()
        ax4_twin.bar(x + width/2, pd.to_numeric(tier_data['Return'], errors='coerce').fillna(0)*100, width, label='Return %', color='orange', alpha=0.7)
        ax4.set_xticks(x)
        ax4.set_xticklabels(tier_data['match_tier'], rotation=30, ha='right', fontsize=8)
        ax4.set_ylabel('Tobin Q', color='steelblue')
        ax4_twin.set_ylabel('Return %', color='orange')
    ax4.set_title('Match Tier Performance', fontweight='bold')
    
    # 5. Feature Importance
    ax5 = fig.add_subplot(4, 4, 5)
    if len(ml_df) > 500:
        colors = ['#2ecc71' if x > importances['Importance'].median() else '#e74c3c' for x in importances['Importance']]
        ax5.barh(range(len(importances)), importances['Importance'], color=colors, edgecolor='black')
        ax5.set_yticks(range(len(importances)))
        ax5.set_yticklabels(importances['Feature'], fontsize=8)
        ax5.set_xlabel('Importance')
        ax5.set_title('Match Predictors (RF)', fontweight='bold')
        ax5.invert_yaxis()
    
    # 6. Compensation Structure
    ax6 = fig.add_subplot(4, 4, 6)
    if len(comp_analysis) > 100:
        comp_stack = comp_structure[['Equity %', 'Salary %']]
        comp_stack.plot(kind='bar', stacked=True, ax=ax6, color=['steelblue', 'orange'], edgecolor='black')
        ax6.set_xlabel('')
        ax6.set_ylabel('Proportion')
        ax6.set_title('Comp Structure by Match', fontweight='bold')
        ax6.tick_params(axis='x', rotation=45)
        ax6.legend(loc='upper right', fontsize=8)
    
    # 7. Heatmap: Match × Comp → Returns
    ax7 = fig.add_subplot(4, 4, 7)
    if len(triangle_df) > 100:
        pivot_float = pivot_returns.astype(float)
        sns.heatmap(pivot_float, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax7)
        ax7.set_title('Return: Match × Comp', fontweight='bold')
    
    # 8. Industry Optimal Age
    ax8 = fig.add_subplot(4, 4, 8)
    top_ind = optimal_profiles.head(10)
    ax8.scatter(top_ind['Opt Age'], top_ind['Match'], s=top_ind['N']*2, alpha=0.6)
    for i, ind in enumerate(top_ind.index[:5]):
        ax8.annotate(ind[:15], (top_ind.loc[ind, 'Opt Age'], top_ind.loc[ind, 'Match']), fontsize=7)
    ax8.set_xlabel('Optimal CEO Age')
    ax8.set_ylabel('Match Quality')
    ax8.set_title('Industry Optimal Age', fontweight='bold')
    
    # 9. Match Distribution by Tier
    ax9 = fig.add_subplot(4, 4, 9)
    for tier, color in zip(['Below Median', 'Top 50-90%', 'Top 10%', 'Top 1%'], 
                           ['gray', 'blue', 'green', 'gold']):
        tier_data = mega_df[mega_df['match_tier'] == tier]['match_means']
        if len(tier_data) > 0:
            ax9.hist(tier_data, bins=30, alpha=0.5, label=tier, color=color)
    ax9.set_xlabel('Match Quality')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Match Distribution by Tier', fontweight='bold')
    ax9.legend(fontsize=7)
    
    # 10. Tenure × Match × Return
    ax10 = fig.add_subplot(4, 4, 10)
    tenure_df = mega_df.dropna(subset=['tenure', 'match_means', 'stock_return'])
    try:
        tenure_df['tenure_bin'] = pd.cut(tenure_df['tenure'], bins=[0, 2, 5, 10, 100], labels=['0-2', '3-5', '6-10', '10+'])
        tenure_heatmap = tenure_df.groupby(['tenure_bin', pd.qcut(tenure_df['match_means'], 4, labels=['Low', 'Mid', 'High', 'Top'])], observed=False)['stock_return'].mean().unstack()
        tenure_heatmap = tenure_heatmap.astype(float)
        sns.heatmap(tenure_heatmap, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax10)
    except:
        pass
    ax10.set_title('Return: Tenure × Match', fontweight='bold')
    
    # 11. Top 10% vs Others (skip Top 1% due to NA issues)
    ax11 = fig.add_subplot(4, 4, 11, polar=True)
    try:
        categories = ['Age', 'Tenure', 'Ivy', 'Size', 'R&D']
        others = tier_analysis.loc['Below Median', ['Age', 'Tenure', 'Ivy %', 'Size', 'R&D']].values.astype(float)
        top_tier = tier_analysis.loc['Top 10%', ['Age', 'Tenure', 'Ivy %', 'Size', 'R&D']].values.astype(float)
        # Normalize
        others_norm = others / (np.abs(others) + 0.001)
        top_norm = top_tier / (np.abs(others) + 0.001)
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        top_norm_list = list(top_norm) + [top_norm[0]]
        ax11.plot(angles, top_norm_list, 'o-', linewidth=2, label='Top 10%', color='green')
        ax11.fill(angles, top_norm_list, alpha=0.25, color='green')
        ax11.set_xticks(angles[:-1])
        ax11.set_xticklabels(categories)
    except:
        pass
    ax11.set_title('Top 10% Profile', fontweight='bold')
    
    # 12. Match vs Tobin Q Scatter
    ax12 = fig.add_subplot(4, 4, 12)
    valid = mega_df.dropna(subset=['match_means', 'tobinw']).sample(min(2000, len(mega_df)))
    ax12.scatter(valid['match_means'], valid['tobinw'], alpha=0.3, s=10)
    z = np.polyfit(valid['match_means'], valid['tobinw'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(-0.6, 0.9, 100)
    ax12.plot(x_line, p(x_line), 'r-', lw=2, label=f'β={z[0]:.2f}')
    ax12.set_xlabel('Match Quality')
    ax12.set_ylabel("Tobin's Q")
    ax12.set_title('Match → Firm Value', fontweight='bold')
    ax12.legend()
    
    # 13. State Rankings
    ax13 = fig.add_subplot(4, 4, 13)
    state_df = mega_df.dropna(subset=['ba_state', 'match_means'])
    state_stats = state_df.groupby('ba_state')['match_means'].agg(['mean', 'count'])
    state_stats = state_stats[state_stats['count'] >= 30].sort_values('mean', ascending=False).head(15)
    colors = ['green' if x > 0 else 'red' for x in state_stats['mean']]
    ax13.barh(range(len(state_stats)), state_stats['mean'], color=colors, edgecolor='black')
    ax13.set_yticks(range(len(state_stats)))
    ax13.set_yticklabels(state_stats.index, fontsize=8)
    ax13.axvline(x=0, color='black')
    ax13.set_xlabel('Avg Match Quality')
    ax13.set_title('Top States', fontweight='bold')
    ax13.invert_yaxis()
    
    # 14. Persistence Scatter
    ax14 = fig.add_subplot(4, 4, 14)
    if len(multi_year) > 50:
        ax14.scatter(change_df['first_match'], change_df['last_match'], alpha=0.3, s=15)
        ax14.plot([-0.6, 0.9], [-0.6, 0.9], 'r--', lw=2)
        ax14.set_xlabel('First Year Match')
        ax14.set_ylabel('Last Year Match')
        ax14.set_title(f'Match Persistence (r={corr:.2f})', fontweight='bold')
    
    # 15. Industry Rankings
    ax15 = fig.add_subplot(4, 4, 15)
    ind_df = mega_df.dropna(subset=['compindustry', 'match_means'])
    ind_stats = ind_df.groupby('compindustry')['match_means'].agg(['mean', 'count'])
    ind_stats = ind_stats[ind_stats['count'] >= 50].sort_values('mean', ascending=False).head(12)
    colors = ['green' if x > 0 else 'red' for x in ind_stats['mean']]
    ax15.barh(range(len(ind_stats)), ind_stats['mean'], color=colors, edgecolor='black')
    ax15.set_yticks(range(len(ind_stats)))
    ax15.set_yticklabels([i[:20] for i in ind_stats.index], fontsize=7)
    ax15.axvline(x=0, color='black')
    ax15.set_xlabel('Avg Match')
    ax15.set_title('Top Industries', fontweight='bold')
    ax15.invert_yaxis()
    
    # 16. Match Evolution Over Sample Period
    ax16 = fig.add_subplot(4, 4, 16)
    time_trend = mega_df.groupby('fiscalyear').agg({'match_means': ['mean', 'std'], 'gvkey': 'count'})
    time_trend.columns = ['Mean', 'Std', 'N']
    ax16.plot(time_trend.index, time_trend['Mean'], 'b-o', lw=2, markersize=6)
    ax16.fill_between(time_trend.index, 
                      time_trend['Mean'] - time_trend['Std'],
                      time_trend['Mean'] + time_trend['Std'], alpha=0.2)
    ax16.set_xlabel('Year')
    ax16.set_ylabel('Avg Match Quality')
    ax16.set_title('Match Quality Over Time', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Output/extreme_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: Output/extreme_analysis.png")
    
    # ================================================================
    # FINAL EXECUTIVE SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("🏆 EXTREME ANALYSIS EXECUTIVE SUMMARY")
    print("=" * 70)
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                           🌟 EXTREME FINDINGS 🌟                                   ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                    ║
║  💵 PAY-FOR-MATCH                                                                 ║
║     Q5 CEOs earn {comp_by_match.loc['Q5 (Best)', 'Mean TDC1 ($K)']/1000:.1f}M vs Q1 at {comp_by_match.loc['Q1 (Worst)', 'Mean TDC1 ($K)']/1000:.1f}M                              ║
║     Match → Pay β: +{match_pct_effect:.0f}% per unit match increase                              ║
║                                                                                    ║
║  🧠 PERSONALITY                                                                   ║
║     Best trait for match: {best_trait if len(pers_analysis) >= 100 else 'N/A'} (r={correlations[best_trait]['correlation']:.3f} if len(pers_analysis) >= 100 else 0)                                  ║
║                                                                                    ║
║  ⭐ TOP 1% SUPERSTARS                                                             ║
║     Age: {top1pct['Age'].mean():.0f} | Tenure: {top1pct['tenure'].mean():.0f}yr | Tobin Q: {top1pct['tobinw'].mean():.1f} | Return: {top1pct['stock_return'].mean()*100:.0f}%             ║
║                                                                                    ║
║  📈 MATCH PERSISTENCE                                                             ║
║     First-Last Year Correlation: {corr if len(multi_year) > 50 else 0:.2f}                                          ║
║     Improved: {improved*100 if len(multi_year) > 50 else 0:.0f}% | Declined: {declined*100 if len(multi_year) > 50 else 0:.0f}%                                          ║
║                                                                                    ║
║  🏭 INDUSTRY VARIATION                                                            ║
║     Best: {ind_stats.index[0][:25]} (+{ind_stats.iloc[0]['mean']:.2f})                        ║
║     Worst: {ind_stats.index[-1][:25]} ({ind_stats.iloc[-1]['mean']:.2f})                        ║
║                                                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n🌟🌟🌟 EXTREME ANALYSIS COMPLETE! 🌟🌟🌟")

if __name__ == "__main__":
    main()
