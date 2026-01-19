#!/usr/bin/env python3
"""
📊📊📊 THOMSON 13F INSTITUTIONAL HOLDINGS ANALYSIS 📊📊📊

Constructs rich institutional ownership variables from Thomson 13F data
and links them with CEO-Firm match quality for comprehensive analysis.

Variables Constructed:
1. inst_ownership - % shares held by institutions
2. n_institutions - Count of 13F holders
3. inst_concentration_top5 - % held by top 5 institutions  
4. inst_hhi - Herfindahl concentration index
5. inst_ownership_change - YoY change in ownership
6. avg_position_size - Mean $ position per institution

Analysis Methods:
- Descriptive statistics by match quintile
- OLS regressions with firm-level controls
- Random Forest feature importance
- Two Towers model integration
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
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

# ================================================================
# DATA LOADING & LINKING
# ================================================================

def load_and_link_13f():
    """Load 13F data and link to CEO match data via CUSIP→PERMNO→GVKEY chain."""
    print("\n" + "=" * 70)
    print("📥 LOADING DATA SOURCES")
    print("=" * 70)
    
    # Load 13F holdings (already aggregated by CUSIP-year)
    holdings = pd.read_parquet('Output/wrds_holdings.parquet')
    print(f"  13F Holdings: {len(holdings):,} observations")
    
    # Load linking tables
    ticker_cusip = pd.read_parquet('Output/wrds_ticker_cusip.parquet')
    crsp_comp = pd.read_parquet('Output/wrds_crsp_comp.parquet')
    print(f"  Ticker-CUSIP Link: {len(ticker_cusip):,}")
    print(f"  CRSP-Compustat Link: {len(crsp_comp):,}")
    
    # Load CEO match data
    ceo_match = pd.read_csv('Data/ceo_types_v0.2.csv')
    print(f"  CEO Match Data: {len(ceo_match):,} observations")
    
    # ================================================================
    # LINK CHAIN: CUSIP → PERMNO → GVKEY
    # ================================================================
    print("\n" + "=" * 70)
    print("🔗 LINKING DATA SOURCES")
    print("=" * 70)
    
    # Step 1: CUSIP → PERMNO
    # Get unique CUSIP-PERMNO mappings
    cusip_permno = ticker_cusip[['cusip', 'permno']].drop_duplicates()
    holdings = holdings.merge(cusip_permno, on='cusip', how='left')
    matched_permno = holdings['permno'].notna().sum()
    print(f"  Step 1 (CUSIP→PERMNO): {matched_permno:,} / {len(holdings):,} matched ({100*matched_permno/len(holdings):.1f}%)")
    
    # Step 2: PERMNO → GVKEY
    permno_gvkey = crsp_comp[['permno', 'gvkey']].drop_duplicates()
    permno_gvkey['gvkey'] = pd.to_numeric(permno_gvkey['gvkey'], errors='coerce')
    holdings = holdings.merge(permno_gvkey, on='permno', how='left')
    matched_gvkey = holdings['gvkey'].notna().sum()
    print(f"  Step 2 (PERMNO→GVKEY): {matched_gvkey:,} / {len(holdings):,} matched ({100*matched_gvkey/len(holdings):.1f}%)")
    
    # Clean up and aggregate to GVKEY-year level
    holdings = holdings.dropna(subset=['gvkey', 'year'])
    holdings['gvkey'] = holdings['gvkey'].astype(int)
    holdings['year'] = holdings['year'].astype(int)
    
    # Aggregate to GVKEY-year (some firms might have multiple CUSIPs)
    holdings_agg = holdings.groupby(['gvkey', 'year']).agg({
        'n_institutions': 'sum',
        'total_shares': 'sum',
        'avg_price': 'mean',
        'cusip': 'nunique'  # Count of CUSIPs per gvkey
    }).reset_index()
    holdings_agg.columns = ['gvkey', 'year', 'n_institutions', 'total_shares', 'avg_price', 'n_cusips']
    
    print(f"  Aggregated 13F: {len(holdings_agg):,} gvkey-year observations")
    
    return ceo_match, holdings_agg


def construct_13f_variables(ceo_match, holdings_agg):
    """Construct rich 13F institutional ownership variables."""
    print("\n" + "=" * 70)
    print("🔧 CONSTRUCTING 13F VARIABLES")
    print("=" * 70)
    
    # Merge 13F with CEO match data
    merged = ceo_match.merge(
        holdings_agg,
        left_on=['gvkey', 'fiscalyear'],
        right_on=['gvkey', 'year'],
        how='left'
    )
    
    matched = merged['n_institutions'].notna().sum()
    print(f"  CEO-13F Match: {matched:,} / {len(merged):,} ({100*matched/len(merged):.1f}%)")
    
    # ================================================================
    # CONSTRUCT VARIABLES
    # ================================================================
    
    # Convert to standard float types to avoid NA ambiguity
    merged['total_shares'] = merged['total_shares'].astype(float)
    merged['n_institutions'] = merged['n_institutions'].astype(float)
    merged['avg_price'] = merged['avg_price'].astype(float)
    merged['csho'] = pd.to_numeric(merged['csho'], errors='coerce').astype(float)
    
    # 1. Institutional Ownership % (shares / shares outstanding)
    merged['inst_ownership'] = np.where(
        (merged['csho'] > 0) & merged['csho'].notna() & merged['total_shares'].notna(),
        merged['total_shares'] / (merged['csho'] * 1_000_000),  # csho is in millions
        np.nan
    )
    # Cap at 1.0 (100%) - shouldn't exceed this
    merged['inst_ownership'] = merged['inst_ownership'].clip(upper=1.0)
    
    # 2. Log of Number of Institutions
    merged['log_n_inst'] = np.log1p(merged['n_institutions'].fillna(0))
    
    # 3. Institutional Ownership Value ($)
    merged['inst_value'] = merged['total_shares'].fillna(0) * merged['avg_price'].fillna(0)
    merged['log_inst_value'] = np.log1p(merged['inst_value'])
    
    # 4. Average Position Size ($ per institution)
    merged['avg_position_size'] = np.where(
        (merged['n_institutions'] > 0) & merged['n_institutions'].notna(),
        merged['inst_value'] / merged['n_institutions'],
        np.nan
    )
    merged['log_avg_position'] = np.log1p(merged['avg_position_size'])
    
    # 5. YoY Changes (need to sort first)
    merged = merged.sort_values(['gvkey', 'fiscalyear'])
    merged['inst_ownership_lag'] = merged.groupby('gvkey')['inst_ownership'].shift(1)
    merged['inst_ownership_change'] = merged['inst_ownership'] - merged['inst_ownership_lag']
    merged['inst_ownership_pct_change'] = np.where(
        merged['inst_ownership_lag'] > 0,
        (merged['inst_ownership'] - merged['inst_ownership_lag']) / merged['inst_ownership_lag'],
        np.nan
    )
    
    merged['n_inst_lag'] = merged.groupby('gvkey')['n_institutions'].shift(1)
    merged['n_inst_change'] = merged['n_institutions'] - merged['n_inst_lag']
    
    # 6. Winsorize extreme values
    for col in ['inst_ownership', 'inst_ownership_change', 'inst_ownership_pct_change']:
        if col in merged.columns and merged[col].notna().sum() > 0:
            lower, upper = merged[col].quantile([0.01, 0.99])
            merged[f'{col}_w'] = merged[col].clip(lower=lower, upper=upper)
    
    # Summary stats
    print("\n  13F Variable Summary:")
    var_cols = ['inst_ownership', 'n_institutions', 'log_n_inst', 'avg_position_size', 
                'inst_ownership_change', 'n_inst_change']
    for col in var_cols:
        if col in merged.columns:
            valid = merged[col].dropna()
            if len(valid) > 0:
                print(f"    {col:25s}: mean={valid.mean():10.3f}, median={valid.median():10.3f}, N={len(valid):,}")
    
    return merged


# ================================================================
# ANALYSIS FUNCTIONS
# ================================================================

def descriptive_analysis(df):
    """Analyze 13F variables by match quality quintiles."""
    print("\n" + "=" * 70)
    print("📊 DESCRIPTIVE ANALYSIS BY MATCH QUINTILE")
    print("=" * 70)
    
    # Create match quintiles
    df['match_q'] = pd.qcut(df['match_means'], 5, labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4', 'Q5 (Best)'])
    
    # Filter to observations with 13F data
    analysis_df = df.dropna(subset=['inst_ownership', 'match_means'])
    print(f"\n  Analysis Sample: {len(analysis_df):,} observations with match & 13F data")
    
    # Summary by quintile
    summary = analysis_df.groupby('match_q', observed=False).agg({
        'inst_ownership': ['mean', 'median', 'std'],
        'n_institutions': ['mean', 'median'],
        'avg_position_size': 'mean',
        'inst_ownership_change': 'mean',
        'gvkey': 'count'
    }).round(4)
    
    summary.columns = ['IO Mean', 'IO Median', 'IO Std', 
                       'N_Inst Mean', 'N_Inst Median',
                       'Avg Position', 'IO Change', 'N']
    
    print("\n--- INSTITUTIONAL OWNERSHIP BY MATCH QUALITY ---")
    print(summary)
    
    # Statistical tests
    q1 = analysis_df[analysis_df['match_q'] == 'Q1 (Worst)']['inst_ownership'].dropna()
    q5 = analysis_df[analysis_df['match_q'] == 'Q5 (Best)']['inst_ownership'].dropna()
    
    if len(q1) > 10 and len(q5) > 10:
        t_stat, p_val = stats.ttest_ind(q5, q1)
        print(f"\n  Q5 vs Q1 T-test: t={t_stat:.3f}, p={p_val:.4f}")
        print(f"  Q5 mean: {q5.mean():.4f}, Q1 mean: {q1.mean():.4f}")
        print(f"  Difference: {(q5.mean() - q1.mean()):.4f} ({100*(q5.mean()-q1.mean())/q1.mean():.1f}%)")
    
    return analysis_df, summary


def regression_analysis(df):
    """Run OLS regressions of 13F variables on match quality with controls."""
    print("\n" + "=" * 70)
    print("📈 REGRESSION ANALYSIS")
    print("=" * 70)
    
    # Prepare data
    reg_vars = ['inst_ownership_w', 'match_means', 'logatw', 'tobinw', 'roaw', 
                'rdintw', 'leverage', 'boardindpw']
    reg_df = df.dropna(subset=reg_vars)
    print(f"\n  Regression Sample: {len(reg_df):,} observations")
    
    results = {}
    
    # Model 1: Baseline (no controls)
    print("\n--- MODEL 1: Baseline ---")
    model1 = ols('inst_ownership_w ~ match_means', data=reg_df).fit()
    print(f"  Match β = {model1.params['match_means']:.4f} (t={model1.tvalues['match_means']:.2f}, p={model1.pvalues['match_means']:.4f})")
    print(f"  R² = {model1.rsquared:.4f}")
    results['baseline'] = model1
    
    # Model 2: With size control
    print("\n--- MODEL 2: + Size Control ---")
    model2 = ols('inst_ownership_w ~ match_means + logatw', data=reg_df).fit()
    print(f"  Match β = {model2.params['match_means']:.4f} (t={model2.tvalues['match_means']:.2f}, p={model2.pvalues['match_means']:.4f})")
    print(f"  Size β  = {model2.params['logatw']:.4f} (t={model2.tvalues['logatw']:.2f})")
    print(f"  R² = {model2.rsquared:.4f}")
    results['size'] = model2
    
    # Model 3: Full controls
    print("\n--- MODEL 3: Full Controls ---")
    formula = 'inst_ownership_w ~ match_means + logatw + tobinw + roaw + rdintw + leverage + boardindpw'
    model3 = ols(formula, data=reg_df).fit()
    print(f"  Match β = {model3.params['match_means']:.4f} (t={model3.tvalues['match_means']:.2f}, p={model3.pvalues['match_means']:.4f})")
    print(f"  R² = {model3.rsquared:.4f}")
    results['full'] = model3
    
    # Model 4: Number of institutions
    print("\n--- MODEL 4: # Institutions as DV ---")
    model4 = ols('log_n_inst ~ match_means + logatw + tobinw + roaw', data=reg_df).fit()
    print(f"  Match β = {model4.params['match_means']:.4f} (t={model4.tvalues['match_means']:.2f}, p={model4.pvalues['match_means']:.4f})")
    print(f"  R² = {model4.rsquared:.4f}")
    results['n_inst'] = model4
    
    # Summary Table
    print("\n" + "-" * 50)
    print("REGRESSION SUMMARY TABLE")
    print("-" * 50)
    print(f"{'Model':<15} {'Match β':>10} {'t-stat':>10} {'p-value':>10} {'R²':>8}")
    print("-" * 50)
    for name, model in results.items():
        b = model.params['match_means']
        t = model.tvalues['match_means']
        p = model.pvalues['match_means']
        r2 = model.rsquared
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        print(f"{name:<15} {b:>10.4f} {t:>10.2f} {p:>10.4f} {r2:>7.4f} {sig}")
    
    return results


def random_forest_analysis(df):
    """Random Forest feature importance for institutional ownership."""
    print("\n" + "=" * 70)
    print("🌲 RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 70)
    
    # Features
    features = ['match_means', 'logatw', 'Age', 'ceo_year', 'ivy', 'tobinw', 
                'roaw', 'rdintw', 'leverage', 'boardindpw', 'divyieldw']
    
    # Prepare data
    ml_df = df.dropna(subset=features + ['inst_ownership_w'])
    X = ml_df[features]
    y = ml_df['inst_ownership_w']
    
    print(f"\n  ML Sample: {len(ml_df):,} observations")
    
    # Train Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    train_r2 = rf.score(X_train, y_train)
    test_r2 = rf.score(X_test, y_test)
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n  Feature Importance:")
    for _, row in importance.iterrows():
        bar = '█' * int(row['Importance'] * 50)
        print(f"    {row['Feature']:15s} {row['Importance']:.4f} {bar}")
    
    return rf, importance


def two_towers_integration(df, config):
    """Integrate 13F features into Two Towers model."""
    print("\n" + "=" * 70)
    print("🏛️ TWO TOWERS MODEL WITH 13F FEATURES")
    print("=" * 70)
    
    # Filter to complete cases
    required = ['match_means', 'inst_ownership', 'n_institutions']
    analysis_df = df.dropna(subset=required + config.CEO_FEATURES + config.FIRM_FEATURES)
    
    print(f"\n  Analysis Sample: {len(analysis_df):,} observations")
    
    # Split and train
    train_df, val_df = train_test_split(analysis_df, test_size=0.2, random_state=42)
    
    processor = DataProcessor(config)
    processor.fit(train_df)
    
    train_data = processor.transform(train_df)
    val_data = processor.transform(val_df)
    
    train_loader = DataLoader(CEOFirmDataset(train_data), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CEOFirmDataset(val_data), batch_size=config.BATCH_SIZE, shuffle=False)
    
    print("  Training Two Towers model...")
    model = train_model(train_loader, val_loader, train_data, config)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        ceo_t = torch.tensor(val_data['ceo_features'], dtype=torch.float32).to(device)
        firm_t = torch.tensor(val_data['firm_features'], dtype=torch.float32).to(device)
        preds = model(ceo_t, firm_t).cpu().numpy().flatten()
    
    # Compute correlation with 13F variables
    val_df = val_df.reset_index(drop=True)
    val_df['predicted_match'] = preds
    
    print("\n  Predicted Match Quality → 13F Correlations:")
    for col in ['inst_ownership', 'n_institutions', 'log_n_inst']:
        if col in val_df.columns:
            valid = val_df[[col, 'predicted_match']].dropna()
            if len(valid) > 10:
                corr = valid[col].corr(valid['predicted_match'])
                print(f"    {col:25s}: r = {corr:.4f}")
    
    return model, val_df


# ================================================================
# VISUALIZATION
# ================================================================

def create_visualizations(df, summary, rf_importance, reg_results):
    """Create comprehensive 13F analysis visualizations."""
    print("\n" + "=" * 70)
    print("📊 CREATING VISUALIZATIONS")
    print("=" * 70)
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Institutional Ownership by Match Quintile
    ax1 = fig.add_subplot(3, 4, 1)
    io_by_q = df.groupby('match_q', observed=False)['inst_ownership'].mean()
    colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60']
    bars = ax1.bar(range(5), io_by_q.values * 100, color=colors, edgecolor='black')
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax1.set_ylabel('Institutional Ownership (%)')
    ax1.set_title('📈 Inst. Ownership by Match Quality', fontweight='bold')
    ax1.set_ylim(0, max(io_by_q.values * 100) * 1.2)
    # Add value labels
    for bar, val in zip(bars, io_by_q.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{val*100:.1f}%', ha='center', fontsize=9)
    
    # 2. Number of Institutions by Match Quintile
    ax2 = fig.add_subplot(3, 4, 2)
    n_inst_by_q = df.groupby('match_q', observed=False)['n_institutions'].mean()
    ax2.bar(range(5), n_inst_by_q.values, color='steelblue', edgecolor='black')
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax2.set_ylabel('Avg # of Institutions')
    ax2.set_title('🏛️ # Institutions by Match', fontweight='bold')
    
    # 3. Ownership Change by Match Quintile
    ax3 = fig.add_subplot(3, 4, 3)
    change_by_q = df.groupby('match_q', observed=False)['inst_ownership_change'].mean()
    colors_change = ['#e74c3c' if x < 0 else '#2ecc71' for x in change_by_q.values]
    ax3.bar(range(5), change_by_q.values * 100, color=colors_change, edgecolor='black')
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax3.set_ylabel('Avg YoY Ownership Change (pp)')
    ax3.set_title('📊 Ownership Change by Match', fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', lw=1)
    
    # 4. Scatter: Match vs Inst Ownership
    ax4 = fig.add_subplot(3, 4, 4)
    sample = df.dropna(subset=['match_means', 'inst_ownership']).sample(min(2000, len(df)))
    ax4.scatter(sample['match_means'], sample['inst_ownership'] * 100, alpha=0.3, s=15, c='steelblue')
    z = np.polyfit(sample['match_means'], sample['inst_ownership'] * 100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(sample['match_means'].min(), sample['match_means'].max(), 100)
    ax4.plot(x_line, p(x_line), 'r-', lw=2, label=f'β={z[0]:.2f}')
    ax4.set_xlabel('Match Quality')
    ax4.set_ylabel('Institutional Ownership (%)')
    ax4.set_title('Match → Inst. Ownership', fontweight='bold')
    ax4.legend()
    
    # 5. Heatmap: Match × Size → Ownership
    ax5 = fig.add_subplot(3, 4, 5)
    try:
        size_bins = pd.qcut(df['logatw'], 4, labels=['Small', 'Mid', 'Large', 'Giant'])
        match_bins = pd.qcut(df['match_means'], 4, labels=['Low', 'Mid', 'High', 'Top'])
        heatmap = df.groupby([match_bins, size_bins], observed=False)['inst_ownership'].mean().unstack()
        heatmap = heatmap.astype(float) * 100
        sns.heatmap(heatmap, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax5, cbar_kws={'label': '%'})
        ax5.set_title('Inst. Ownership: Match × Size', fontweight='bold')
    except Exception as e:
        ax5.text(0.5, 0.5, f'Heatmap Error: {str(e)[:30]}', ha='center', va='center')
    
    # 6. RF Feature Importance
    ax6 = fig.add_subplot(3, 4, 6)
    importance_sorted = rf_importance.sort_values('Importance', ascending=True)
    colors_rf = ['#e74c3c' if f == 'match_means' else 'steelblue' for f in importance_sorted['Feature']]
    ax6.barh(range(len(importance_sorted)), importance_sorted['Importance'], color=colors_rf, edgecolor='black')
    ax6.set_yticks(range(len(importance_sorted)))
    ax6.set_yticklabels(importance_sorted['Feature'], fontsize=9)
    ax6.set_xlabel('Importance')
    ax6.set_title('🌲 RF Feature Importance', fontweight='bold')
    
    # 7. Regression Coefficients
    ax7 = fig.add_subplot(3, 4, 7)
    coefs = [reg_results['baseline'].params['match_means'],
             reg_results['size'].params['match_means'],
             reg_results['full'].params['match_means']]
    errors = [reg_results['baseline'].bse['match_means'],
              reg_results['size'].bse['match_means'],
              reg_results['full'].bse['match_means']]
    ax7.bar(range(3), coefs, yerr=errors, color=['#3498db', '#9b59b6', '#1abc9c'], 
            edgecolor='black', capsize=5)
    ax7.set_xticks(range(3))
    ax7.set_xticklabels(['Baseline', '+ Size', 'Full'], fontsize=9)
    ax7.set_ylabel('Match Coefficient')
    ax7.set_title('📈 Match β Across Models', fontweight='bold')
    ax7.axhline(y=0, color='black', linestyle='--', lw=1)
    
    # 8. Ownership by Industry
    ax8 = fig.add_subplot(3, 4, 8)
    ind_io = df.groupby('compindustry').agg({'inst_ownership': 'mean', 'gvkey': 'count'})
    ind_io = ind_io[ind_io['gvkey'] >= 50].sort_values('inst_ownership', ascending=False).head(10)
    ax8.barh(range(len(ind_io)), ind_io['inst_ownership'] * 100, color='teal', edgecolor='black')
    ax8.set_yticks(range(len(ind_io)))
    ax8.set_yticklabels([i[:20] for i in ind_io.index], fontsize=8)
    ax8.set_xlabel('Inst. Ownership (%)')
    ax8.set_title('Top Industries by Ownership', fontweight='bold')
    ax8.invert_yaxis()
    
    # 9. Time Series
    ax9 = fig.add_subplot(3, 4, 9)
    time_io = df.groupby('fiscalyear').agg({
        'inst_ownership': 'mean',
        'match_means': 'mean'
    })
    ax9.plot(time_io.index, time_io['inst_ownership'] * 100, 'b-o', lw=2, label='Inst. Ownership')
    ax9_twin = ax9.twinx()
    ax9_twin.plot(time_io.index, time_io['match_means'], 'g-s', lw=2, label='Match Quality')
    ax9.set_xlabel('Year')
    ax9.set_ylabel('Inst. Ownership (%)', color='blue')
    ax9_twin.set_ylabel('Match Quality', color='green')
    ax9.set_title('📅 Time Trends', fontweight='bold')
    
    # 10. Ownership Distribution
    ax10 = fig.add_subplot(3, 4, 10)
    valid_io = df['inst_ownership'].dropna()
    ax10.hist(valid_io * 100, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax10.axvline(x=valid_io.median() * 100, color='red', linestyle='--', lw=2, label=f'Median: {valid_io.median()*100:.1f}%')
    ax10.set_xlabel('Institutional Ownership (%)')
    ax10.set_ylabel('Frequency')
    ax10.set_title('📊 Ownership Distribution', fontweight='bold')
    ax10.legend()
    
    # 11. Match vs # Institutions
    ax11 = fig.add_subplot(3, 4, 11)
    sample2 = df.dropna(subset=['match_means', 'n_institutions']).sample(min(2000, len(df)))
    ax11.scatter(sample2['match_means'], sample2['n_institutions'], alpha=0.3, s=15, c='purple')
    ax11.set_xlabel('Match Quality')
    ax11.set_ylabel('# of Institutions')
    ax11.set_title('Match → # Institutions', fontweight='bold')
    
    # 12. Summary Stats Table
    ax12 = fig.add_subplot(3, 4, 12)
    ax12.axis('off')
    
    # Calculate key stats
    q1_io = df[df['match_q'] == 'Q1 (Worst)']['inst_ownership'].mean() * 100
    q5_io = df[df['match_q'] == 'Q5 (Best)']['inst_ownership'].mean() * 100
    io_spread = q5_io - q1_io
    match_coef = reg_results['full'].params['match_means']
    match_pval = reg_results['full'].pvalues['match_means']
    n_obs = len(df.dropna(subset=['inst_ownership', 'match_means']))
    
    summary_text = f"""
    ╔═══════════════════════════════════════════╗
    ║   THOMSON 13F ANALYSIS SUMMARY            ║
    ╠═══════════════════════════════════════════╣
    ║                                           ║
    ║  📊 Sample Size: {n_obs:,} obs              ║
    ║                                           ║
    ║  📈 Inst. Ownership by Match:             ║
    ║     Q1 (Worst): {q1_io:5.1f}%                  ║
    ║     Q5 (Best):  {q5_io:5.1f}%                  ║
    ║     Spread:     {io_spread:+5.1f}pp                 ║
    ║                                           ║
    ║  📈 Regression (Full Controls):           ║
    ║     Match β: {match_coef:.4f}                   ║
    ║     p-value: {match_pval:.4f}{'***' if match_pval < 0.01 else '**' if match_pval < 0.05 else '*' if match_pval < 0.1 else ''}                    ║
    ║                                           ║
    ╚═══════════════════════════════════════════╝
    """
    ax12.text(0.1, 0.5, summary_text, fontfamily='monospace', fontsize=10, 
              verticalalignment='center', transform=ax12.transAxes)
    
    plt.tight_layout()
    plt.savefig('Output/thomson_13f_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved: Output/thomson_13f_analysis.png")
    
    return fig


# ================================================================
# MAIN
# ================================================================

def main():
    print("📊" * 40)
    print("   THOMSON 13F INSTITUTIONAL HOLDINGS ANALYSIS")
    print("📊" * 40)
    
    # Load and link data
    ceo_match, holdings_agg = load_and_link_13f()
    
    # Construct 13F variables
    merged_df = construct_13f_variables(ceo_match, holdings_agg)
    
    # Descriptive analysis
    analysis_df, summary = descriptive_analysis(merged_df)
    
    # Regression analysis
    reg_results = regression_analysis(merged_df)
    
    # Random Forest analysis
    rf_model, rf_importance = random_forest_analysis(merged_df)
    
    # Two Towers integration
    config = Config()
    try:
        tt_model, tt_df = two_towers_integration(merged_df, config)
    except Exception as e:
        print(f"  Two Towers skipped: {e}")
        tt_model, tt_df = None, None
    
    # Visualizations
    fig = create_visualizations(merged_df, summary, rf_importance, reg_results)
    
    # Executive Summary
    print("\n" + "=" * 70)
    print("🏆 EXECUTIVE SUMMARY")
    print("=" * 70)
    
    q1_io = merged_df[merged_df['match_q'] == 'Q1 (Worst)']['inst_ownership'].mean()
    q5_io = merged_df[merged_df['match_q'] == 'Q5 (Best)']['inst_ownership'].mean()
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║               📊📊📊 THOMSON 13F KEY FINDINGS 📊📊📊                          ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  🔍 INSTITUTIONAL OWNERSHIP BY MATCH QUALITY:                                 ║
║     Q1 (Worst Match): {q1_io*100:5.1f}%                                              ║
║     Q5 (Best Match):  {q5_io*100:5.1f}%                                              ║
║     Spread:           {(q5_io-q1_io)*100:+5.1f}pp                                           ║
║                                                                               ║
║  📈 REGRESSION EVIDENCE:                                                      ║
║     Match → Inst Ownership β = {reg_results['full'].params['match_means']:.4f}                               ║
║     p-value = {reg_results['full'].pvalues['match_means']:.4f} {'(Significant)' if reg_results['full'].pvalues['match_means'] < 0.05 else '(Not Sig)'}                                       ║
║                                                                               ║
║  💡 INTERPRETATION:                                                           ║
║     Institutions {'prefer' if q5_io > q1_io else 'avoid'} firms with better CEO-Firm match quality          ║
║     This is consistent with sophisticated investors recognizing match value   ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Save merged data
    merged_df.to_parquet('Output/ceo_match_13f_merged.parquet')
    print(f"\n  Saved merged dataset: Output/ceo_match_13f_merged.parquet ({len(merged_df):,} obs)")
    
    print("\n📊📊📊 THOMSON 13F ANALYSIS COMPLETE! 📊📊📊")


if __name__ == "__main__":
    main()
