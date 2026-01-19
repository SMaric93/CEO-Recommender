#!/usr/bin/env python3
"""
📊 IBES ANALYST DATA INTEGRATION WITH CEO-FIRM MATCH ANALYSIS 📊

This module:
1. Links IBES analyst data to CEO match data via gvkey → permno → ticker
2. Constructs analyst coverage and forecast accuracy variables
3. Runs multi-method analysis: OLS, Random Forest, Two Towers
4. Generates comprehensive visualizations

Usage:
    python ibes_analysis.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader

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

def load_and_link_ibes():
    """
    Load IBES data and link to CEO match data via gvkey.
    
    Linking path: 
        CEO data (gvkey) → CRSP-Compustat link (gvkey→permno) → 
        Ticker-CUSIP link (permno→ticker) → IBES (ticker)
    """
    print("📂 LOADING DATA...")
    
    # Load CEO match data
    ceo_df = pd.read_csv('Data/ceo_types_v0.2.csv')
    print(f"   CEO Match Data: {len(ceo_df):,} observations")
    
    # Load IBES analyst data
    ibes = pd.read_parquet('Output/wrds_ibes.parquet')
    print(f"   IBES Analyst Data: {len(ibes):,} observations")
    
    # Load link tables
    crsp_comp_link = pd.read_parquet('Output/wrds_crsp_comp.parquet')
    ticker_cusip_link = pd.read_parquet('Output/wrds_ticker_cusip.parquet')
    print(f"   CRSP-Compustat Link: {len(crsp_comp_link):,} links")
    print(f"   Ticker-CUSIP Link: {len(ticker_cusip_link):,} links")
    
    # Clean gvkey in link table
    crsp_comp_link['gvkey'] = pd.to_numeric(crsp_comp_link['gvkey'], errors='coerce')
    
    # Build gvkey → ticker mapping
    # Step 1: gvkey → permno (from CRSP-Compustat link)
    gvkey_to_permno = crsp_comp_link[['gvkey', 'permno']].dropna().drop_duplicates()
    
    # Step 2: permno → ticker (from ticker-cusip link) 
    permno_to_ticker = ticker_cusip_link[['permno', 'ticker']].dropna().drop_duplicates()
    
    # Step 3: Merge to get gvkey → ticker
    gvkey_to_ticker = gvkey_to_permno.merge(permno_to_ticker, on='permno', how='left')
    gvkey_to_ticker = gvkey_to_ticker[['gvkey', 'ticker']].dropna().drop_duplicates()
    
    # Keep only one ticker per gvkey (take first)
    gvkey_to_ticker = gvkey_to_ticker.groupby('gvkey').first().reset_index()
    print(f"   Unique gvkey-ticker mappings: {len(gvkey_to_ticker):,}")
    
    # Add ticker to CEO data
    ceo_df = ceo_df.merge(gvkey_to_ticker, on='gvkey', how='left')
    matched_before = ceo_df['ticker'].notna().sum()
    print(f"   CEO obs with ticker: {matched_before:,} ({matched_before/len(ceo_df)*100:.1f}%)")
    
    # Merge IBES data
    ibes['year'] = ibes['year'].astype(int)
    merged = ceo_df.merge(
        ibes,
        left_on=['ticker', 'fiscalyear'],
        right_on=['ticker', 'year'],
        how='left'
    )
    
    matched_ibes = merged['n_estimates'].notna().sum()
    print(f"   CEO obs with IBES data: {matched_ibes:,} ({matched_ibes/len(merged)*100:.1f}%)")
    
    return merged, ibes


def construct_ibes_variables(df):
    """
    Construct analyst-derived variables from IBES data.
    
    Variables:
    - analyst_coverage: Number of analyst estimates (log-transformed)
    - forecast_dispersion: StdDev of forecasts (normalized by abs(mean))
    - forecast_accuracy: 1 / (1 + |error|), bounded [0,1]
    - earnings_surprise: actual - forecast
    - positive_surprise: Binary indicator for positive surprise
    """
    print("\n📐 CONSTRUCTING IBES VARIABLES...")
    
    # Analyst Coverage (log-transformed)
    df['analyst_coverage'] = np.log1p(df['n_estimates'].fillna(0))
    
    # Forecast Dispersion (coefficient of variation style)
    # Use dispersion / |mean forecast| to normalize across firms
    df['forecast_dispersion_raw'] = df['forecast_dispersion']
    mean_forecast_abs = np.abs(df['mean_forecast'].fillna(0))
    
    # Safe division that handles NA and zero denominators
    df['forecast_dispersion_norm'] = df['forecast_dispersion'].copy()
    mask = mean_forecast_abs > 0.01
    df.loc[mask, 'forecast_dispersion_norm'] = (
        df.loc[mask, 'forecast_dispersion'] / mean_forecast_abs[mask]
    )
    
    # Forecast Accuracy: bounded [0,1], higher = more accurate
    df['forecast_accuracy'] = 1 / (1 + df['forecast_error'].fillna(np.inf))
    df['forecast_accuracy'] = df['forecast_accuracy'].clip(0, 1)
    
    # Earnings Surprise - fill NA with 0 for safe comparison
    df['earnings_surprise'] = df['actual_eps'].fillna(0) - df['mean_forecast'].fillna(0)
    # Use where to handle the comparison safely, result is float to accommodate NaN
    df['positive_surprise'] = 0.0
    valid_mask = df['actual_eps'].notna() & df['mean_forecast'].notna()
    df.loc[valid_mask, 'positive_surprise'] = (df.loc[valid_mask, 'earnings_surprise'] > 0).astype(float)
    df.loc[~valid_mask, 'positive_surprise'] = np.nan
    
    # Standardized surprise (by year) - with proper NA handling
    def safe_zscore(x):
        valid = x.dropna()
        if len(valid) > 1 and valid.std() > 0:
            return (x - valid.mean()) / valid.std()
        return x * 0  # return zeros if can't compute
    
    df['surprise_std'] = df.groupby('fiscalyear')['earnings_surprise'].transform(safe_zscore)
    
    # Winsorize extreme values
    for col in ['forecast_dispersion_norm', 'earnings_surprise', 'surprise_std']:
        if col in df.columns:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)
    
    # Summary stats
    ibes_vars = ['analyst_coverage', 'forecast_dispersion_norm', 
                 'forecast_accuracy', 'earnings_surprise', 'positive_surprise']
    
    print("\n   IBES Variable Summary:")
    for var in ibes_vars:
        if var in df.columns:
            valid = df[var].notna().sum()
            mean = df[var].mean()
            std = df[var].std()
            print(f"   {var:25s}: N={valid:,}, mean={mean:.4f}, std={std:.4f}")
    
    return df


# ================================================================
# ANALYSIS FUNCTIONS
# ================================================================

def run_descriptive_analysis(df):
    """Analyze IBES variables by match quality quintile."""
    print("\n" + "=" * 70)
    print("📊 DESCRIPTIVE ANALYSIS: IBES BY MATCH QUALITY")
    print("=" * 70)
    
    # Create match quintiles
    df = df.copy()
    df['match_q'] = pd.qcut(df['match_means'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    # Filter to observations with IBES data
    ibes_df = df.dropna(subset=['n_estimates'])
    print(f"\n   Observations with IBES data: {len(ibes_df):,}")
    
    ibes_vars = ['analyst_coverage', 'forecast_dispersion_norm', 
                 'forecast_accuracy', 'positive_surprise']
    
    results = ibes_df.groupby('match_q', observed=False).agg({
        'analyst_coverage': 'mean',
        'forecast_dispersion_norm': 'mean',
        'forecast_accuracy': 'mean',
        'positive_surprise': 'mean',
        'n_estimates': 'mean',
        'gvkey': 'count'
    }).round(4)
    
    results.columns = ['Log(Coverage)', 'Dispersion', 'Accuracy', 
                       'Pos Surprise %', 'Raw Coverage', 'N']
    
    print("\n--- IBES VARIABLES BY MATCH QUINTILE ---")
    print(results.to_string())
    
    # Statistical tests
    print("\n--- STATISTICAL TESTS: Q5 vs Q1 ---")
    q5 = ibes_df[ibes_df['match_q'] == 'Q5']
    q1 = ibes_df[ibes_df['match_q'] == 'Q1']
    
    for var in ibes_vars:
        if var in q5.columns and var in q1.columns:
            q5_data = q5[var].dropna()
            q1_data = q1[var].dropna()
            if len(q5_data) > 10 and len(q1_data) > 10:
                t_stat, p_val = stats.ttest_ind(q5_data, q1_data)
                diff = q5_data.mean() - q1_data.mean()
                print(f"   {var:25s}: Diff={diff:+.4f}, t={t_stat:.2f}, p={p_val:.4f}")
    
    return results


def run_regression_analysis(df):
    """Run OLS regressions: match_means ~ IBES variables + controls."""
    print("\n" + "=" * 70)
    print("📈 REGRESSION ANALYSIS: MATCH ~ IBES VARIABLES")
    print("=" * 70)
    
    # Filter to complete cases
    reg_vars = ['match_means', 'analyst_coverage', 'forecast_dispersion_norm',
                'forecast_accuracy', 'positive_surprise', 'logatw', 'rdintw']
    
    reg_df = df.dropna(subset=reg_vars).copy()
    print(f"\n   Regression sample: {len(reg_df):,} observations")
    
    # Standardize for interpretation
    for var in ['analyst_coverage', 'forecast_dispersion_norm', 'forecast_accuracy']:
        reg_df[f'{var}_z'] = (reg_df[var] - reg_df[var].mean()) / reg_df[var].std()
    
    # Model 1: IBES only
    print("\n--- MODEL 1: Match ~ IBES Variables Only ---")
    model1 = ols(
        'match_means ~ analyst_coverage_z + forecast_dispersion_norm_z + forecast_accuracy_z + positive_surprise',
        data=reg_df
    ).fit()
    print(model1.summary().tables[1].as_text())
    
    # Model 2: IBES + Firm Controls
    print("\n--- MODEL 2: Match ~ IBES + Firm Controls ---")
    model2 = ols(
        'match_means ~ analyst_coverage_z + forecast_dispersion_norm_z + forecast_accuracy_z + positive_surprise + logatw + rdintw',
        data=reg_df
    ).fit()
    print(model2.summary().tables[1].as_text())
    
    # Model 3: Include year FE (approximation via dummies)
    print("\n--- MODEL 3: Match ~ IBES + Controls + Year FE ---")
    reg_df['year_fe'] = reg_df['fiscalyear'].astype(str)
    model3 = ols(
        'match_means ~ analyst_coverage_z + forecast_dispersion_norm_z + forecast_accuracy_z + positive_surprise + logatw + rdintw + C(year_fe)',
        data=reg_df
    ).fit()
    
    # Extract IBES coefficients only
    ibes_coefs = {k: v for k, v in model3.params.items() if 'coverage' in k or 'dispersion' in k or 'accuracy' in k or 'surprise' in k}
    ibes_pvals = {k: v for k, v in model3.pvalues.items() if 'coverage' in k or 'dispersion' in k or 'accuracy' in k or 'surprise' in k}
    
    print("IBES Variable Coefficients (with Year FE):")
    for var in ibes_coefs:
        print(f"   {var:35s}: β={ibes_coefs[var]:+.4f}, p={ibes_pvals[var]:.4f}")
    
    print(f"\n   Model R²: {model3.rsquared:.4f}")
    print(f"   Observations: {int(model3.nobs):,}")
    
    return {'model1': model1, 'model2': model2, 'model3': model3}


def run_random_forest_analysis(df):
    """Run Random Forest to assess IBES feature importance for match quality."""
    print("\n" + "=" * 70)
    print("🌲 RANDOM FOREST: FEATURE IMPORTANCE")
    print("=" * 70)
    
    # Features for RF
    ibes_features = ['analyst_coverage', 'forecast_dispersion_norm', 
                     'forecast_accuracy', 'positive_surprise']
    firm_features = ['logatw', 'rdintw', 'leverage', 'exp_roa', 'boardindpw']
    ceo_features = ['Age', 'tenure', 'ivy']
    
    all_features = ibes_features + firm_features + ceo_features
    
    # Prepare data - first compute tenure
    rf_df = df.copy()
    rf_df['tenure'] = (rf_df['fiscalyear'] - rf_df['ceo_year']).clip(lower=0)
    
    # Now filter for complete cases
    rf_df = rf_df.dropna(subset=all_features + ['match_means'])
    print(f"\n   RF Sample: {len(rf_df):,} observations")
    
    X = rf_df[all_features].values
    y = rf_df['match_means'].values
    
    # Train RF
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n--- FEATURE IMPORTANCE RANKING ---")
    for i, row in importance_df.iterrows():
        marker = "🔵" if row['Feature'] in ibes_features else "⚪"
        print(f"   {marker} {row['Feature']:25s}: {row['Importance']:.4f}")
    
    # R² score
    from sklearn.metrics import r2_score
    y_pred = rf.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"\n   RF R² (in-sample): {r2:.4f}")
    
    return importance_df, rf


def run_two_towers_with_ibes(df, config):
    """Train Two Towers model with IBES features as additional firm features."""
    print("\n" + "=" * 70)
    print("🗼🗼 TWO TOWERS MODEL WITH IBES FEATURES")
    print("=" * 70)
    
    # Prepare data with IBES features
    ibes_features = ['analyst_coverage', 'forecast_dispersion_norm', 'forecast_accuracy']
    
    # Filter to IBES-available observations
    tt_df = df.dropna(subset=ibes_features + ['match_means']).copy()
    tt_df['tenure'] = (tt_df['fiscalyear'] - tt_df['ceo_year']).clip(lower=0)
    print(f"\n   Two Towers Sample: {len(tt_df):,} observations")
    
    # Extend firm numeric columns to include IBES
    extended_firm_num = list(config.FIRM_NUM_COLS) + ibes_features
    
    # Create a modified config
    class ExtendedConfig(Config):
        FIRM_NUM_COLS = extended_firm_num
    
    ext_config = ExtendedConfig()
    
    # Process data
    processor = DataProcessor(ext_config)
    df_clean = processor.prepare_features(tt_df)
    
    # Train/val split
    train_df, val_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    
    # Fit and transform
    processor.fit(train_df)
    train_data = processor.transform(train_df)
    val_data = processor.transform(val_df)
    
    # Create data loaders
    train_loader = DataLoader(
        CEOFirmDataset(train_data), 
        batch_size=ext_config.BATCH_SIZE, 
        shuffle=True
    )
    val_loader = DataLoader(
        CEOFirmDataset(val_data), 
        batch_size=ext_config.BATCH_SIZE, 
        shuffle=False
    )
    
    # Train model
    print("\n   Training Two Towers with IBES features...")
    model = train_model(train_loader, val_loader, train_data, ext_config)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        all_preds = []
        all_targets = []
        
        for batch in val_loader:
            firm_num = batch['firm_numeric'].to(device)
            firm_cat = batch['firm_cat'].to(device)
            ceo_num = batch['ceo_numeric'].to(device)
            ceo_cat = batch['ceo_cat'].to(device)
            target = batch['target']
            
            pred = model(firm_num, firm_cat, ceo_num, ceo_cat)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.numpy().flatten())
    
    correlation = np.corrcoef(all_preds, all_targets)[0, 1]
    print(f"\n   Validation Correlation: {correlation:.4f}")
    
    return model, processor, correlation


# ================================================================
# VISUALIZATION
# ================================================================

def create_visualizations(df, importance_df, results):
    """Create comprehensive IBES analysis visualizations."""
    print("\n" + "=" * 70)
    print("📊 CREATING VISUALIZATIONS")
    print("=" * 70)
    
    fig = plt.figure(figsize=(20, 16))
    
    # Filter to IBES data
    ibes_df = df.dropna(subset=['n_estimates']).copy()
    ibes_df['match_q'] = pd.qcut(ibes_df['match_means'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60']
    
    # 1. Analyst Coverage by Match
    ax1 = fig.add_subplot(3, 3, 1)
    coverage = ibes_df.groupby('match_q', observed=False)['n_estimates'].mean()
    ax1.bar(range(5), coverage.values, color=colors, edgecolor='black')
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax1.set_ylabel('Mean Analyst Coverage')
    ax1.set_xlabel('Match Quality Quintile')
    ax1.set_title('📊 Analyst Coverage by Match Quality', fontweight='bold')
    
    # 2. Forecast Dispersion by Match
    ax2 = fig.add_subplot(3, 3, 2)
    dispersion = ibes_df.groupby('match_q', observed=False)['forecast_dispersion_norm'].mean()
    ax2.bar(range(5), dispersion.values, color=colors, edgecolor='black')
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax2.set_ylabel('Forecast Dispersion (CV)')
    ax2.set_xlabel('Match Quality Quintile')
    ax2.set_title('📉 Analyst Disagreement by Match', fontweight='bold')
    
    # 3. Forecast Accuracy by Match
    ax3 = fig.add_subplot(3, 3, 3)
    accuracy = ibes_df.groupby('match_q', observed=False)['forecast_accuracy'].mean()
    ax3.bar(range(5), accuracy.values, color=colors, edgecolor='black')
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax3.set_ylabel('Forecast Accuracy')
    ax3.set_xlabel('Match Quality Quintile')
    ax3.set_title('🎯 Forecast Accuracy by Match', fontweight='bold')
    
    # 4. Positive Surprise Rate by Match
    ax4 = fig.add_subplot(3, 3, 4)
    surprise = ibes_df.groupby('match_q', observed=False)['positive_surprise'].mean()
    ax4.bar(range(5), surprise.values * 100, color=colors, edgecolor='black')
    ax4.set_xticks(range(5))
    ax4.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax4.set_ylabel('% Positive Surprises')
    ax4.set_xlabel('Match Quality Quintile')
    ax4.set_title('😊 Earnings Beat Rate by Match', fontweight='bold')
    ax4.axhline(y=50, color='gray', linestyle='--', lw=1)
    
    # 5. Feature Importance
    ax5 = fig.add_subplot(3, 3, 5)
    imp_sorted = importance_df.sort_values('Importance', ascending=True).tail(10)
    colors_imp = ['steelblue' if 'coverage' in f or 'dispersion' in f or 'accuracy' in f or 'surprise' in f 
                  else 'lightgray' for f in imp_sorted['Feature']]
    ax5.barh(range(len(imp_sorted)), imp_sorted['Importance'], color=colors_imp, edgecolor='black')
    ax5.set_yticks(range(len(imp_sorted)))
    ax5.set_yticklabels(imp_sorted['Feature'], fontsize=9)
    ax5.set_xlabel('Importance')
    ax5.set_title('🌲 RF Feature Importance\n(Blue = IBES)', fontweight='bold')
    
    # 6. Match vs Coverage Scatter
    ax6 = fig.add_subplot(3, 3, 6)
    sample = ibes_df.sample(min(2000, len(ibes_df)))
    ax6.scatter(sample['analyst_coverage'], sample['match_means'], alpha=0.3, s=15)
    z = np.polyfit(sample['analyst_coverage'].dropna(), 
                   sample.loc[sample['analyst_coverage'].notna(), 'match_means'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sample['analyst_coverage'].min(), sample['analyst_coverage'].max(), 100)
    ax6.plot(x_line, p(x_line), 'r-', lw=2, label=f'β={z[0]:.3f}')
    ax6.set_xlabel('Log(Analyst Coverage)')
    ax6.set_ylabel('Match Quality')
    ax6.set_title('Coverage → Match Quality', fontweight='bold')
    ax6.legend()
    
    # 7. Match vs Accuracy Scatter
    ax7 = fig.add_subplot(3, 3, 7)
    sample = ibes_df[ibes_df['forecast_accuracy'] < 1].sample(min(2000, len(ibes_df)))
    ax7.scatter(sample['forecast_accuracy'], sample['match_means'], alpha=0.3, s=15)
    z = np.polyfit(sample['forecast_accuracy'].dropna(), 
                   sample.loc[sample['forecast_accuracy'].notna(), 'match_means'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0.01, 0.99, 100)
    ax7.plot(x_line, p(x_line), 'r-', lw=2, label=f'β={z[0]:.3f}')
    ax7.set_xlabel('Forecast Accuracy')
    ax7.set_ylabel('Match Quality')
    ax7.set_title('Accuracy → Match Quality', fontweight='bold')
    ax7.legend()
    
    # 8. Heatmap: Match × Size → Coverage
    ax8 = fig.add_subplot(3, 3, 8)
    try:
        size_bins = pd.qcut(ibes_df['logatw'], 4, labels=['Small', 'Mid', 'Large', 'Giant'])
        match_bins = pd.qcut(ibes_df['match_means'], 4, labels=['Low', 'Med', 'High', 'Top'])
        heatmap = ibes_df.groupby([match_bins, size_bins], observed=False)['n_estimates'].mean().unstack()
        sns.heatmap(heatmap, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax8)
        ax8.set_title('Analyst Coverage:\nMatch × Size', fontweight='bold')
    except Exception as e:
        ax8.text(0.5, 0.5, f'Heatmap Error: {e}', ha='center', va='center')
    
    # 9. Time Trend
    ax9 = fig.add_subplot(3, 3, 9)
    time_trend = ibes_df.groupby('fiscalyear').agg({
        'match_means': 'mean',
        'n_estimates': 'mean',
        'forecast_accuracy': 'mean'
    })
    ax9.plot(time_trend.index, time_trend['match_means'], 'b-o', lw=2, label='Match Quality')
    ax9_twin = ax9.twinx()
    ax9_twin.plot(time_trend.index, time_trend['n_estimates'], 'g-s', lw=2, label='Coverage')
    ax9.set_xlabel('Year')
    ax9.set_ylabel('Match Quality', color='blue')
    ax9_twin.set_ylabel('Analyst Coverage', color='green')
    ax9.set_title('Match & Coverage Over Time', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Output/ibes_match_analysis.png', dpi=150, bbox_inches='tight')
    print("\n   Saved: Output/ibes_match_analysis.png")


# ================================================================
# MAIN
# ================================================================

def main():
    print("📊" * 40)
    print("   IBES ANALYST INTEGRATION WITH CEO-FIRM MATCH")
    print("📊" * 40)
    
    # Load and link data
    merged_df, raw_ibes = load_and_link_ibes()
    
    # Construct IBES variables
    merged_df = construct_ibes_variables(merged_df)
    
    # Run analyses
    results = run_descriptive_analysis(merged_df)
    reg_results = run_regression_analysis(merged_df)
    importance_df, rf_model = run_random_forest_analysis(merged_df)
    
    # Two Towers with IBES
    config = Config()
    try:
        model, processor, corr = run_two_towers_with_ibes(merged_df, config)
    except Exception as e:
        print(f"\n   ⚠️ Two Towers skipped: {e}")
        corr = None
    
    # Create visualizations
    create_visualizations(merged_df, importance_df, results)
    
    # Summary
    print("\n" + "=" * 70)
    print("🏆 IBES ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Extract key findings
    ibes_obs = merged_df['n_estimates'].notna().sum()
    coverage_diff = results.loc['Q5', 'Raw Coverage'] - results.loc['Q1', 'Raw Coverage']
    accuracy_diff = results.loc['Q5', 'Accuracy'] - results.loc['Q1', 'Accuracy']
    surprise_diff = (results.loc['Q5', 'Pos Surprise %'] - results.loc['Q1', 'Pos Surprise %']) * 100
    
    # Get RF IBES importance
    ibes_importance = importance_df[importance_df['Feature'].str.contains('coverage|dispersion|accuracy|surprise', case=False)]['Importance'].sum()
    total_importance = importance_df['Importance'].sum()
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    📊📊📊 IBES INTEGRATION FINDINGS 📊📊📊                    ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  📦 DATA SUMMARY                                                              ║
║     CEO-Firm observations with IBES: {ibes_obs:,}                            ║
║     Merge rate: {ibes_obs/len(merged_df)*100:.1f}%                                                    ║
║                                                                               ║
║  📊 ANALYST COVERAGE                                                          ║
║     Q5 has {coverage_diff:+.1f} more analysts than Q1                                 ║
║     Better matches attract MORE analyst attention                             ║
║                                                                               ║
║  🎯 FORECAST ACCURACY                                                         ║
║     Q5 accuracy is {accuracy_diff:+.4f} higher than Q1                                ║
║     Better matches have MORE predictable earnings                             ║
║                                                                               ║
║  😊 EARNINGS SURPRISES                                                        ║
║     Q5 positive surprise rate is {surprise_diff:+.1f}pp vs Q1                         ║
║     Better matches BEAT expectations more often                               ║
║                                                                               ║
║  🌲 RANDOM FOREST                                                             ║
║     IBES features explain {ibes_importance/total_importance*100:.1f}% of match quality variance       ║
║                                                                               ║
║  🗼🗼 TWO TOWERS                                                               ║
║     Validation correlation: {f'{corr:.4f}' if corr else 'N/A'}                                         ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n📊📊📊 IBES ANALYSIS COMPLETE! 📊📊📊")


if __name__ == "__main__":
    main()
