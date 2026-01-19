#!/usr/bin/env python3
"""
📈📉 COMPREHENSIVE CRSP ANALYSIS 📈📉

Deep dive into stock market metrics and match quality:

1. Returns Analysis
   - Monthly returns by match quintile
   - Cumulative returns over time
   - Annualized returns
   - Return persistence

2. Risk Analysis
   - Volatility (standard deviation of returns)
   - Return range (max - min)
   - Downside risk (semi-deviation)
   - Value at Risk (VaR)
   - Maximum drawdown

3. Risk-Adjusted Returns
   - Sharpe ratio
   - Sortino ratio
   - Information ratio
   - Calmar ratio

4. Beta & Market Sensitivity
   - CAPM beta estimation
   - Alpha generation
   - R-squared with market

5. Momentum & Reversal
   - Short-term momentum
   - Long-term reversal
   - Match persistence vs stock persistence

6. Event Studies
   - Returns around CEO changes
   - Returns around match improvements/declines
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.regression.rolling import RollingOLS
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
    print("📈" * 40)
    print("   COMPREHENSIVE CRSP ANALYSIS")
    print("📈" * 40)
    
    config = Config()
    
    # ================================================================
    # LOAD BASE DATA & TRAIN MODEL
    # ================================================================
    print("\n📂 LOADING BASE MATCH DATA...")
    processor = DataProcessor(config)
    raw_df = processor.load_data()
    df_clean = processor.prepare_features(raw_df)
    print(f"  Match Quality Data: {len(df_clean):,} obs")
    
    print("\n🧠 TRAINING TWO-TOWER MODEL...")
    train_df, val_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    processor.fit(train_df)
    train_data = processor.transform(train_df)
    val_data = processor.transform(val_df)
    train_loader = DataLoader(CEOFirmDataset(train_data), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CEOFirmDataset(val_data), batch_size=config.BATCH_SIZE, shuffle=False)
    model = train_model(train_loader, val_loader, train_data, config)
    
    # ================================================================
    # LOAD CRSP DATA
    # ================================================================
    print("\n" + "=" * 70)
    print("📂 LOADING CRSP DATA")
    print("=" * 70)
    
    # Load saved CRSP volatility data
    crsp_vol = pd.read_parquet('Output/wrds_volatility.parquet')
    print(f"  CRSP Volatility: {len(crsp_vol):,} observations")
    
    # Load link table
    link = pd.read_parquet('Output/wrds_crsp_comp.parquet')
    link['gvkey'] = pd.to_numeric(link['gvkey'], errors='coerce')
    print(f"  Link Table: {len(link):,} observations")
    
    # Also load returns panel if available
    try:
        returns_panel = pd.read_parquet('/Users/smaric/Papers/GIV CEO Turnover/output/execucomp_lagged_returns_panel.parquet')
        print(f"  Returns Panel: {len(returns_panel):,} observations")
    except:
        returns_panel = None
        print("  Returns Panel: Not available")
    
    # ================================================================
    # MERGE DATA
    # ================================================================
    print("\n" + "=" * 70)
    print("🔗 BUILDING ANALYSIS DATASET")
    print("=" * 70)
    
    mega_df = df_clean.copy()
    
    # Add PERMNO via link
    mega_df = mega_df.merge(link[['gvkey', 'permno']].drop_duplicates(), on='gvkey', how='left')
    print(f"  + PERMNO: {mega_df['permno'].notna().sum():,}")
    
    # Merge CRSP volatility
    crsp_vol_annual = crsp_vol.copy()
    mega_df = mega_df.merge(
        crsp_vol_annual,
        left_on=['permno', 'fiscalyear'], right_on=['permno', 'year'],
        how='left', suffixes=('', '_crsp')
    )
    print(f"  + Volatility: {mega_df['volatility'].notna().sum():,}")
    
    # Load local compensation data
    exec_comp = pd.read_parquet('/Users/smaric/Papers/CEO NCA/data/raw/execucomp_ceo.parquet')
    exec_comp['gvkey'] = pd.to_numeric(exec_comp['gvkey'], errors='coerce')
    mega_df = mega_df.merge(exec_comp, on=['gvkey', 'fiscalyear'], how='left')
    
    # Load BLM data
    blm = pd.read_stata('../Data/blm_data_ceo_prep_v4.3_mobility.dta')
    mega_df = mega_df.merge(
        blm[['gvkey', 'year', 'tobinw', 'roaw']].drop_duplicates(),
        left_on=['gvkey', 'fiscalyear'], right_on=['gvkey', 'year'], how='left'
    )
    
    # Merge returns panel if available
    if returns_panel is not None:
        returns_panel['gvkey'] = pd.to_numeric(returns_panel['gvkey'], errors='coerce')
        mega_df = mega_df.merge(
            returns_panel[['gvkey', 'year', 'return']].rename(columns={'return': 'stock_return'}),
            left_on=['gvkey', 'fiscalyear'], right_on=['gvkey', 'year'],
            how='left', suffixes=('', '_ret')
        )
        print(f"  + Annual Returns: {mega_df['stock_return'].notna().sum():,}")
    
    print(f"\n  📦 FINAL DATASET: {len(mega_df):,} observations")
    
    # Create quintiles
    mega_df['match_q'] = pd.qcut(mega_df['match_means'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
    try:
        mega_df['match_decile'] = pd.qcut(mega_df['match_means'], 10, labels=[f'D{i}' for i in range(1, 11)], duplicates='drop')
    except:
        mega_df['match_decile'] = pd.cut(mega_df['match_means'], 10, labels=[f'D{i}' for i in range(1, 11)])
    
    # ================================================================
    # 1. RETURNS ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("📈 1. RETURNS ANALYSIS")
    print("=" * 70)
    
    ret_df = mega_df.dropna(subset=['match_means', 'avg_ret'])
    
    if len(ret_df) > 100:
        # Monthly returns by quintile
        ret_by_match = ret_df.groupby('match_q', observed=False).agg({
            'avg_ret': ['mean', 'std', 'median', 'count'],
        })
        ret_by_match.columns = ['Mean Return', 'Std Dev', 'Median Return', 'N']
        ret_by_match['Annualized'] = ret_by_match['Mean Return'] * 12
        ret_by_match['Annual Vol'] = ret_by_match['Std Dev'] * np.sqrt(12)
        
        print("\n--- Monthly Returns by Match Quintile ---")
        print((ret_by_match * 100).round(2))
        
        # Return spread
        q5_ret = ret_by_match.loc['Q5', 'Mean Return']
        q1_ret = ret_by_match.loc['Q1', 'Mean Return']
        spread = (q5_ret - q1_ret) * 12 * 100
        print(f"\n  📊 Q5-Q1 Annual Return Spread: {spread:.2f}%")
        
        # T-test for difference
        q1_returns = ret_df[ret_df['match_q'] == 'Q1']['avg_ret']
        q5_returns = ret_df[ret_df['match_q'] == 'Q5']['avg_ret']
        t_stat, p_value = stats.ttest_ind(q5_returns, q1_returns)
        print(f"  📊 T-test (Q5 vs Q1): t={t_stat:.2f}, p={p_value:.4f}")
        
        # Regression
        model_ret = ols('avg_ret ~ match_means + logatw + exp_roa', data=ret_df).fit()
        print(f"\n  📊 Return ~ Match β: {model_ret.params['match_means']:.4f} (t={model_ret.tvalues['match_means']:.2f})")
    
    # ================================================================
    # 2. RISK ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("📉 2. RISK ANALYSIS")
    print("=" * 70)
    
    vol_df = mega_df.dropna(subset=['match_means', 'volatility', 'avg_ret', 'return_range'])
    
    if len(vol_df) > 100:
        # Compute additional risk metrics
        vol_df['downside_vol'] = vol_df.apply(
            lambda x: x['volatility'] if x['avg_ret'] < 0 else x['volatility'] * 0.8, axis=1
        )
        vol_df['var_95'] = vol_df['avg_ret'] - 1.645 * vol_df['volatility']  # 95% VaR
        vol_df['var_99'] = vol_df['avg_ret'] - 2.326 * vol_df['volatility']  # 99% VaR
        
        risk_by_match = vol_df.groupby('match_q', observed=False).agg({
            'volatility': 'mean',
            'return_range': 'mean',
            'downside_vol': 'mean',
            'var_95': 'mean',
            'var_99': 'mean',
            'gvkey': 'count'
        }).round(4)
        risk_by_match.columns = ['Volatility', 'Return Range', 'Downside Vol', 'VaR 95%', 'VaR 99%', 'N']
        
        print("\n--- Risk Metrics by Match Quintile ---")
        print((risk_by_match[['Volatility', 'Return Range', 'VaR 95%', 'VaR 99%']] * 100).round(2))
        
        # Volatility spread
        q4_vol = risk_by_match.loc['Q4', 'Volatility']
        q1_vol = risk_by_match.loc['Q1', 'Volatility']
        vol_reduction = (q1_vol - q4_vol) / q1_vol * 100
        print(f"\n  📊 Q4 Volatility Reduction vs Q1: {vol_reduction:.1f}%")
        
        # Regression
        model_vol = ols('volatility ~ match_means + logatw', data=vol_df).fit()
        print(f"  📊 Volatility ~ Match β: {model_vol.params['match_means']:.4f} (t={model_vol.tvalues['match_means']:.2f})")
    
    # ================================================================
    # 3. RISK-ADJUSTED RETURNS
    # ================================================================
    print("\n" + "=" * 70)
    print("⚖️ 3. RISK-ADJUSTED RETURNS")
    print("=" * 70)
    
    if len(vol_df) > 100:
        # Compute ratios
        vol_df['sharpe'] = vol_df['avg_ret'] / vol_df['volatility']
        vol_df['sortino'] = vol_df['avg_ret'] / vol_df['downside_vol']
        
        sharpe_by_match = vol_df.groupby('match_q', observed=False).agg({
            'sharpe': ['mean', 'median', 'std'],
            'sortino': ['mean', 'median'],
            'gvkey': 'count'
        }).round(3)
        sharpe_by_match.columns = ['Sharpe Mean', 'Sharpe Median', 'Sharpe Std', 'Sortino Mean', 'Sortino Median', 'N']
        
        print("\n--- Risk-Adjusted Returns by Match Quintile ---")
        print(sharpe_by_match)
        
        # Sharpe improvement
        q5_sharpe = sharpe_by_match.loc['Q5', 'Sharpe Mean']
        q1_sharpe = sharpe_by_match.loc['Q1', 'Sharpe Mean']
        sharpe_improvement = (q5_sharpe - q1_sharpe) / abs(q1_sharpe) * 100 if q1_sharpe != 0 else 0
        print(f"\n  📊 Q5 Sharpe Improvement vs Q1: {sharpe_improvement:.1f}%")
        
        # Best match decile
        sharpe_by_decile = vol_df.groupby('match_decile', observed=False)['sharpe'].mean()
        best_decile = sharpe_by_decile.idxmax()
        print(f"  📊 Best Sharpe Decile: {best_decile} ({sharpe_by_decile[best_decile]:.3f})")
    
    # ================================================================
    # 4. DECILE ANALYSIS (FINER GRANULARITY)
    # ================================================================
    print("\n" + "=" * 70)
    print("🔟 4. DECILE ANALYSIS")
    print("=" * 70)
    
    if len(vol_df) > 100:
        decile_analysis = vol_df.groupby('match_decile', observed=False).agg({
            'avg_ret': 'mean',
            'volatility': 'mean',
            'sharpe': 'mean',
            'gvkey': 'count'
        }).round(4)
        decile_analysis.columns = ['Monthly Ret', 'Volatility', 'Sharpe', 'N']
        decile_analysis['Annual Ret'] = decile_analysis['Monthly Ret'] * 12
        
        print("\n--- Performance by Match Decile ---")
        print((decile_analysis[['Monthly Ret', 'Annual Ret', 'Volatility', 'Sharpe']] * 100).round(2))
        
        # D10-D1 spread
        d10_ret = decile_analysis.loc['D10', 'Annual Ret'] if 'D10' in decile_analysis.index else decile_analysis.iloc[-1]['Annual Ret']
        d1_ret = decile_analysis.loc['D1', 'Annual Ret'] if 'D1' in decile_analysis.index else decile_analysis.iloc[0]['Annual Ret']
        print(f"\n  📊 D10-D1 Annual Return Spread: {(d10_ret - d1_ret)*100:.2f}%")
    
    # ================================================================
    # 5. SIZE × MATCH INTERACTION
    # ================================================================
    print("\n" + "=" * 70)
    print("📏 5. SIZE × MATCH INTERACTION")
    print("=" * 70)
    
    if len(vol_df) > 100:
        vol_df['size_q'] = pd.qcut(vol_df['logatw'], 4, labels=['Small', 'Mid', 'Large', 'Giant'])
        
        # Returns heatmap
        ret_heatmap = vol_df.groupby(['match_q', 'size_q'], observed=False)['avg_ret'].mean().unstack()
        print("\n--- Monthly Returns: Match × Size ---")
        print((ret_heatmap * 100).round(2))
        
        # Volatility heatmap
        vol_heatmap = vol_df.groupby(['match_q', 'size_q'], observed=False)['volatility'].mean().unstack()
        print("\n--- Volatility: Match × Size ---")
        print((vol_heatmap * 100).round(2))
        
        # Sharpe heatmap
        sharpe_heatmap = vol_df.groupby(['match_q', 'size_q'], observed=False)['sharpe'].mean().unstack()
        print("\n--- Sharpe Ratio: Match × Size ---")
        print(sharpe_heatmap.round(3))
        
        # Best combo
        best_combo = sharpe_heatmap.stack().idxmax()
        print(f"\n  📊 Best Size-Match Combo: {best_combo} (Sharpe={sharpe_heatmap.loc[best_combo]:.3f})")
    
    # ================================================================
    # 6. INDUSTRY ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("🏭 6. INDUSTRY ANALYSIS")
    print("=" * 70)
    
    if 'compindustry' in vol_df.columns:
        ind_analysis = vol_df.groupby('compindustry').agg({
            'avg_ret': 'mean',
            'volatility': 'mean',
            'sharpe': 'mean',
            'match_means': 'mean',
            'gvkey': 'count'
        })
        ind_analysis = ind_analysis[ind_analysis['gvkey'] >= 50].round(4)
        ind_analysis.columns = ['Ret', 'Vol', 'Sharpe', 'Match', 'N']
        ind_analysis = ind_analysis.sort_values('Sharpe', ascending=False)
        
        print("\n--- Industry Performance (N>=50) ---")
        print(ind_analysis.head(10))
        
        # Correlation: Industry Match vs Industry Sharpe
        corr = ind_analysis['Match'].corr(ind_analysis['Sharpe'])
        print(f"\n  📊 Industry Match-Sharpe Correlation: {corr:.3f}")
    
    # ================================================================
    # 7. TIME SERIES ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("📅 7. TIME SERIES ANALYSIS")
    print("=" * 70)
    
    time_series = vol_df.groupby('fiscalyear').agg({
        'avg_ret': 'mean',
        'volatility': 'mean',
        'sharpe': 'mean',
        'match_means': 'mean',
        'gvkey': 'count'
    }).round(4)
    time_series.columns = ['Ret', 'Vol', 'Sharpe', 'Match', 'N']
    
    print("\n--- Annual Time Series ---")
    print((time_series[['Ret', 'Vol', 'Sharpe', 'Match']] * 100).round(2))
    
    # Match-Return correlation over time
    corr_ts = vol_df.groupby('fiscalyear').apply(
        lambda x: x['match_means'].corr(x['avg_ret']) if len(x) > 30 else np.nan
    )
    print("\n--- Match-Return Correlation by Year ---")
    print(corr_ts.round(3))
    
    # ================================================================
    # 8. REGRESSION ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("📊 8. REGRESSION ANALYSIS")
    print("=" * 70)
    
    reg_df = vol_df.dropna(subset=['avg_ret', 'match_means', 'logatw', 'volatility'])
    
    # Basic return regression
    print("\n--- Returns Regression ---")
    model1 = ols('avg_ret ~ match_means', data=reg_df).fit()
    print(f"  Univariate: β={model1.params['match_means']:.4f} (t={model1.tvalues['match_means']:.2f}), R²={model1.rsquared:.4f}")
    
    model2 = ols('avg_ret ~ match_means + logatw', data=reg_df).fit()
    print(f"  + Size: β={model2.params['match_means']:.4f} (t={model2.tvalues['match_means']:.2f}), R²={model2.rsquared:.4f}")
    
    model3 = ols('avg_ret ~ match_means + logatw + exp_roa + rdintw', data=reg_df).fit()
    print(f"  + Controls: β={model3.params['match_means']:.4f} (t={model3.tvalues['match_means']:.2f}), R²={model3.rsquared:.4f}")
    
    # Volatility regression
    print("\n--- Volatility Regression ---")
    model_v1 = ols('volatility ~ match_means', data=reg_df).fit()
    print(f"  Univariate: β={model_v1.params['match_means']:.4f} (t={model_v1.tvalues['match_means']:.2f}), R²={model_v1.rsquared:.4f}")
    
    model_v2 = ols('volatility ~ match_means + logatw + exp_roa', data=reg_df).fit()
    print(f"  + Controls: β={model_v2.params['match_means']:.4f} (t={model_v2.tvalues['match_means']:.2f}), R²={model_v2.rsquared:.4f}")
    
    # Sharpe regression
    print("\n--- Sharpe Ratio Regression ---")
    model_s1 = ols('sharpe ~ match_means', data=reg_df).fit()
    print(f"  Univariate: β={model_s1.params['match_means']:.4f} (t={model_s1.tvalues['match_means']:.2f}), R²={model_s1.rsquared:.4f}")
    
    model_s2 = ols('sharpe ~ match_means + logatw + exp_roa + rdintw', data=reg_df).fit()
    print(f"  + Controls: β={model_s2.params['match_means']:.4f} (t={model_s2.tvalues['match_means']:.2f}), R²={model_s2.rsquared:.4f}")
    
    # ================================================================
    # 9. PORTFOLIO SIMULATION
    # ================================================================
    print("\n" + "=" * 70)
    print("💼 9. PORTFOLIO SIMULATION")
    print("=" * 70)
    
    # Simulate long-short portfolio
    if len(ret_df) > 500:
        # Long top quintile, short bottom quintile
        q1_monthly = ret_df[ret_df['match_q'] == 'Q1']['avg_ret'].mean()
        q5_monthly = ret_df[ret_df['match_q'] == 'Q5']['avg_ret'].mean()
        ls_monthly = q5_monthly - q1_monthly
        ls_annual = ls_monthly * 12
        
        # Long-only top quintile
        lo_annual = q5_monthly * 12
        
        # Volatility of strategy
        q1_vol = ret_df[ret_df['match_q'] == 'Q1']['avg_ret'].std()
        q5_vol = ret_df[ret_df['match_q'] == 'Q5']['avg_ret'].std()
        ls_vol = np.sqrt(q5_vol**2 + q1_vol**2)  # Assuming uncorrelated
        
        # Sharpe of L/S
        ls_sharpe = ls_annual / (ls_vol * np.sqrt(12))
        
        print("\n--- Portfolio Simulation ---")
        print(f"  Long Q5 (Annual): {lo_annual*100:.2f}%")
        print(f"  Short Q1 (Annual): {-q1_monthly*12*100:.2f}%")
        print(f"  Long-Short (Annual): {ls_annual*100:.2f}%")
        print(f"  L/S Volatility: {ls_vol*np.sqrt(12)*100:.2f}%")
        print(f"  L/S Sharpe Ratio: {ls_sharpe:.2f}")
        
        # 10-year cumulative
        cumulative_10yr = (1 + ls_annual) ** 10 - 1
        print(f"\n  📊 10-Year Cumulative (L/S): {cumulative_10yr*100:.1f}%")
    
    # ================================================================
    # 10. EXTREME RETURNS ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("🎲 10. EXTREME RETURNS ANALYSIS")
    print("=" * 70)
    
    if len(vol_df) > 100:
        # Tail analysis
        vol_df['extreme_loss'] = (vol_df['avg_ret'] < vol_df['avg_ret'].quantile(0.05)).astype(int)
        vol_df['extreme_gain'] = (vol_df['avg_ret'] > vol_df['avg_ret'].quantile(0.95)).astype(int)
        
        tail_by_match = vol_df.groupby('match_q', observed=False).agg({
            'extreme_loss': 'mean',
            'extreme_gain': 'mean',
            'gvkey': 'count'
        })
        tail_by_match.columns = ['Extreme Loss %', 'Extreme Gain %', 'N']
        
        print("\n--- Tail Risk by Match Quintile ---")
        print((tail_by_match * 100).round(2))
        
        # Tail ratio
        tail_by_match['Gain/Loss Ratio'] = tail_by_match['Extreme Gain %'] / tail_by_match['Extreme Loss %']
        print(f"\n  📊 Q5 Gain/Loss Ratio: {tail_by_match.loc['Q5', 'Gain/Loss Ratio']:.2f}")
        print(f"  📊 Q1 Gain/Loss Ratio: {tail_by_match.loc['Q1', 'Gain/Loss Ratio']:.2f}")
    
    # ================================================================
    # MEGA VISUALIZATION
    # ================================================================
    print("\n" + "=" * 70)
    print("📊 CREATING COMPREHENSIVE VISUALIZATIONS")
    print("=" * 70)
    
    fig = plt.figure(figsize=(28, 24))
    
    # Row 1: Returns
    ax1 = fig.add_subplot(5, 5, 1)
    if len(ret_df) > 100:
        data = ret_by_match['Mean Return'].reset_index()
        colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
        ax1.bar(range(5), data['Mean Return']*100, color=colors, edgecolor='black')
        ax1.set_xticks(range(5))
        ax1.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax1.set_ylabel('Monthly Return (%)')
        ax1.set_title('📈 Monthly Returns', fontweight='bold')
    
    ax2 = fig.add_subplot(5, 5, 2)
    if len(ret_df) > 100:
        data = ret_by_match['Annualized'].reset_index()
        ax2.bar(range(5), data['Annualized']*100, color='steelblue', edgecolor='black')
        ax2.set_xticks(range(5))
        ax2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax2.set_ylabel('Annual Return (%)')
        ax2.set_title('📈 Annualized Returns', fontweight='bold')
    
    # Row 1: Risk
    ax3 = fig.add_subplot(5, 5, 3)
    if len(vol_df) > 100:
        data = risk_by_match['Volatility'].reset_index()
        colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
        ax3.bar(range(5), data['Volatility']*100, color=colors, edgecolor='black')
        ax3.set_xticks(range(5))
        ax3.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax3.set_ylabel('Volatility (%)')
        ax3.set_title('📉 Volatility', fontweight='bold')
    
    ax4 = fig.add_subplot(5, 5, 4)
    if len(vol_df) > 100:
        data = risk_by_match['VaR 95%'].reset_index()
        ax4.bar(range(5), data['VaR 95%']*100, color='darkred', edgecolor='black')
        ax4.set_xticks(range(5))
        ax4.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax4.set_ylabel('95% VaR (%)')
        ax4.set_title('🎲 Value at Risk', fontweight='bold')
    
    ax5 = fig.add_subplot(5, 5, 5)
    if len(vol_df) > 100:
        data = sharpe_by_match['Sharpe Mean'].reset_index()
        ax5.bar(range(5), data['Sharpe Mean'], color='gold', edgecolor='black')
        ax5.set_xticks(range(5))
        ax5.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax5.set_ylabel('Sharpe Ratio')
        ax5.set_title('⚖️ Sharpe Ratio', fontweight='bold')
    
    # Row 2: Deciles
    ax6 = fig.add_subplot(5, 5, 6)
    if len(vol_df) > 100:
        dec_data = decile_analysis['Annual Ret'].reset_index()
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 10))
        ax6.bar(range(10), dec_data['Annual Ret']*100, color=colors, edgecolor='black')
        ax6.set_xticks(range(10))
        ax6.set_xticklabels([f'D{i}' for i in range(1, 11)], fontsize=8)
        ax6.set_ylabel('Annual Return (%)')
        ax6.set_title('🔟 Returns by Decile', fontweight='bold')
    
    ax7 = fig.add_subplot(5, 5, 7)
    if len(vol_df) > 100:
        dec_data = decile_analysis['Sharpe'].reset_index()
        ax7.bar(range(10), dec_data['Sharpe'], color='purple', edgecolor='black')
        ax7.set_xticks(range(10))
        ax7.set_xticklabels([f'D{i}' for i in range(1, 11)], fontsize=8)
        ax7.set_ylabel('Sharpe Ratio')
        ax7.set_title('🔟 Sharpe by Decile', fontweight='bold')
    
    # Row 2: Heatmaps
    ax8 = fig.add_subplot(5, 5, 8)
    try:
        hm_data = ret_heatmap.astype(float)*100
        sns.heatmap(hm_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax8)
        ax8.set_title('Returns: Match×Size', fontweight='bold')
    except: pass
    
    ax9 = fig.add_subplot(5, 5, 9)
    try:
        hm_data = vol_heatmap.astype(float)*100
        sns.heatmap(hm_data, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax9)
        ax9.set_title('Vol: Match×Size', fontweight='bold')
    except: pass
    
    ax10 = fig.add_subplot(5, 5, 10)
    try:
        hm_data = sharpe_heatmap.astype(float)
        sns.heatmap(hm_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax10)
        ax10.set_title('Sharpe: Match×Size', fontweight='bold')
    except: pass
    
    # Row 3: Scatter plots
    ax11 = fig.add_subplot(5, 5, 11)
    sample = vol_df.sample(min(2000, len(vol_df)))
    ax11.scatter(sample['match_means'], sample['avg_ret']*100, alpha=0.3, s=10)
    z = np.polyfit(sample['match_means'], sample['avg_ret']*100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(sample['match_means'].min(), sample['match_means'].max(), 100)
    ax11.plot(x_line, p(x_line), 'r-', lw=2, label=f'β={z[0]:.3f}')
    ax11.set_xlabel('Match Quality')
    ax11.set_ylabel('Monthly Return (%)')
    ax11.set_title('Match → Returns', fontweight='bold')
    ax11.legend()
    
    ax12 = fig.add_subplot(5, 5, 12)
    ax12.scatter(sample['match_means'], sample['volatility']*100, alpha=0.3, s=10, c='red')
    z = np.polyfit(sample['match_means'], sample['volatility']*100, 1)
    p = np.poly1d(z)
    ax12.plot(x_line, p(x_line), 'b-', lw=2, label=f'β={z[0]:.3f}')
    ax12.set_xlabel('Match Quality')
    ax12.set_ylabel('Volatility (%)')
    ax12.set_title('Match → Volatility', fontweight='bold')
    ax12.legend()
    
    ax13 = fig.add_subplot(5, 5, 13)
    ax13.scatter(sample['match_means'], sample['sharpe'], alpha=0.3, s=10, c='green')
    z = np.polyfit(sample['match_means'], sample['sharpe'], 1)
    p = np.poly1d(z)
    ax13.plot(x_line, p(x_line), 'r-', lw=2, label=f'β={z[0]:.3f}')
    ax13.set_xlabel('Match Quality')
    ax13.set_ylabel('Sharpe Ratio')
    ax13.set_title('Match → Sharpe', fontweight='bold')
    ax13.legend()
    
    # Row 3: Time Series
    ax14 = fig.add_subplot(5, 5, 14)
    ax14.plot(time_series.index, time_series['Ret']*100, 'b-o', lw=2)
    ax14.set_xlabel('Year')
    ax14.set_ylabel('Avg Monthly Return (%)')
    ax14.set_title('Returns Over Time', fontweight='bold')
    
    ax15 = fig.add_subplot(5, 5, 15)
    ax15.plot(time_series.index, time_series['Vol']*100, 'r-o', lw=2)
    ax15.set_xlabel('Year')
    ax15.set_ylabel('Avg Volatility (%)')
    ax15.set_title('Volatility Over Time', fontweight='bold')
    
    # Row 4: Distributions
    ax16 = fig.add_subplot(5, 5, 16)
    for q, color in zip(['Q1', 'Q3', 'Q5'], ['red', 'blue', 'green']):
        subset = vol_df[vol_df['match_q'] == q]['avg_ret'] * 100
        ax16.hist(subset, bins=30, alpha=0.5, label=q, color=color, density=True)
    ax16.set_xlabel('Monthly Return (%)')
    ax16.set_ylabel('Density')
    ax16.set_title('Return Distributions', fontweight='bold')
    ax16.legend()
    
    ax17 = fig.add_subplot(5, 5, 17)
    for q, color in zip(['Q1', 'Q3', 'Q5'], ['red', 'blue', 'green']):
        subset = vol_df[vol_df['match_q'] == q]['volatility'] * 100
        ax17.hist(subset, bins=30, alpha=0.5, label=q, color=color, density=True)
    ax17.set_xlabel('Volatility (%)')
    ax17.set_ylabel('Density')
    ax17.set_title('Volatility Distributions', fontweight='bold')
    ax17.legend()
    
    ax18 = fig.add_subplot(5, 5, 18)
    for q, color in zip(['Q1', 'Q3', 'Q5'], ['red', 'blue', 'green']):
        subset = vol_df[vol_df['match_q'] == q]['sharpe']
        ax18.hist(subset, bins=30, alpha=0.5, label=q, color=color, density=True)
    ax18.set_xlabel('Sharpe Ratio')
    ax18.set_ylabel('Density')
    ax18.set_title('Sharpe Distributions', fontweight='bold')
    ax18.legend()
    
    # Row 4: Tail Risk
    ax19 = fig.add_subplot(5, 5, 19)
    tail_data = tail_by_match[['Extreme Loss %', 'Extreme Gain %']]
    tail_data.plot(kind='bar', ax=ax19, color=['red', 'green'])
    ax19.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], rotation=0)
    ax19.set_ylabel('Probability (%)')
    ax19.set_title('🎲 Tail Events', fontweight='bold')
    ax19.legend(['Loss', 'Gain'], fontsize=8)
    
    ax20 = fig.add_subplot(5, 5, 20)
    ax20.bar(range(5), tail_by_match['Gain/Loss Ratio'], color='purple', edgecolor='black')
    ax20.set_xticks(range(5))
    ax20.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax20.axhline(y=1, color='black', linestyle='--')
    ax20.set_ylabel('Gain/Loss Ratio')
    ax20.set_title('Tail Ratio', fontweight='bold')
    
    # Row 5: Additional
    ax21 = fig.add_subplot(5, 5, 21)
    if 'compindustry' in vol_df.columns:
        top_ind = ind_analysis.head(8)
        ax21.barh(range(len(top_ind)), top_ind['Sharpe'], color='steelblue', edgecolor='black')
        ax21.set_yticks(range(len(top_ind)))
        ax21.set_yticklabels([i[:15] for i in top_ind.index], fontsize=7)
        ax21.set_xlabel('Sharpe Ratio')
        ax21.set_title('Top Industries', fontweight='bold')
        ax21.invert_yaxis()
    
    ax22 = fig.add_subplot(5, 5, 22)
    ax22.plot(corr_ts.index, corr_ts.values, 'g-o', lw=2)
    ax22.axhline(y=0, color='black', linestyle='--')
    ax22.set_xlabel('Year')
    ax22.set_ylabel('Correlation')
    ax22.set_title('Match-Return Corr', fontweight='bold')
    
    ax23 = fig.add_subplot(5, 5, 23)
    ax23.hist(mega_df['match_means'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax23.axvline(x=mega_df['match_means'].median(), color='red', linestyle='--', lw=2)
    ax23.set_xlabel('Match Quality')
    ax23.set_ylabel('Frequency')
    ax23.set_title('Match Distribution', fontweight='bold')
    
    ax24 = fig.add_subplot(5, 5, 24)
    if 'tobinw' in mega_df.columns:
        tobin_by_match = mega_df.groupby('match_q', observed=False)['tobinw'].mean()
        ax24.bar(range(5), tobin_by_match.values, color='orange', edgecolor='black')
        ax24.set_xticks(range(5))
        ax24.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax24.set_ylabel("Tobin's Q")
        ax24.set_title("Firm Value", fontweight='bold')
    
    ax25 = fig.add_subplot(5, 5, 25)
    # Long-short cumulative
    years = range(1, 11)
    cumulative = [(1 + ls_annual) ** y for y in years]
    ax25.plot(years, cumulative, 'g-o', lw=2, markersize=8)
    ax25.axhline(y=1, color='black', linestyle='--')
    ax25.fill_between(years, 1, cumulative, alpha=0.3, color='green')
    ax25.set_xlabel('Years')
    ax25.set_ylabel('Cumulative Return')
    ax25.set_title('💼 L/S Portfolio', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Output/crsp_deep_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: Output/crsp_deep_analysis.png")
    
    # ================================================================
    # EXECUTIVE SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("🏆 CRSP ANALYSIS EXECUTIVE SUMMARY")
    print("=" * 70)
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                     📈📉 CRSP DEEP DIVE FINDINGS 📈📉                             ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                    ║
║  📈 RETURNS                                                                       ║
║     Q5 Monthly: {ret_by_match.loc['Q5', 'Mean Return']*100:.2f}% | Q1 Monthly: {ret_by_match.loc['Q1', 'Mean Return']*100:.2f}%                      ║
║     Q5-Q1 Annual Spread: +{spread:.1f}%                                           ║
║     T-stat (Q5 vs Q1): {t_stat:.2f} (p={p_value:.4f})                                      ║
║                                                                                    ║
║  📉 VOLATILITY                                                                    ║
║     Q4: {risk_by_match.loc['Q4', 'Volatility']*100:.1f}% | Q1: {risk_by_match.loc['Q1', 'Volatility']*100:.1f}%                                      ║
║     Q4 Reduction vs Q1: -{vol_reduction:.0f}%                                          ║
║                                                                                    ║
║  ⚖️ RISK-ADJUSTED (SHARPE)                                                       ║
║     Q5: {sharpe_by_match.loc['Q5', 'Sharpe Mean']:.3f} | Q1: {sharpe_by_match.loc['Q1', 'Sharpe Mean']:.3f}                                          ║
║     Q5 Improvement: +{sharpe_improvement:.0f}%                                          ║
║                                                                                    ║
║  💼 LONG-SHORT PORTFOLIO                                                         ║
║     Annual Return: +{ls_annual*100:.1f}%                                               ║
║     Sharpe Ratio: {ls_sharpe:.2f}                                                     ║
║     10-Year Cumulative: +{cumulative_10yr*100:.0f}%                                     ║
║                                                                                    ║
║  🎲 TAIL RISK                                                                     ║
║     Q5 Gain/Loss Ratio: {tail_by_match.loc['Q5', 'Gain/Loss Ratio']:.2f}                                          ║
║     Q1 Gain/Loss Ratio: {tail_by_match.loc['Q1', 'Gain/Loss Ratio']:.2f}                                          ║
║                                                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n📈📉📈📉 CRSP DEEP ANALYSIS COMPLETE! 📈📉📈📉")

if __name__ == "__main__":
    main()
