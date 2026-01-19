#!/usr/bin/env python3
"""
🔌🔌🔌 FULL WRDS INTEGRATION ANALYSIS 🔌🔌🔌

Leverages ALL pulled WRDS datasets:
1. IBES - Analyst coverage & forecast accuracy
2. 13F - Institutional ownership % and # holders
3. ISS - Governance provisions interaction
4. CRSP Volatility - Risk metrics
5. Compustat Quarterly - Growth rates
6. Link Tables - For merging everything
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import seaborn as sns
from scipy import stats
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
    print("🔌" * 40)
    print("   FULL WRDS INTEGRATION ANALYSIS")
    print("🔌" * 40)
    
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
    # LOAD ALL WRDS PARQUET FILES
    # ================================================================
    print("\n" + "=" * 70)
    print("📂 LOADING SAVED WRDS DATA")
    print("=" * 70)
    
    wrds = {}
    
    # Load each saved dataset
    try:
        wrds['ibes'] = pd.read_parquet('Output/wrds_ibes.parquet')
        print(f"  ✅ IBES Analysts: {len(wrds['ibes']):,} observations")
    except: print("  ❌ IBES not found")
    
    try:
        wrds['holdings'] = pd.read_parquet('Output/wrds_holdings.parquet')
        print(f"  ✅ 13F Holdings: {len(wrds['holdings']):,} observations")
    except: print("  ❌ Holdings not found")
    
    try:
        wrds['governance'] = pd.read_parquet('Output/wrds_governance.parquet')
        print(f"  ✅ ISS Governance: {len(wrds['governance']):,} observations")
    except: print("  ❌ Governance not found")
    
    try:
        wrds['volatility'] = pd.read_parquet('Output/wrds_volatility.parquet')
        print(f"  ✅ CRSP Volatility: {len(wrds['volatility']):,} observations")
    except: print("  ❌ Volatility not found")
    
    try:
        wrds['quarterly'] = pd.read_parquet('Output/wrds_quarterly.parquet')
        print(f"  ✅ Compustat Quarterly: {len(wrds['quarterly']):,} observations")
    except: print("  ❌ Quarterly not found")
    
    try:
        wrds['crsp_comp'] = pd.read_parquet('Output/wrds_crsp_comp.parquet')
        print(f"  ✅ CRSP-Comp Link: {len(wrds['crsp_comp']):,} observations")
    except: print("  ❌ CRSP-Comp link not found")
    
    try:
        wrds['ticker_cusip'] = pd.read_parquet('Output/wrds_ticker_cusip.parquet')
        print(f"  ✅ Ticker-CUSIP Link: {len(wrds['ticker_cusip']):,} observations")
    except: print("  ❌ Ticker-CUSIP link not found")
    
    # ================================================================
    # MEGA MERGE
    # ================================================================
    print("\n" + "=" * 70)
    print("🔗 BUILDING MEGA MERGED DATASET")
    print("=" * 70)
    
    mega_df = df_clean.copy()
    
    # Load local compensation data
    exec_comp = pd.read_parquet('/Users/smaric/Papers/CEO NCA/data/raw/execucomp_ceo.parquet')
    exec_comp['gvkey'] = pd.to_numeric(exec_comp['gvkey'], errors='coerce')
    mega_df = mega_df.merge(exec_comp, on=['gvkey', 'fiscalyear'], how='left')
    print(f"  + Compensation: {mega_df['tdc1'].notna().sum():,}")
    
    # Load BLM data
    blm = pd.read_stata('../Data/blm_data_ceo_prep_v4.3_mobility.dta')
    mega_df = mega_df.merge(
        blm[['gvkey', 'year', 'tobinw', 'roaw']].drop_duplicates(),
        left_on=['gvkey', 'fiscalyear'], right_on=['gvkey', 'year'], how='left'
    )
    print(f"  + BLM (Tobin Q): {mega_df['tobinw'].notna().sum():,}")
    
    # Merge CRSP-Compustat link
    if 'crsp_comp' in wrds:
        link = wrds['crsp_comp']
        link['gvkey'] = pd.to_numeric(link['gvkey'], errors='coerce')
        mega_df = mega_df.merge(link[['gvkey', 'permno']].drop_duplicates(), on='gvkey', how='left')
        print(f"  + PERMNO: {mega_df['permno'].notna().sum():,}")
    
    # Merge Volatility
    if 'volatility' in wrds and 'permno' in mega_df.columns:
        vol = wrds['volatility']
        mega_df = mega_df.merge(
            vol[['permno', 'year', 'volatility', 'avg_ret', 'return_range']],
            left_on=['permno', 'fiscalyear'], right_on=['permno', 'year'],
            how='left', suffixes=('', '_vol')
        )
        print(f"  + Volatility: {mega_df['volatility'].notna().sum():,}")
    
    # Merge IBES via ticker
    if 'ibes' in wrds and 'tic' in mega_df.columns:
        ibes = wrds['ibes']
        mega_df = mega_df.merge(
            ibes[['ticker', 'year', 'n_estimates', 'forecast_error', 'forecast_dispersion']],
            left_on=['tic', 'fiscalyear'], right_on=['ticker', 'year'],
            how='left', suffixes=('', '_ibes')
        )
        print(f"  + IBES Coverage: {mega_df['n_estimates'].notna().sum():,}")
    
    # Merge 13F Holdings via CUSIP
    if 'holdings' in wrds and 'ticker_cusip' in wrds and 'permno' in mega_df.columns:
        link2 = wrds['ticker_cusip']
        holdings = wrds['holdings']
        
        # Link PERMNO -> CUSIP
        mega_df = mega_df.merge(link2[['permno', 'cusip']].drop_duplicates(), on='permno', how='left')
        
        # Merge holdings via 8-char CUSIP
        if 'cusip' in mega_df.columns:
            mega_df['cusip8'] = mega_df['cusip'].astype(str).str[:8]
            holdings['cusip8'] = holdings['cusip'].astype(str).str[:8]
            mega_df = mega_df.merge(
                holdings[['cusip8', 'year', 'n_institutions', 'total_shares']],
                left_on=['cusip8', 'fiscalyear'], right_on=['cusip8', 'year'],
                how='left', suffixes=('', '_inst')
            )
            print(f"  + Inst. Holdings: {mega_df['n_institutions'].notna().sum():,}")
    
    # Merge ISS Governance via ticker
    if 'governance' in wrds and 'tic' in mega_df.columns:
        gov = wrds['governance']
        mega_df = mega_df.merge(
            gov,
            left_on=['tic', 'fiscalyear'], right_on=['ticker', 'year'],
            how='left', suffixes=('', '_gov')
        )
        print(f"  + Governance: {mega_df['classified_board'].notna().sum():,}")
    
    # Merge Quarterly Fundamentals
    if 'quarterly' in wrds:
        quarterly = wrds['quarterly']
        quarterly['gvkey'] = pd.to_numeric(quarterly['gvkey'], errors='coerce')
        annual = quarterly.groupby(['gvkey', 'year']).agg({
            'sales': 'sum',
            'income': 'sum',
            'sales_growth': 'mean',
            'income_growth': 'mean'
        }).reset_index()
        mega_df = mega_df.merge(
            annual,
            left_on=['gvkey', 'fiscalyear'], right_on=['gvkey', 'year'],
            how='left', suffixes=('', '_q')
        )
        print(f"  + Quarterly Growth: {mega_df['sales_growth'].notna().sum():,}")
    
    print(f"\n  📦 FINAL MEGA DATASET: {len(mega_df):,} observations")
    
    # Create quintiles
    mega_df['match_q'] = pd.qcut(mega_df['match_means'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    # ================================================================
    # COMPREHENSIVE ANALYSES
    # ================================================================
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE WRDS ANALYSES")
    print("=" * 70)
    
    results = {}
    
    # Analysis 1: Analyst Coverage by Match
    if 'n_estimates' in mega_df.columns:
        ibes_df = mega_df.dropna(subset=['match_means', 'n_estimates'])
        if len(ibes_df) > 100:
            ibes_by_match = ibes_df.groupby('match_q', observed=False).agg({
                'n_estimates': 'mean',
                'forecast_error': 'mean',
                'forecast_dispersion': 'mean',
                'gvkey': 'count'
            }).round(3)
            ibes_by_match.columns = ['Analyst Count', 'Forecast Error', 'Dispersion', 'N']
            results['ibes'] = ibes_by_match
            
            print("\n--- 📊 ANALYST COVERAGE BY MATCH QUALITY ---")
            print(ibes_by_match)
            
            # Regression
            model_ibes = ols('n_estimates ~ match_means + logatw', data=ibes_df).fit()
            print(f"\n  Coverage ~ Match β: {model_ibes.params['match_means']:.3f} (p={model_ibes.pvalues['match_means']:.4f})")
    
    # Analysis 2: Institutional Ownership by Match
    if 'n_institutions' in mega_df.columns:
        inst_df = mega_df.dropna(subset=['match_means', 'n_institutions'])
        if len(inst_df) > 100:
            inst_by_match = inst_df.groupby('match_q', observed=False).agg({
                'n_institutions': 'mean',
                'total_shares': 'mean',
                'gvkey': 'count'
            }).round(0)
            inst_by_match.columns = ['# Institutions', 'Total Shares (M)', 'N']
            inst_by_match['Total Shares (M)'] = inst_by_match['Total Shares (M)'] / 1e6
            results['inst'] = inst_by_match
            
            print("\n--- 🏦 INSTITUTIONAL OWNERSHIP BY MATCH QUALITY ---")
            print(inst_by_match.round(1))
            
            # Regression
            model_inst = ols('n_institutions ~ match_means + logatw', data=inst_df).fit()
            print(f"\n  Inst. Count ~ Match β: {model_inst.params['match_means']:.3f} (p={model_inst.pvalues['match_means']:.4f})")
    
    # Analysis 3: Governance by Match
    if 'classified_board' in mega_df.columns:
        gov_df = mega_df.dropna(subset=['match_means', 'classified_board'])
        if len(gov_df) > 50:
            # Convert to numeric
            gov_df['classified_board'] = pd.to_numeric(gov_df['classified_board'], errors='coerce')
            gov_df['poison_pill'] = pd.to_numeric(gov_df.get('poison_pill', 0), errors='coerce')
            gov_df['dual_class'] = pd.to_numeric(gov_df.get('dual_class', 0), errors='coerce')
            
            gov_by_match = gov_df.groupby('match_q', observed=False).agg({
                'classified_board': 'mean',
                'poison_pill': 'mean',
                'dual_class': 'mean',
                'gvkey': 'count'
            }).round(3)
            gov_by_match.columns = ['Classified Board %', 'Poison Pill %', 'Dual Class %', 'N']
            results['gov'] = gov_by_match
            
            print("\n--- 🏛️ GOVERNANCE PROVISIONS BY MATCH QUALITY ---")
            print(gov_by_match)
    
    # Analysis 4: Volatility by Match (with Sharpe ratio)
    if 'volatility' in mega_df.columns:
        vol_df = mega_df.dropna(subset=['match_means', 'volatility', 'avg_ret'])
        if len(vol_df) > 100:
            vol_df['sharpe'] = vol_df['avg_ret'] / vol_df['volatility']
            
            vol_by_match = vol_df.groupby('match_q', observed=False).agg({
                'volatility': 'mean',
                'avg_ret': 'mean',
                'sharpe': 'mean',
                'return_range': 'mean',
                'gvkey': 'count'
            }).round(4)
            vol_by_match.columns = ['Volatility', 'Avg Return', 'Sharpe', 'Return Range', 'N']
            results['vol'] = vol_by_match
            
            print("\n--- 📉 STOCK VOLATILITY & RISK-ADJUSTED RETURNS ---")
            print(vol_by_match)
    
    # Analysis 5: Quarterly Growth by Match
    if 'sales_growth' in mega_df.columns:
        growth_df = mega_df.dropna(subset=['match_means', 'sales_growth'])
        growth_df = growth_df[growth_df['sales_growth'].between(-1, 2)]
        if len(growth_df) > 100:
            growth_by_match = growth_df.groupby('match_q', observed=False).agg({
                'sales_growth': 'mean',
                'sales': 'mean',
                'income': 'mean',
                'gvkey': 'count'
            }).round(4)
            growth_by_match.columns = ['Sales Growth', 'Avg Sales', 'Avg Income', 'N']
            results['growth'] = growth_by_match
            
            print("\n--- 📈 QUARTERLY GROWTH BY MATCH QUALITY ---")
            print(growth_by_match)
    
    # Analysis 6: Compensation by Match
    comp_df = mega_df.dropna(subset=['match_means', 'tdc1'])
    comp_df = comp_df[comp_df['tdc1'] > 0]
    if len(comp_df) > 100:
        comp_df['equity_ratio'] = (comp_df['stock_awards_fv'].fillna(0) + 
                                   comp_df['option_awards_fv'].fillna(0)) / comp_df['tdc1']
        
        comp_by_match = comp_df.groupby('match_q', observed=False).agg({
            'tdc1': 'mean',
            'equity_ratio': 'mean',
            'salary': 'mean',
            'bonus': 'mean',
            'stock_awards_fv': 'mean',
            'gvkey': 'count'
        }).round(0)
        comp_by_match.columns = ['TDC1', 'Equity%', 'Salary', 'Bonus', 'Stock Awards', 'N']
        results['comp'] = comp_by_match
        
        print("\n--- 💰 CEO COMPENSATION BY MATCH QUALITY ---")
        print(comp_by_match)
    
    # ================================================================
    # CROSS-VARIABLE INTERACTIONS
    # ================================================================
    print("\n" + "=" * 70)
    print("🔬 CROSS-VARIABLE INTERACTION ANALYSIS")
    print("=" * 70)
    
    # Match × Inst Ownership → Returns
    if 'n_institutions' in mega_df.columns and 'avg_ret' in mega_df.columns:
        interact_df = mega_df.dropna(subset=['match_means', 'n_institutions', 'avg_ret'])
        if len(interact_df) > 500:
            interact_df['match_high'] = (interact_df['match_means'] > interact_df['match_means'].median()).astype(int)
            interact_df['inst_high'] = (interact_df['n_institutions'] > interact_df['n_institutions'].median()).astype(int)
            
            interaction = interact_df.groupby(['match_high', 'inst_high'])['avg_ret'].mean().unstack()
            interaction.index = ['Low Match', 'High Match']
            interaction.columns = ['Low Inst', 'High Inst']
            
            print("\n--- 📊 Returns: Match × Institutional Ownership ---")
            print((interaction * 100).round(2))
            print("   (Monthly returns in %)")
    
    # Match × Governance → Tobin Q
    if 'classified_board' in mega_df.columns and 'tobinw' in mega_df.columns:
        gov_interact = mega_df.dropna(subset=['match_means', 'classified_board', 'tobinw'])
        if len(gov_interact) > 100:
            gov_interact['match_high'] = (gov_interact['match_means'] > gov_interact['match_means'].median()).astype(int)
            gov_interact['classified_board'] = pd.to_numeric(gov_interact['classified_board'], errors='coerce')
            
            gov_tobin = gov_interact.groupby(['match_high', 'classified_board'])['tobinw'].mean().unstack()
            if not gov_tobin.empty:
                gov_tobin.index = ['Low Match', 'High Match']
                
                print("\n--- 🏛️ Tobin's Q: Match × Classified Board ---")
                print(gov_tobin.round(2))
    
    # Match × Analyst Coverage → Volatility
    if 'n_estimates' in mega_df.columns and 'volatility' in mega_df.columns:
        analyst_vol = mega_df.dropna(subset=['match_means', 'n_estimates', 'volatility'])
        if len(analyst_vol) > 200:
            analyst_vol['match_high'] = (analyst_vol['match_means'] > analyst_vol['match_means'].median()).astype(int)
            analyst_vol['coverage_high'] = (analyst_vol['n_estimates'] > analyst_vol['n_estimates'].median()).astype(int)
            
            cov_vol = analyst_vol.groupby(['match_high', 'coverage_high'])['volatility'].mean().unstack()
            cov_vol.index = ['Low Match', 'High Match']
            cov_vol.columns = ['Low Coverage', 'High Coverage']
            
            print("\n--- 📊 Volatility: Match × Analyst Coverage ---")
            print((cov_vol * 100).round(2))
            print("   (Monthly volatility in %)")
    
    # ================================================================
    # MEGA VISUALIZATION
    # ================================================================
    print("\n" + "=" * 70)
    print("📊 CREATING COMPREHENSIVE VISUALIZATIONS")
    print("=" * 70)
    
    fig = plt.figure(figsize=(28, 24))
    
    # Row 1: Core WRDS metrics
    # 1. Analyst Coverage
    ax1 = fig.add_subplot(5, 5, 1)
    if 'ibes' in results:
        data = results['ibes']['Analyst Count'].reset_index()
        ax1.bar(range(5), data['Analyst Count'], color='steelblue', edgecolor='black')
        ax1.set_xticks(range(5))
        ax1.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax1.set_ylabel('# Analysts')
        ax1.set_title('📊 Analyst Coverage', fontweight='bold')
    
    # 2. Forecast Error
    ax2 = fig.add_subplot(5, 5, 2)
    if 'ibes' in results:
        data = results['ibes']['Forecast Error'].reset_index()
        colors = ['red' if x > data['Forecast Error'].median() else 'green' for x in data['Forecast Error']]
        ax2.bar(range(5), data['Forecast Error'], color=colors, edgecolor='black')
        ax2.set_xticks(range(5))
        ax2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax2.set_ylabel('Forecast Error')
        ax2.set_title('📉 Analyst Accuracy', fontweight='bold')
    
    # 3. Institutional Ownership
    ax3 = fig.add_subplot(5, 5, 3)
    if 'inst' in results:
        data = results['inst']['# Institutions'].reset_index()
        ax3.bar(range(5), data['# Institutions'], color='purple', edgecolor='black')
        ax3.set_xticks(range(5))
        ax3.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax3.set_ylabel('# Institutions')
        ax3.set_title('🏦 Institutional Holders', fontweight='bold')
    
    # 4. Volatility
    ax4 = fig.add_subplot(5, 5, 4)
    if 'vol' in results:
        data = results['vol']['Volatility'].reset_index()
        colors = ['#e74c3c' if i < 2 else '#f39c12' if i < 4 else '#2ecc71' for i in range(5)]
        ax4.bar(range(5), data['Volatility']*100, color=colors, edgecolor='black')
        ax4.set_xticks(range(5))
        ax4.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax4.set_ylabel('Volatility (%)')
        ax4.set_title('📉 Stock Volatility', fontweight='bold')
    
    # 5. Sharpe Ratio
    ax5 = fig.add_subplot(5, 5, 5)
    if 'vol' in results:
        data = results['vol']['Sharpe'].reset_index()
        ax5.bar(range(5), data['Sharpe'], color='gold', edgecolor='black')
        ax5.set_xticks(range(5))
        ax5.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax5.set_ylabel('Sharpe Ratio')
        ax5.set_title('⚖️ Risk-Adj Returns', fontweight='bold')
    
    # Row 2: Growth and Performance
    # 6. Sales Growth
    ax6 = fig.add_subplot(5, 5, 6)
    if 'growth' in results:
        data = results['growth']['Sales Growth'].reset_index()
        colors = ['green' if x > 0 else 'red' for x in data['Sales Growth']]
        ax6.bar(range(5), data['Sales Growth']*100, color=colors, edgecolor='black')
        ax6.set_xticks(range(5))
        ax6.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax6.set_ylabel('YoY Growth (%)')
        ax6.set_title('📈 Sales Growth', fontweight='bold')
        ax6.axhline(y=0, color='black', lw=1)
    
    # 7. Returns
    ax7 = fig.add_subplot(5, 5, 7)
    if 'vol' in results:
        data = results['vol']['Avg Return'].reset_index()
        ax7.bar(range(5), data['Avg Return']*100, color='steelblue', edgecolor='black')
        ax7.set_xticks(range(5))
        ax7.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax7.set_ylabel('Monthly Return (%)')
        ax7.set_title('📈 Stock Returns', fontweight='bold')
    
    # 8. Compensation
    ax8 = fig.add_subplot(5, 5, 8)
    if 'comp' in results:
        data = results['comp']['TDC1'].reset_index()
        ax8.bar(range(5), data['TDC1']/1000, color='green', edgecolor='black')
        ax8.set_xticks(range(5))
        ax8.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax8.set_ylabel('Total Comp ($M)')
        ax8.set_title('💰 CEO Compensation', fontweight='bold')
    
    # 9. Tobin Q
    ax9 = fig.add_subplot(5, 5, 9)
    tobin_df = mega_df.dropna(subset=['match_means', 'tobinw'])
    if len(tobin_df) > 100:
        tobin_by_match = tobin_df.groupby('match_q', observed=False)['tobinw'].mean()
        ax9.bar(range(5), tobin_by_match.values, color='orange', edgecolor='black')
        ax9.set_xticks(range(5))
        ax9.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax9.set_ylabel("Tobin's Q")
        ax9.set_title("🏛️ Firm Value", fontweight='bold')
    
    # 10. Compensation Stacked
    ax10 = fig.add_subplot(5, 5, 10)
    if 'comp' in results:
        comp_stack = results['comp'][['Salary', 'Bonus', 'Stock Awards']]
        comp_stack.plot(kind='bar', stacked=True, ax=ax10, 
                        color=['#3498db', '#e74c3c', '#2ecc71'])
        ax10.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], rotation=0)
        ax10.set_ylabel('Compensation ($K)')
        ax10.set_title('💵 Comp Structure', fontweight='bold')
        ax10.legend(fontsize=7)
    
    # Row 3: Heatmaps
    # 11. Match × Size → Comp
    ax11 = fig.add_subplot(5, 5, 11)
    try:
        size_bins = pd.qcut(mega_df['logatw'], 4, labels=['Small', 'Mid', 'Large', 'Giant'])
        match_bins = pd.qcut(mega_df['match_means'], 4, labels=['Low', 'Mid', 'High', 'Top'])
        heatmap = mega_df.groupby([match_bins, size_bins], observed=False)['tdc1'].mean().unstack()
        heatmap = heatmap.astype(float) / 1000
        sns.heatmap(heatmap, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax11)
        ax11.set_title('Pay ($M): Match×Size', fontweight='bold')
    except: pass
    
    # 12. Match × Size → Volatility
    ax12 = fig.add_subplot(5, 5, 12)
    if 'volatility' in mega_df.columns:
        try:
            vol_heat = mega_df.dropna(subset=['volatility']).groupby([match_bins, size_bins], observed=False)['volatility'].mean().unstack()
            vol_heat = vol_heat.astype(float) * 100
            sns.heatmap(vol_heat, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax12)
            ax12.set_title('Vol (%): Match×Size', fontweight='bold')
        except: pass
    
    # 13. Match × Size → Returns
    ax13 = fig.add_subplot(5, 5, 13)
    if 'avg_ret' in mega_df.columns:
        try:
            ret_heat = mega_df.dropna(subset=['avg_ret']).groupby([match_bins, size_bins], observed=False)['avg_ret'].mean().unstack()
            ret_heat = ret_heat.astype(float) * 100
            sns.heatmap(ret_heat, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax13)
            ax13.set_title('Ret (%): Match×Size', fontweight='bold')
        except: pass
    
    # 14. Match × Size → Tobin Q
    ax14 = fig.add_subplot(5, 5, 14)
    try:
        q_heat = mega_df.dropna(subset=['tobinw']).groupby([match_bins, size_bins], observed=False)['tobinw'].mean().unstack()
        q_heat = q_heat.astype(float)
        sns.heatmap(q_heat, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax14)
        ax14.set_title("Tobin Q: Match×Size", fontweight='bold')
    except: pass
    
    # 15. Match × Size → Growth
    ax15 = fig.add_subplot(5, 5, 15)
    if 'sales_growth' in mega_df.columns:
        try:
            g_df = mega_df[mega_df['sales_growth'].between(-1, 2)]
            g_heat = g_df.groupby([match_bins, size_bins], observed=False)['sales_growth'].mean().unstack()
            g_heat = g_heat.astype(float) * 100
            sns.heatmap(g_heat, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax15)
            ax15.set_title('Growth (%): Match×Size', fontweight='bold')
        except: pass
    
    # Row 4: Scatter plots and distributions
    # 16. Match vs Volatility Scatter
    ax16 = fig.add_subplot(5, 5, 16)
    if 'volatility' in mega_df.columns:
        vol_scatter = mega_df.dropna(subset=['match_means', 'volatility']).sample(min(2000, len(mega_df)))
        ax16.scatter(vol_scatter['match_means'], vol_scatter['volatility']*100, alpha=0.3, s=10)
        z = np.polyfit(vol_scatter['match_means'], vol_scatter['volatility']*100, 1)
        p = np.poly1d(z)
        x_line = np.linspace(vol_scatter['match_means'].min(), vol_scatter['match_means'].max(), 100)
        ax16.plot(x_line, p(x_line), 'r-', lw=2, label=f'β={z[0]:.2f}')
        ax16.set_xlabel('Match Quality')
        ax16.set_ylabel('Volatility (%)')
        ax16.set_title('Match → Volatility', fontweight='bold')
        ax16.legend()
    
    # 17. Match vs Returns Scatter
    ax17 = fig.add_subplot(5, 5, 17)
    if 'avg_ret' in mega_df.columns:
        ret_scatter = mega_df.dropna(subset=['match_means', 'avg_ret']).sample(min(2000, len(mega_df)))
        ax17.scatter(ret_scatter['match_means'], ret_scatter['avg_ret']*100, alpha=0.3, s=10)
        z = np.polyfit(ret_scatter['match_means'], ret_scatter['avg_ret']*100, 1)
        p = np.poly1d(z)
        x_line = np.linspace(ret_scatter['match_means'].min(), ret_scatter['match_means'].max(), 100)
        ax17.plot(x_line, p(x_line), 'r-', lw=2, label=f'β={z[0]:.2f}')
        ax17.set_xlabel('Match Quality')
        ax17.set_ylabel('Monthly Return (%)')
        ax17.set_title('Match → Returns', fontweight='bold')
        ax17.legend()
    
    # 18. Match vs Tobin Q Scatter
    ax18 = fig.add_subplot(5, 5, 18)
    valid = mega_df.dropna(subset=['match_means', 'tobinw'])
    if len(valid) > 100:
        sample = valid.sample(min(2000, len(valid)))
        ax18.scatter(sample['match_means'], sample['tobinw'], alpha=0.3, s=10)
        z = np.polyfit(sample['match_means'], sample['tobinw'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(sample['match_means'].min(), sample['match_means'].max(), 100)
        ax18.plot(x_line, p(x_line), 'r-', lw=2, label=f'β={z[0]:.2f}')
        ax18.set_xlabel('Match Quality')
        ax18.set_ylabel("Tobin's Q")
        ax18.set_title('Match → Firm Value', fontweight='bold')
        ax18.legend()
    
    # 19. Match Distribution
    ax19 = fig.add_subplot(5, 5, 19)
    ax19.hist(mega_df['match_means'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax19.axvline(x=mega_df['match_means'].median(), color='red', linestyle='--', lw=2)
    ax19.axvline(x=mega_df['match_means'].quantile(0.9), color='gold', linestyle='--', lw=2)
    ax19.set_xlabel('Match Quality')
    ax19.set_ylabel('Frequency')
    ax19.set_title('Match Distribution', fontweight='bold')
    
    # 20. Volatility Distribution by Match
    ax20 = fig.add_subplot(5, 5, 20)
    if 'volatility' in mega_df.columns:
        for q, color in zip(['Q1', 'Q3', 'Q5'], ['red', 'blue', 'green']):
            subset = mega_df[mega_df['match_q'] == q]['volatility'].dropna() * 100
            if len(subset) > 50:
                ax20.hist(subset, bins=30, alpha=0.5, label=q, color=color)
        ax20.set_xlabel('Volatility (%)')
        ax20.set_ylabel('Frequency')
        ax20.set_title('Volatility by Match', fontweight='bold')
        ax20.legend()
    
    # Row 5: Rankings and time series
    # 21. Top Industries
    ax21 = fig.add_subplot(5, 5, 21)
    ind_stats = mega_df.groupby('compindustry')['match_means'].agg(['mean', 'count'])
    ind_stats = ind_stats[ind_stats['count'] >= 50].sort_values('mean', ascending=False).head(10)
    colors = ['green' if x > 0 else 'red' for x in ind_stats['mean']]
    ax21.barh(range(len(ind_stats)), ind_stats['mean'], color=colors, edgecolor='black')
    ax21.set_yticks(range(len(ind_stats)))
    ax21.set_yticklabels([i[:15] for i in ind_stats.index], fontsize=7)
    ax21.axvline(x=0, color='black')
    ax21.set_xlabel('Avg Match')
    ax21.set_title('Top Industries', fontweight='bold')
    ax21.invert_yaxis()
    
    # 22. Top States
    ax22 = fig.add_subplot(5, 5, 22)
    state_stats = mega_df.groupby('ba_state')['match_means'].agg(['mean', 'count'])
    state_stats = state_stats[state_stats['count'] >= 30].sort_values('mean', ascending=False).head(10)
    colors = ['green' if x > 0 else 'red' for x in state_stats['mean']]
    ax22.barh(range(len(state_stats)), state_stats['mean'], color=colors, edgecolor='black')
    ax22.set_yticks(range(len(state_stats)))
    ax22.set_yticklabels(state_stats.index, fontsize=8)
    ax22.axvline(x=0, color='black')
    ax22.set_xlabel('Avg Match')
    ax22.set_title('Top States', fontweight='bold')
    ax22.invert_yaxis()
    
    # 23. Time Trend - Match
    ax23 = fig.add_subplot(5, 5, 23)
    time_trend = mega_df.groupby('fiscalyear').agg({
        'match_means': 'mean',
        'tdc1': 'mean',
        'tobinw': 'mean'
    })
    ax23.plot(time_trend.index, time_trend['match_means'], 'b-o', lw=2)
    ax23.set_xlabel('Year')
    ax23.set_ylabel('Avg Match Quality')
    ax23.set_title('Match Over Time', fontweight='bold')
    
    # 24. Time Trend - Volatility
    ax24 = fig.add_subplot(5, 5, 24)
    if 'volatility' in mega_df.columns:
        vol_trend = mega_df.groupby('fiscalyear')['volatility'].mean() * 100
        ax24.plot(vol_trend.index, vol_trend.values, 'r-o', lw=2)
        ax24.set_xlabel('Year')
        ax24.set_ylabel('Avg Volatility (%)')
        ax24.set_title('Volatility Over Time', fontweight='bold')
    
    # 25. Feature Importance
    ax25 = fig.add_subplot(5, 5, 25)
    features = ['Age', 'tenure', 'ivy', 'logatw', 'rdintw', 'exp_roa', 'leverage', 'boardindpw']
    ml_df = mega_df[features + ['match_means']].dropna()
    if len(ml_df) > 500:
        rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
        rf.fit(ml_df[features], ml_df['match_means'])
        imp = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
        imp = imp.sort_values('Importance', ascending=True)
        ax25.barh(range(len(imp)), imp['Importance'], color='steelblue', edgecolor='black')
        ax25.set_yticks(range(len(imp)))
        ax25.set_yticklabels(imp['Feature'], fontsize=8)
        ax25.set_xlabel('Importance')
        ax25.set_title('🎯 Match Predictors', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Output/wrds_full_integration.png', dpi=150, bbox_inches='tight')
    print("Saved: Output/wrds_full_integration.png")
    
    # ================================================================
    # EXECUTIVE SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("🏆 FULL WRDS INTEGRATION SUMMARY")
    print("=" * 70)
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                   🔌🔌🔌 FULL WRDS INTEGRATION FINDINGS 🔌🔌🔌                     ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                    ║
║  📊 DATA INTEGRATED:                                                              ║
║     • IBES Analysts, 13F Holdings, ISS Governance                                ║
║     • CRSP Volatility, Compustat Quarterly, Link Tables                          ║
║                                                                                    ║
║  📉 VOLATILITY: Q4 is {(results['vol'].loc['Q1', 'Volatility'] - results['vol'].loc['Q4', 'Volatility'])/results['vol'].loc['Q1', 'Volatility']*100:.0f}% LOWER than Q1                                  ║
║  📈 RETURNS: Q5 earns {(results['vol'].loc['Q5', 'Avg Return'] - results['vol'].loc['Q1', 'Avg Return'])*1200:.1f}%/yr MORE than Q1                               ║
║  💰 COMPENSATION: Q4 gets ${results['comp'].loc['Q4', 'TDC1']/1000:.1f}M vs Q1 ${results['comp'].loc['Q1', 'TDC1']/1000:.1f}M                      ║
║                                                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n🔌🔌🔌 FULL WRDS INTEGRATION COMPLETE! 🔌🔌🔌")

if __name__ == "__main__":
    main()
