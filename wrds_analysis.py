#!/usr/bin/env python3
"""
🔌 WRDS DATA PULL & ANALYSIS 🔌

Pulls fresh data from WRDS databases for CEO-Firm Matching analysis:
1. IBES (Analyst Forecasts)
2. Thomson 13F (Institutional Holdings)
3. ISS Governance
4. SDC M&A Activity
5. CRSP Volatility
6. ExecuComp Incentives
"""
import pandas as pd
import numpy as np
import wrds
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

def connect_wrds():
    """Connect to WRDS with stored credentials."""
    try:
        # Use environment variable or pgpass file for password
        db = wrds.Connection(wrds_username='maricste93')
        print("✅ WRDS Connection Successful!")
        return db
    except Exception as e:
        print(f"❌ WRDS Connection Failed: {e}")
        return None

def pull_ibes_coverage(db):
    """Pull analyst coverage from IBES."""
    print("\n📊 Pulling IBES Analyst Coverage...")
    query = """
    SELECT a.ticker, 
           EXTRACT(YEAR FROM a.statpers) as year,
           COUNT(DISTINCT a.analys) as n_analysts,
           AVG(a.actual) as actual_eps,
           AVG(a.value) as forecast_eps
    FROM ibes.statsum_epsus a
    WHERE a.measure = 'EPS' 
      AND a.fpi = '1'
      AND a.statpers >= '2006-01-01'
    GROUP BY a.ticker, EXTRACT(YEAR FROM a.statpers)
    """
    try:
        df = db.raw_sql(query)
        df['forecast_error'] = abs(df['actual_eps'] - df['forecast_eps'])
        print(f"   Retrieved {len(df):,} analyst-year observations")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_institutional_holdings(db):
    """Pull institutional holdings from Thomson 13F."""
    print("\n📊 Pulling Institutional Holdings (13F)...")
    query = """
    SELECT a.cusip, 
           EXTRACT(YEAR FROM a.rdate) as year,
           COUNT(DISTINCT a.mgrno) as n_institutions,
           SUM(a.shares) as total_shares
    FROM tfn.s34 a
    WHERE a.rdate >= '2006-01-01'
    GROUP BY a.cusip, EXTRACT(YEAR FROM a.rdate)
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} firm-year observations")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_governance(db):
    """Pull governance data from ISS."""
    print("\n📊 Pulling Governance Data (ISS)...")
    query = """
    SELECT ticker, year, 
           cboard as classified_board,
           dualclass as dual_class,
           ppill as poison_pill
    FROM risk.gset
    WHERE year >= 2006
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} governance observations")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_monthly_returns(db):
    """Pull monthly stock returns from CRSP."""
    print("\n📊 Pulling Monthly Returns (CRSP)...")
    query = """
    SELECT a.permno, 
           EXTRACT(YEAR FROM a.date) as year,
           STDDEV(a.ret) as volatility,
           AVG(a.ret) as avg_ret,
           COUNT(*) as n_months
    FROM crsp.msf a
    WHERE a.date >= '2006-01-01'
      AND a.ret IS NOT NULL
    GROUP BY a.permno, EXTRACT(YEAR FROM a.date)
    HAVING COUNT(*) >= 10
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} firm-year volatility observations")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_execucomp_full(db):
    """Pull full ExecuComp data with incentives."""
    print("\n📊 Pulling ExecuComp Incentives...")
    query = """
    SELECT gvkey, year, execid,
           tdc1, tdc2, salary, bonus,
           stock_awards_fv, option_awards_fv,
           shrown_excl_opts as shares_owned,
           opt_unex_exer_num as options_exer,
           opt_unex_unexer_num as options_unexer
    FROM comp.execcomp
    WHERE ceoann = 'CEO'
      AND year >= 2006
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} CEO-year observations")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_permno_gvkey_link(db):
    """Pull PERMNO-GVKEY link table."""
    print("\n📊 Pulling PERMNO-GVKEY Link...")
    query = """
    SELECT DISTINCT gvkey, lpermno as permno, linkdt, linkenddt
    FROM crsp.ccmxpf_linktable
    WHERE linktype IN ('LU', 'LC')
      AND linkprim IN ('P', 'C')
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} link observations")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def main():
    print("🔌" * 35)
    print("      WRDS DATA PULL & ANALYSIS")
    print("🔌" * 35)
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       WRDS DATA BRAINSTORM                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  1. IBES: Analyst coverage → Do good matches attract attention?              ║
║  2. 13F: Institutional holdings → Do investors recognize matches?            ║
║  3. ISS: Governance → Does governance interact with match?                   ║
║  4. CRSP: Volatility → Are good matches lower risk?                          ║
║  5. ExecuComp: Incentives → Does incentive design matter?                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
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
    
    df_clean['match_quintile'] = pd.qcut(df_clean['match_means'], 5, 
        labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4', 'Q5 (Best)'])
    
    # ================================================================
    # CONNECT TO WRDS
    # ================================================================
    print("\n🔌 CONNECTING TO WRDS...")
    db = connect_wrds()
    
    wrds_data = {}
    
    if db is not None:
        print("\n" + "=" * 70)
        print("📥 PULLING WRDS DATA")
        print("=" * 70)
        
        wrds_data['ibes'] = pull_ibes_coverage(db)
        wrds_data['holdings'] = pull_institutional_holdings(db)
        wrds_data['governance'] = pull_governance(db)
        wrds_data['volatility'] = pull_monthly_returns(db)
        wrds_data['execucomp'] = pull_execucomp_full(db)
        wrds_data['link'] = pull_permno_gvkey_link(db)
        
        db.close()
        print("\n✅ WRDS Connection Closed")
        
        # Save WRDS data locally for future use
        print("\n💾 SAVING WRDS DATA...")
        for name, data in wrds_data.items():
            if data is not None:
                data.to_parquet(f'Output/wrds_{name}.parquet')
                print(f"   Saved: Output/wrds_{name}.parquet")
    
    # ================================================================
    # MERGE AND ANALYZE
    # ================================================================
    print("\n" + "=" * 70)
    print("🔗 MERGING AND ANALYZING")
    print("=" * 70)
    
    mega_df = df_clean.copy()
    
    # Load local compensation data
    exec_comp = pd.read_parquet('/Users/smaric/Papers/CEO NCA/data/raw/execucomp_ceo.parquet')
    exec_comp['gvkey'] = pd.to_numeric(exec_comp['gvkey'], errors='coerce')
    mega_df = mega_df.merge(exec_comp, on=['gvkey', 'fiscalyear'], how='left')
    
    # Load BLM data
    blm_mobility = pd.read_stata('../Data/blm_data_ceo_prep_v4.3_mobility.dta')
    mega_df = mega_df.merge(
        blm_mobility[['gvkey', 'year', 'tobinw', 'roaw']].drop_duplicates(),
        left_on=['gvkey', 'fiscalyear'], right_on=['gvkey', 'year'], how='left'
    )
    
    # Merge WRDS ExecuComp if available
    if wrds_data.get('execucomp') is not None:
        wrds_exec = wrds_data['execucomp'].copy()
        wrds_exec['gvkey'] = pd.to_numeric(wrds_exec['gvkey'], errors='coerce')
        mega_df = mega_df.merge(
            wrds_exec[['gvkey', 'year', 'shares_owned', 'options_exer', 'options_unexer']],
            left_on=['gvkey', 'fiscalyear'], right_on=['gvkey', 'year'],
            how='left', suffixes=('', '_wrds')
        )
        print(f"  Merged WRDS ExecuComp: {mega_df['shares_owned'].notna().sum():,} obs with ownership data")
    
    # Merge WRDS Volatility if available
    if wrds_data.get('volatility') is not None and wrds_data.get('link') is not None:
        link = wrds_data['link']
        link['gvkey'] = pd.to_numeric(link['gvkey'], errors='coerce')
        vol = wrds_data['volatility'].merge(link[['gvkey', 'permno']], on='permno', how='left')
        vol = vol.dropna(subset=['gvkey'])
        mega_df = mega_df.merge(
            vol[['gvkey', 'year', 'volatility', 'avg_ret']],
            left_on=['gvkey', 'fiscalyear'], right_on=['gvkey', 'year'],
            how='left', suffixes=('', '_vol')
        )
        print(f"  Merged WRDS Volatility: {mega_df['volatility'].notna().sum():,} obs")
    
    print(f"\n  Final Merged Dataset: {len(mega_df):,} observations")
    print(f"  With Compensation: {mega_df['tdc1'].notna().sum():,}")
    
    # ================================================================
    # ANALYSES
    # ================================================================
    print("\n" + "=" * 70)
    print("📊 WRDS-ENHANCED ANALYSES")
    print("=" * 70)
    
    # Analysis 1: CEO Ownership by Match Quality
    if 'shares_owned' in mega_df.columns:
        own_df = mega_df.dropna(subset=['match_means', 'shares_owned'])
        own_df = own_df[own_df['shares_owned'] > 0]
        if len(own_df) > 100:
            own_df['match_q'] = pd.qcut(own_df['match_means'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ownership_by_match = own_df.groupby('match_q', observed=False).agg({
                'shares_owned': ['mean', 'median'],
                'options_exer': 'mean',
                'gvkey': 'count'
            }).round(0)
            ownership_by_match.columns = ['Mean Shares', 'Median Shares', 'Mean Options', 'N']
            print("\n--- CEO Equity Ownership by Match Quality ---")
            print(ownership_by_match)
    
    # Analysis 2: Volatility by Match Quality
    if 'volatility' in mega_df.columns:
        vol_df = mega_df.dropna(subset=['match_means', 'volatility'])
        if len(vol_df) > 100:
            vol_df['match_q'] = pd.qcut(vol_df['match_means'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            vol_by_match = vol_df.groupby('match_q', observed=False).agg({
                'volatility': 'mean',
                'avg_ret': 'mean',
                'gvkey': 'count'
            }).round(4)
            vol_by_match.columns = ['Volatility', 'Avg Return', 'N']
            print("\n--- Stock Volatility by Match Quality ---")
            print(vol_by_match)
    
    # Analysis 3: Full Compensation Breakdown
    comp_df = mega_df.dropna(subset=['match_means', 'tdc1', 'salary'])
    comp_df = comp_df[comp_df['tdc1'] > 0]
    if len(comp_df) > 100:
        comp_df['equity_ratio'] = (comp_df['stock_awards_fv'].fillna(0) + 
                                    comp_df['option_awards_fv'].fillna(0)) / comp_df['tdc1']
        comp_df['match_q'] = pd.qcut(comp_df['match_means'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        comp_by_match = comp_df.groupby('match_q', observed=False).agg({
            'tdc1': 'mean',
            'equity_ratio': 'mean',
            'salary': 'mean',
            'bonus': 'mean',
            'stock_awards_fv': 'mean',
            'gvkey': 'count'
        }).round(0)
        comp_by_match.columns = ['TDC1', 'Equity%', 'Salary', 'Bonus', 'Stock Awards', 'N']
        
        print("\n--- Full Compensation by Match Quality ---")
        print(comp_by_match)
    
    # ================================================================
    # VISUALIZATIONS
    # ================================================================
    print("\n" + "=" * 70)
    print("📊 CREATING VISUALIZATIONS")
    print("=" * 70)
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Compensation by Match
    ax1 = fig.add_subplot(3, 3, 1)
    if len(comp_df) > 100:
        tdc_data = comp_by_match['TDC1'].reset_index()
        ax1.bar(range(5), tdc_data['TDC1']/1000, color='steelblue', edgecolor='black')
        ax1.set_xticks(range(5))
        ax1.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax1.set_ylabel('Mean Total Comp ($M)')
        ax1.set_title('CEO Pay by Match Quality', fontweight='bold')
    
    # 2. Equity Ratio by Match
    ax2 = fig.add_subplot(3, 3, 2)
    if len(comp_df) > 100:
        eq_data = comp_by_match['Equity%'].reset_index()
        ax2.bar(range(5), eq_data['Equity%'], color='green', edgecolor='black')
        ax2.set_xticks(range(5))
        ax2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax2.set_ylabel('Equity % of Total Comp')
        ax2.set_title('Equity Ratio by Match', fontweight='bold')
    
    # 3. Volatility by Match
    ax3 = fig.add_subplot(3, 3, 3)
    if 'volatility' in mega_df.columns and len(vol_df) > 100:
        vol_data = vol_by_match['Volatility'].reset_index()
        ax3.bar(range(5), vol_data['Volatility']*100, color='red', edgecolor='black')
        ax3.set_xticks(range(5))
        ax3.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax3.set_ylabel('Monthly Volatility (%)')
        ax3.set_title('Stock Volatility by Match', fontweight='bold')
    
    # 4. Ownership by Match
    ax4 = fig.add_subplot(3, 3, 4)
    if 'shares_owned' in mega_df.columns and len(own_df) > 100:
        own_data = ownership_by_match['Mean Shares'].reset_index()
        ax4.bar(range(5), own_data['Mean Shares']/1e6, color='purple', edgecolor='black')
        ax4.set_xticks(range(5))
        ax4.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax4.set_ylabel('Mean Shares Owned (M)')
        ax4.set_title('CEO Ownership by Match', fontweight='bold')
    
    # 5. Match × Tobin Q Scatter
    ax5 = fig.add_subplot(3, 3, 5)
    valid = mega_df.dropna(subset=['match_means', 'tobinw'])
    if len(valid) > 100:
        sample = valid.sample(min(2000, len(valid)))
        ax5.scatter(sample['match_means'], sample['tobinw'], alpha=0.3, s=15)
        z = np.polyfit(sample['match_means'], sample['tobinw'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(sample['match_means'].min(), sample['match_means'].max(), 100)
        ax5.plot(x_line, p(x_line), 'r-', lw=2, label=f'β={z[0]:.2f}')
        ax5.set_xlabel('Match Quality')
        ax5.set_ylabel("Tobin's Q")
        ax5.set_title('Match → Firm Value', fontweight='bold')
        ax5.legend()
    
    # 6. Match Distribution
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.hist(mega_df['match_means'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax6.axvline(x=mega_df['match_means'].median(), color='red', linestyle='--', lw=2, label='Median')
    ax6.axvline(x=mega_df['match_means'].quantile(0.9), color='gold', linestyle='--', lw=2, label='Top 10%')
    ax6.set_xlabel('Match Quality')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Match Quality Distribution', fontweight='bold')
    ax6.legend()
    
    # 7. Industry Pay
    ax7 = fig.add_subplot(3, 3, 7)
    ind_pay = mega_df.dropna(subset=['compindustry', 'tdc1'])
    ind_stats = ind_pay.groupby('compindustry')['tdc1'].agg(['mean', 'count'])
    ind_stats = ind_stats[ind_stats['count'] >= 50].sort_values('mean', ascending=False).head(10)
    ax7.barh(range(len(ind_stats)), ind_stats['mean']/1000, color='steelblue', edgecolor='black')
    ax7.set_yticks(range(len(ind_stats)))
    ax7.set_yticklabels([i[:18] for i in ind_stats.index], fontsize=8)
    ax7.set_xlabel('Mean CEO Pay ($M)')
    ax7.set_title('Top Industries by Pay', fontweight='bold')
    ax7.invert_yaxis()
    
    # 8. Match × Size Heatmap
    ax8 = fig.add_subplot(3, 3, 8)
    try:
        size_bins = pd.qcut(mega_df['logatw'], 4, labels=['Small', 'Med', 'Large', 'Giant'])
        match_bins = pd.qcut(mega_df['match_means'], 4, labels=['Low', 'Mid', 'High', 'Top'])
        heatmap = mega_df.groupby([match_bins, size_bins], observed=False)['tdc1'].mean().unstack()
        heatmap = heatmap.astype(float) / 1000
        sns.heatmap(heatmap, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax8)
        ax8.set_title('CEO Pay ($M): Match × Size', fontweight='bold')
    except:
        pass
    
    # 9. Time Trend
    ax9 = fig.add_subplot(3, 3, 9)
    time_trend = mega_df.groupby('fiscalyear').agg({'match_means': 'mean', 'tdc1': 'mean'})
    ax9.plot(time_trend.index, time_trend['match_means'], 'b-o', lw=2)
    ax9_twin = ax9.twinx()
    ax9_twin.plot(time_trend.index, time_trend['tdc1']/1000, 'g-s', lw=2)
    ax9.set_xlabel('Year')
    ax9.set_ylabel('Avg Match', color='blue')
    ax9_twin.set_ylabel('Avg Pay ($M)', color='green')
    ax9.set_title('Match & Pay Over Time', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Output/wrds_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: Output/wrds_analysis.png")
    
    # ================================================================
    # EXECUTIVE SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("🏆 WRDS ANALYSIS SUMMARY")
    print("=" * 70)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       🔌 WRDS ANALYSIS FINDINGS 🔌                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📊 DATA PULLED:                                                              ║
║     - IBES analyst coverage                                                   ║
║     - 13F institutional holdings                                              ║
║     - ISS governance provisions                                               ║
║     - CRSP monthly volatility                                                 ║
║     - ExecuComp CEO ownership                                                 ║
║                                                                               ║
║  💰 COMPENSATION:                                                             ║
║     Q5 (Best): ${comp_by_match.loc['Q5', 'TDC1']/1000:.1f}M | Equity: {comp_by_match.loc['Q5', 'Equity%']*100:.0f}%                               ║
║     Q1 (Worst): ${comp_by_match.loc['Q1', 'TDC1']/1000:.1f}M | Equity: {comp_by_match.loc['Q1', 'Equity%']*100:.0f}%                              ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n🔌🔌🔌 WRDS ANALYSIS COMPLETE! 🔌🔌🔌")

if __name__ == "__main__":
    main()
