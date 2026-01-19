#!/usr/bin/env python3
"""
🔌🔌🔌 COMPREHENSIVE WRDS PULL & MEGA ANALYSIS 🔌🔌🔌

Pulls ALL available WRDS databases for CEO-Firm Match analysis:
1. IBES - Analyst coverage & forecasts
2. Thomson 13F - Institutional holdings  
3. ISS Governance - Board provisions
4. CRSP - Stock returns & volatility
5. SDC M&A - Mergers & acquisitions
6. Compustat Quarterly - Operating performance
7. Audit Analytics - Restatements & SOX
8. ExecuComp Full - CEO incentives & ownership
"""
import pandas as pd
import numpy as np
import wrds
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')
import os

from ceo_firm_matching import (
    Config,
    DataProcessor,
    CEOFirmDataset,
    train_model,
)

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# WRDS Credentials
WRDS_USER = 'maricste93'
WRDS_PASS = 'jexvar-manryn-6Cosky'

def connect_wrds():
    """Connect to WRDS."""
    try:
        os.environ['PGPASSWORD'] = WRDS_PASS
        db = wrds.Connection(wrds_username=WRDS_USER)
        print("✅ WRDS Connection Successful!")
        return db
    except Exception as e:
        print(f"❌ WRDS Connection Failed: {e}")
        return None

# ================================================================
# WRDS PULL FUNCTIONS
# ================================================================

def pull_ibes(db):
    """Pull IBES analyst data."""
    print("\n📊 Pulling IBES Analyst Data...")
    query = """
    SELECT a.ticker, 
           EXTRACT(YEAR FROM a.statpers) as year,
           COUNT(*) as n_estimates,
           AVG(a.actual) as actual_eps,
           AVG(a.meanest) as mean_forecast,
           STDDEV(a.meanest) as forecast_dispersion
    FROM ibes.statsum_epsus a
    WHERE a.measure = 'EPS' 
      AND a.fpi = '1'
      AND a.statpers >= '2006-01-01'
    GROUP BY a.ticker, EXTRACT(YEAR FROM a.statpers)
    """
    try:
        df = db.raw_sql(query)
        df['forecast_error'] = abs(df['actual_eps'] - df['mean_forecast'])
        print(f"   Retrieved {len(df):,} analyst-year observations")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_13f_holdings(db):
    """Pull institutional holdings."""
    print("\n📊 Pulling 13F Institutional Holdings...")
    query = """
    SELECT a.cusip, 
           EXTRACT(YEAR FROM a.rdate) as year,
           COUNT(DISTINCT a.mgrno) as n_institutions,
           SUM(a.shares) as total_shares,
           AVG(a.prc) as avg_price
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
    """Pull ISS governance provisions."""
    print("\n📊 Pulling ISS Governance...")
    query = """
    SELECT ticker, year, 
           cboard as classified_board,
           dualclass as dual_class,
           ppill as poison_pill,
           lspmt as limit_special_mtg,
           supermajor as supermajority
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

def pull_crsp_volatility(db):
    """Pull CRSP monthly returns and compute volatility."""
    print("\n📊 Pulling CRSP Monthly Returns...")
    query = """
    SELECT a.permno, 
           EXTRACT(YEAR FROM a.date) as year,
           STDDEV(a.ret) as volatility,
           AVG(a.ret) as avg_ret,
           MIN(a.ret) as min_ret,
           MAX(a.ret) as max_ret,
           COUNT(*) as n_months
    FROM crsp.msf a
    WHERE a.date >= '2006-01-01'
      AND a.ret IS NOT NULL
    GROUP BY a.permno, EXTRACT(YEAR FROM a.date)
    HAVING COUNT(*) >= 10
    """
    try:
        df = db.raw_sql(query)
        df['return_range'] = df['max_ret'] - df['min_ret']
        print(f"   Retrieved {len(df):,} firm-year volatility observations")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_sdc_mna(db):
    """Pull SDC M&A activity."""
    print("\n📊 Pulling SDC M&A Deals...")
    query = """
    SELECT acu as acquiror_cusip,
           EXTRACT(YEAR FROM da) as year,
           COUNT(*) as n_deals,
           SUM(val) as total_deal_value,
           AVG(val) as avg_deal_value,
           AVG(pr) as avg_premium
    FROM sdc.mergers
    WHERE da >= '2006-01-01'
      AND acu IS NOT NULL
      AND val > 0
    GROUP BY acu, EXTRACT(YEAR FROM da)
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} acquiror-year observations")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_compustat_quarterly(db):
    """Pull Compustat quarterly fundamentals."""
    print("\n📊 Pulling Compustat Quarterly...")
    query = """
    SELECT gvkey, 
           EXTRACT(YEAR FROM datadate) as year,
           EXTRACT(QUARTER FROM datadate) as quarter,
           saleq as sales,
           ibq as income,
           atq as assets,
           revtq as revenue
    FROM comp.fundq
    WHERE datadate >= '2006-01-01'
      AND indfmt = 'INDL'
      AND datafmt = 'STD'
      AND popsrc = 'D'
      AND consol = 'C'
    """
    try:
        df = db.raw_sql(query)
        # Compute YoY growth
        df = df.sort_values(['gvkey', 'year', 'quarter'])
        df['sales_growth'] = df.groupby('gvkey')['sales'].pct_change(4)  # YoY
        df['income_growth'] = df.groupby('gvkey')['income'].pct_change(4)
        print(f"   Retrieved {len(df):,} firm-quarter observations")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_audit_analytics(db):
    """Pull Audit Analytics restatements."""
    print("\n📊 Pulling Audit Analytics Restatements...")
    query = """
    SELECT company_fkey as cik,
           EXTRACT(YEAR FROM res_begin_date) as year,
           COUNT(*) as n_restatements,
           SUM(CASE WHEN res_fraud = 'Y' THEN 1 ELSE 0 END) as n_fraud
    FROM audit.auditnonreli
    WHERE res_begin_date >= '2006-01-01'
    GROUP BY company_fkey, EXTRACT(YEAR FROM res_begin_date)
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} firm-year restatement observations")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_execucomp(db):
    """Pull ExecuComp CEO data."""
    print("\n📊 Pulling ExecuComp CEO Data...")
    query = """
    SELECT gvkey, year, execid,
           tdc1, tdc2, salary, bonus,
           stock_awards_fv, option_awards_fv,
           shrown_excl_opts as shares_owned,
           opt_unex_exer_num as options_exer,
           opt_unex_unexer_num as options_unexer,
           pcttotalt as pct_total
    FROM execcomp.anncomp
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

def pull_link_tables(db):
    """Pull all linking tables."""
    print("\n📊 Pulling Link Tables...")
    
    # CRSP-Compustat link
    query1 = """
    SELECT DISTINCT gvkey, lpermno as permno, linkdt, linkenddt
    FROM crsp.ccmxpf_linktable
    WHERE linktype IN ('LU', 'LC')
      AND linkprim IN ('P', 'C')
    """
    
    # Ticker-CUSIP link from CRSP
    query2 = """
    SELECT DISTINCT permno, ncusip as cusip, ticker
    FROM crsp.msenames
    WHERE ncusip IS NOT NULL
    """
    
    try:
        link1 = db.raw_sql(query1)
        link2 = db.raw_sql(query2)
        print(f"   CRSP-Compustat link: {len(link1):,} observations")
        print(f"   Ticker-CUSIP link: {len(link2):,} observations")
        return {'crsp_comp': link1, 'ticker_cusip': link2}
    except Exception as e:
        print(f"   Error: {e}")
        return None

def main():
    print("🔌" * 40)
    print("   COMPREHENSIVE WRDS PULL & MEGA ANALYSIS")
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
    
    df_clean['match_quintile'] = pd.qcut(df_clean['match_means'], 5, 
        labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4', 'Q5 (Best)'])
    
    # ================================================================
    # CONNECT TO WRDS AND PULL ALL DATA
    # ================================================================
    print("\n🔌 CONNECTING TO WRDS...")
    db = connect_wrds()
    
    if db is None:
        print("❌ Could not connect to WRDS")
        return
    
    print("\n" + "=" * 70)
    print("📥 PULLING ALL WRDS DATABASES")
    print("=" * 70)
    
    wrds_data = {}
    
    # Pull all databases
    wrds_data['ibes'] = pull_ibes(db)
    wrds_data['holdings'] = pull_13f_holdings(db)
    wrds_data['governance'] = pull_governance(db)
    wrds_data['volatility'] = pull_crsp_volatility(db)
    wrds_data['mna'] = pull_sdc_mna(db)
    wrds_data['quarterly'] = pull_compustat_quarterly(db)
    wrds_data['restatements'] = pull_audit_analytics(db)
    wrds_data['execucomp'] = pull_execucomp(db)
    wrds_data['links'] = pull_link_tables(db)
    
    db.close()
    print("\n✅ WRDS Connection Closed")
    
    # Count successful pulls
    total_obs = 0
    print("\n" + "=" * 70)
    print("📊 WRDS DATA SUMMARY")
    print("=" * 70)
    for name, data in wrds_data.items():
        if data is not None and not isinstance(data, dict):
            print(f"  {name}: {len(data):,} observations")
            total_obs += len(data)
            # Save to parquet
            data.to_parquet(f'Output/wrds_{name}.parquet')
        elif isinstance(data, dict):
            for subname, subdata in data.items():
                if subdata is not None:
                    print(f"  {name}/{subname}: {len(subdata):,} observations")
                    total_obs += len(subdata)
                    subdata.to_parquet(f'Output/wrds_{subname}.parquet')
    
    print(f"\n  📦 TOTAL WRDS DATA: {total_obs:,} observations")
    
    # ================================================================
    # MERGE ALL DATA
    # ================================================================
    print("\n" + "=" * 70)
    print("🔗 CREATING MEGA MERGED DATASET")
    print("=" * 70)
    
    mega_df = df_clean.copy()
    
    # Load local compensation data
    exec_comp = pd.read_parquet('/Users/smaric/Papers/CEO NCA/data/raw/execucomp_ceo.parquet')
    exec_comp['gvkey'] = pd.to_numeric(exec_comp['gvkey'], errors='coerce')
    mega_df = mega_df.merge(exec_comp, on=['gvkey', 'fiscalyear'], how='left')
    print(f"  + Compensation: {mega_df['tdc1'].notna().sum():,}")
    
    # Load BLM data
    blm_mobility = pd.read_stata('../Data/blm_data_ceo_prep_v4.3_mobility.dta')
    mega_df = mega_df.merge(
        blm_mobility[['gvkey', 'year', 'tobinw', 'roaw']].drop_duplicates(),
        left_on=['gvkey', 'fiscalyear'], right_on=['gvkey', 'year'], how='left'
    )
    print(f"  + BLM (Tobin Q): {mega_df['tobinw'].notna().sum():,}")
    
    # Merge volatility via PERMNO link
    if wrds_data.get('volatility') is not None and wrds_data.get('links') is not None:
        link = wrds_data['links']['crsp_comp']
        link['gvkey'] = pd.to_numeric(link['gvkey'], errors='coerce')
        vol = wrds_data['volatility'].merge(link[['gvkey', 'permno']], on='permno', how='left')
        vol = vol.dropna(subset=['gvkey'])
        mega_df = mega_df.merge(
            vol[['gvkey', 'year', 'volatility', 'avg_ret', 'return_range']].drop_duplicates(),
            left_on=['gvkey', 'fiscalyear'], right_on=['gvkey', 'year'],
            how='left', suffixes=('', '_vol')
        )
        print(f"  + Volatility: {mega_df['volatility'].notna().sum():,}")
    
    # Merge quarterly data (annualized)
    if wrds_data.get('quarterly') is not None:
        quarterly = wrds_data['quarterly']
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
        print(f"  + Quarterly Fundamentals: {mega_df['sales_growth'].notna().sum():,}")
    
    # Merge M&A via CUSIP link
    if wrds_data.get('mna') is not None and wrds_data.get('links') is not None:
        mna = wrds_data['mna']
        link2 = wrds_data['links']['ticker_cusip']
        link1 = wrds_data['links']['crsp_comp']
        # Link CUSIP -> PERMNO -> GVKEY
        link2 = link2.merge(link1[['gvkey', 'permno']], on='permno', how='left')
        mna = mna.merge(link2[['cusip', 'gvkey']].drop_duplicates(), 
                        left_on='acquiror_cusip', right_on='cusip', how='left')
        mna['gvkey'] = pd.to_numeric(mna['gvkey'], errors='coerce')
        mega_df = mega_df.merge(
            mna[['gvkey', 'year', 'n_deals', 'total_deal_value', 'avg_premium']].drop_duplicates(),
            left_on=['gvkey', 'fiscalyear'], right_on=['gvkey', 'year'],
            how='left', suffixes=('', '_mna')
        )
        mega_df['n_deals'] = mega_df['n_deals'].fillna(0)
        mega_df['is_acquiror'] = (mega_df['n_deals'] > 0).astype(int)
        print(f"  + M&A Activity: {(mega_df['n_deals'] > 0).sum():,} firm-years with deals")
    
    # Merge IBES via ticker
    if wrds_data.get('ibes') is not None:
        ibes = wrds_data['ibes']
        # Try to match via ticker from Compustat
        if 'tic' in mega_df.columns:
            mega_df = mega_df.merge(
                ibes[['ticker', 'year', 'n_estimates', 'forecast_error', 'forecast_dispersion']],
                left_on=['tic', 'fiscalyear'], right_on=['ticker', 'year'],
                how='left', suffixes=('', '_ibes')
            )
            print(f"  + IBES Analysts: {mega_df['n_estimates'].notna().sum():,}")
    
    # Merge ExecuComp WRDS (ownership)
    if wrds_data.get('execucomp') is not None:
        exec_wrds = wrds_data['execucomp']
        exec_wrds['gvkey'] = pd.to_numeric(exec_wrds['gvkey'], errors='coerce')
        mega_df = mega_df.merge(
            exec_wrds[['gvkey', 'year', 'shares_owned', 'options_exer', 'pct_total']].drop_duplicates(),
            left_on=['gvkey', 'fiscalyear'], right_on=['gvkey', 'year'],
            how='left', suffixes=('', '_own')
        )
        print(f"  + CEO Ownership: {mega_df['shares_owned'].notna().sum():,}")
    
    print(f"\n  📦 FINAL MEGA DATASET: {len(mega_df):,} observations")
    
    # ================================================================
    # COMPREHENSIVE ANALYSES
    # ================================================================
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE WRDS ANALYSES")
    print("=" * 70)
    
    # Create match quintiles
    mega_df['match_q'] = pd.qcut(mega_df['match_means'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    # Analysis 1: Volatility by Match
    if 'volatility' in mega_df.columns:
        vol_df = mega_df.dropna(subset=['match_means', 'volatility'])
        if len(vol_df) > 100:
            vol_by_match = vol_df.groupby('match_q', observed=False).agg({
                'volatility': 'mean',
                'avg_ret': 'mean',
                'return_range': 'mean',
                'gvkey': 'count'
            }).round(4)
            vol_by_match.columns = ['Volatility', 'Avg Return', 'Return Range', 'N']
            print("\n--- STOCK VOLATILITY BY MATCH QUALITY ---")
            print(vol_by_match)
            
            # Regression
            model_vol = ols('volatility ~ match_means + logatw', data=vol_df).fit()
            print(f"\n  Volatility ~ Match β: {model_vol.params['match_means']:.4f} (p={model_vol.pvalues['match_means']:.4f})")
    
    # Analysis 2: M&A Activity by Match
    if 'n_deals' in mega_df.columns:
        mna_df = mega_df.dropna(subset=['match_means'])
        mna_by_match = mna_df.groupby('match_q', observed=False).agg({
            'is_acquiror': 'mean',
            'n_deals': 'mean',
            'total_deal_value': 'mean',
            'avg_premium': 'mean',
            'gvkey': 'count'
        }).round(3)
        mna_by_match.columns = ['Acquiror Rate', 'Avg Deals', 'Total Value', 'Avg Premium', 'N']
        print("\n--- M&A ACTIVITY BY MATCH QUALITY ---")
        print(mna_by_match)
    
    # Analysis 3: Quarterly Growth by Match
    if 'sales_growth' in mega_df.columns:
        growth_df = mega_df.dropna(subset=['match_means', 'sales_growth'])
        growth_df = growth_df[growth_df['sales_growth'].between(-1, 2)]  # Winsorize
        if len(growth_df) > 100:
            growth_by_match = growth_df.groupby('match_q', observed=False).agg({
                'sales_growth': 'mean',
                'income_growth': 'mean',
                'gvkey': 'count'
            }).round(4)
            growth_by_match.columns = ['Sales Growth', 'Income Growth', 'N']
            print("\n--- QUARTERLY GROWTH BY MATCH QUALITY ---")
            print(growth_by_match)
    
    # Analysis 4: CEO Ownership by Match
    if 'shares_owned' in mega_df.columns:
        own_df = mega_df.dropna(subset=['match_means', 'shares_owned'])
        own_df = own_df[own_df['shares_owned'] > 0]
        if len(own_df) > 100:
            own_by_match = own_df.groupby('match_q', observed=False).agg({
                'shares_owned': ['mean', 'median'],
                'options_exer': 'mean',
                'gvkey': 'count'
            }).round(0)
            own_by_match.columns = ['Mean Shares', 'Median Shares', 'Mean Options', 'N']
            print("\n--- CEO OWNERSHIP BY MATCH QUALITY ---")
            print(own_by_match)
    
    # Analysis 5: Compensation by Match
    comp_df = mega_df.dropna(subset=['match_means', 'tdc1'])
    comp_df = comp_df[comp_df['tdc1'] > 0]
    if len(comp_df) > 100:
        comp_df['equity_ratio'] = (comp_df['stock_awards_fv'].fillna(0) + 
                                    comp_df['option_awards_fv'].fillna(0)) / comp_df['tdc1']
        
        comp_by_match = comp_df.groupby('match_q', observed=False).agg({
            'tdc1': 'mean',
            'equity_ratio': 'mean',
            'salary': 'mean',
            'stock_awards_fv': 'mean',
            'gvkey': 'count'
        }).round(0)
        comp_by_match.columns = ['TDC1', 'Equity%', 'Salary', 'Stock Awards', 'N']
        print("\n--- COMPENSATION BY MATCH QUALITY ---")
        print(comp_by_match)
    
    # ================================================================
    # MEGA VISUALIZATION
    # ================================================================
    print("\n" + "=" * 70)
    print("📊 CREATING MEGA VISUALIZATIONS")
    print("=" * 70)
    
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Volatility by Match
    ax1 = fig.add_subplot(4, 4, 1)
    if 'volatility' in mega_df.columns and len(vol_df) > 100:
        vol_data = vol_by_match['Volatility'].reset_index()
        colors = ['#e74c3c' if i < 2 else '#f39c12' if i < 4 else '#2ecc71' for i in range(5)]
        ax1.bar(range(5), vol_data['Volatility']*100, color=colors, edgecolor='black')
        ax1.set_xticks(range(5))
        ax1.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax1.set_ylabel('Monthly Volatility (%)')
        ax1.set_title('📉 Stock Volatility by Match', fontweight='bold')
    
    # 2. Returns by Match
    ax2 = fig.add_subplot(4, 4, 2)
    if 'volatility' in mega_df.columns and len(vol_df) > 100:
        ret_data = vol_by_match['Avg Return'].reset_index()
        ax2.bar(range(5), ret_data['Avg Return']*100, color='steelblue', edgecolor='black')
        ax2.set_xticks(range(5))
        ax2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax2.set_ylabel('Avg Monthly Return (%)')
        ax2.set_title('📈 Returns by Match', fontweight='bold')
    
    # 3. M&A Rate by Match
    ax3 = fig.add_subplot(4, 4, 3)
    if 'n_deals' in mega_df.columns:
        mna_data = mna_by_match['Acquiror Rate'].reset_index()
        ax3.bar(range(5), mna_data['Acquiror Rate']*100, color='purple', edgecolor='black')
        ax3.set_xticks(range(5))
        ax3.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax3.set_ylabel('% Making Acquisitions')
        ax3.set_title('🤝 M&A Activity by Match', fontweight='bold')
    
    # 4. Sales Growth by Match
    ax4 = fig.add_subplot(4, 4, 4)
    if 'sales_growth' in mega_df.columns and len(growth_df) > 100:
        growth_data = growth_by_match['Sales Growth'].reset_index()
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in growth_data['Sales Growth']]
        ax4.bar(range(5), growth_data['Sales Growth']*100, color=colors, edgecolor='black')
        ax4.set_xticks(range(5))
        ax4.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax4.set_ylabel('YoY Sales Growth (%)')
        ax4.set_title('📊 Sales Growth by Match', fontweight='bold')
        ax4.axhline(y=0, color='black', linestyle='-', lw=1)
    
    # 5. Compensation by Match
    ax5 = fig.add_subplot(4, 4, 5)
    if len(comp_df) > 100:
        comp_data = comp_by_match['TDC1'].reset_index()
        ax5.bar(range(5), comp_data['TDC1']/1000, color='green', edgecolor='black')
        ax5.set_xticks(range(5))
        ax5.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax5.set_ylabel('Mean Total Comp ($M)')
        ax5.set_title('💰 CEO Pay by Match', fontweight='bold')
    
    # 6. Equity Ratio by Match
    ax6 = fig.add_subplot(4, 4, 6)
    if len(comp_df) > 100:
        eq_data = comp_by_match['Equity%'].reset_index()
        ax6.bar(range(5), eq_data['Equity%'], color='orange', edgecolor='black')
        ax6.set_xticks(range(5))
        ax6.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax6.set_ylabel('Equity % of Comp')
        ax6.set_title('📊 Equity Ratio by Match', fontweight='bold')
    
    # 7. Tobin Q by Match
    ax7 = fig.add_subplot(4, 4, 7)
    tobin_df = mega_df.dropna(subset=['match_means', 'tobinw'])
    if len(tobin_df) > 100:
        tobin_by_match = tobin_df.groupby('match_q', observed=False)['tobinw'].mean()
        ax7.bar(range(5), tobin_by_match.values, color='steelblue', edgecolor='black')
        ax7.set_xticks(range(5))
        ax7.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax7.set_ylabel("Tobin's Q")
        ax7.set_title("🏛️ Firm Value by Match", fontweight='bold')
    
    # 8. CEO Ownership by Match
    ax8 = fig.add_subplot(4, 4, 8)
    if 'shares_owned' in mega_df.columns and len(own_df) > 100:
        own_data = own_by_match['Mean Shares'].reset_index()
        ax8.bar(range(5), own_data['Mean Shares']/1e6, color='purple', edgecolor='black')
        ax8.set_xticks(range(5))
        ax8.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax8.set_ylabel('Mean Shares (M)')
        ax8.set_title('👤 CEO Ownership by Match', fontweight='bold')
    
    # 9. Sharpe Ratio by Match
    ax9 = fig.add_subplot(4, 4, 9)
    if 'volatility' in mega_df.columns and len(vol_df) > 100:
        sharpe = vol_by_match['Avg Return'] / vol_by_match['Volatility']
        ax9.bar(range(5), sharpe.values, color='gold', edgecolor='black')
        ax9.set_xticks(range(5))
        ax9.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax9.set_ylabel('Sharpe Ratio')
        ax9.set_title('⚖️ Risk-Adjusted Returns', fontweight='bold')
    
    # 10. Match × Size Heatmap
    ax10 = fig.add_subplot(4, 4, 10)
    try:
        size_bins = pd.qcut(mega_df['logatw'], 4, labels=['Small', 'Mid', 'Large', 'Giant'])
        match_bins = pd.qcut(mega_df['match_means'], 4, labels=['Low', 'Mid', 'High', 'Top'])
        heatmap = mega_df.groupby([match_bins, size_bins], observed=False)['tdc1'].mean().unstack()
        heatmap = heatmap.astype(float) / 1000
        sns.heatmap(heatmap, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax10)
        ax10.set_title('💵 Pay ($M): Match × Size', fontweight='bold')
    except:
        pass
    
    # 11. Match vs Tobin Q Scatter
    ax11 = fig.add_subplot(4, 4, 11)
    valid = mega_df.dropna(subset=['match_means', 'tobinw'])
    if len(valid) > 100:
        sample = valid.sample(min(2000, len(valid)))
        ax11.scatter(sample['match_means'], sample['tobinw'], alpha=0.3, s=15)
        z = np.polyfit(sample['match_means'], sample['tobinw'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(sample['match_means'].min(), sample['match_means'].max(), 100)
        ax11.plot(x_line, p(x_line), 'r-', lw=2, label=f'β={z[0]:.2f}')
        ax11.set_xlabel('Match Quality')
        ax11.set_ylabel("Tobin's Q")
        ax11.set_title('Match → Firm Value', fontweight='bold')
        ax11.legend()
    
    # 12. Match Distribution
    ax12 = fig.add_subplot(4, 4, 12)
    ax12.hist(mega_df['match_means'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax12.axvline(x=mega_df['match_means'].median(), color='red', linestyle='--', lw=2)
    ax12.axvline(x=mega_df['match_means'].quantile(0.9), color='gold', linestyle='--', lw=2)
    ax12.set_xlabel('Match Quality')
    ax12.set_ylabel('Frequency')
    ax12.set_title('Match Distribution', fontweight='bold')
    
    # 13. Industry Rankings
    ax13 = fig.add_subplot(4, 4, 13)
    ind_stats = mega_df.groupby('compindustry')['match_means'].agg(['mean', 'count'])
    ind_stats = ind_stats[ind_stats['count'] >= 50].sort_values('mean', ascending=False).head(10)
    colors = ['green' if x > 0 else 'red' for x in ind_stats['mean']]
    ax13.barh(range(len(ind_stats)), ind_stats['mean'], color=colors, edgecolor='black')
    ax13.set_yticks(range(len(ind_stats)))
    ax13.set_yticklabels([i[:18] for i in ind_stats.index], fontsize=8)
    ax13.axvline(x=0, color='black')
    ax13.set_xlabel('Avg Match Quality')
    ax13.set_title('Top Industries', fontweight='bold')
    ax13.invert_yaxis()
    
    # 14. State Rankings
    ax14 = fig.add_subplot(4, 4, 14)
    state_stats = mega_df.groupby('ba_state')['match_means'].agg(['mean', 'count'])
    state_stats = state_stats[state_stats['count'] >= 30].sort_values('mean', ascending=False).head(10)
    colors = ['green' if x > 0 else 'red' for x in state_stats['mean']]
    ax14.barh(range(len(state_stats)), state_stats['mean'], color=colors, edgecolor='black')
    ax14.set_yticks(range(len(state_stats)))
    ax14.set_yticklabels(state_stats.index, fontsize=8)
    ax14.axvline(x=0, color='black')
    ax14.set_xlabel('Avg Match Quality')
    ax14.set_title('Top States', fontweight='bold')
    ax14.invert_yaxis()
    
    # 15. Time Trend
    ax15 = fig.add_subplot(4, 4, 15)
    time_trend = mega_df.groupby('fiscalyear').agg({
        'match_means': 'mean',
        'tdc1': 'mean',
        'tobinw': 'mean'
    })
    ax15.plot(time_trend.index, time_trend['match_means'], 'b-o', lw=2, label='Match')
    ax15_twin = ax15.twinx()
    ax15_twin.plot(time_trend.index, time_trend['tobinw'], 'g-s', lw=2, label="Tobin's Q")
    ax15.set_xlabel('Year')
    ax15.set_ylabel('Avg Match', color='blue')
    ax15_twin.set_ylabel("Tobin's Q", color='green')
    ax15.set_title('Match & Value Over Time', fontweight='bold')
    
    # 16. Feature Importance
    ax16 = fig.add_subplot(4, 4, 16)
    ml_features = ['Age', 'tenure', 'ivy', 'logatw', 'rdintw', 'exp_roa', 'leverage', 'boardindpw']
    ml_df = mega_df[ml_features + ['match_means']].dropna()
    if len(ml_df) > 500:
        rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
        rf.fit(ml_df[ml_features], ml_df['match_means'])
        importances = pd.DataFrame({
            'Feature': ml_features,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=True)
        ax16.barh(range(len(importances)), importances['Importance'], color='steelblue', edgecolor='black')
        ax16.set_yticks(range(len(importances)))
        ax16.set_yticklabels(importances['Feature'], fontsize=8)
        ax16.set_xlabel('Importance')
        ax16.set_title('🎯 Match Predictors', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Output/wrds_mega_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: Output/wrds_mega_analysis.png")
    
    # ================================================================
    # EXECUTIVE SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("🏆 MEGA WRDS ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Compute key stats
    vol_spread = (vol_by_match.loc['Q1', 'Volatility'] - vol_by_match.loc['Q4', 'Volatility']) / vol_by_match.loc['Q1', 'Volatility'] * 100 if 'volatility' in mega_df.columns else 0
    ret_spread = (vol_by_match.loc['Q5', 'Avg Return'] - vol_by_match.loc['Q1', 'Avg Return']) * 1200 if 'volatility' in mega_df.columns else 0
    mna_spread = mna_by_match.loc['Q5', 'Acquiror Rate'] - mna_by_match.loc['Q1', 'Acquiror Rate'] if 'n_deals' in mega_df.columns else 0
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                     🔌🔌🔌 MEGA WRDS FINDINGS 🔌🔌🔌                               ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                    ║
║  📦 TOTAL WRDS DATA PULLED: {total_obs:,} observations                           ║
║                                                                                    ║
║  📉 VOLATILITY (RISK)                                                            ║
║     Q4 volatility is {vol_spread:.0f}% LOWER than Q1                                    ║
║     Better matches = Lower risk                                                   ║
║                                                                                    ║
║  📈 RETURNS                                                                       ║
║     Q5 returns are {ret_spread:.1f}% per year HIGHER than Q1                           ║
║     Better matches = Higher returns                                               ║
║                                                                                    ║
║  🤝 M&A ACTIVITY                                                                  ║
║     Q5 acquiror rate is {mna_spread*100:.1f}pp higher than Q1                          ║
║     Better matches = More acquisitions                                            ║
║                                                                                    ║
║  💰 COMPENSATION                                                                  ║
║     Q5: ${comp_by_match.loc['Q5', 'TDC1']/1000:.1f}M  |  Q1: ${comp_by_match.loc['Q1', 'TDC1']/1000:.1f}M                               ║
║                                                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n🔌🔌🔌 MEGA WRDS ANALYSIS COMPLETE! 🔌🔌🔌")

if __name__ == "__main__":
    main()
