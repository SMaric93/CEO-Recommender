#!/usr/bin/env python3
"""
🏢🏢🏢 BOARDEX NETWORK ANALYSIS 🏢🏢🏢

Pulls BoardEx executive network data from WRDS, constructs research variables,
and analyzes the relationship with CEO-firm match quality.

Key Tables:
- boardex.na_wrds_individual_profile: Demographics (age, gender, network size)
- boardex.na_wrds_dir_profile_emp: Employment history (boards, roles)
- boardex.na_wrds_company_profile: Company details (for linking)
- boardex.na_wrds_annual_compensation: Executive pay data
"""
import pandas as pd
import numpy as np
import wrds
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
import os
import argparse

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
# BOARDEX PULL FUNCTIONS
# ================================================================

def pull_individual_profiles(db, limit=None):
    """Pull BoardEx individual demographic profiles."""
    print("\n📊 Pulling BoardEx Individual Profiles...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT directorid, 
           directorname,
           dob,
           gender,
           nationality,
           networksize,
           age
    FROM boardex.na_dir_profile_details
    WHERE dob IS NOT NULL
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} individual profiles")
        # Compute birth year
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['birth_year'] = df['dob'].dt.year
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_employment_history(db, limit=None):
    """Pull BoardEx employment/directorship history."""
    print("\n📊 Pulling BoardEx Employment History...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT directorid,
           companyid,
           companyname,
           rolename,
           datestartrole,
           dateendrole,
           ned,
           brdposition
    FROM boardex.na_dir_profile_emp
    WHERE datestartrole IS NOT NULL
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} employment records")
        df['datestartrole'] = pd.to_datetime(df['datestartrole'], errors='coerce')
        df['dateendrole'] = pd.to_datetime(df['dateendrole'], errors='coerce')
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_company_profiles(db, limit=None):
    """Pull BoardEx company profile for linking."""
    print("\n📊 Pulling BoardEx Company Profiles...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT boardid as companyid,
           boardname as companyname,
           isin,
           ticker,
           sector,
           index as stock_index
    FROM boardex.na_wrds_company_profile
    WHERE ticker IS NOT NULL
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} company profiles")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_compensation(db, limit=None):
    """Pull BoardEx annual compensation data from standard remuneration table."""
    print("\n📊 Pulling BoardEx Standard Remuneration...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT directorid,
           companyid,
           annualreportdate,
           basesalary as salary,
           bonus,
           totalannualremun as totalcompensation
    FROM boardex.na_dir_standard_remun
    WHERE annualreportdate IS NOT NULL
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} compensation records")
        df['annualreportdate'] = pd.to_datetime(df['annualreportdate'], errors='coerce')
        df['comp_year'] = df['annualreportdate'].dt.year
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_link_tables(db):
    """Pull linking tables for BoardEx → Compustat matching."""
    print("\n📊 Pulling Link Tables for BoardEx → Compustat...")
    
    # CRSP-Compustat link
    query1 = """
    SELECT DISTINCT gvkey, lpermno as permno, linkdt, linkenddt
    FROM crsp.ccmxpf_linktable
    WHERE linktype IN ('LU', 'LC')
      AND linkprim IN ('P', 'C')
    """
    
    # Ticker link from CRSP
    query2 = """
    SELECT DISTINCT permno, ncusip as cusip, ticker
    FROM crsp.msenames
    WHERE ticker IS NOT NULL
    """
    
    try:
        link1 = db.raw_sql(query1)
        link2 = db.raw_sql(query2)
        print(f"   CRSP-Compustat link: {len(link1):,} observations")
        print(f"   Ticker-CUSIP link: {len(link2):,} observations")
        return {'crsp_comp': link1, 'ticker_link': link2}
    except Exception as e:
        print(f"   Error: {e}")
        return None

# ================================================================
# VARIABLE CONSTRUCTION
# ================================================================

def construct_boardex_variables(individuals, employment, companies, compensation):
    """Construct derived research variables from BoardEx data."""
    print("\n🔧 Constructing BoardEx Variables...")
    
    if employment is None or individuals is None:
        print("   ❌ Missing required data for variable construction")
        return None
    
    # 1. Count prior board/executive positions per director
    emp = employment.copy()
    emp['is_board'] = emp['rolename'].str.lower().str.contains('director|board', na=False)
    emp['is_ceo'] = emp['rolename'].str.lower().str.contains('ceo|chief executive', na=False)
    emp['is_cfo'] = emp['rolename'].str.lower().str.contains('cfo|chief financial', na=False)
    emp['is_coo'] = emp['rolename'].str.lower().str.contains('coo|chief operating', na=False)
    
    # Tenure in days
    emp['tenure_days'] = (emp['dateendrole'] - emp['datestartrole']).dt.days
    emp['tenure_days'] = emp['tenure_days'].fillna(0).clip(lower=0)
    
    # Aggregate by director
    director_stats = emp.groupby('directorid').agg({
        'companyid': 'nunique',           # Unique companies
        'is_board': 'sum',                # Board seats count
        'is_ceo': 'sum',                  # CEO positions
        'is_cfo': 'sum',                  # CFO positions
        'is_coo': 'sum',                  # COO positions
        'tenure_days': 'mean',            # Avg tenure
        'rolename': 'count'               # Total roles
    }).reset_index()
    
    director_stats.columns = [
        'directorid', 'n_companies', 'n_board_seats', 
        'n_ceo_positions', 'n_cfo_positions', 'n_coo_positions',
        'avg_tenure_days', 'n_total_roles'
    ]
    
    # 2. Industry breadth (if sector available)
    if companies is not None and 'sector' in companies.columns:
        emp_with_sector = emp.merge(companies[['companyid', 'sector']], on='companyid', how='left')
        industry_breadth = emp_with_sector.groupby('directorid')['sector'].nunique().reset_index()
        industry_breadth.columns = ['directorid', 'industry_breadth']
        director_stats = director_stats.merge(industry_breadth, on='directorid', how='left')
    else:
        director_stats['industry_breadth'] = np.nan
    
    # 3. Merge with individual profile (network size, demographics)
    if individuals is not None:
        director_stats = director_stats.merge(
            individuals[['directorid', 'gender', 'nationality', 'networksize', 'birth_year']],
            on='directorid', how='left'
        )
    
    # 4. Compensation aggregates (if available)
    if compensation is not None and len(compensation) > 0:
        comp_agg = compensation.groupby('directorid').agg({
            'totalcompensation': ['mean', 'max'],
            'salary': 'mean',
            'bonus': 'mean',
            'equity': 'mean'
        }).reset_index()
        comp_agg.columns = ['directorid', 'avg_total_comp', 'max_total_comp', 
                           'avg_salary', 'avg_bonus', 'avg_equity']
        director_stats = director_stats.merge(comp_agg, on='directorid', how='left')
    
    # 5. Create career progression indicator
    director_stats['c_suite_exp'] = (
        (director_stats['n_ceo_positions'] > 0).astype(int) +
        (director_stats['n_cfo_positions'] > 0).astype(int) +
        (director_stats['n_coo_positions'] > 0).astype(int)
    )
    
    # 6. "Well-connected" flag (top quartile network size)
    if 'networksize' in director_stats.columns:
        network_q75 = director_stats['networksize'].quantile(0.75)
        director_stats['well_connected'] = (director_stats['networksize'] >= network_q75).fillna(False).astype(int)
    
    print(f"   ✅ Created variables for {len(director_stats):,} directors")
    print(f"   Variables: {list(director_stats.columns)}")
    
    return director_stats

def link_boardex_to_match(boardex_df, employment, companies, links, match_df):
    """Link BoardEx data to CEO match data via gvkey."""
    print("\n🔗 Linking BoardEx to CEO Match Data...")
    
    if links is None or employment is None:
        print("   ❌ Missing link tables or employment data")
        return None
    
    # Step 1: Get ticker → permno → gvkey mapping
    ticker_link = links.get('ticker_link', pd.DataFrame())
    crsp_link = links.get('crsp_comp', pd.DataFrame())
    
    if len(ticker_link) == 0 or len(crsp_link) == 0:
        print("   ❌ Empty link tables")
        return None
    
    # Merge ticker → permno → gvkey
    ticker_to_gvkey = ticker_link.merge(
        crsp_link[['permno', 'gvkey']].drop_duplicates(), 
        on='permno', how='left'
    )
    ticker_to_gvkey = ticker_to_gvkey.dropna(subset=['gvkey'])
    ticker_to_gvkey['gvkey'] = pd.to_numeric(ticker_to_gvkey['gvkey'], errors='coerce')
    ticker_to_gvkey = ticker_to_gvkey[['ticker', 'gvkey']].drop_duplicates()
    print(f"   Ticker → GVKEY map: {len(ticker_to_gvkey):,} unique mappings")
    
    # Step 2: Get BoardEx companyid → ticker
    if companies is not None and 'ticker' in companies.columns:
        company_to_ticker = companies[['companyid', 'ticker']].drop_duplicates()
        # Standardize ticker format
        company_to_ticker['ticker'] = company_to_ticker['ticker'].str.upper().str.strip()
        ticker_to_gvkey['ticker'] = ticker_to_gvkey['ticker'].str.upper().str.strip()
        
        # Link companyid → gvkey
        company_to_gvkey = company_to_ticker.merge(ticker_to_gvkey, on='ticker', how='inner')
        company_to_gvkey = company_to_gvkey[['companyid', 'gvkey']].drop_duplicates()
        print(f"   BoardEx CompanyID → GVKEY: {len(company_to_gvkey):,} mappings")
    else:
        print("   ❌ No ticker column in companies data")
        return None
    
    # Step 3: Find most recent CEO position per director per company
    ceo_emp = employment[employment['rolename'].str.lower().str.contains('ceo|chief executive', na=False)].copy()
    ceo_emp = ceo_emp.sort_values('datestartrole', ascending=False)
    ceo_emp = ceo_emp.drop_duplicates(subset=['directorid', 'companyid'], keep='first')
    ceo_emp['role_year'] = ceo_emp['datestartrole'].dt.year
    
    # Step 4: Merge director-company → gvkey
    ceo_with_gvkey = ceo_emp.merge(company_to_gvkey, on='companyid', how='inner')
    print(f"   CEO roles with GVKEY: {len(ceo_with_gvkey):,}")
    
    # Step 5: Merge with BoardEx stats
    ceo_with_stats = ceo_with_gvkey.merge(boardex_df, on='directorid', how='left')
    
    # Step 6: Merge with match data
    # Match on gvkey and approximate year
    match_df = match_df.copy()
    match_df['gvkey'] = pd.to_numeric(match_df['gvkey'], errors='coerce')
    
    # Prepare for merge
    ceo_with_stats = ceo_with_stats.rename(columns={'role_year': 'fiscalyear'})
    ceo_with_stats = ceo_with_stats[ceo_with_stats['fiscalyear'].notna()]
    ceo_with_stats['fiscalyear'] = ceo_with_stats['fiscalyear'].astype(int)
    
    # Merge
    merged = match_df.merge(
        ceo_with_stats.drop(columns=['companyid', 'companyname', 'rolename', 
                                      'datestartrole', 'dateendrole', 'seniority'], 
                           errors='ignore'),
        on=['gvkey', 'fiscalyear'], how='left'
    )
    
    boardex_cols = ['n_companies', 'n_board_seats', 'n_ceo_positions', 
                   'industry_breadth', 'networksize', 'c_suite_exp', 'well_connected']
    
    matched = merged['n_board_seats'].notna().sum()
    print(f"   ✅ Matched {matched:,} / {len(merged):,} observations ({100*matched/len(merged):.1f}%)")
    
    return merged

# ================================================================
# ANALYSIS FUNCTIONS
# ================================================================

def run_descriptive_analysis(df):
    """Descriptive analysis of BoardEx variables by match quintile."""
    print("\n📊 Descriptive Analysis by Match Quintile...")
    
    if 'match_means' not in df.columns:
        print("   ❌ match_means column not found")
        return None
    
    df = df.copy()
    df['match_q'] = pd.qcut(df['match_means'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    boardex_vars = ['n_board_seats', 'n_companies', 'n_ceo_positions', 
                   'industry_breadth', 'networksize', 'c_suite_exp', 'well_connected']
    
    available_vars = [v for v in boardex_vars if v in df.columns and df[v].notna().sum() > 100]
    
    if len(available_vars) == 0:
        print("   ❌ No BoardEx variables with sufficient data")
        return None
    
    stats_by_q = df.groupby('match_q', observed=False)[available_vars].mean().round(2)
    print("\n--- BOARDEX VARIABLES BY MATCH QUALITY ---")
    print(stats_by_q)
    
    # Test for monotonic relationship
    print("\n--- CORRELATION WITH MATCH QUALITY ---")
    for var in available_vars:
        valid = df.dropna(subset=['match_means', var])
        if len(valid) > 100:
            corr, pval = stats.pearsonr(valid['match_means'], valid[var])
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"   {var}: r = {corr:.3f} (p = {pval:.4f}) {sig}")
    
    return stats_by_q

def run_regression_analysis(df):
    """OLS regression: Match Quality ~ BoardEx Variables."""
    print("\n📈 Regression Analysis...")
    
    # Check data availability
    boardex_vars = ['n_board_seats', 'n_companies', 'networksize', 'industry_breadth', 'c_suite_exp']
    controls = ['logatw', 'exp_roa', 'Age']
    
    available = [v for v in boardex_vars if v in df.columns]
    available_controls = [c for c in controls if c in df.columns]
    
    if len(available) == 0:
        print("   ❌ No BoardEx variables available for regression")
        return None
    
    # Prepare data
    all_vars = ['match_means'] + available + available_controls
    reg_df = df[all_vars].dropna()
    
    if len(reg_df) < 100:
        print(f"   ❌ Insufficient observations: {len(reg_df)}")
        return None
    
    print(f"   Running on N = {len(reg_df):,} observations")
    
    # Model 1: BoardEx only
    formula1 = f"match_means ~ {' + '.join(available)}"
    model1 = ols(formula1, data=reg_df).fit(cov_type='HC1')
    
    print("\n--- MODEL 1: BOARDEX VARIABLES ONLY ---")
    print(f"   R² = {model1.rsquared:.4f}")
    for var in available:
        coef = model1.params[var]
        pval = model1.pvalues[var]
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"   {var}: β = {coef:.4f} (p = {pval:.4f}) {sig}")
    
    # Model 2: With controls
    if len(available_controls) > 0:
        formula2 = f"match_means ~ {' + '.join(available)} + {' + '.join(available_controls)}"
        model2 = ols(formula2, data=reg_df).fit(cov_type='HC1')
        
        print("\n--- MODEL 2: WITH CONTROLS ---")
        print(f"   R² = {model2.rsquared:.4f}")
        for var in available:
            coef = model2.params[var]
            pval = model2.pvalues[var]
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"   {var}: β = {coef:.4f} (p = {pval:.4f}) {sig}")
    
    return model1

def run_random_forest(df):
    """Random Forest for feature importance."""
    print("\n🌲 Random Forest Feature Importance...")
    
    boardex_vars = ['n_board_seats', 'n_companies', 'networksize', 
                   'industry_breadth', 'c_suite_exp', 'well_connected']
    other_vars = ['Age', 'logatw', 'exp_roa', 'leverage', 'boardindpw', 'rdintw']
    
    all_features = [v for v in boardex_vars + other_vars if v in df.columns]
    
    ml_df = df[all_features + ['match_means']].dropna()
    
    if len(ml_df) < 200:
        print(f"   ❌ Insufficient data: {len(ml_df)} obs")
        return None
    
    print(f"   Training on N = {len(ml_df):,}")
    
    X = ml_df[all_features]
    y = ml_df['match_means']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    train_r2 = rf.score(X_train, y_train)
    test_r2 = rf.score(X_test, y_test)
    print(f"   Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}")
    
    # Feature importance
    importances = pd.DataFrame({
        'Feature': all_features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n--- FEATURE IMPORTANCE ---")
    for _, row in importances.iterrows():
        star = "★" if row['Feature'] in boardex_vars else " "
        print(f"   {star} {row['Feature']}: {row['Importance']:.4f}")
    
    return rf, importances

# ================================================================
# VISUALIZATION
# ================================================================

def create_visualization_dashboard(df, output_dir='Output'):
    """Create comprehensive BoardEx analysis dashboard."""
    print("\n🎨 Creating Visualization Dashboard...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 16))
    
    df = df.copy()
    if 'match_means' in df.columns:
        df['match_q'] = pd.qcut(df['match_means'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    # 1. Network Size by Match Quintile
    ax1 = fig.add_subplot(3, 4, 1)
    if 'networksize' in df.columns:
        net_data = df.dropna(subset=['match_q', 'networksize'])
        if len(net_data) > 100:
            net_by_q = net_data.groupby('match_q', observed=False)['networksize'].mean()
            colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
            ax1.bar(range(5), net_by_q.values, color=colors, edgecolor='black')
            ax1.set_xticks(range(5))
            ax1.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax1.set_ylabel('Network Size')
            ax1.set_title('🔗 Network Size by Match', fontweight='bold')
    
    # 2. Board Seats by Match
    ax2 = fig.add_subplot(3, 4, 2)
    if 'n_board_seats' in df.columns:
        board_data = df.dropna(subset=['match_q', 'n_board_seats'])
        if len(board_data) > 100:
            board_by_q = board_data.groupby('match_q', observed=False)['n_board_seats'].mean()
            ax2.bar(range(5), board_by_q.values, color='steelblue', edgecolor='black')
            ax2.set_xticks(range(5))
            ax2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax2.set_ylabel('Avg Board Seats')
            ax2.set_title('🪑 Board Experience by Match', fontweight='bold')
    
    # 3. Prior CEO Positions
    ax3 = fig.add_subplot(3, 4, 3)
    if 'n_ceo_positions' in df.columns:
        ceo_data = df.dropna(subset=['match_q', 'n_ceo_positions'])
        if len(ceo_data) > 100:
            ceo_by_q = ceo_data.groupby('match_q', observed=False)['n_ceo_positions'].mean()
            ax3.bar(range(5), ceo_by_q.values, color='purple', edgecolor='black')
            ax3.set_xticks(range(5))
            ax3.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax3.set_ylabel('Prior CEO Positions')
            ax3.set_title('👔 Prior CEO Experience', fontweight='bold')
    
    # 4. Industry Breadth
    ax4 = fig.add_subplot(3, 4, 4)
    if 'industry_breadth' in df.columns:
        ind_data = df.dropna(subset=['match_q', 'industry_breadth'])
        if len(ind_data) > 100:
            ind_by_q = ind_data.groupby('match_q', observed=False)['industry_breadth'].mean()
            ax4.bar(range(5), ind_by_q.values, color='teal', edgecolor='black')
            ax4.set_xticks(range(5))
            ax4.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax4.set_ylabel('Unique Industries')
            ax4.set_title('🏭 Industry Experience', fontweight='bold')
    
    # 5. C-Suite Experience
    ax5 = fig.add_subplot(3, 4, 5)
    if 'c_suite_exp' in df.columns:
        cs_data = df.dropna(subset=['match_q', 'c_suite_exp'])
        if len(cs_data) > 100:
            cs_by_q = cs_data.groupby('match_q', observed=False)['c_suite_exp'].mean()
            ax5.bar(range(5), cs_by_q.values, color='gold', edgecolor='black')
            ax5.set_xticks(range(5))
            ax5.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax5.set_ylabel('C-Suite Roles (0-3)')
            ax5.set_title('⭐ C-Suite Experience', fontweight='bold')
    
    # 6. Well-Connected Rate
    ax6 = fig.add_subplot(3, 4, 6)
    if 'well_connected' in df.columns:
        wc_data = df.dropna(subset=['match_q', 'well_connected'])
        if len(wc_data) > 100:
            wc_by_q = wc_data.groupby('match_q', observed=False)['well_connected'].mean()
            ax6.bar(range(5), wc_by_q.values * 100, color='coral', edgecolor='black')
            ax6.set_xticks(range(5))
            ax6.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax6.set_ylabel('% Well Connected')
            ax6.set_title('🌐 Connection Rate', fontweight='bold')
    
    # 7. Network Size Distribution
    ax7 = fig.add_subplot(3, 4, 7)
    if 'networksize' in df.columns:
        valid = df['networksize'].dropna()
        if len(valid) > 100:
            ax7.hist(valid.clip(upper=valid.quantile(0.99)), bins=50, 
                    color='steelblue', edgecolor='black', alpha=0.7)
            ax7.axvline(x=valid.median(), color='red', linestyle='--', lw=2, label='Median')
            ax7.set_xlabel('Network Size')
            ax7.set_ylabel('Frequency')
            ax7.set_title('Network Size Distribution', fontweight='bold')
            ax7.legend()
    
    # 8. Network vs Match Scatter
    ax8 = fig.add_subplot(3, 4, 8)
    if 'networksize' in df.columns and 'match_means' in df.columns:
        valid = df.dropna(subset=['networksize', 'match_means'])
        if len(valid) > 100:
            sample = valid.sample(min(2000, len(valid)))
            ax8.scatter(sample['networksize'], sample['match_means'], alpha=0.3, s=15)
            z = np.polyfit(sample['networksize'], sample['match_means'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(sample['networksize'].min(), sample['networksize'].max(), 100)
            ax8.plot(x_line, p(x_line), 'r-', lw=2, label=f'β={z[0]:.4f}')
            ax8.set_xlabel('Network Size')
            ax8.set_ylabel('Match Quality')
            ax8.set_title('Network → Match', fontweight='bold')
            ax8.legend()
    
    # 9. Match Quality Distribution
    ax9 = fig.add_subplot(3, 4, 9)
    if 'match_means' in df.columns:
        ax9.hist(df['match_means'].dropna(), bins=50, color='steelblue', 
                edgecolor='black', alpha=0.7)
        ax9.axvline(x=df['match_means'].median(), color='red', linestyle='--', lw=2)
        ax9.set_xlabel('Match Quality')
        ax9.set_ylabel('Frequency')
        ax9.set_title('Match Quality Distribution', fontweight='bold')
    
    # 10. Correlation Heatmap
    ax10 = fig.add_subplot(3, 4, 10)
    boardex_cols = ['match_means', 'n_board_seats', 'n_companies', 'networksize',
                   'industry_breadth', 'c_suite_exp']
    available_cols = [c for c in boardex_cols if c in df.columns]
    if len(available_cols) > 2:
        corr_df = df[available_cols].dropna()
        if len(corr_df) > 50:
            corr_matrix = corr_df.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                       center=0, ax=ax10, vmin=-1, vmax=1)
            ax10.set_title('Correlation Matrix', fontweight='bold')
    
    # 11. Gender Distribution (if available)
    ax11 = fig.add_subplot(3, 4, 11)
    if 'gender' in df.columns:
        gender_data = df.dropna(subset=['gender', 'match_means'])
        if len(gender_data) > 100:
            gender_stats = gender_data.groupby('gender')['match_means'].agg(['mean', 'count'])
            colors = ['steelblue', 'coral'][:len(gender_stats)]
            ax11.bar(range(len(gender_stats)), gender_stats['mean'], color=colors, edgecolor='black')
            ax11.set_xticks(range(len(gender_stats)))
            ax11.set_xticklabels(gender_stats.index)
            ax11.set_ylabel('Avg Match Quality')
            ax11.set_title('Match by Gender', fontweight='bold')
            for i, (idx, row) in enumerate(gender_stats.iterrows()):
                ax11.annotate(f'n={int(row["count"])}', xy=(i, row['mean']), 
                            ha='center', va='bottom', fontsize=9)
    
    # 12. Summary Stats Box
    ax12 = fig.add_subplot(3, 4, 12)
    ax12.axis('off')
    stats_text = "📊 BOARDEX DATA SUMMARY\n" + "=" * 30 + "\n\n"
    
    if 'networksize' in df.columns:
        stats_text += f"Network Size:\n  Mean: {df['networksize'].mean():.1f}\n  Median: {df['networksize'].median():.1f}\n\n"
    if 'n_board_seats' in df.columns:
        stats_text += f"Board Seats:\n  Mean: {df['n_board_seats'].mean():.2f}\n\n"
    if 'n_ceo_positions' in df.columns:
        stats_text += f"Prior CEO Roles:\n  Mean: {df['n_ceo_positions'].mean():.2f}\n\n"
    
    boardex_match = df['n_board_seats'].notna().sum() if 'n_board_seats' in df.columns else 0
    stats_text += f"Observations:\n  Total: {len(df):,}\n  w/ BoardEx: {boardex_match:,}"
    
    ax12.text(0.1, 0.9, stats_text, transform=ax12.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'boardex_analysis_dashboard.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()
    
    return output_path

# ================================================================
# MAIN EXECUTION
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='BoardEx WRDS Analysis')
    parser.add_argument('--test', action='store_true', help='Test mode (limit 1000 rows)')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    args = parser.parse_args()
    
    print("🏢" * 40)
    print("   BOARDEX EXECUTIVE NETWORK ANALYSIS")
    print("🏢" * 40)
    
    config = Config()
    
    # ================================================================
    # LOAD BASE MATCH DATA
    # ================================================================
    print("\n📂 LOADING BASE MATCH DATA...")
    processor = DataProcessor(config)
    raw_df = processor.load_data()
    df_clean = processor.prepare_features(raw_df)
    print(f"  Match Quality Data: {len(df_clean):,} observations")
    
    if args.synthetic:
        print("\n⚠️ Synthetic mode: Generating fake BoardEx data...")
        # Create synthetic BoardEx variables
        np.random.seed(42)
        n = len(df_clean)
        df_clean['n_board_seats'] = np.random.poisson(3, n)
        df_clean['n_companies'] = np.random.poisson(4, n) + 1
        df_clean['n_ceo_positions'] = np.random.poisson(0.5, n)
        df_clean['networksize'] = np.random.lognormal(5, 1, n).astype(int)
        df_clean['industry_breadth'] = np.random.randint(1, 8, n)
        df_clean['c_suite_exp'] = np.random.randint(0, 4, n)
        df_clean['well_connected'] = (np.random.random(n) > 0.75).astype(int)
        
        mega_df = df_clean
        
    else:
        # ================================================================
        # CONNECT TO WRDS AND PULL BOARDEX DATA
        # ================================================================
        print("\n🔌 CONNECTING TO WRDS...")
        db = connect_wrds()
        
        if db is None:
            print("❌ Could not connect to WRDS. Run with --synthetic for testing.")
            return
        
        limit = 1000 if args.test else None
        
        print("\n" + "=" * 70)
        print("📥 PULLING BOARDEX DATA")
        print("=" * 70)
        
        # Pull all BoardEx tables
        individuals = pull_individual_profiles(db, limit)
        employment = pull_employment_history(db, limit)
        companies = pull_company_profiles(db, limit)
        compensation = pull_compensation(db, limit)
        links = pull_link_tables(db)
        
        db.close()
        print("\n✅ WRDS Connection Closed")
        
        # Save raw data
        os.makedirs('Output', exist_ok=True)
        if individuals is not None:
            individuals.to_parquet('Output/wrds_boardex_individuals.parquet')
        if employment is not None:
            employment.to_parquet('Output/wrds_boardex_employment.parquet')
        if companies is not None:
            companies.to_parquet('Output/wrds_boardex_companies.parquet')
        if compensation is not None:
            compensation.to_parquet('Output/wrds_boardex_compensation.parquet')
        
        # ================================================================
        # CONSTRUCT VARIABLES
        # ================================================================
        print("\n" + "=" * 70)
        print("🔧 CONSTRUCTING BOARDEX VARIABLES")
        print("=" * 70)
        
        boardex_stats = construct_boardex_variables(individuals, employment, companies, compensation)
        
        if boardex_stats is not None:
            boardex_stats.to_parquet('Output/wrds_boardex_director_stats.parquet')
            print(f"\n  Director-level stats: {len(boardex_stats):,} directors")
        
        # ================================================================
        # LINK TO MATCH DATA
        # ================================================================
        print("\n" + "=" * 70)
        print("🔗 LINKING TO MATCH DATA")
        print("=" * 70)
        
        if boardex_stats is not None:
            mega_df = link_boardex_to_match(boardex_stats, employment, companies, links, df_clean)
        else:
            mega_df = df_clean
    
    if mega_df is None:
        print("❌ Failed to create merged dataset")
        return
    
    # ================================================================
    # RUN ANALYSES
    # ================================================================
    print("\n" + "=" * 70)
    print("📊 RUNNING ANALYSES")
    print("=" * 70)
    
    # Descriptive
    stats_by_q = run_descriptive_analysis(mega_df)
    
    # Regression
    reg_model = run_regression_analysis(mega_df)
    
    # Random Forest
    rf_result = run_random_forest(mega_df)
    
    # ================================================================
    # VISUALIZATIONS
    # ================================================================
    print("\n" + "=" * 70)
    print("🎨 CREATING VISUALIZATIONS")
    print("=" * 70)
    
    create_visualization_dashboard(mega_df)
    
    # ================================================================
    # EXECUTIVE SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("🏆 BOARDEX ANALYSIS SUMMARY")
    print("=" * 70)
    
    n_matched = mega_df['n_board_seats'].notna().sum() if 'n_board_seats' in mega_df.columns else 0
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                     🏢🏢🏢 BOARDEX FINDINGS 🏢🏢🏢                               ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                   ║
║  📦 DATA COVERAGE                                                                 ║
║     Total Match Observations: {len(mega_df):,}                                   ║
║     With BoardEx Data: {n_matched:,} ({100*n_matched/len(mega_df):.1f}%)         ║
║                                                                                   ║""")
    
    if stats_by_q is not None and 'networksize' in stats_by_q.columns:
        q5_net = stats_by_q.loc['Q5', 'networksize'] if 'Q5' in stats_by_q.index else 0
        q1_net = stats_by_q.loc['Q1', 'networksize'] if 'Q1' in stats_by_q.index else 0
        net_diff = q5_net - q1_net
        print(f"""║  🔗 NETWORK SIZE                                                                  ║
║     Q5 avg: {q5_net:.1f} vs Q1 avg: {q1_net:.1f}                                ║
║     Difference: {net_diff:+.1f}                                                   ║
║                                                                                   ║""")
    
    if stats_by_q is not None and 'n_board_seats' in stats_by_q.columns:
        q5_board = stats_by_q.loc['Q5', 'n_board_seats'] if 'Q5' in stats_by_q.index else 0
        q1_board = stats_by_q.loc['Q1', 'n_board_seats'] if 'Q1' in stats_by_q.index else 0
        print(f"""║  🪑 BOARD EXPERIENCE                                                              ║
║     Q5 avg seats: {q5_board:.2f} vs Q1: {q1_board:.2f}                          ║
║                                                                                   ║""")
    
    print("""╚═══════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Save final dataset
    mega_df.to_parquet('Output/ceo_match_with_boardex.parquet')
    print("📦 Saved: Output/ceo_match_with_boardex.parquet")
    
    print("\n🏢🏢🏢 BOARDEX ANALYSIS COMPLETE! 🏢🏢🏢")

if __name__ == "__main__":
    main()
