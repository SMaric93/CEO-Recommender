#!/usr/bin/env python3
"""
👥👥👥 PEOPLE MODULE 👥👥👥

Comprehensive CEO People Data Integration:
1. BoardEx - Executive network, board experience, demographics
2. Capital IQ People Intelligence - Education, biography, compensation history

Uses WRDS People Link crosswalk for linking BoardEx directorid ↔ CIQ personid.

Key Output Variables (20+):
- Network: network_size, n_board_seats, n_interlocks
- Experience: n_ceo_roles, n_csuite_roles, industry_breadth
- Education: has_mba, has_phd, elite_school, stem_degree
- Demographics: age, gender, nationality
- Compensation: prior_total_comp, equity_orientation
- Career: internal_promotion, career_velocity, functional_background
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

# ================================================================
# WRDS CONNECTION
# ================================================================

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

def test_connection(db):
    """Test access to BoardEx and Capital IQ schemas."""
    print("\n🔍 Testing Schema Access...")
    
    # List available schemas
    try:
        # Test BoardEx
        test1 = db.raw_sql("SELECT COUNT(*) as n FROM boardex.na_dir_profile_details LIMIT 1")
        print(f"   ✅ BoardEx: Accessible")
    except Exception as e:
        print(f"   ❌ BoardEx: {e}")
    
    # Test Capital IQ People
    try:
        test2 = db.raw_sql("SELECT COUNT(*) as n FROM ciq.ciqprofessional LIMIT 1")
        print(f"   ✅ Capital IQ People: Accessible")
    except Exception as e:
        print(f"   ❌ Capital IQ People: {e}")
    
    # Test WRDS People Link
    try:
        test3 = db.raw_sql("SELECT COUNT(*) as n FROM wrdsapps.peoplelink LIMIT 1")
        print(f"   ✅ WRDS People Link: Accessible")
    except Exception as e:
        print(f"   ℹ️ WRDS People Link not found, will use alternative linking")

# ================================================================
# BOARDEX PULL FUNCTIONS
# ================================================================

def pull_boardex_profiles(db, limit=None):
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
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['birth_year'] = df['dob'].dt.year
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_boardex_employment(db, limit=None):
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

def pull_boardex_education(db, limit=None):
    """Pull BoardEx education data."""
    print("\n📊 Pulling BoardEx Education...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT directorid,
           qualification,
           institutionname
    FROM boardex.na_dir_profile_edu
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} education records")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_boardex_companies(db, limit=None):
    """Pull BoardEx company profiles for linking."""
    print("\n📊 Pulling BoardEx Company Profiles...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT boardid as companyid,
           boardname as companyname,
           isin,
           ticker,
           sector
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

# ================================================================
# CAPITAL IQ PEOPLE PULL FUNCTIONS
# ================================================================

def pull_ciq_professionals(db, limit=None):
    """Pull Capital IQ professional profiles."""
    print("\n📊 Pulling Capital IQ Professionals...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT personid,
           firstname,
           lastname,
           suffix,
           prefix
    FROM ciq.ciqprofessional
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} CIQ professionals")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_ciq_affiliations(db, limit=None):
    """Pull Capital IQ professional-company affiliations (roles)."""
    print("\n📊 Pulling Capital IQ Affiliations...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT a.personid,
           a.companyid,
           f.profunction as job_function,
           a.title,
           a.startdate,
           a.enddate
    FROM ciq.ciqprofessionalaffil a
    LEFT JOIN ciq.ciqprofunction f ON a.profunctionid = f.profunctionid
    WHERE a.startdate IS NOT NULL
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} CIQ affiliations")
        df['startdate'] = pd.to_datetime(df['startdate'], errors='coerce')
        df['enddate'] = pd.to_datetime(df['enddate'], errors='coerce')
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_ciq_education(db, limit=None):
    """Pull Capital IQ education data."""
    print("\n📊 Pulling Capital IQ Education...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT personid,
           degree,
           schoolname,
           major,
           graduationyear
    FROM ciq.ciqproeducation
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} CIQ education records")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_ciq_compensation(db, limit=None):
    """Pull Capital IQ compensation data."""
    print("\n📊 Pulling Capital IQ Compensation...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT personid,
           companyid,
           fiscalyear,
           salary,
           bonus,
           stockawards,
           optionawards,
           nonequityincentive,
           totalcompensation
    FROM ciq.ciqprocompensation
    WHERE fiscalyear IS NOT NULL
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} CIQ compensation records")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

# ================================================================
# WRDS CROSSWALK / LINKING
# ================================================================

def pull_wrds_people_link(db):
    """Pull WRDS People Link crosswalk (BoardEx directorid ↔ CIQ personid)."""
    print("\n📊 Pulling WRDS People Link Crosswalk...")
    
    # Try the direct people link table
    try:
        query = """
        SELECT boardex_directorid as directorid,
               ciq_personid as personid
        FROM wrdsapps.peoplelink
        WHERE boardex_directorid IS NOT NULL
          AND ciq_personid IS NOT NULL
        """
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} person links")
        return df
    except Exception as e:
        print(f"   People Link error: {e}")
    
    # Alternative: Try wrdsapps_link_boardex_ciq
    try:
        query = """
        SELECT directorid, personid
        FROM wrdsapps.link_boardex_ciq
        WHERE directorid IS NOT NULL AND personid IS NOT NULL
        """
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} person links (via link_boardex_ciq)")
        return df
    except Exception as e:
        print(f"   Alternative link error: {e}")
    
    return None

def pull_exec_boardex_link(db):
    """Pull ExecuComp-BoardEx crosswalk (execid ↔ directorid)."""
    print("\n📊 Pulling ExecuComp-BoardEx Link...")
    
    query = """
    SELECT execid, directorid, exec_fullname, directorname, score
    FROM wrdsapps.exec_boardex_link
    WHERE execid IS NOT NULL AND directorid IS NOT NULL
    """
    try:
        df = db.raw_sql(query)
        df['execid'] = pd.to_numeric(df['execid'], errors='coerce').astype('Int64')
        df['directorid'] = pd.to_numeric(df['directorid'], errors='coerce').astype('Int64')
        print(f"   Retrieved {len(df):,} exec-boardex links")
        print(f"   Unique execids: {df['execid'].nunique():,}")
        print(f"   Unique directorids: {df['directorid'].nunique():,}")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_execucomp_ceos(db):
    """Pull ExecuComp CEO records with gvkey-year mapping."""
    print("\n📊 Pulling ExecuComp CEO Data...")
    
    query = """
    SELECT execid, gvkey, year, ceoann, exec_fullname, titleann
    FROM execcomp.anncomp
    WHERE ceoann = 'CEO'
    AND year >= 2000
    """
    try:
        df = db.raw_sql(query)
        df['execid'] = pd.to_numeric(df['execid'], errors='coerce').astype('Int64')
        df['gvkey'] = pd.to_numeric(df['gvkey'], errors='coerce').astype('Int64')
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        print(f"   Retrieved {len(df):,} CEO-year records")
        print(f"   Unique CEOs (execid): {df['execid'].nunique():,}")
        print(f"   Unique firms (gvkey): {df['gvkey'].nunique():,}")
        print(f"   Year range: {df['year'].min()} - {df['year'].max()}")
        return df
    except Exception as e:
        print(f"   Error: {e}")
        return None

def pull_link_tables(db):
    """Pull standard CRSP-Compustat linking tables."""
    print("\n📊 Pulling Link Tables...")
    
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
    
    # CIQ company to GVKEY
    query3 = """
    SELECT companyid, gvkey
    FROM ciq.wrds_gvkey
    WHERE gvkey IS NOT NULL
    """
    
    try:
        link1 = db.raw_sql(query1)
        link2 = db.raw_sql(query2)
        print(f"   CRSP-Compustat link: {len(link1):,}")
        print(f"   Ticker-CUSIP link: {len(link2):,}")
    except Exception as e:
        print(f"   Standard link error: {e}")
        link1 = pd.DataFrame()
        link2 = pd.DataFrame()
    
    try:
        link3 = db.raw_sql(query3)
        print(f"   CIQ-GVKEY link: {len(link3):,}")
    except Exception as e:
        print(f"   CIQ-GVKEY link error: {e}")
        link3 = pd.DataFrame()
    
    return {
        'crsp_comp': link1,
        'ticker_link': link2,
        'ciq_gvkey': link3
    }

# ================================================================
# VARIABLE CONSTRUCTION
# ================================================================

def construct_boardex_variables(profiles, employment, education, companies):
    """Construct BoardEx-derived variables per director (50+ CEO-specific variables)."""
    print("\n🔧 Constructing BoardEx Variables (Extended)...")
    
    if employment is None or profiles is None:
        print("   ❌ Missing required BoardEx data")
        return None
    
    emp = employment.copy()
    
    # ==========================================================
    # BASIC ROLE FLAGS
    # ==========================================================
    emp['is_board'] = emp['rolename'].str.lower().str.contains('director|board', na=False)
    emp['is_ceo'] = emp['rolename'].str.lower().str.contains('ceo|chief executive', na=False)
    emp['is_cfo'] = emp['rolename'].str.lower().str.contains('cfo|chief financial', na=False)
    emp['is_coo'] = emp['rolename'].str.lower().str.contains('coo|chief operating', na=False)
    emp['is_cto'] = emp['rolename'].str.lower().str.contains('cto|chief technology|chief tech', na=False)
    emp['is_cmo'] = emp['rolename'].str.lower().str.contains('cmo|chief marketing', na=False)
    emp['is_cio'] = emp['rolename'].str.lower().str.contains('cio|chief information', na=False)
    emp['is_chro'] = emp['rolename'].str.lower().str.contains('chro|chief human|chief people', na=False)
    emp['is_clo'] = emp['rolename'].str.lower().str.contains('clo|general counsel|chief legal', na=False)
    emp['is_csuite'] = emp['rolename'].str.lower().str.contains('chief|ceo|cfo|coo|cto|cmo|cio', na=False)
    emp['is_president'] = emp['rolename'].str.lower().str.contains('president', na=False)
    emp['is_chairman'] = emp['rolename'].str.lower().str.contains('chairman|chair of', na=False)
    emp['is_founder'] = emp['rolename'].str.lower().str.contains('founder|co-founder', na=False)
    emp['is_vp'] = emp['rolename'].str.lower().str.contains('vice president|vp|svp|evp', na=False)
    
    # Functional background flags
    emp['is_finance_role'] = emp['rolename'].str.lower().str.contains('finance|financial|treasury|controller|accounting', na=False)
    emp['is_ops_role'] = emp['rolename'].str.lower().str.contains('operations|operating|manufacturing|supply chain|logistics', na=False)
    emp['is_sales_role'] = emp['rolename'].str.lower().str.contains('sales|commercial|business development|revenue', na=False)
    emp['is_marketing_role'] = emp['rolename'].str.lower().str.contains('marketing|brand|advertising|communications', na=False)
    emp['is_tech_role'] = emp['rolename'].str.lower().str.contains('technology|engineering|r&d|research|development|product', na=False)
    emp['is_legal_role'] = emp['rolename'].str.lower().str.contains('legal|counsel|compliance|regulatory', na=False)
    emp['is_hr_role'] = emp['rolename'].str.lower().str.contains('human|hr|people|talent|personnel', na=False)
    emp['is_strategy_role'] = emp['rolename'].str.lower().str.contains('strategy|strategic|planning|corp dev', na=False)
    
    # NED (Non-Executive Director) flag
    emp['is_ned'] = emp['ned'].fillna(0).astype(bool) | emp['rolename'].str.lower().str.contains('non-executive|non executive', na=False)
    
    # ==========================================================
    # TENURE CALCULATIONS
    # ==========================================================
    emp['tenure_days'] = (emp['dateendrole'] - emp['datestartrole']).dt.days
    emp['tenure_days'] = emp['tenure_days'].fillna(0).clip(lower=0)
    emp['tenure_years'] = emp['tenure_days'] / 365.25
    emp['start_year'] = emp['datestartrole'].dt.year
    emp['end_year'] = emp['dateendrole'].dt.year
    
    # ==========================================================
    # AGGREGATE STATISTICS PER DIRECTOR
    # ==========================================================
    agg_dict = {
        'companyid': 'nunique',
        'is_board': 'sum',
        'is_ceo': 'sum',
        'is_cfo': 'sum',
        'is_coo': 'sum',
        'is_cto': 'sum',
        'is_cmo': 'sum',
        'is_cio': 'sum',
        'is_chro': 'sum',
        'is_clo': 'sum',
        'is_csuite': 'sum',
        'is_president': 'sum',
        'is_chairman': 'sum',
        'is_founder': 'sum',
        'is_vp': 'sum',
        'is_ned': 'sum',
        'is_finance_role': 'sum',
        'is_ops_role': 'sum',
        'is_sales_role': 'sum',
        'is_marketing_role': 'sum',
        'is_tech_role': 'sum',
        'is_legal_role': 'sum',
        'is_hr_role': 'sum',
        'is_strategy_role': 'sum',
        'tenure_days': ['mean', 'max', 'sum'],
        'tenure_years': 'mean',
        'start_year': 'min',
        'rolename': 'count'
    }
    
    stats = emp.groupby('directorid').agg(agg_dict).reset_index()
    stats.columns = [
        'directorid', 'n_companies', 'n_board_seats',
        'n_ceo_roles', 'n_cfo_roles', 'n_coo_roles', 'n_cto_roles', 'n_cmo_roles',
        'n_cio_roles', 'n_chro_roles', 'n_clo_roles', 'n_csuite_roles',
        'n_president_roles', 'n_chairman_roles', 'n_founder_roles', 'n_vp_roles', 'n_ned_roles',
        'n_finance_roles', 'n_ops_roles', 'n_sales_roles', 'n_marketing_roles',
        'n_tech_roles', 'n_legal_roles', 'n_hr_roles', 'n_strategy_roles',
        'avg_tenure_days', 'max_tenure_days', 'total_tenure_days',
        'avg_tenure_years', 'career_start_year', 'n_total_roles'
    ]
    
    # ==========================================================
    # DERIVED CEO-SPECIFIC VARIABLES
    # ==========================================================
    
    # Career span
    current_year = 2024
    stats['career_length_years'] = current_year - stats['career_start_year']
    stats['career_length_years'] = stats['career_length_years'].clip(lower=0)
    
    # C-suite breadth (how many different C-suite roles)
    stats['csuite_breadth'] = (
        (stats['n_ceo_roles'] > 0).astype(int) +
        (stats['n_cfo_roles'] > 0).astype(int) +
        (stats['n_coo_roles'] > 0).astype(int) +
        (stats['n_cto_roles'] > 0).astype(int) +
        (stats['n_cmo_roles'] > 0).astype(int) +
        (stats['n_cio_roles'] > 0).astype(int)
    )
    
    # Functional expertise flags (dominant background)
    stats['finance_background'] = (stats['n_finance_roles'] > 0) | (stats['n_cfo_roles'] > 0)
    stats['ops_background'] = (stats['n_ops_roles'] > 0) | (stats['n_coo_roles'] > 0)
    stats['tech_background'] = (stats['n_tech_roles'] > 0) | (stats['n_cto_roles'] > 0)
    stats['marketing_background'] = (stats['n_marketing_roles'] > 0) | (stats['n_cmo_roles'] > 0)
    stats['sales_background'] = stats['n_sales_roles'] > 0
    stats['legal_background'] = (stats['n_legal_roles'] > 0) | (stats['n_clo_roles'] > 0)
    stats['hr_background'] = (stats['n_hr_roles'] > 0) | (stats['n_chro_roles'] > 0)
    stats['strategy_background'] = stats['n_strategy_roles'] > 0
    
    # Primary functional background (most roles)
    func_cols = ['n_finance_roles', 'n_ops_roles', 'n_sales_roles', 'n_marketing_roles', 
                 'n_tech_roles', 'n_legal_roles', 'n_hr_roles', 'n_strategy_roles']
    func_names = ['Finance', 'Operations', 'Sales', 'Marketing', 'Technology', 'Legal', 'HR', 'Strategy']
    stats['primary_function'] = stats[func_cols].idxmax(axis=1).map(dict(zip(func_cols, func_names)))
    stats.loc[stats[func_cols].sum(axis=1) == 0, 'primary_function'] = 'General'
    
    # Serial CEO flag
    stats['serial_ceo'] = (stats['n_ceo_roles'] >= 2).astype(int)
    
    # Career velocity: time from career start to first C-suite role
    csuite_emp = emp[emp['is_csuite']].copy()
    if len(csuite_emp) > 0:
        first_csuite = csuite_emp.groupby('directorid')['start_year'].min().reset_index()
        first_csuite.columns = ['directorid', 'first_csuite_year']
        stats = stats.merge(first_csuite, on='directorid', how='left')
        stats['years_to_csuite'] = stats['first_csuite_year'] - stats['career_start_year']
        stats['years_to_csuite'] = stats['years_to_csuite'].clip(lower=0)
    else:
        stats['first_csuite_year'] = np.nan
        stats['years_to_csuite'] = np.nan
    
    # First CEO year and time to CEO
    ceo_emp = emp[emp['is_ceo']].copy()
    if len(ceo_emp) > 0:
        first_ceo = ceo_emp.groupby('directorid')['start_year'].min().reset_index()
        first_ceo.columns = ['directorid', 'first_ceo_year']
        stats = stats.merge(first_ceo, on='directorid', how='left')
        stats['years_to_ceo'] = stats['first_ceo_year'] - stats['career_start_year']
        stats['years_to_ceo'] = stats['years_to_ceo'].clip(lower=0)
        stats['age_first_ceo'] = stats['first_ceo_year'] - stats.get('birth_year', np.nan)
    else:
        stats['first_ceo_year'] = np.nan
        stats['years_to_ceo'] = np.nan
        stats['age_first_ceo'] = np.nan
    
    # Leadership experience score (weighted roles)
    stats['leadership_score'] = (
        stats['n_ceo_roles'] * 5 +
        stats['n_president_roles'] * 4 +
        stats['n_coo_roles'] * 3 +
        stats['n_cfo_roles'] * 3 +
        stats['n_chairman_roles'] * 3 +
        stats['n_csuite_roles'] * 2 +
        stats['n_vp_roles'] * 1
    )
    
    # Board experience depth
    stats['heavy_board_experience'] = (stats['n_board_seats'] >= 3).astype(int)
    stats['board_to_executive_ratio'] = stats['n_board_seats'] / (stats['n_csuite_roles'] + 1)
    
    # Tenure patterns
    stats['avg_tenure_years'] = stats['avg_tenure_days'] / 365.25
    stats['max_tenure_years'] = stats['max_tenure_days'] / 365.25
    stats['job_hopper'] = (stats['avg_tenure_years'] < 2).astype(int)
    stats['long_tenure'] = (stats['avg_tenure_years'] > 5).astype(int)
    
    # Company diversity
    stats['companies_per_role'] = stats['n_companies'] / (stats['n_total_roles'] + 1)
    stats['multi_company'] = (stats['n_companies'] >= 3).astype(int)
    
    # Founder vs professional manager
    stats['is_founder'] = (stats['n_founder_roles'] > 0).astype(int)
    stats['professional_manager'] = ((stats['n_founder_roles'] == 0) & (stats['n_ceo_roles'] > 0)).astype(int)
    
    # ==========================================================
    # INDUSTRY BREADTH (if companies data available)
    # ==========================================================
    if companies is not None and 'sector' in companies.columns:
        emp_sector = emp.merge(companies[['companyid', 'sector']], on='companyid', how='left')
        ind_breadth = emp_sector.groupby('directorid')['sector'].nunique().reset_index()
        ind_breadth.columns = ['directorid', 'industry_breadth']
        stats = stats.merge(ind_breadth, on='directorid', how='left')
        
        # Industry specialist vs generalist
        stats['industry_specialist'] = (stats['industry_breadth'] == 1).astype(int)
        stats['industry_generalist'] = (stats['industry_breadth'] >= 3).astype(int)
    else:
        stats['industry_breadth'] = np.nan
        stats['industry_specialist'] = np.nan
        stats['industry_generalist'] = np.nan
    
    # ==========================================================
    # MERGE PROFILES (network, demographics)
    # ==========================================================
    if profiles is not None:
        stats = stats.merge(
            profiles[['directorid', 'gender', 'nationality', 'networksize', 'birth_year', 'age']],
            on='directorid', how='left'
        )
        
        # Convert to numeric types
        stats['age'] = pd.to_numeric(stats['age'], errors='coerce')
        stats['networksize'] = pd.to_numeric(stats['networksize'], errors='coerce')
        
        # Network-derived variables
        if 'networksize' in stats.columns:
            q75 = stats['networksize'].quantile(0.75)
            q90 = stats['networksize'].quantile(0.90)
            q50 = stats['networksize'].quantile(0.50)
            
            stats['well_connected'] = (stats['networksize'] >= q75).fillna(False).astype(int)
            stats['super_connected'] = (stats['networksize'] >= q90).fillna(False).astype(int)
            stats['low_connected'] = (stats['networksize'] < q50).fillna(False).astype(int)
            stats['network_quintile'] = pd.qcut(stats['networksize'].fillna(0), 5, labels=False, duplicates='drop')
            
            # Network per year of career
            stats['network_per_career_year'] = stats['networksize'] / (stats['career_length_years'] + 1)
        
        # Age-derived variables
        if 'age' in stats.columns:
            stats['young_ceo'] = (stats['age'] < 45).fillna(False).astype(int)
            stats['experienced_ceo'] = (stats['age'] >= 55).fillna(False).astype(int)
            stats['prime_age'] = ((stats['age'] >= 45) & (stats['age'] < 55)).fillna(False).astype(int)
        
        # Gender
        if 'gender' in stats.columns:
            stats['is_female'] = (stats['gender'].str.lower() == 'female').fillna(False).astype(int)
    
    # ==========================================================
    # EDUCATION VARIABLES (from BoardEx)
    # ==========================================================
    if education is not None and len(education) > 0:
        edu = education.copy()
        edu['qualification'] = edu['qualification'].str.lower().fillna('')
        edu['institutionname'] = edu['institutionname'].str.lower().fillna('')
        
        # Degree flags
        edu['has_mba_bx'] = edu['qualification'].str.contains('mba|master.*business', na=False)
        edu['has_phd_bx'] = edu['qualification'].str.contains('phd|doctorate|d.phil', na=False)
        edu['has_jd_bx'] = edu['qualification'].str.contains('j.d.|juris|law degree|llb|llm', na=False)
        edu['has_cpa_bx'] = edu['qualification'].str.contains('cpa|chartered accountant|aca|acca', na=False)
        edu['has_cfa_bx'] = edu['qualification'].str.contains('cfa|chartered financial analyst', na=False)
        
        # Elite schools (expanded list)
        ivy_pattern = 'harvard|yale|princeton|columbia|brown|dartmouth|cornell|penn|upenn'
        top_bschool = 'wharton|hbs|stanford gsb|kellogg|booth|sloan|haas|tuck|columbia business|insead'
        top_global = 'oxford|cambridge|lse|imperial|mit|stanford|berkeley|chicago|northwestern'
        
        edu['ivy_league'] = edu['institutionname'].str.contains(ivy_pattern, na=False)
        edu['top_bschool'] = edu['institutionname'].str.contains(top_bschool, na=False)
        edu['elite_school_bx'] = edu['institutionname'].str.contains(f'{ivy_pattern}|{top_global}', na=False)
        
        # STEM majors
        stem_pattern = 'engineer|computer|math|physics|chemistry|biology|science|technology'
        edu['stem_edu_bx'] = edu['qualification'].str.contains(stem_pattern, na=False)
        
        # Aggregate education
        edu_agg = edu.groupby('directorid').agg({
            'has_mba_bx': 'max',
            'has_phd_bx': 'max',
            'has_jd_bx': 'max',
            'has_cpa_bx': 'max',
            'has_cfa_bx': 'max',
            'ivy_league': 'max',
            'top_bschool': 'max',
            'elite_school_bx': 'max',
            'stem_edu_bx': 'max',
            'qualification': 'count'
        }).reset_index()
        edu_agg.columns = [
            'directorid', 'has_mba_bx', 'has_phd_bx', 'has_jd_bx', 
            'has_cpa_bx', 'has_cfa_bx', 'ivy_league', 'top_bschool',
            'elite_school_bx', 'stem_edu_bx', 'n_degrees_bx'
        ]
        
        stats = stats.merge(edu_agg, on='directorid', how='left')
        
        # Education quality score
        stats['education_quality_score'] = (
            stats['has_phd_bx'].fillna(0).astype(int) * 3 +
            stats['has_mba_bx'].fillna(0).astype(int) * 2 +
            stats['elite_school_bx'].fillna(0).astype(int) * 2 +
            stats['top_bschool'].fillna(0).astype(int) * 2 +
            stats['ivy_league'].fillna(0).astype(int) * 1
        )
    
    # ==========================================================
    # COMPOSITE CEO QUALITY INDICATORS
    # ==========================================================
    
    # C-suite experience flag (0-6 scale)
    stats['c_suite_exp'] = stats['csuite_breadth']
    
    # Overall CEO quality score
    stats['ceo_quality_score'] = (
        stats['n_ceo_roles'].clip(upper=5) * 3 +
        stats['csuite_breadth'] * 2 +
        stats['n_board_seats'].clip(upper=10) * 1 +
        stats.get('education_quality_score', 0) +
        (stats['networksize'].fillna(0) / 100).clip(upper=10)
    )
    
    print(f"   ✅ BoardEx variables for {len(stats):,} directors ({len(stats.columns)} variables)")
    return stats

def construct_ciq_variables(professionals, affiliations, education, compensation):
    """Construct Capital IQ-derived variables per person."""
    print("\n🔧 Constructing Capital IQ Variables...")
    
    if professionals is None:
        print("   ❌ Missing CIQ professionals data")
        return None
    
    stats = professionals[['personid']].drop_duplicates().copy()
    
    # Affiliation-based variables
    if affiliations is not None and len(affiliations) > 0:
        aff = affiliations.copy()
        aff['job_function'] = aff['job_function'].str.lower().fillna('')
        aff['title'] = aff['title'].str.lower().fillna('')
        
        # Role flags
        aff['is_ceo_ciq'] = aff['job_function'].str.contains('chief executive|ceo', na=False)
        aff['is_cfo_ciq'] = aff['job_function'].str.contains('chief financial|cfo', na=False)
        aff['is_csuite_ciq'] = aff['job_function'].str.contains('chief|^c[a-z]o$', na=False)
        
        # Tenure
        aff['tenure_days_ciq'] = (aff['enddate'] - aff['startdate']).dt.days
        aff['tenure_days_ciq'] = aff['tenure_days_ciq'].fillna(0).clip(lower=0)
        
        # Career velocity: year of first C-suite role
        csuite_aff = aff[aff['is_csuite_ciq']]
        if len(csuite_aff) > 0:
            first_csuite = csuite_aff.groupby('personid')['startdate'].min().reset_index()
            first_csuite['first_csuite_year'] = first_csuite['startdate'].dt.year
            first_csuite = first_csuite[['personid', 'first_csuite_year']]
            stats = stats.merge(first_csuite, on='personid', how='left')
        
        aff_agg = aff.groupby('personid').agg({
            'companyid': 'nunique',
            'is_ceo_ciq': 'sum',
            'is_cfo_ciq': 'sum',
            'is_csuite_ciq': 'sum',
            'tenure_days_ciq': 'mean',
            'title': 'count'
        }).reset_index()
        aff_agg.columns = [
            'personid', 'n_companies_ciq', 'n_ceo_ciq', 'n_cfo_ciq',
            'n_csuite_ciq', 'avg_tenure_ciq', 'n_roles_ciq'
        ]
        stats = stats.merge(aff_agg, on='personid', how='left')
    
    # Education variables from CIQ
    if education is not None and len(education) > 0:
        edu = education.copy()
        edu['degree'] = edu['degree'].str.lower().fillna('')
        edu['schoolname'] = edu['schoolname'].str.lower().fillna('')
        edu['major'] = edu['major'].str.lower().fillna('')
        
        # Degree flags
        edu['has_mba_ciq'] = edu['degree'].str.contains('mba|master.*business', na=False)
        edu['has_phd_ciq'] = edu['degree'].str.contains('phd|doctorate|d.phil', na=False)
        edu['has_jd_ciq'] = edu['degree'].str.contains('j.d.|juris doctor|law', na=False)
        
        # STEM degrees
        stem_pattern = 'engineer|computer|math|physics|chemistry|biology|science'
        edu['stem_degree'] = (
            edu['major'].str.contains(stem_pattern, na=False) |
            edu['degree'].str.contains(stem_pattern, na=False)
        )
        
        # Elite schools
        elite_pattern = 'harvard|stanford|wharton|mit|yale|princeton|columbia|chicago|berkeley|northwestern|oxford|cambridge'
        edu['elite_school_ciq'] = edu['schoolname'].str.contains(elite_pattern, na=False)
        
        # Max education level
        def edu_level(d):
            if 'phd' in d or 'doctorate' in d: return 5
            if 'mba' in d or 'mba' in d: return 4
            if 'master' in d or 'ms' in d or 'ma' in d: return 3
            if 'bachelor' in d or 'bs' in d or 'ba' in d: return 2
            return 1
        edu['edu_level'] = edu['degree'].apply(edu_level)
        
        edu_agg = edu.groupby('personid').agg({
            'has_mba_ciq': 'max',
            'has_phd_ciq': 'max',
            'has_jd_ciq': 'max',
            'stem_degree': 'max',
            'elite_school_ciq': 'max',
            'edu_level': 'max',
            'degree': 'count'
        }).reset_index()
        edu_agg.columns = [
            'personid', 'has_mba_ciq', 'has_phd_ciq', 'has_jd_ciq',
            'stem_degree', 'elite_school_ciq', 'max_education', 'n_degrees_ciq'
        ]
        stats = stats.merge(edu_agg, on='personid', how='left')
    
    # Compensation variables
    if compensation is not None and len(compensation) > 0:
        comp = compensation.copy()
        comp['equity_comp'] = comp['stockawards'].fillna(0) + comp['optionawards'].fillna(0)
        comp['equity_ratio'] = comp['equity_comp'] / comp['totalcompensation'].replace(0, np.nan)
        
        comp_agg = comp.groupby('personid').agg({
            'totalcompensation': ['mean', 'max'],
            'salary': 'mean',
            'equity_ratio': 'mean'
        }).reset_index()
        comp_agg.columns = [
            'personid', 'avg_total_comp_ciq', 'max_total_comp_ciq',
            'avg_salary_ciq', 'equity_orientation'
        ]
        stats = stats.merge(comp_agg, on='personid', how='left')
    
    print(f"   ✅ CIQ variables for {len(stats):,} persons")
    return stats

def merge_boardex_ciq(boardex_vars, ciq_vars, people_link):
    """Merge BoardEx and CIQ variables using WRDS People Link."""
    print("\n🔗 Merging BoardEx + CIQ via People Link...")
    
    if people_link is None or len(people_link) == 0:
        print("   ⚠️ No people link available, returning BoardEx only")
        return boardex_vars
    
    if boardex_vars is None:
        print("   ❌ No BoardEx data")
        return None
    
    # Merge BoardEx with link
    merged = boardex_vars.merge(people_link, on='directorid', how='left')
    
    # Merge with CIQ where linked
    if ciq_vars is not None:
        merged = merged.merge(ciq_vars, on='personid', how='left')
        linked = merged['personid'].notna().sum()
        print(f"   ✅ {linked:,} / {len(merged):,} directors linked to CIQ ({100*linked/len(merged):.1f}%)")
    
    # Combine overlapping variables (prefer CIQ, fallback to BoardEx)
    if 'has_mba_ciq' in merged.columns and 'has_mba_bx' in merged.columns:
        merged['has_mba'] = merged['has_mba_ciq'].fillna(merged['has_mba_bx'])
    elif 'has_mba_ciq' in merged.columns:
        merged['has_mba'] = merged['has_mba_ciq']
    elif 'has_mba_bx' in merged.columns:
        merged['has_mba'] = merged['has_mba_bx']
    
    if 'has_phd_ciq' in merged.columns and 'has_phd_bx' in merged.columns:
        merged['has_phd'] = merged['has_phd_ciq'].fillna(merged['has_phd_bx'])
    elif 'has_phd_ciq' in merged.columns:
        merged['has_phd'] = merged['has_phd_ciq']
    elif 'has_phd_bx' in merged.columns:
        merged['has_phd'] = merged['has_phd_bx']
    
    if 'elite_school_ciq' in merged.columns and 'elite_school_bx' in merged.columns:
        merged['elite_school'] = merged['elite_school_ciq'].fillna(merged['elite_school_bx'])
    elif 'elite_school_ciq' in merged.columns:
        merged['elite_school'] = merged['elite_school_ciq']
    elif 'elite_school_bx' in merged.columns:
        merged['elite_school'] = merged['elite_school_bx']
    
    print(f"   Final merged: {len(merged):,} directors with {len(merged.columns)} variables")
    return merged

# ================================================================
# LINK TO MATCH DATA
# ================================================================

def link_people_via_execucomp(people_df, exec_boardex_link, execucomp_ceos, match_df):
    """
    Link people data to CEO-firm match data via ExecuComp.
    
    Chain: match_df (gvkey, fiscalyear) → ExecuComp (execid) → BoardEx (directorid) → people_df
    """
    print("\n🔗 Linking People Data via ExecuComp (gvkey-year method)...")
    
    if exec_boardex_link is None or execucomp_ceos is None or people_df is None:
        print("   ❌ Missing required data for ExecuComp linking")
        return None
    
    # Step 1: Create gvkey-year → execid mapping from ExecuComp
    print(f"   ExecuComp CEOs: {len(execucomp_ceos):,} records")
    exec_ceos = execucomp_ceos[['execid', 'gvkey', 'year', 'exec_fullname']].copy()
    exec_ceos = exec_ceos.dropna(subset=['execid', 'gvkey', 'year'])
    exec_ceos['gvkey'] = exec_ceos['gvkey'].astype('Int64')
    exec_ceos['year'] = exec_ceos['year'].astype('Int64')
    
    # Step 2: Add directorid via exec_boardex_link
    print(f"   Exec-BoardEx links: {len(exec_boardex_link):,} records")
    exec_bx = exec_boardex_link[['execid', 'directorid']].drop_duplicates()
    exec_ceos_with_bx = exec_ceos.merge(exec_bx, on='execid', how='inner')
    print(f"   CEOs with BoardEx link: {len(exec_ceos_with_bx):,}")
    
    # Step 3: Add people variables via directorid
    exec_ceos_with_vars = exec_ceos_with_bx.merge(people_df, on='directorid', how='left')
    print(f"   CEOs with people variables: {len(exec_ceos_with_vars):,}")
    
    # Step 4: Prepare match data
    match_df = match_df.copy()
    match_df['gvkey'] = pd.to_numeric(match_df['gvkey'], errors='coerce').astype('Int64')
    match_df['fiscalyear'] = pd.to_numeric(match_df['fiscalyear'], errors='coerce').astype('Int64')
    
    # Step 5: Merge with match data on gvkey-year
    exec_ceos_with_vars = exec_ceos_with_vars.rename(columns={'year': 'fiscalyear'})
    
    # Select columns for merge (drop duplicates on gvkey-fiscalyear)
    drop_cols = ['execid', 'exec_fullname']
    merge_cols = [c for c in exec_ceos_with_vars.columns if c not in drop_cols]
    
    # Keep only one CEO per gvkey-year (most recent link)
    merge_df = exec_ceos_with_vars[merge_cols].drop_duplicates(
        subset=['gvkey', 'fiscalyear'], keep='first'
    )
    
    merged = match_df.merge(merge_df, on=['gvkey', 'fiscalyear'], how='left')
    
    # Report match rate
    matched = merged['directorid'].notna().sum()
    print(f"   ✅ ExecuComp Linking: {matched:,} / {len(merged):,} ({100*matched/len(merged):.1f}%)")
    
    return merged

def link_people_to_match(people_df, employment, companies, links, match_df):
    """Link people data to CEO-firm match data via gvkey (ticker-based fallback)."""
    print("\n🔗 Linking People Data to Match Dataset (ticker method)...")
    
    if links is None or employment is None:
        print("   ❌ Missing link tables or employment")
        return None
    
    # Build BoardEx companyid → gvkey mapping
    ticker_link = links.get('ticker_link', pd.DataFrame())
    crsp_link = links.get('crsp_comp', pd.DataFrame())
    
    if len(ticker_link) > 0 and len(crsp_link) > 0:
        ticker_to_gvkey = ticker_link.merge(
            crsp_link[['permno', 'gvkey']].drop_duplicates(),
            on='permno', how='left'
        ).dropna(subset=['gvkey'])
        ticker_to_gvkey['gvkey'] = pd.to_numeric(ticker_to_gvkey['gvkey'], errors='coerce')
        ticker_to_gvkey['ticker'] = ticker_to_gvkey['ticker'].str.upper().str.strip()
        ticker_to_gvkey = ticker_to_gvkey[['ticker', 'gvkey']].drop_duplicates()
    else:
        ticker_to_gvkey = pd.DataFrame()
    
    if companies is not None and 'ticker' in companies.columns and len(ticker_to_gvkey) > 0:
        company_map = companies[['companyid', 'ticker']].copy()
        company_map['ticker'] = company_map['ticker'].str.upper().str.strip()
        company_map = company_map.merge(ticker_to_gvkey, on='ticker', how='inner')
        company_map = company_map[['companyid', 'gvkey']].drop_duplicates()
        print(f"   BoardEx companyid → gvkey: {len(company_map):,}")
    else:
        company_map = pd.DataFrame()
    
    # Find CEO roles
    ceo_emp = employment[
        employment['rolename'].str.lower().str.contains('ceo|chief executive', na=False)
    ].copy()
    ceo_emp = ceo_emp.sort_values('datestartrole', ascending=False)
    ceo_emp = ceo_emp.drop_duplicates(subset=['directorid', 'companyid'], keep='first')
    ceo_emp['role_year'] = ceo_emp['datestartrole'].dt.year
    
    # Add gvkey
    if len(company_map) > 0:
        ceo_gvkey = ceo_emp.merge(company_map, on='companyid', how='inner')
    else:
        print("   ❌ No company mapping available")
        return None
    
    # Merge with people variables
    ceo_with_vars = ceo_gvkey.merge(people_df, on='directorid', how='left')
    ceo_with_vars = ceo_with_vars.rename(columns={'role_year': 'fiscalyear'})
    ceo_with_vars = ceo_with_vars[ceo_with_vars['fiscalyear'].notna()]
    ceo_with_vars['fiscalyear'] = ceo_with_vars['fiscalyear'].astype(int)
    ceo_with_vars['gvkey'] = ceo_with_vars['gvkey'].astype(int)
    
    # Merge with match data
    match_df = match_df.copy()
    match_df['gvkey'] = pd.to_numeric(match_df['gvkey'], errors='coerce')
    
    # Select columns for merge
    drop_cols = ['companyid', 'companyname', 'rolename', 'datestartrole', 'dateendrole']
    merge_cols = [c for c in ceo_with_vars.columns if c not in drop_cols]
    
    merged = match_df.merge(
        ceo_with_vars[merge_cols].drop_duplicates(subset=['gvkey', 'fiscalyear']),
        on=['gvkey', 'fiscalyear'], how='left'
    )
    
    matched = merged['n_board_seats'].notna().sum() if 'n_board_seats' in merged.columns else 0
    print(f"   ✅ Matched {matched:,} / {len(merged):,} ({100*matched/len(merged):.1f}%)")
    
    return merged

# ================================================================
# ANALYSIS FUNCTIONS
# ================================================================

def run_descriptive_analysis(df):
    """Descriptive analysis by match quintile."""
    print("\n📊 Descriptive Analysis by Match Quintile...")
    
    df = df.copy()
    df['match_q'] = pd.qcut(df['match_means'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    people_vars = [
        'n_board_seats', 'n_companies', 'n_ceo_roles', 'n_csuite_roles',
        'industry_breadth', 'networksize', 'c_suite_exp', 'well_connected',
        'has_mba', 'has_phd', 'elite_school', 'stem_degree', 'max_education',
        'equity_orientation', 'avg_total_comp_ciq'
    ]
    
    available = [v for v in people_vars if v in df.columns and df[v].notna().sum() > 50]
    
    if len(available) == 0:
        print("   ❌ No people variables with sufficient data")
        return None
    
    stats_df = df.groupby('match_q', observed=False)[available].mean().round(3)
    print("\n--- PEOPLE VARIABLES BY MATCH QUALITY ---")
    print(stats_df.T)
    
    print("\n--- CORRELATION WITH MATCH ---")
    for var in available:
        valid = df.dropna(subset=['match_means', var])
        if len(valid) > 50:
            corr, pval = stats.pearsonr(valid['match_means'], valid[var])
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"   {var}: r = {corr:.3f} (p = {pval:.4f}) {sig}")
    
    return stats_df

def run_regression_analysis(df):
    """OLS regressions."""
    print("\n📈 Regression Analysis...")
    
    people_vars = ['n_board_seats', 'networksize', 'has_mba', 'has_phd', 'elite_school', 'n_ceo_roles']
    controls = ['logatw', 'exp_roa', 'Age']
    
    available = [v for v in people_vars if v in df.columns and df[v].notna().sum() > 100]
    available_ctrl = [c for c in controls if c in df.columns]
    
    if len(available) == 0:
        print("   ❌ No people variables for regression")
        return None
    
    reg_df = df[['match_means'] + available + available_ctrl].dropna()
    print(f"   N = {len(reg_df):,}")
    
    formula = f"match_means ~ {' + '.join(available)}"
    if available_ctrl:
        formula += f" + {' + '.join(available_ctrl)}"
    
    model = ols(formula, data=reg_df).fit(cov_type='HC1')
    
    print(f"\n   R² = {model.rsquared:.4f}")
    for var in available:
        coef = model.params.get(var, 0)
        pval = model.pvalues.get(var, 1)
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"   {var}: β = {coef:.4f} (p = {pval:.4f}) {sig}")
    
    return model

def run_random_forest(df):
    """Random Forest feature importance."""
    print("\n🌲 Random Forest Analysis...")
    
    people_vars = [
        'n_board_seats', 'n_companies', 'networksize', 'industry_breadth',
        'n_ceo_roles', 'n_csuite_roles', 'has_mba', 'has_phd', 'elite_school',
        'stem_degree', 'max_education', 'well_connected', 'c_suite_exp'
    ]
    other_vars = ['Age', 'logatw', 'exp_roa', 'leverage', 'boardindpw']
    
    all_features = [v for v in people_vars + other_vars if v in df.columns]
    ml_df = df[all_features + ['match_means']].dropna()
    
    if len(ml_df) < 200:
        print(f"   ❌ Insufficient data: {len(ml_df)}")
        return None
    
    print(f"   Training on N = {len(ml_df):,}")
    
    X = ml_df[all_features]
    y = ml_df['match_means']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    print(f"   Train R² = {rf.score(X_train, y_train):.4f}")
    print(f"   Test R² = {rf.score(X_test, y_test):.4f}")
    
    importances = pd.DataFrame({
        'Feature': all_features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n--- FEATURE IMPORTANCE ---")
    for _, row in importances.head(15).iterrows():
        star = "★" if row['Feature'] in people_vars else " "
        print(f"   {star} {row['Feature']}: {row['Importance']:.4f}")
    
    return rf, importances

# ================================================================
# VISUALIZATION
# ================================================================

def create_visualization_dashboard(df, output_dir='Output'):
    """Create comprehensive visualization dashboard."""
    print("\n🎨 Creating Visualization Dashboard...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(24, 20))
    
    df = df.copy()
    df['match_q'] = pd.qcut(df['match_means'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
    
    # 1. Network Size
    ax1 = fig.add_subplot(4, 4, 1)
    if 'networksize' in df.columns:
        data = df.dropna(subset=['match_q', 'networksize'])
        if len(data) > 50:
            by_q = data.groupby('match_q', observed=False)['networksize'].mean()
            ax1.bar(range(5), by_q.values, color=colors, edgecolor='black')
            ax1.set_xticks(range(5))
            ax1.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax1.set_ylabel('Network Size')
            ax1.set_title('🔗 Network Size by Match', fontweight='bold')
    
    # 2. Board Seats
    ax2 = fig.add_subplot(4, 4, 2)
    if 'n_board_seats' in df.columns:
        data = df.dropna(subset=['match_q', 'n_board_seats'])
        if len(data) > 50:
            by_q = data.groupby('match_q', observed=False)['n_board_seats'].mean()
            ax2.bar(range(5), by_q.values, color='steelblue', edgecolor='black')
            ax2.set_xticks(range(5))
            ax2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax2.set_ylabel('Board Seats')
            ax2.set_title('🪑 Board Experience', fontweight='bold')
    
    # 3. Prior CEO Roles
    ax3 = fig.add_subplot(4, 4, 3)
    if 'n_ceo_roles' in df.columns:
        data = df.dropna(subset=['match_q', 'n_ceo_roles'])
        if len(data) > 50:
            by_q = data.groupby('match_q', observed=False)['n_ceo_roles'].mean()
            ax3.bar(range(5), by_q.values, color='purple', edgecolor='black')
            ax3.set_xticks(range(5))
            ax3.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax3.set_ylabel('Prior CEO Roles')
            ax3.set_title('👔 CEO Experience', fontweight='bold')
    
    # 4. MBA Rate
    ax4 = fig.add_subplot(4, 4, 4)
    if 'has_mba' in df.columns:
        data = df.dropna(subset=['match_q', 'has_mba'])
        if len(data) > 50:
            by_q = data.groupby('match_q', observed=False)['has_mba'].mean()
            ax4.bar(range(5), by_q.values * 100, color='gold', edgecolor='black')
            ax4.set_xticks(range(5))
            ax4.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax4.set_ylabel('% with MBA')
            ax4.set_title('🎓 MBA Rate', fontweight='bold')
    
    # 5. Elite School Rate
    ax5 = fig.add_subplot(4, 4, 5)
    if 'elite_school' in df.columns:
        data = df.dropna(subset=['match_q', 'elite_school'])
        if len(data) > 50:
            by_q = data.groupby('match_q', observed=False)['elite_school'].mean()
            ax5.bar(range(5), by_q.values * 100, color='coral', edgecolor='black')
            ax5.set_xticks(range(5))
            ax5.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax5.set_ylabel('% Elite School')
            ax5.set_title('🏛️ Elite Education', fontweight='bold')
    
    # 6. PhD Rate
    ax6 = fig.add_subplot(4, 4, 6)
    if 'has_phd' in df.columns:
        data = df.dropna(subset=['match_q', 'has_phd'])
        if len(data) > 50:
            by_q = data.groupby('match_q', observed=False)['has_phd'].mean()
            ax6.bar(range(5), by_q.values * 100, color='teal', edgecolor='black')
            ax6.set_xticks(range(5))
            ax6.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax6.set_ylabel('% with PhD')
            ax6.set_title('📚 PhD Rate', fontweight='bold')
    
    # 7. STEM Degree Rate
    ax7 = fig.add_subplot(4, 4, 7)
    if 'stem_degree' in df.columns:
        data = df.dropna(subset=['match_q', 'stem_degree'])
        if len(data) > 50:
            by_q = data.groupby('match_q', observed=False)['stem_degree'].mean()
            ax7.bar(range(5), by_q.values * 100, color='darkorange', edgecolor='black')
            ax7.set_xticks(range(5))
            ax7.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax7.set_ylabel('% STEM Degree')
            ax7.set_title('🔬 STEM Background', fontweight='bold')
    
    # 8. Industry Breadth
    ax8 = fig.add_subplot(4, 4, 8)
    if 'industry_breadth' in df.columns:
        data = df.dropna(subset=['match_q', 'industry_breadth'])
        if len(data) > 50:
            by_q = data.groupby('match_q', observed=False)['industry_breadth'].mean()
            ax8.bar(range(5), by_q.values, color='seagreen', edgecolor='black')
            ax8.set_xticks(range(5))
            ax8.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax8.set_ylabel('Unique Industries')
            ax8.set_title('🏭 Industry Experience', fontweight='bold')
    
    # 9. C-Suite Experience
    ax9 = fig.add_subplot(4, 4, 9)
    if 'c_suite_exp' in df.columns:
        data = df.dropna(subset=['match_q', 'c_suite_exp'])
        if len(data) > 50:
            by_q = data.groupby('match_q', observed=False)['c_suite_exp'].mean()
            ax9.bar(range(5), by_q.values, color='crimson', edgecolor='black')
            ax9.set_xticks(range(5))
            ax9.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax9.set_ylabel('C-Suite Roles (0-3)')
            ax9.set_title('⭐ C-Suite Experience', fontweight='bold')
    
    # 10. Well-Connected Rate
    ax10 = fig.add_subplot(4, 4, 10)
    if 'well_connected' in df.columns:
        data = df.dropna(subset=['match_q', 'well_connected'])
        if len(data) > 50:
            by_q = data.groupby('match_q', observed=False)['well_connected'].mean()
            ax10.bar(range(5), by_q.values * 100, color='navy', edgecolor='black')
            ax10.set_xticks(range(5))
            ax10.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax10.set_ylabel('% Well Connected')
            ax10.set_title('🌐 Connection Rate', fontweight='bold')
    
    # 11. Network vs Match Scatter
    ax11 = fig.add_subplot(4, 4, 11)
    if 'networksize' in df.columns:
        valid = df.dropna(subset=['networksize', 'match_means'])
        if len(valid) > 100:
            sample = valid.sample(min(2000, len(valid)))
            ax11.scatter(sample['networksize'], sample['match_means'], alpha=0.3, s=15)
            z = np.polyfit(sample['networksize'], sample['match_means'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(0, sample['networksize'].quantile(0.95), 100)
            ax11.plot(x_line, p(x_line), 'r-', lw=2, label=f'β={z[0]:.4f}')
            ax11.set_xlabel('Network Size')
            ax11.set_ylabel('Match Quality')
            ax11.set_title('Network → Match', fontweight='bold')
            ax11.legend()
    
    # 12. Correlation Heatmap
    ax12 = fig.add_subplot(4, 4, 12)
    corr_vars = ['match_means', 'networksize', 'n_board_seats', 'n_ceo_roles', 
                 'industry_breadth', 'has_mba', 'elite_school']
    avail_corr = [v for v in corr_vars if v in df.columns]
    if len(avail_corr) > 2:
        corr_df = df[avail_corr].dropna()
        if len(corr_df) > 50:
            corr_mat = corr_df.corr()
            sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, ax=ax12, vmin=-1, vmax=1)
            ax12.set_title('Correlation Matrix', fontweight='bold')
    
    # 13. Gender Distribution
    ax13 = fig.add_subplot(4, 4, 13)
    if 'gender' in df.columns:
        data = df.dropna(subset=['gender', 'match_means'])
        if len(data) > 50:
            gender_stats = data.groupby('gender')['match_means'].agg(['mean', 'count'])
            colors_g = ['steelblue', 'coral'][:len(gender_stats)]
            ax13.bar(range(len(gender_stats)), gender_stats['mean'], color=colors_g, edgecolor='black')
            ax13.set_xticks(range(len(gender_stats)))
            ax13.set_xticklabels(gender_stats.index)
            ax13.set_ylabel('Avg Match')
            ax13.set_title('Match by Gender', fontweight='bold')
    
    # 14. Max Education Level
    ax14 = fig.add_subplot(4, 4, 14)
    if 'max_education' in df.columns:
        data = df.dropna(subset=['match_q', 'max_education'])
        if len(data) > 50:
            by_q = data.groupby('match_q', observed=False)['max_education'].mean()
            ax14.bar(range(5), by_q.values, color='mediumpurple', edgecolor='black')
            ax14.set_xticks(range(5))
            ax14.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax14.set_ylabel('Max Edu Level (1-5)')
            ax14.set_title('📖 Education Level', fontweight='bold')
    
    # 15. Equity Orientation
    ax15 = fig.add_subplot(4, 4, 15)
    if 'equity_orientation' in df.columns:
        data = df.dropna(subset=['match_q', 'equity_orientation'])
        if len(data) > 50:
            by_q = data.groupby('match_q', observed=False)['equity_orientation'].mean()
            ax15.bar(range(5), by_q.values * 100, color='darkgreen', edgecolor='black')
            ax15.set_xticks(range(5))
            ax15.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            ax15.set_ylabel('Equity % of Comp')
            ax15.set_title('💹 Equity Orientation', fontweight='bold')
    
    # 16. Summary Stats
    ax16 = fig.add_subplot(4, 4, 16)
    ax16.axis('off')
    summary = "👥 PEOPLE MODULE SUMMARY\n" + "=" * 30 + "\n\n"
    
    people_matched = df['n_board_seats'].notna().sum() if 'n_board_seats' in df.columns else 0
    summary += f"Total Observations: {len(df):,}\n"
    summary += f"With People Data: {people_matched:,}\n"
    summary += f"Coverage: {100*people_matched/len(df):.1f}%\n\n"
    
    if 'networksize' in df.columns:
        summary += f"Avg Network: {df['networksize'].mean():.0f}\n"
    if 'n_board_seats' in df.columns:
        summary += f"Avg Boards: {df['n_board_seats'].mean():.1f}\n"
    if 'has_mba' in df.columns:
        summary += f"MBA Rate: {df['has_mba'].mean()*100:.1f}%\n"
    if 'elite_school' in df.columns:
        summary += f"Elite School: {df['elite_school'].mean()*100:.1f}%\n"
    
    ax16.text(0.1, 0.9, summary, transform=ax16.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'people_analysis_dashboard.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()
    
    return output_path

# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='People Module: BoardEx + Capital IQ Integration')
    parser.add_argument('--test', action='store_true', help='Test mode (limit 5000 rows)')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--test-connection', action='store_true', help='Only test WRDS connection')
    args = parser.parse_args()
    
    print("👥" * 40)
    print("   PEOPLE MODULE: BOARDEX + CAPITAL IQ")
    print("👥" * 40)
    
    config = Config()
    
    # Test connection only
    if args.test_connection:
        db = connect_wrds()
        if db:
            test_connection(db)
            db.close()
        return
    
    # Load match data
    print("\n📂 LOADING BASE MATCH DATA...")
    processor = DataProcessor(config)
    raw_df = processor.load_data()
    df_clean = processor.prepare_features(raw_df)
    print(f"  Match Data: {len(df_clean):,} observations")
    
    if args.synthetic:
        print("\n⚠️ SYNTHETIC MODE: Generating 50+ CEO variables...")
        np.random.seed(42)
        n = len(df_clean)
        
        # ==========================================================
        # CORE EXPERIENCE VARIABLES
        # ==========================================================
        df_clean['n_board_seats'] = np.random.poisson(3, n)
        df_clean['n_companies'] = np.random.poisson(4, n) + 1
        df_clean['n_ceo_roles'] = np.random.poisson(0.5, n)
        df_clean['n_cfo_roles'] = np.random.poisson(0.3, n)
        df_clean['n_coo_roles'] = np.random.poisson(0.2, n)
        df_clean['n_cto_roles'] = np.random.poisson(0.1, n)
        df_clean['n_cmo_roles'] = np.random.poisson(0.1, n)
        df_clean['n_csuite_roles'] = np.random.poisson(2, n)
        df_clean['n_president_roles'] = np.random.poisson(0.4, n)
        df_clean['n_chairman_roles'] = np.random.poisson(0.2, n)
        df_clean['n_vp_roles'] = np.random.poisson(1.5, n)
        df_clean['n_total_roles'] = np.random.poisson(8, n)
        
        # ==========================================================
        # FUNCTIONAL BACKGROUND
        # ==========================================================
        df_clean['n_finance_roles'] = np.random.poisson(0.8, n)
        df_clean['n_ops_roles'] = np.random.poisson(0.5, n)
        df_clean['n_sales_roles'] = np.random.poisson(0.4, n)
        df_clean['n_marketing_roles'] = np.random.poisson(0.3, n)
        df_clean['n_tech_roles'] = np.random.poisson(0.4, n)
        df_clean['n_legal_roles'] = np.random.poisson(0.1, n)
        df_clean['n_strategy_roles'] = np.random.poisson(0.3, n)
        
        df_clean['finance_background'] = (np.random.random(n) > 0.7).astype(int)
        df_clean['ops_background'] = (np.random.random(n) > 0.75).astype(int)
        df_clean['tech_background'] = (np.random.random(n) > 0.8).astype(int)
        df_clean['sales_background'] = (np.random.random(n) > 0.85).astype(int)
        df_clean['marketing_background'] = (np.random.random(n) > 0.88).astype(int)
        df_clean['legal_background'] = (np.random.random(n) > 0.92).astype(int)
        df_clean['strategy_background'] = (np.random.random(n) > 0.85).astype(int)
        
        df_clean['primary_function'] = np.random.choice(
            ['Finance', 'Operations', 'Sales', 'Marketing', 'Technology', 'Legal', 'Strategy', 'General'],
            n, p=[0.25, 0.15, 0.12, 0.08, 0.12, 0.05, 0.08, 0.15]
        )
        
        # ==========================================================
        # CAREER TRAJECTORY
        # ==========================================================
        df_clean['career_start_year'] = np.random.randint(1980, 2010, n)
        df_clean['career_length_years'] = 2024 - df_clean['career_start_year']
        df_clean['first_csuite_year'] = df_clean['career_start_year'] + np.random.randint(5, 20, n)
        df_clean['first_ceo_year'] = df_clean['first_csuite_year'] + np.random.randint(0, 10, n)
        df_clean['years_to_csuite'] = df_clean['first_csuite_year'] - df_clean['career_start_year']
        df_clean['years_to_ceo'] = df_clean['first_ceo_year'] - df_clean['career_start_year']
        
        df_clean['serial_ceo'] = (df_clean['n_ceo_roles'] >= 2).astype(int)
        df_clean['is_founder'] = (np.random.random(n) > 0.9).astype(int)
        df_clean['professional_manager'] = ((df_clean['is_founder'] == 0) & (df_clean['n_ceo_roles'] > 0)).astype(int)
        
        df_clean['csuite_breadth'] = np.random.randint(0, 5, n)
        df_clean['c_suite_exp'] = df_clean['csuite_breadth']
        df_clean['leadership_score'] = np.random.lognormal(2, 0.8, n).astype(int)
        
        # ==========================================================
        # TENURE PATTERNS
        # ==========================================================
        df_clean['avg_tenure_days'] = np.random.lognormal(6.5, 0.8, n)
        df_clean['max_tenure_days'] = df_clean['avg_tenure_days'] * np.random.uniform(1.5, 3, n)
        df_clean['avg_tenure_years'] = df_clean['avg_tenure_days'] / 365.25
        df_clean['max_tenure_years'] = df_clean['max_tenure_days'] / 365.25
        df_clean['job_hopper'] = (df_clean['avg_tenure_years'] < 2).astype(int)
        df_clean['long_tenure'] = (df_clean['avg_tenure_years'] > 5).astype(int)
        df_clean['companies_per_role'] = df_clean['n_companies'] / (df_clean['n_total_roles'] + 1)
        df_clean['multi_company'] = (df_clean['n_companies'] >= 3).astype(int)
        
        # ==========================================================
        # NETWORK & CONNECTIONS
        # ==========================================================
        df_clean['networksize'] = np.random.lognormal(5, 1, n).astype(int)
        df_clean['well_connected'] = (np.random.random(n) > 0.75).astype(int)
        df_clean['super_connected'] = (np.random.random(n) > 0.90).astype(int)
        df_clean['low_connected'] = (np.random.random(n) > 0.50).astype(int)
        df_clean['network_quintile'] = np.random.randint(0, 5, n)
        df_clean['network_per_career_year'] = df_clean['networksize'] / (df_clean['career_length_years'] + 1)
        
        # ==========================================================
        # BOARD EXPERIENCE
        # ==========================================================
        df_clean['heavy_board_experience'] = (df_clean['n_board_seats'] >= 3).astype(int)
        df_clean['board_to_executive_ratio'] = df_clean['n_board_seats'] / (df_clean['n_csuite_roles'] + 1)
        df_clean['industry_breadth'] = np.random.randint(1, 8, n)
        df_clean['industry_specialist'] = (df_clean['industry_breadth'] == 1).astype(int)
        df_clean['industry_generalist'] = (df_clean['industry_breadth'] >= 3).astype(int)
        
        # ==========================================================
        # DEMOGRAPHICS
        # ==========================================================
        df_clean['gender'] = np.random.choice(['Male', 'Female'], n, p=[0.92, 0.08])
        df_clean['is_female'] = (df_clean['gender'] == 'Female').astype(int)
        df_clean['young_ceo'] = (np.random.random(n) > 0.85).astype(int)
        df_clean['experienced_ceo'] = (np.random.random(n) > 0.6).astype(int)
        df_clean['prime_age'] = (np.random.random(n) > 0.5).astype(int)
        
        # ==========================================================
        # EDUCATION
        # ==========================================================
        df_clean['has_mba'] = (np.random.random(n) > 0.55).astype(int)
        df_clean['has_phd'] = (np.random.random(n) > 0.92).astype(int)
        df_clean['has_jd'] = (np.random.random(n) > 0.95).astype(int)
        df_clean['has_cpa'] = (np.random.random(n) > 0.9).astype(int)
        df_clean['has_cfa'] = (np.random.random(n) > 0.95).astype(int)
        df_clean['elite_school'] = (np.random.random(n) > 0.75).astype(int)
        df_clean['ivy_league'] = (np.random.random(n) > 0.85).astype(int)
        df_clean['top_bschool'] = (np.random.random(n) > 0.8).astype(int)
        df_clean['stem_degree'] = (np.random.random(n) > 0.7).astype(int)
        df_clean['max_education'] = np.random.choice([2, 3, 4, 5], n, p=[0.15, 0.30, 0.40, 0.15])
        df_clean['n_degrees'] = np.random.choice([1, 2, 3, 4], n, p=[0.3, 0.45, 0.2, 0.05])
        df_clean['education_quality_score'] = np.random.poisson(4, n)
        
        # ==========================================================
        # COMPENSATION HISTORY
        # ==========================================================
        df_clean['equity_orientation'] = np.random.beta(2, 3, n)
        df_clean['avg_total_comp_ciq'] = np.random.lognormal(14, 1, n)
        df_clean['max_total_comp_ciq'] = df_clean['avg_total_comp_ciq'] * np.random.uniform(1.2, 2, n)
        
        # ==========================================================
        # COMPOSITE SCORES
        # ==========================================================
        df_clean['ceo_quality_score'] = (
            df_clean['n_ceo_roles'].clip(upper=5) * 3 +
            df_clean['csuite_breadth'] * 2 +
            df_clean['n_board_seats'].clip(upper=10) * 1 +
            df_clean['education_quality_score'] +
            (df_clean['networksize'] / 100).clip(upper=10)
        )
        
        mega_df = df_clean
        print(f"   Generated {len(mega_df.columns)} variables")
        
    else:
        # Connect to WRDS
        print("\n🔌 CONNECTING TO WRDS...")
        db = connect_wrds()
        
        if db is None:
            print("❌ Could not connect. Use --synthetic for testing.")
            return
        
        test_connection(db)
        limit = 5000 if args.test else None
        
        print("\n" + "=" * 70)
        print("📥 PULLING BOARDEX DATA")
        print("=" * 70)
        
        bx_profiles = pull_boardex_profiles(db, limit)
        bx_employment = pull_boardex_employment(db, limit)
        bx_education = pull_boardex_education(db, limit)
        bx_companies = pull_boardex_companies(db, limit)
        
        print("\n" + "=" * 70)
        print("📥 PULLING CAPITAL IQ DATA")
        print("=" * 70)
        
        ciq_professionals = pull_ciq_professionals(db, limit)
        ciq_affiliations = pull_ciq_affiliations(db, limit)
        ciq_education = pull_ciq_education(db, limit)
        ciq_compensation = pull_ciq_compensation(db, limit)
        
        print("\n" + "=" * 70)
        print("📥 PULLING CROSSWALK TABLES")
        print("=" * 70)
        
        people_link = pull_wrds_people_link(db)
        links = pull_link_tables(db)
        
        # NEW: Pull ExecuComp-BoardEx link for enhanced gvkey-year linking
        print("\n" + "=" * 70)
        print("📥 PULLING EXECUCOMP DATA (for gvkey-year linking)")
        print("=" * 70)
        
        exec_boardex_link = pull_exec_boardex_link(db)
        execucomp_ceos = pull_execucomp_ceos(db)
        
        # Save ExecuComp link data
        if exec_boardex_link is not None:
            exec_boardex_link.to_parquet('Output/exec_boardex_link.parquet')
        if execucomp_ceos is not None:
            execucomp_ceos.to_parquet('Output/execucomp_ceos.parquet')
        
        db.close()
        print("\n✅ WRDS Connection Closed")

        
        # Save raw data
        os.makedirs('Output', exist_ok=True)
        if bx_profiles is not None:
            bx_profiles.to_parquet('Output/people_boardex_profiles.parquet')
        if bx_employment is not None:
            bx_employment.to_parquet('Output/people_boardex_employment.parquet')
        if ciq_professionals is not None:
            ciq_professionals.to_parquet('Output/people_ciq_professionals.parquet')
        if ciq_education is not None:
            ciq_education.to_parquet('Output/people_ciq_education.parquet')
        
        # Construct variables
        print("\n" + "=" * 70)
        print("🔧 CONSTRUCTING VARIABLES")
        print("=" * 70)
        
        boardex_vars = construct_boardex_variables(bx_profiles, bx_employment, bx_education, bx_companies)
        ciq_vars = construct_ciq_variables(ciq_professionals, ciq_affiliations, ciq_education, ciq_compensation)
        
        # Merge sources
        people_vars = merge_boardex_ciq(boardex_vars, ciq_vars, people_link)
        
        if people_vars is not None:
            people_vars.to_parquet('Output/people_combined_variables.parquet')
        
        # Link to match data
        print("\n" + "=" * 70)
        print("🔗 LINKING TO MATCH DATA")
        print("=" * 70)
        
        mega_df = None
        
        # PRIMARY: Use ExecuComp-based linking (gvkey-year → execid → directorid)
        if people_vars is not None and exec_boardex_link is not None and execucomp_ceos is not None:
            mega_df = link_people_via_execucomp(people_vars, exec_boardex_link, execucomp_ceos, df_clean)
        
        # FALLBACK: Use ticker-based linking if ExecuComp linking didn't work
        if mega_df is None and people_vars is not None:
            print("\n   ⚠️ ExecuComp linking unavailable, using ticker-based fallback...")
            mega_df = link_people_to_match(people_vars, bx_employment, bx_companies, links, df_clean)
        
        if mega_df is None:
            mega_df = df_clean

    
    if mega_df is None:
        print("❌ Failed to create dataset")
        return
    
    # Save merged data
    mega_df.to_parquet('Output/ceo_match_with_people.parquet')
    print(f"\n✅ Saved: Output/ceo_match_with_people.parquet")
    
    # Run analyses
    print("\n" + "=" * 70)
    print("📊 RUNNING ANALYSES")
    print("=" * 70)
    
    run_descriptive_analysis(mega_df)
    run_regression_analysis(mega_df)
    run_random_forest(mega_df)
    
    # Visualizations
    print("\n" + "=" * 70)
    print("🎨 CREATING VISUALIZATIONS")
    print("=" * 70)
    
    create_visualization_dashboard(mega_df)
    
    print("\n👥👥👥 PEOPLE MODULE COMPLETE! 👥👥👥")

if __name__ == "__main__":
    main()
