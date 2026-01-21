#!/usr/bin/env python3
"""
📊 CEO DATA PIPELINE 📊

A modular pipeline to merge BoardEx, CapitalIQ, and ExecuComp data via WRDS cross-walks.

Pipeline Stages:
1. Pull fresh data from WRDS (BoardEx, CIQ, ExecuComp)
2. Construct ~74 CEO-level variables
3. Link to user-provided core panel (gvkey-year)
4. Merge with match quality dataset for regressions/ML

Usage:
    python ceo_data_pipeline.py --stage 1  # Pull and merge raw data
    python ceo_data_pipeline.py --stage 2  # Construct variables
    python ceo_data_pipeline.py --stage 3 --core_panel PATH  # Link to core panel
    python ceo_data_pipeline.py --stage 4 --match_dataset PATH  # Final merge
    python ceo_data_pipeline.py --all  # Run stages 1-2 (data-dependent stages)
"""

import pandas as pd
import numpy as np
import wrds
import os
import argparse
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# CONFIGURATION
# ================================================================

class PipelineConfig:
    """Central configuration for the CEO data pipeline."""
    
    # WRDS Credentials
    WRDS_USER = 'maricste93'
    WRDS_PASS = 'jexvar-manryn-6Cosky'
    
    # Output paths
    BASE_DIR = Path(__file__).parent
    OUTPUT_DIR = BASE_DIR / 'Output' / 'CEO_Pipeline'
    DATA_DIR = BASE_DIR / 'Data'
    
    # Output file names
    BOARDEX_RAW = 'boardex_raw.parquet'
    CIQ_RAW = 'ciq_raw.parquet'
    EXECUCOMP_RAW = 'execucomp_raw.parquet'
    CROSSWALK_RAW = 'crosswalk_raw.parquet'
    CEO_VARIABLES = 'ceo_variables.parquet'
    MERGED_PANEL = 'ceo_panel_merged.parquet'
    
    # Year filters
    MIN_YEAR = 1990
    MAX_YEAR = 2025

# Ensure output directory exists
PipelineConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# WRDS CONNECTION
# ================================================================

def connect_wrds():
    """Establish WRDS connection."""
    try:
        os.environ['PGPASSWORD'] = PipelineConfig.WRDS_PASS
        db = wrds.Connection(wrds_username=PipelineConfig.WRDS_USER)
        print("✅ WRDS Connection Established")
        return db
    except Exception as e:
        print(f"❌ WRDS Connection Failed: {e}")
        return None

def test_wrds_access(db):
    """Test access to required WRDS schemas."""
    print("\n🔍 Testing WRDS Schema Access...")
    schemas = {
        'BoardEx': "SELECT COUNT(*) FROM boardex.na_dir_profile_details LIMIT 1",
        'CIQ PPL Intel': "SELECT COUNT(*) FROM ciq_pplintel.wrds_professional LIMIT 1",
        'CIQ Person': "SELECT COUNT(*) FROM ciq_pplintel.ciqperson LIMIT 1",
        'CIQ Compensation': "SELECT COUNT(*) FROM ciq_pplintel.wrds_compensation LIMIT 1",
        'ExecuComp': "SELECT COUNT(*) FROM execcomp.anncomp LIMIT 1",
        'CrossWalk': "SELECT COUNT(*) FROM wrdsapps.exec_boardex_link LIMIT 1",
        'CIQ-GVKEY': "SELECT COUNT(*) FROM ciq.wrds_gvkey LIMIT 1"
    }
    for name, query in schemas.items():
        try:
            db.raw_sql(query)
            print(f"   ✅ {name}")
        except Exception as e:
            print(f"   ❌ {name}: {str(e)[:50]}")

# ================================================================
# STAGE 1: WRDS DATA PULL FUNCTIONS
# ================================================================

# -- BoardEx --

def pull_boardex_profiles(db, limit=None):
    """Pull BoardEx director profiles."""
    print("\n📊 Pulling BoardEx Profiles...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT directorid, directorname, dob, gender, nationality, networksize, age
    FROM boardex.na_dir_profile_details
    WHERE dob IS NOT NULL
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['birth_year'] = df['dob'].dt.year
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['networksize'] = pd.to_numeric(df['networksize'], errors='coerce')
        print(f"   Retrieved {len(df):,} profiles")
        return df
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()

def pull_boardex_employment(db, limit=None):
    """Pull BoardEx employment/directorship history."""
    print("\n📊 Pulling BoardEx Employment...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT directorid, companyid, companyname, rolename, 
           datestartrole, dateendrole, ned, brdposition
    FROM boardex.na_dir_profile_emp
    WHERE datestartrole IS NOT NULL
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        df['datestartrole'] = pd.to_datetime(df['datestartrole'], errors='coerce')
        df['dateendrole'] = pd.to_datetime(df['dateendrole'], errors='coerce')
        print(f"   Retrieved {len(df):,} employment records")
        return df
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()

def pull_boardex_education(db, limit=None):
    """Pull BoardEx education records."""
    print("\n📊 Pulling BoardEx Education...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT directorid, qualification, institutionname
    FROM boardex.na_dir_profile_edu
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} education records")
        return df
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()

def pull_boardex_companies(db, limit=None):
    """Pull BoardEx company profiles."""
    print("\n📊 Pulling BoardEx Companies...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT boardid as companyid, boardname as companyname, isin, ticker, sector
    FROM boardex.na_wrds_company_profile
    WHERE ticker IS NOT NULL
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} companies")
        return df
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()

# -- Capital IQ People Intelligence (ciq_pplintel schema) --

def pull_ciq_persons(db, limit=None):
    """Pull CIQ person demographics from ciq_pplintel.ciqperson."""
    print("\n📊 Pulling CIQ Persons...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT personid, firstname, middlename, lastname, suffix, prefix, yearborn
    FROM ciq_pplintel.ciqperson
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        df['personid'] = pd.to_numeric(df['personid'], errors='coerce').astype('Int64')
        df['yearborn'] = pd.to_numeric(df['yearborn'], errors='coerce').astype('Int64')
        print(f"   Retrieved {len(df):,} persons")
        return df
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()

def pull_ciq_professionals(db, limit=None):
    """Pull CIQ professional affiliations from ciq_pplintel.wrds_professional."""
    print("\n📊 Pulling CIQ Professional Affiliations...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT companyid, personid, proid, profunctionid, companyname, personname,
           profunctionname, yearfounded, yearborn, title, country, state,
           startyear, endyear, rank, prorank, boardrank,
           proflag, currentproflag, boardflag, currentboardflag,
           keyexecflag, topkeyexecflag
    FROM ciq_pplintel.wrds_professional
    WHERE startyear IS NOT NULL
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        df['personid'] = pd.to_numeric(df['personid'], errors='coerce').astype('Int64')
        df['companyid'] = pd.to_numeric(df['companyid'], errors='coerce').astype('Int64')
        df['startyear'] = pd.to_numeric(df['startyear'], errors='coerce').astype('Int64')
        df['endyear'] = pd.to_numeric(df['endyear'], errors='coerce').astype('Int64')
        print(f"   Retrieved {len(df):,} professional affiliations")
        print(f"   Unique persons: {df['personid'].nunique():,}")
        print(f"   Unique companies: {df['companyid'].nunique():,}")
        return df
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()

def pull_ciq_compensation(db, limit=None):
    """Pull CIQ compensation from ciq_pplintel.wrds_compensation (has gvkey!)."""
    print("\n📊 Pulling CIQ Compensation...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    # ctype1=Salary, ctype2=Bonus, ctype3=Stock Awards, ctype4=Option Awards, etc.
    query = f"""
    SELECT companyid, gvkey, year as fiscalyear, personid, personname, title,
           profunctionname, yearborn, rank, keyexecflag, topkeyexecflag,
           ctype1 as salary, ctype2 as bonus, ctype3 as stock_awards,
           ctype4 as option_awards, ctype5 as non_equity_incentive,
           ctype7 as all_other_comp, ctype8 as total_comp
    FROM ciq_pplintel.wrds_compensation
    WHERE year >= {PipelineConfig.MIN_YEAR}
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        df['personid'] = pd.to_numeric(df['personid'], errors='coerce').astype('Int64')
        df['companyid'] = pd.to_numeric(df['companyid'], errors='coerce').astype('Int64')
        df['gvkey'] = pd.to_numeric(df['gvkey'], errors='coerce').astype('Int64')
        df['fiscalyear'] = pd.to_numeric(df['fiscalyear'], errors='coerce').astype('Int64')
        print(f"   Retrieved {len(df):,} compensation records")
        print(f"   Unique persons: {df['personid'].nunique():,}")
        print(f"   Unique gvkeys: {df['gvkey'].nunique():,}")
        return df
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()

# -- ExecuComp --

def pull_execucomp_ceos(db):
    """Pull ExecuComp CEO records."""
    print("\n📊 Pulling ExecuComp CEOs...")
    query = f"""
    SELECT execid, gvkey, year as fiscalyear, ceoann, exec_fullname, titleann,
           tdc1, salary, bonus, stock_awards_fv, option_awards_fv
    FROM execcomp.anncomp
    WHERE ceoann = 'CEO'
    AND year >= {PipelineConfig.MIN_YEAR}
    """
    try:
        df = db.raw_sql(query)
        df['execid'] = pd.to_numeric(df['execid'], errors='coerce').astype('Int64')
        df['gvkey'] = pd.to_numeric(df['gvkey'], errors='coerce').astype('Int64')
        df['fiscalyear'] = pd.to_numeric(df['fiscalyear'], errors='coerce').astype('Int64')
        print(f"   Retrieved {len(df):,} CEO-year records")
        print(f"   Unique CEOs: {df['execid'].nunique():,}")
        print(f"   Unique firms: {df['gvkey'].nunique():,}")
        print(f"   Year range: {df['fiscalyear'].min()} - {df['fiscalyear'].max()}")
        return df
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()

def pull_execucomp_full(db):
    """Pull all ExecuComp records (not just CEO) for additional comp variables."""
    print("\n📊 Pulling ExecuComp Full Compensation...")
    query = f"""
    SELECT execid, gvkey, year as fiscalyear, exec_fullname, titleann,
           tdc1, tdc2, salary, bonus, stock_awards_fv, option_awards_fv,
           othcomp, ltip, allothtot, pension_chg
    FROM execcomp.anncomp
    WHERE year >= {PipelineConfig.MIN_YEAR}
    """
    try:
        df = db.raw_sql(query)
        df['execid'] = pd.to_numeric(df['execid'], errors='coerce').astype('Int64')
        df['gvkey'] = pd.to_numeric(df['gvkey'], errors='coerce').astype('Int64')
        df['fiscalyear'] = pd.to_numeric(df['fiscalyear'], errors='coerce').astype('Int64')
        print(f"   Retrieved {len(df):,} exec-year records")
        return df
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()

# -- Cross-Walk Tables --

def pull_exec_boardex_link(db):
    """Pull ExecuComp-BoardEx crosswalk."""
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
        print(f"   Retrieved {len(df):,} links")
        print(f"   Unique execids: {df['execid'].nunique():,}")
        print(f"   Unique directorids: {df['directorid'].nunique():,}")
        return df
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()

def pull_boardex_ciq_company_match(db):
    """Pull BoardEx-CIQ company-level match (the main cross-walk)."""
    print("\n📊 Pulling BoardEx-CIQ Company Match...")
    query = """
    SELECT boardex_boardid as boardex_companyid, 
           ciq_companyid,
           matchscore
    FROM wrdsapps.boardex_ciq_company_match
    WHERE boardex_boardid IS NOT NULL AND ciq_companyid IS NOT NULL
    """
    try:
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} company matches")
        return df
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()

def pull_wrds_people_link(db):
    """Pull direct BoardEx-CIQ person link."""
    print("\n📊 Pulling WRDS People Link...")
    try:
        query = """
        SELECT boardex_directorid as directorid, ciq_personid as personid
        FROM wrdsapps.peoplelink
        WHERE boardex_directorid IS NOT NULL AND ciq_personid IS NOT NULL
        """
        df = db.raw_sql(query)
        print(f"   Retrieved {len(df):,} person links")
        return df
    except Exception as e:
        print(f"   ℹ️ peoplelink not accessible: {e}")
        try:
            query = """
            SELECT directorid, personid
            FROM wrdsapps.link_boardex_ciq
            WHERE directorid IS NOT NULL AND personid IS NOT NULL
            """
            df = db.raw_sql(query)
            print(f"   Retrieved {len(df):,} person links (via link_boardex_ciq)")
            return df
        except:
            return pd.DataFrame()

def pull_ciq_gvkey(db):
    """Pull CIQ company to GVKEY mapping."""
    print("\n📊 Pulling CIQ-GVKEY Mapping...")
    query = """
    SELECT companyid as ciq_companyid, gvkey
    FROM ciq.wrds_gvkey
    WHERE gvkey IS NOT NULL
    """
    try:
        df = db.raw_sql(query)
        df['gvkey'] = pd.to_numeric(df['gvkey'], errors='coerce').astype('Int64')
        print(f"   Retrieved {len(df):,} CIQ-GVKEY mappings")
        return df
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()

# ================================================================
# STAGE 2: VARIABLE CONSTRUCTION
# ================================================================

def construct_boardex_variables(profiles, employment, education, companies):
    """
    Construct BoardEx-derived CEO variables.
    
    Returns DataFrame with directorid as key and 50+ features.
    """
    print("\n🔧 Constructing BoardEx Variables...")
    
    if employment.empty or profiles.empty:
        print("   ❌ Missing required BoardEx data")
        return pd.DataFrame()
    
    emp = employment.copy()
    
    # Role flags
    role_patterns = {
        'is_board': r'director|board',
        'is_ceo': r'ceo|chief executive',
        'is_cfo': r'cfo|chief financial',
        'is_coo': r'coo|chief operating',
        'is_cto': r'cto|chief technology',
        'is_cmo': r'cmo|chief marketing',
        'is_cio': r'cio|chief information',
        'is_chro': r'chro|chief human|chief people',
        'is_clo': r'clo|general counsel|chief legal',
        'is_csuite': r'chief|ceo|cfo|coo|cto|cmo|cio',
        'is_president': r'president',
        'is_chairman': r'chairman|chair of',
        'is_founder': r'founder|co-founder',
        'is_vp': r'vice president|vp|svp|evp'
    }
    
    for col, pattern in role_patterns.items():
        emp[col] = emp['rolename'].str.lower().str.contains(pattern, na=False)
    
    # Functional backgrounds
    func_patterns = {
        'is_finance_role': r'finance|financial|treasury|controller|accounting',
        'is_ops_role': r'operations|operating|manufacturing|supply chain',
        'is_sales_role': r'sales|commercial|business development|revenue',
        'is_marketing_role': r'marketing|brand|advertising|communications',
        'is_tech_role': r'technology|engineering|r&d|research|development|product',
        'is_legal_role': r'legal|counsel|compliance|regulatory',
        'is_hr_role': r'human|hr|people|talent|personnel',
        'is_strategy_role': r'strategy|strategic|planning|corp dev'
    }
    
    for col, pattern in func_patterns.items():
        emp[col] = emp['rolename'].str.lower().str.contains(pattern, na=False)
    
    # Tenure
    emp['tenure_days'] = (emp['dateendrole'] - emp['datestartrole']).dt.days.fillna(0).clip(lower=0)
    emp['tenure_years'] = emp['tenure_days'] / 365.25
    emp['start_year'] = emp['datestartrole'].dt.year
    
    # Aggregate per director
    agg_cols = ['is_board', 'is_ceo', 'is_cfo', 'is_coo', 'is_cto', 'is_cmo', 
                'is_cio', 'is_chro', 'is_clo', 'is_csuite', 'is_president', 
                'is_chairman', 'is_founder', 'is_vp',
                'is_finance_role', 'is_ops_role', 'is_sales_role', 'is_marketing_role',
                'is_tech_role', 'is_legal_role', 'is_hr_role', 'is_strategy_role']
    
    agg_dict = {col: 'sum' for col in agg_cols}
    agg_dict.update({
        'companyid': 'nunique',
        'tenure_days': ['mean', 'max', 'sum'],
        'tenure_years': 'mean',
        'start_year': 'min',
        'rolename': 'count'
    })
    
    stats = emp.groupby('directorid').agg(agg_dict).reset_index()
    stats.columns = ['directorid'] + [f'n_{c[3:]}' if c.startswith('is_') else c for c in agg_cols] + [
        'n_companies', 'avg_tenure_days', 'max_tenure_days', 'total_tenure_days',
        'avg_tenure_years', 'career_start_year', 'n_total_roles'
    ]
    
    # Derived variables
    current_year = datetime.now().year
    stats['career_length_years'] = (current_year - stats['career_start_year']).clip(lower=0)
    
    # C-suite breadth
    csuite_cols = ['n_ceo', 'n_cfo', 'n_coo', 'n_cto', 'n_cmo', 'n_cio']
    stats['csuite_breadth'] = sum((stats[c] > 0).astype(int) for c in csuite_cols)
    
    # Functional backgrounds (binary)
    stats['finance_background'] = ((stats['n_finance_role'] > 0) | (stats['n_cfo'] > 0)).astype(int)
    stats['ops_background'] = ((stats['n_ops_role'] > 0) | (stats['n_coo'] > 0)).astype(int)
    stats['tech_background'] = ((stats['n_tech_role'] > 0) | (stats['n_cto'] > 0)).astype(int)
    stats['marketing_background'] = ((stats['n_marketing_role'] > 0) | (stats['n_cmo'] > 0)).astype(int)
    stats['sales_background'] = (stats['n_sales_role'] > 0).astype(int)
    stats['legal_background'] = ((stats['n_legal_role'] > 0) | (stats['n_clo'] > 0)).astype(int)
    stats['hr_background'] = ((stats['n_hr_role'] > 0) | (stats['n_chro'] > 0)).astype(int)
    stats['strategy_background'] = (stats['n_strategy_role'] > 0).astype(int)
    
    # Serial CEO
    stats['serial_ceo'] = (stats['n_ceo'] >= 2).astype(int)
    
    # Leadership score
    stats['leadership_score'] = (
        stats['n_ceo'] * 5 +
        stats['n_president'] * 4 +
        stats['n_coo'] * 3 +
        stats['n_cfo'] * 3 +
        stats['n_chairman'] * 3 +
        stats['n_csuite'] * 2 +
        stats['n_vp'] * 1
    )
    
    # Board experience
    stats['heavy_board_experience'] = (stats['n_board'] >= 3).astype(int)
    stats['board_to_exec_ratio'] = stats['n_board'] / (stats['n_csuite'] + 1)
    
    # Tenure patterns
    stats['avg_tenure_years'] = stats['avg_tenure_days'] / 365.25
    stats['max_tenure_years'] = stats['max_tenure_days'] / 365.25
    stats['job_hopper'] = (stats['avg_tenure_years'] < 2).astype(int)
    stats['long_tenure'] = (stats['avg_tenure_years'] > 5).astype(int)
    
    # Company diversity
    stats['companies_per_role'] = stats['n_companies'] / (stats['n_total_roles'] + 1)
    stats['multi_company'] = (stats['n_companies'] >= 3).astype(int)
    
    # Founder vs professional
    stats['is_founder'] = (stats['n_founder'] > 0).astype(int)
    stats['professional_manager'] = ((stats['n_founder'] == 0) & (stats['n_ceo'] > 0)).astype(int)
    
    # Industry breadth
    if not companies.empty and 'sector' in companies.columns:
        emp_sector = emp.merge(companies[['companyid', 'sector']], on='companyid', how='left')
        ind_breadth = emp_sector.groupby('directorid')['sector'].nunique().reset_index()
        ind_breadth.columns = ['directorid', 'industry_breadth']
        stats = stats.merge(ind_breadth, on='directorid', how='left')
        stats['industry_specialist'] = (stats['industry_breadth'] == 1).astype(int)
        stats['industry_generalist'] = (stats['industry_breadth'] >= 3).astype(int)
    
    # Merge profiles
    if not profiles.empty:
        stats = stats.merge(
            profiles[['directorid', 'gender', 'nationality', 'networksize', 'birth_year', 'age']],
            on='directorid', how='left'
        )
        
        # Network variables
        if 'networksize' in stats.columns:
            q75 = stats['networksize'].quantile(0.75)
            q90 = stats['networksize'].quantile(0.90)
            q50 = stats['networksize'].quantile(0.50)
            stats['well_connected'] = (stats['networksize'] >= q75).fillna(False).astype(int)
            stats['super_connected'] = (stats['networksize'] >= q90).fillna(False).astype(int)
            stats['low_connected'] = (stats['networksize'] < q50).fillna(False).astype(int)
            stats['network_per_career_year'] = stats['networksize'] / (stats['career_length_years'] + 1)
        
        # Age variables
        if 'age' in stats.columns:
            stats['young_ceo'] = (stats['age'] < 45).fillna(False).astype(int)
            stats['experienced_ceo'] = (stats['age'] >= 55).fillna(False).astype(int)
            stats['prime_age'] = ((stats['age'] >= 45) & (stats['age'] < 55)).fillna(False).astype(int)
        
        # Gender
        if 'gender' in stats.columns:
            stats['is_female'] = (stats['gender'].str.lower() == 'female').fillna(False).astype(int)
    
    # Education (BoardEx)
    if not education.empty:
        edu = education.copy()
        edu['qualification'] = edu['qualification'].str.lower().fillna('')
        edu['institutionname'] = edu['institutionname'].str.lower().fillna('')
        
        edu['has_mba_bx'] = edu['qualification'].str.contains(r'mba|master.*business', na=False)
        edu['has_phd_bx'] = edu['qualification'].str.contains(r'phd|doctorate|d\.phil', na=False)
        edu['has_jd_bx'] = edu['qualification'].str.contains(r'j\.d\.|juris|law degree|llb|llm', na=False)
        edu['has_cpa_bx'] = edu['qualification'].str.contains(r'cpa|chartered accountant', na=False)
        edu['has_cfa_bx'] = edu['qualification'].str.contains(r'cfa|chartered financial analyst', na=False)
        
        ivy = r'harvard|yale|princeton|columbia|brown|dartmouth|cornell|penn|upenn'
        top_bschool = r'wharton|hbs|stanford gsb|kellogg|booth|sloan|haas|tuck|insead'
        top_global = r'oxford|cambridge|lse|imperial|mit|stanford|berkeley|chicago|northwestern'
        
        edu['ivy_league'] = edu['institutionname'].str.contains(ivy, na=False)
        edu['top_bschool'] = edu['institutionname'].str.contains(top_bschool, na=False)
        edu['elite_school_bx'] = edu['institutionname'].str.contains(f'{ivy}|{top_global}', na=False)
        edu['stem_edu_bx'] = edu['qualification'].str.contains(r'engineer|computer|math|physics|chemistry|biology|science', na=False)
        
        edu_agg = edu.groupby('directorid').agg({
            'has_mba_bx': 'max', 'has_phd_bx': 'max', 'has_jd_bx': 'max',
            'has_cpa_bx': 'max', 'has_cfa_bx': 'max', 'ivy_league': 'max',
            'top_bschool': 'max', 'elite_school_bx': 'max', 'stem_edu_bx': 'max',
            'qualification': 'count'
        }).reset_index()
        edu_agg.columns = ['directorid', 'has_mba_bx', 'has_phd_bx', 'has_jd_bx', 
                          'has_cpa_bx', 'has_cfa_bx', 'ivy_league', 'top_bschool',
                          'elite_school_bx', 'stem_edu_bx', 'n_degrees_bx']
        stats = stats.merge(edu_agg, on='directorid', how='left')
        
        # Education quality score
        stats['education_quality_score'] = (
            stats['has_phd_bx'].fillna(0).astype(int) * 3 +
            stats['has_mba_bx'].fillna(0).astype(int) * 2 +
            stats['elite_school_bx'].fillna(0).astype(int) * 2 +
            stats['top_bschool'].fillna(0).astype(int) * 2 +
            stats['ivy_league'].fillna(0).astype(int) * 1
        )
    
    # CEO quality score
    stats['ceo_quality_score'] = (
        stats['n_ceo'].clip(upper=5) * 3 +
        stats['csuite_breadth'] * 2 +
        stats['n_board'].clip(upper=10) * 1 +
        stats.get('education_quality_score', pd.Series(0, index=stats.index)) +
        (stats['networksize'].fillna(0) / 100).clip(upper=10)
    )
    
    print(f"   ✅ BoardEx variables: {len(stats):,} directors, {len(stats.columns)} features")
    return stats

def construct_ciq_variables(ciq_persons, ciq_professionals, ciq_compensation):
    """
    Construct Capital IQ-derived CEO variables from ciq_pplintel tables.
    
    Args:
        ciq_persons: DataFrame from ciqperson (demographics)
        ciq_professionals: DataFrame from wrds_professional (affiliations)
        ciq_compensation: DataFrame from wrds_compensation (comp with gvkey)
    
    Returns DataFrame with personid as key.
    """
    print("\n🔧 Constructing CIQ Variables...")
    
    # Start with persons if available, else professionals
    if not ciq_persons.empty:
        stats = ciq_persons[['personid', 'yearborn']].drop_duplicates().copy()
    elif not ciq_professionals.empty:
        stats = ciq_professionals[['personid', 'yearborn']].drop_duplicates().copy()
    else:
        print("   ❌ No CIQ data available")
        return pd.DataFrame()
    
    # Affiliation variables from wrds_professional
    if not ciq_professionals.empty:
        aff = ciq_professionals.copy()
        aff['profunctionname'] = aff['profunctionname'].str.lower().fillna('')
        aff['title'] = aff['title'].str.lower().fillna('')
        
        # Role detection
        aff['is_ceo_ciq'] = aff['profunctionname'].str.contains(r'chief executive|ceo', na=False)
        aff['is_cfo_ciq'] = aff['profunctionname'].str.contains(r'chief financial|cfo', na=False)
        aff['is_coo_ciq'] = aff['profunctionname'].str.contains(r'chief operating|coo', na=False)
        aff['is_csuite_ciq'] = aff['profunctionname'].str.contains(r'chief', na=False)
        aff['is_board_ciq'] = (aff['boardflag'] == 1) | (aff['boardrank'].notna() & (aff['boardrank'] > 0))
        aff['is_key_exec_ciq'] = (aff['keyexecflag'] == 1) | (aff['topkeyexecflag'] == 1)
        
        # Tenure
        aff['tenure_years_ciq'] = (aff['endyear'].fillna(2025) - aff['startyear'].fillna(2025)).clip(lower=0)
        
        aff_agg = aff.groupby('personid').agg({
            'companyid': 'nunique',
            'is_ceo_ciq': 'sum',
            'is_cfo_ciq': 'sum',
            'is_coo_ciq': 'sum',
            'is_csuite_ciq': 'sum',
            'is_board_ciq': 'sum',
            'is_key_exec_ciq': 'sum',
            'tenure_years_ciq': 'mean',
            'startyear': 'min',
            'title': 'count',
            'country': lambda x: x.nunique()
        }).reset_index()
        aff_agg.columns = ['personid', 'n_companies_ciq', 'n_ceo_ciq', 'n_cfo_ciq', 'n_coo_ciq',
                          'n_csuite_ciq', 'n_board_ciq', 'n_key_exec_ciq',
                          'avg_tenure_ciq', 'career_start_ciq', 'n_roles_ciq', 'n_countries_ciq']
        stats = stats.merge(aff_agg, on='personid', how='left')
        
        # Serial CEO (use fillna to handle NA values from left merge)
        stats['serial_ceo_ciq'] = (stats['n_ceo_ciq'].fillna(0) >= 2).astype(int)
        stats['international_exp_ciq'] = (stats['n_countries_ciq'].fillna(0) > 1).astype(int)
    
    # Compensation variables from wrds_compensation
    if not ciq_compensation.empty:
        comp = ciq_compensation.copy()
        comp['salary'] = pd.to_numeric(comp['salary'], errors='coerce')
        comp['bonus'] = pd.to_numeric(comp['bonus'], errors='coerce')
        comp['stock_awards'] = pd.to_numeric(comp['stock_awards'], errors='coerce')
        comp['option_awards'] = pd.to_numeric(comp['option_awards'], errors='coerce')
        comp['total_comp'] = pd.to_numeric(comp['total_comp'], errors='coerce')
        
        comp['equity_comp_ciq'] = comp['stock_awards'].fillna(0) + comp['option_awards'].fillna(0)
        comp['equity_ratio_ciq'] = comp['equity_comp_ciq'] / comp['total_comp'].replace(0, np.nan)
        
        comp_agg = comp.groupby('personid').agg({
            'total_comp': ['mean', 'max'],
            'salary': 'mean',
            'equity_ratio_ciq': 'mean',
            'gvkey': 'nunique',
            'fiscalyear': ['min', 'max']
        }).reset_index()
        comp_agg.columns = ['personid', 'avg_total_comp_ciq', 'max_total_comp_ciq',
                           'avg_salary_ciq', 'equity_orientation_ciq',
                           'n_gvkeys_ciq', 'first_year_ciq', 'last_year_ciq']
        stats = stats.merge(comp_agg, on='personid', how='left')
        
        # Compensation trajectory
        stats['comp_span_ciq'] = stats['last_year_ciq'] - stats['first_year_ciq']
    
    print(f"   ✅ CIQ variables: {len(stats):,} persons, {len(stats.columns)} features")
    return stats

def construct_execucomp_variables(execucomp_ceos, execucomp_full=None):
    """
    Construct ExecuComp-derived compensation variables.
    
    Returns DataFrame with (execid, fiscalyear) as key.
    """
    print("\n🔧 Constructing ExecuComp Variables...")
    
    if execucomp_ceos.empty:
        print("   ❌ Missing ExecuComp data")
        return pd.DataFrame()
    
    df = execucomp_ceos.copy()
    
    # Numeric conversion
    for col in ['tdc1', 'salary', 'bonus', 'stock_awards_fv', 'option_awards_fv']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Derived variables
    if 'stock_awards_fv' in df.columns and 'option_awards_fv' in df.columns:
        df['equity_comp_exec'] = df['stock_awards_fv'].fillna(0) + df['option_awards_fv'].fillna(0)
    if 'tdc1' in df.columns:
        df['equity_ratio_exec'] = df['equity_comp_exec'] / df['tdc1'].replace(0, np.nan)
        df['salary_ratio_exec'] = df['salary'] / df['tdc1'].replace(0, np.nan)
    
    # Rename for clarity
    rename_map = {
        'tdc1': 'total_comp_exec',
        'salary': 'salary_exec',
        'bonus': 'bonus_exec',
        'stock_awards_fv': 'stock_awards_exec',
        'option_awards_fv': 'option_awards_exec'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    print(f"   ✅ ExecuComp variables: {len(df):,} CEO-years, {len(df.columns)} features")
    return df

# ================================================================
# CROSS-WALK MERGE FUNCTIONS
# ================================================================

def merge_boardex_ciq(boardex_vars, ciq_vars, people_link):
    """Merge BoardEx and CIQ variables using person-level cross-walk."""
    print("\n🔗 Merging BoardEx + CIQ (person-level)...")
    
    if people_link.empty:
        print("   ⚠️ No people link, returning BoardEx only")
        return boardex_vars
    
    if boardex_vars.empty:
        return pd.DataFrame()
    
    merged = boardex_vars.merge(people_link, on='directorid', how='left')
    
    if not ciq_vars.empty:
        merged = merged.merge(ciq_vars, on='personid', how='left')
        linked = merged['personid'].notna().sum()
        print(f"   ✅ {linked:,}/{len(merged):,} directors linked ({100*linked/len(merged):.1f}%)")
    
    # Combine overlapping variables (prefer CIQ, fallback BoardEx)
    for var in ['has_mba', 'has_phd', 'elite_school']:
        ciq_col = f'{var}_ciq'
        bx_col = f'{var}_bx' if var != 'elite_school' else 'elite_school_bx'
        if ciq_col in merged.columns and bx_col in merged.columns:
            merged[var] = merged[ciq_col].fillna(merged[bx_col])
        elif ciq_col in merged.columns:
            merged[var] = merged[ciq_col]
        elif bx_col in merged.columns:
            merged[var] = merged[bx_col]
    
    print(f"   Final: {len(merged):,} directors, {len(merged.columns)} features")
    return merged

def create_gvkey_year_panel(combined_vars, exec_boardex_link, execucomp_ceos, execucomp_vars=None):
    """
    Create a gvkey-year panel by linking:
    directorid → execid → gvkey-fiscalyear
    """
    print("\n🔗 Creating GVKEY-Year Panel...")
    
    if exec_boardex_link.empty or execucomp_ceos.empty or combined_vars.empty:
        print("   ❌ Missing required data")
        return pd.DataFrame()
    
    # Step 1: directorid → execid
    link = exec_boardex_link[['directorid', 'execid']].drop_duplicates()
    
    # Step 2: execid → gvkey-fiscalyear
    exec_ceos = execucomp_ceos[['execid', 'gvkey', 'fiscalyear', 'exec_fullname']].copy()
    exec_ceos = exec_ceos.dropna(subset=['execid', 'gvkey', 'fiscalyear'])
    
    # Merge chain
    panel = exec_ceos.merge(link, on='execid', how='inner')
    print(f"   CEOs with BoardEx link: {len(panel):,}")
    
    panel = panel.merge(combined_vars, on='directorid', how='left')
    print(f"   After variable merge: {len(panel):,}")
    
    # Add ExecuComp compensation variables
    if execucomp_vars is not None and not execucomp_vars.empty:
        exec_comp_cols = [c for c in execucomp_vars.columns if c.endswith('_exec')]
        if exec_comp_cols:
            panel = panel.merge(
                execucomp_vars[['execid', 'fiscalyear'] + exec_comp_cols],
                on=['execid', 'fiscalyear'], how='left'
            )
    
    # Ensure unique gvkey-year
    panel = panel.drop_duplicates(subset=['gvkey', 'fiscalyear'], keep='first')
    
    print(f"   ✅ Final panel: {len(panel):,} gvkey-years, {len(panel.columns)} features")
    return panel

# ================================================================
# STAGE 3 & 4: PANEL MERGE FUNCTIONS
# ================================================================

def link_to_core_panel(ceo_vars_panel, core_panel_path):
    """
    Merge CEO variables to user-provided core panel.
    
    Args:
        ceo_vars_panel: DataFrame with gvkey, fiscalyear, and CEO variables
        core_panel_path: Path to core panel CSV/parquet
    """
    print(f"\n🔗 Linking to Core Panel: {core_panel_path}")
    
    # Load core panel
    if core_panel_path.endswith('.parquet'):
        core = pd.read_parquet(core_panel_path)
    else:
        core = pd.read_csv(core_panel_path)
    
    print(f"   Core panel: {len(core):,} rows")
    
    # Standardize keys
    core['gvkey'] = pd.to_numeric(core['gvkey'], errors='coerce').astype('Int64')
    
    # Detect year column
    year_col = None
    for col in ['fiscalyear', 'year', 'fyear']:
        if col in core.columns:
            year_col = col
            break
    
    if year_col is None:
        print("   ❌ No year column found in core panel")
        return core
    
    core['fiscalyear'] = pd.to_numeric(core[year_col], errors='coerce').astype('Int64')
    
    # Merge
    ceo_cols = [c for c in ceo_vars_panel.columns if c not in ['execid', 'exec_fullname']]
    merged = core.merge(ceo_vars_panel[ceo_cols], on=['gvkey', 'fiscalyear'], how='left')
    
    # Report coverage
    if 'directorid' in merged.columns:
        matched = merged['directorid'].notna().sum()
        print(f"   ✅ Coverage: {matched:,}/{len(merged):,} ({100*matched/len(merged):.1f}%)")
    
    return merged

def merge_match_quality(panel_with_ceo, match_dataset_path):
    """
    Merge match quality dataset for ML/regressions.
    
    Args:
        panel_with_ceo: DataFrame with CEO variables linked to core panel
        match_dataset_path: Path to match quality CSV/parquet
    """
    print(f"\n🔗 Merging Match Quality: {match_dataset_path}")
    
    # Load match dataset
    if match_dataset_path.endswith('.parquet'):
        match_df = pd.read_parquet(match_dataset_path)
    else:
        match_df = pd.read_csv(match_dataset_path)
    
    print(f"   Match dataset: {len(match_df):,} rows")
    
    # Standardize keys
    match_df['gvkey'] = pd.to_numeric(match_df['gvkey'], errors='coerce').astype('Int64')
    
    year_col = None
    for col in ['fiscalyear', 'year', 'fyear']:
        if col in match_df.columns:
            year_col = col
            break
    
    if year_col:
        match_df['fiscalyear'] = pd.to_numeric(match_df[year_col], errors='coerce').astype('Int64')
    
    # Detect merge keys
    if 'fiscalyear' in match_df.columns:
        merge_keys = ['gvkey', 'fiscalyear']
    else:
        merge_keys = ['gvkey']
    
    # Merge
    merged = panel_with_ceo.merge(match_df, on=merge_keys, how='left', suffixes=('', '_match'))
    
    print(f"   ✅ Final panel: {len(merged):,} rows, {len(merged.columns)} columns")
    return merged

# ================================================================
# MAIN PIPELINE RUNNERS
# ================================================================

def run_stage1(db, save=True):
    """Stage 1: Pull all WRDS data and cross-walks."""
    print("\n" + "="*60)
    print("STAGE 1: WRDS DATA PULL")
    print("="*60)
    
    # BoardEx
    bx_profiles = pull_boardex_profiles(db)
    bx_employment = pull_boardex_employment(db)
    bx_education = pull_boardex_education(db)
    bx_companies = pull_boardex_companies(db)
    
    boardex_data = {
        'profiles': bx_profiles,
        'employment': bx_employment,
        'education': bx_education,
        'companies': bx_companies
    }
    
    # Capital IQ People Intelligence (ciq_pplintel)
    ciq_persons = pull_ciq_persons(db)
    ciq_professionals = pull_ciq_professionals(db)
    ciq_compensation = pull_ciq_compensation(db)
    
    ciq_data = {
        'persons': ciq_persons,
        'professionals': ciq_professionals,
        'compensation': ciq_compensation
    }
    
    # ExecuComp
    execucomp_ceos = pull_execucomp_ceos(db)
    execucomp_full = pull_execucomp_full(db)
    
    execucomp_data = {
        'ceos': execucomp_ceos,
        'full': execucomp_full
    }
    
    # Cross-walks
    exec_bx_link = pull_exec_boardex_link(db)
    people_link = pull_wrds_people_link(db)
    ciq_gvkey = pull_ciq_gvkey(db)
    
    crosswalk_data = {
        'exec_boardex': exec_bx_link,
        'people_link': people_link,
        'ciq_gvkey': ciq_gvkey
    }
    
    if save:
        print("\n💾 Saving raw data...")
        for name, df in boardex_data.items():
            if not df.empty:
                df.to_parquet(PipelineConfig.OUTPUT_DIR / f'boardex_{name}.parquet', index=False)
        for name, df in ciq_data.items():
            if not df.empty:
                df.to_parquet(PipelineConfig.OUTPUT_DIR / f'ciq_{name}.parquet', index=False)
        for name, df in execucomp_data.items():
            if not df.empty:
                df.to_parquet(PipelineConfig.OUTPUT_DIR / f'execucomp_{name}.parquet', index=False)
        for name, df in crosswalk_data.items():
            if not df.empty:
                df.to_parquet(PipelineConfig.OUTPUT_DIR / f'crosswalk_{name}.parquet', index=False)
        print(f"   Saved to {PipelineConfig.OUTPUT_DIR}")
    
    return {
        'boardex': boardex_data,
        'ciq': ciq_data,
        'execucomp': execucomp_data,
        'crosswalk': crosswalk_data
    }

def run_stage2(raw_data=None, save=True):
    """Stage 2: Construct CEO variables."""
    print("\n" + "="*60)
    print("STAGE 2: VARIABLE CONSTRUCTION")
    print("="*60)
    
    # Load raw data if not provided
    if raw_data is None:
        print("\n📂 Loading raw data from disk...")
        
        def safe_load(path):
            try:
                return pd.read_parquet(path)
            except FileNotFoundError:
                print(f"   ⚠️ Not found: {path.name}")
                return pd.DataFrame()
        
        raw_data = {
            'boardex': {
                'profiles': safe_load(PipelineConfig.OUTPUT_DIR / 'boardex_profiles.parquet'),
                'employment': safe_load(PipelineConfig.OUTPUT_DIR / 'boardex_employment.parquet'),
                'education': safe_load(PipelineConfig.OUTPUT_DIR / 'boardex_education.parquet'),
                'companies': safe_load(PipelineConfig.OUTPUT_DIR / 'boardex_companies.parquet')
            },
            'ciq': {
                'persons': safe_load(PipelineConfig.OUTPUT_DIR / 'ciq_persons.parquet'),
                'professionals': safe_load(PipelineConfig.OUTPUT_DIR / 'ciq_professionals.parquet'),
                'compensation': safe_load(PipelineConfig.OUTPUT_DIR / 'ciq_compensation.parquet')
            },
            'execucomp': {
                'ceos': safe_load(PipelineConfig.OUTPUT_DIR / 'execucomp_ceos.parquet'),
                'full': safe_load(PipelineConfig.OUTPUT_DIR / 'execucomp_full.parquet')
            },
            'crosswalk': {
                'exec_boardex': safe_load(PipelineConfig.OUTPUT_DIR / 'crosswalk_exec_boardex.parquet'),
                'people_link': safe_load(PipelineConfig.OUTPUT_DIR / 'crosswalk_people_link.parquet')
            }
        }
    
    # Construct variables
    bx = raw_data['boardex']
    boardex_vars = construct_boardex_variables(
        bx['profiles'], bx['employment'], bx['education'], bx['companies']
    )
    
    ciq = raw_data['ciq']
    ciq_vars = construct_ciq_variables(
        ciq.get('persons', pd.DataFrame()),
        ciq.get('professionals', pd.DataFrame()),
        ciq.get('compensation', pd.DataFrame())
    )
    
    exec_data = raw_data['execucomp']
    execucomp_vars = construct_execucomp_variables(exec_data['ceos'], exec_data.get('full'))
    
    # Merge BoardEx + CIQ
    crosswalk = raw_data['crosswalk']
    combined_vars = merge_boardex_ciq(boardex_vars, ciq_vars, crosswalk.get('people_link', pd.DataFrame()))
    
    # Create gvkey-year panel
    ceo_panel = create_gvkey_year_panel(
        combined_vars,
        crosswalk['exec_boardex'],
        exec_data['ceos'],
        execucomp_vars
    )
    
    if save:
        print("\n💾 Saving constructed variables...")
        combined_vars.to_parquet(PipelineConfig.OUTPUT_DIR / 'ceo_combined_variables.parquet', index=False)
        ceo_panel.to_parquet(PipelineConfig.OUTPUT_DIR / PipelineConfig.CEO_VARIABLES, index=False)
        print(f"   Saved to {PipelineConfig.OUTPUT_DIR}")
    
    return {
        'boardex_vars': boardex_vars,
        'ciq_vars': ciq_vars,
        'execucomp_vars': execucomp_vars,
        'combined_vars': combined_vars,
        'ceo_panel': ceo_panel
    }

def run_stage3(core_panel_path, ceo_panel=None, save=True):
    """Stage 3: Link CEO variables to core panel."""
    print("\n" + "="*60)
    print("STAGE 3: CORE PANEL LINKAGE")
    print("="*60)
    
    if ceo_panel is None:
        ceo_panel = pd.read_parquet(PipelineConfig.OUTPUT_DIR / PipelineConfig.CEO_VARIABLES)
    
    merged = link_to_core_panel(ceo_panel, core_panel_path)
    
    if save:
        merged.to_parquet(PipelineConfig.OUTPUT_DIR / 'ceo_with_core_panel.parquet', index=False)
        print(f"   Saved to {PipelineConfig.OUTPUT_DIR / 'ceo_with_core_panel.parquet'}")
    
    return merged

def run_stage4(match_dataset_path, panel_with_ceo=None, save=True):
    """Stage 4: Merge with match quality dataset."""
    print("\n" + "="*60)
    print("STAGE 4: MATCH QUALITY MERGE")
    print("="*60)
    
    if panel_with_ceo is None:
        panel_with_ceo = pd.read_parquet(PipelineConfig.OUTPUT_DIR / 'ceo_with_core_panel.parquet')
    
    final = merge_match_quality(panel_with_ceo, match_dataset_path)
    
    if save:
        final.to_parquet(PipelineConfig.OUTPUT_DIR / PipelineConfig.MERGED_PANEL, index=False)
        print(f"   Saved to {PipelineConfig.OUTPUT_DIR / PipelineConfig.MERGED_PANEL}")
    
    return final

def run_all_data_stages():
    """Run stages 1-2 (data pull and variable construction)."""
    print("\n" + "="*60)
    print("🚀 CEO DATA PIPELINE - STAGES 1-2")
    print("="*60)
    
    # Connect to WRDS
    db = connect_wrds()
    if db is None:
        return None
    
    test_wrds_access(db)
    
    # Stage 1
    raw_data = run_stage1(db, save=True)
    
    # Stage 2
    results = run_stage2(raw_data, save=True)
    
    # Summary
    print("\n" + "="*60)
    print("📊 PIPELINE COMPLETE - STAGES 1-2")
    print("="*60)
    print(f"\nOutput files saved to: {PipelineConfig.OUTPUT_DIR}")
    print(f"\nCEO Variables Panel: {len(results['ceo_panel']):,} rows")
    print(f"Number of features: {len(results['ceo_panel'].columns)}")
    
    print("\n🔜 NEXT STEPS:")
    print("   1. Provide core panel path:")
    print(f"      python ceo_data_pipeline.py --stage 3 --core_panel YOUR_PANEL.csv")
    print("   2. Provide match quality dataset path:")
    print(f"      python ceo_data_pipeline.py --stage 4 --match_dataset YOUR_MATCH.csv")
    
    return results

# ================================================================
# SERVER-SIDE WRDS MERGE (Efficient Alternative)
# ================================================================

def pull_merged_ceo_panel(db, save=True):
    """
    Pull pre-merged CEO panel directly from WRDS using server-side SQL JOINs.
    
    This is much more efficient than pulling raw tables and merging locally because:
    1. Reduced data transfer (only the final merged result ~50K rows)
    2. Server-side optimization via WRDS PostgreSQL indexes
    3. Memory efficient (avoids loading 180MB+ raw CIQ tables)
    
    Returns DataFrame with gvkey-fiscalyear panel with 80+ CEO features.
    """
    print("\n" + "="*60)
    print("🚀 SERVER-SIDE WRDS MERGE")
    print("="*60)
    
    # Main merged query - joins ExecuComp → BoardEx → CIQ on server
    query = f"""
    WITH exec_ceos AS (
        -- ExecuComp CEO records
        SELECT 
            e.execid::text as execid, e.gvkey, e.year as fiscalyear, e.exec_fullname, 
            e.titleann, e.ceoann,
            e.tdc1 as total_comp_exec, e.salary as salary_exec, 
            e.bonus as bonus_exec,
            e.stock_awards_fv as stock_awards_exec, 
            e.option_awards_fv as option_awards_exec
        FROM execcomp.anncomp e
        WHERE e.ceoann = 'CEO'
          AND e.year >= {PipelineConfig.MIN_YEAR}
          AND e.year <= {PipelineConfig.MAX_YEAR}
    ),
    boardex_link AS (
        -- BoardEx profile linked via exec_boardex_link
        SELECT 
            link.execid::text as execid,
            bp.directorid,
            bp.dob, bp.gender, bp.nationality, bp.networksize, bp.age,
            EXTRACT(YEAR FROM bp.dob) as birth_year
        FROM wrdsapps.exec_boardex_link link
        INNER JOIN boardex.na_dir_profile_details bp 
            ON link.directorid = bp.directorid
    ),
    boardex_emp_stats AS (
        -- BoardEx employment statistics (aggregated per director)
        SELECT 
            emp.directorid,
            COUNT(*) as n_total_roles,
            COUNT(DISTINCT emp.companyid) as n_companies,
            SUM(CASE WHEN LOWER(emp.rolename) LIKE '%%ceo%%' OR LOWER(emp.rolename) LIKE '%%chief executive%%' THEN 1 ELSE 0 END) as n_ceo,
            SUM(CASE WHEN LOWER(emp.rolename) LIKE '%%cfo%%' OR LOWER(emp.rolename) LIKE '%%chief financial%%' THEN 1 ELSE 0 END) as n_cfo,
            SUM(CASE WHEN LOWER(emp.rolename) LIKE '%%coo%%' OR LOWER(emp.rolename) LIKE '%%chief operating%%' THEN 1 ELSE 0 END) as n_coo,
            SUM(CASE WHEN LOWER(emp.rolename) LIKE '%%director%%' OR LOWER(emp.rolename) LIKE '%%board%%' THEN 1 ELSE 0 END) as n_board,
            SUM(CASE WHEN LOWER(emp.rolename) LIKE '%%president%%' THEN 1 ELSE 0 END) as n_president,
            SUM(CASE WHEN LOWER(emp.rolename) LIKE '%%chairman%%' THEN 1 ELSE 0 END) as n_chairman,
            SUM(CASE WHEN LOWER(emp.rolename) LIKE '%%founder%%' THEN 1 ELSE 0 END) as n_founder,
            SUM(CASE WHEN LOWER(emp.rolename) LIKE '%%chief%%' THEN 1 ELSE 0 END) as n_csuite,
            MIN(EXTRACT(YEAR FROM emp.datestartrole::date)) as career_start_year,
            AVG(COALESCE(emp.dateendrole::date, CURRENT_DATE) - emp.datestartrole::date) as avg_tenure_days
        FROM boardex.na_dir_profile_emp emp
        WHERE emp.datestartrole IS NOT NULL
        GROUP BY emp.directorid
    ),
    ciq_comp AS (
        -- CIQ Compensation (has gvkey directly for matching)
        SELECT 
            comp.gvkey::text as ciq_gvkey,
            comp.year as ciq_year,
            comp.personid,
            comp.personname,
            comp.title as ciq_title,
            comp.profunctionname,
            comp.yearborn as ciq_yearborn,
            comp.rank as ciq_rank,
            comp.keyexecflag,
            comp.topkeyexecflag,
            comp.ctype1 as ciq_salary,
            comp.ctype2 as ciq_bonus,
            comp.ctype3 as ciq_stock_awards,
            comp.ctype4 as ciq_option_awards,
            comp.ctype8 as ciq_total_comp
        FROM ciq_pplintel.wrds_compensation comp
        WHERE comp.year >= {PipelineConfig.MIN_YEAR}
          AND (comp.topkeyexecflag = 1 OR comp.keyexecflag = 1)  -- Key executives
    )
    SELECT 
        -- Identifiers
        ec.execid, ec.gvkey, ec.fiscalyear, ec.exec_fullname, ec.titleann,
        
        -- ExecuComp Compensation
        ec.total_comp_exec, ec.salary_exec, ec.bonus_exec,
        ec.stock_awards_exec, ec.option_awards_exec,
        COALESCE(ec.stock_awards_exec, 0) + COALESCE(ec.option_awards_exec, 0) as equity_comp_exec,
        CASE WHEN ec.total_comp_exec > 0 THEN 
            (COALESCE(ec.stock_awards_exec, 0) + COALESCE(ec.option_awards_exec, 0)) / ec.total_comp_exec 
        ELSE NULL END as equity_ratio_exec,
        
        -- BoardEx Profile
        bl.directorid, bl.gender, bl.nationality, bl.networksize, bl.age, bl.birth_year,
        
        -- BoardEx Employment Stats
        bes.n_total_roles, bes.n_companies, bes.n_ceo, bes.n_cfo, bes.n_coo,
        bes.n_board, bes.n_president, bes.n_chairman, bes.n_founder, bes.n_csuite,
        bes.career_start_year, bes.avg_tenure_days,
        bes.avg_tenure_days / 365.25 as avg_tenure_years,
        EXTRACT(YEAR FROM CURRENT_DATE) - bes.career_start_year as career_length_years,
        CASE WHEN bes.n_ceo >= 2 THEN 1 ELSE 0 END as serial_ceo,
        CASE WHEN bes.n_board >= 3 THEN 1 ELSE 0 END as heavy_board_experience,
        CASE WHEN bes.n_companies >= 3 THEN 1 ELSE 0 END as multi_company,
        
        -- CIQ Compensation
        cc.personid as ciq_personid, cc.personname as ciq_personname,
        cc.ciq_title, cc.profunctionname,
        cc.ciq_salary, cc.ciq_bonus, cc.ciq_stock_awards, cc.ciq_option_awards, cc.ciq_total_comp,
        cc.ciq_rank, cc.keyexecflag as ciq_keyexecflag, cc.topkeyexecflag as ciq_topkeyexecflag,
        cc.ciq_yearborn,
        COALESCE(cc.ciq_stock_awards, 0) + COALESCE(cc.ciq_option_awards, 0) as ciq_equity_comp,
        CASE WHEN cc.ciq_total_comp > 0 THEN 
            (COALESCE(cc.ciq_stock_awards, 0) + COALESCE(cc.ciq_option_awards, 0)) / cc.ciq_total_comp 
        ELSE NULL END as ciq_equity_ratio
        
    FROM exec_ceos ec
    
    -- Link to BoardEx
    LEFT JOIN boardex_link bl ON ec.execid = bl.execid
    LEFT JOIN boardex_emp_stats bes ON bl.directorid = bes.directorid
    
    -- Link to CIQ (by gvkey + year)
    LEFT JOIN ciq_comp cc ON ec.gvkey = cc.ciq_gvkey AND ec.fiscalyear = cc.ciq_year
    
    ORDER BY ec.gvkey, ec.fiscalyear
    """
    
    print("\n📊 Executing server-side merge query...")
    print("   (This performs all JOINs on WRDS PostgreSQL)")
    
    try:
        df = db.raw_sql(query)
        print(f"   ✅ Retrieved {len(df):,} rows")
        
        # Type conversions
        for col in ['gvkey', 'execid', 'directorid', 'fiscalyear', 'ciq_personid']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        
        # Derived variables (ensure numeric types)
        if 'networksize' in df.columns:
            df['networksize'] = pd.to_numeric(df['networksize'], errors='coerce')
            q75 = df['networksize'].quantile(0.75)
            q90 = df['networksize'].quantile(0.90)
            df['well_connected'] = (df['networksize'] >= q75).fillna(False).astype(int)
            df['super_connected'] = (df['networksize'] >= q90).fillna(False).astype(int)
        
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['young_ceo'] = (df['age'] < 45).fillna(False).astype(int)
            df['experienced_ceo'] = (df['age'] >= 55).fillna(False).astype(int)
            df['prime_age'] = ((df['age'] >= 45) & (df['age'] < 55)).fillna(False).astype(int)
        
        if 'gender' in df.columns:
            df['is_female'] = (df['gender'].str.lower() == 'female').fillna(False).astype(int)
        
        # Leadership score
        csuite_cols = ['n_ceo', 'n_cfo', 'n_coo']
        if all(c in df.columns for c in csuite_cols):
            df['csuite_breadth'] = sum((df[c].fillna(0) > 0).astype(int) for c in csuite_cols)
            df['leadership_score'] = (
                df['n_ceo'].fillna(0) * 5 +
                df['n_president'].fillna(0) * 4 +
                df['n_coo'].fillna(0) * 3 +
                df['n_cfo'].fillna(0) * 3 +
                df['n_chairman'].fillna(0) * 3 +
                df['n_csuite'].fillna(0) * 2
            )
        
        # Deduplicate to gvkey-year (keep first per year)
        df = df.drop_duplicates(subset=['gvkey', 'fiscalyear'], keep='first')
        
        print(f"   ✅ Final panel: {len(df):,} gvkey-years, {len(df.columns)} features")
        
        # Coverage stats
        boardex_coverage = df['directorid'].notna().sum() / len(df) * 100
        ciq_coverage = df['ciq_personid'].notna().sum() / len(df) * 100
        print(f"   📊 BoardEx coverage: {boardex_coverage:.1f}%")
        print(f"   📊 CIQ coverage: {ciq_coverage:.1f}%")
        
        if save:
            output_path = PipelineConfig.OUTPUT_DIR / 'ceo_panel_serverside.parquet'
            df.to_parquet(output_path, index=False)
            print(f"\n💾 Saved to {output_path}")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()

def run_serverside_merge():
    """Run the server-side WRDS merge pipeline."""
    print("\n" + "="*60)
    print("🚀 CEO DATA PIPELINE - SERVER-SIDE MERGE")
    print("="*60)
    
    db = connect_wrds()
    if db is None:
        return None
    
    test_wrds_access(db)
    
    panel = pull_merged_ceo_panel(db, save=True)
    
    print("\n" + "="*60)
    print("📊 SERVER-SIDE MERGE COMPLETE")
    print("="*60)
    print(f"\nOutput file: {PipelineConfig.OUTPUT_DIR / 'ceo_panel_serverside.parquet'}")
    print(f"Panel size: {len(panel):,} rows x {len(panel.columns)} columns")
    
    return panel

# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='CEO Data Pipeline')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4], 
                        help='Run specific stage (1-4)')
    parser.add_argument('--all', action='store_true',
                        help='Run stages 1-2 (data-dependent stages)')
    parser.add_argument('--serverside', action='store_true',
                        help='Use efficient server-side WRDS merge (recommended)')
    parser.add_argument('--core_panel', type=str,
                        help='Path to core gvkey-year panel (for stage 3)')
    parser.add_argument('--match_dataset', type=str,
                        help='Path to match quality dataset (for stage 4)')
    parser.add_argument('--test', action='store_true',
                        help='Test WRDS connection only')
    
    args = parser.parse_args()
    
    if args.test:
        db = connect_wrds()
        if db:
            test_wrds_access(db)
        return
    
    if args.serverside:
        run_serverside_merge()
    elif args.all:
        run_all_data_stages()
    elif args.stage == 1:
        db = connect_wrds()
        if db:
            run_stage1(db)
    elif args.stage == 2:
        run_stage2()
    elif args.stage == 3:
        if not args.core_panel:
            print("❌ --core_panel required for stage 3")
            return
        run_stage3(args.core_panel)
    elif args.stage == 4:
        if not args.match_dataset:
            print("❌ --match_dataset required for stage 4")
            return
        run_stage4(args.match_dataset)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

