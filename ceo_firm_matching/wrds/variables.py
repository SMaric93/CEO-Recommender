"""
WRDS Variable Construction Functions.

Centralized functions for constructing CEO-level variables from raw WRDS data:
- BoardEx: Career trajectory, network, education, role patterns
- Capital IQ: Professional affiliations, compensation
- ExecuComp: Executive compensation metrics
"""
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd


def construct_boardex_variables(
    profiles: pd.DataFrame,
    employment: pd.DataFrame,
    education: pd.DataFrame,
    companies: pd.DataFrame
) -> pd.DataFrame:
    """
    Construct BoardEx-derived CEO variables.

    Args:
        profiles: Director profile demographics
        employment: Employment/directorship history
        education: Education records
        companies: Company profiles for sector information

    Returns:
        DataFrame with directorid as key and 50+ features
    """
    print("🔧 Constructing BoardEx Variables...")

    if employment.empty or profiles.empty:
        print("   ✗ Missing required BoardEx data")
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
    agg_cols = [
        'is_board', 'is_ceo', 'is_cfo', 'is_coo', 'is_cto', 'is_cmo',
        'is_cio', 'is_chro', 'is_clo', 'is_csuite', 'is_president',
        'is_chairman', 'is_founder', 'is_vp',
        'is_finance_role', 'is_ops_role', 'is_sales_role', 'is_marketing_role',
        'is_tech_role', 'is_legal_role', 'is_hr_role', 'is_strategy_role'
    ]

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
        edu['stem_edu_bx'] = edu['qualification'].str.contains(
            r'engineer|computer|math|physics|chemistry|biology|science', na=False
        )

        edu_agg = edu.groupby('directorid').agg({
            'has_mba_bx': 'max', 'has_phd_bx': 'max', 'has_jd_bx': 'max',
            'has_cpa_bx': 'max', 'has_cfa_bx': 'max', 'ivy_league': 'max',
            'top_bschool': 'max', 'elite_school_bx': 'max', 'stem_edu_bx': 'max',
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

    # CEO quality score
    stats['ceo_quality_score'] = (
        stats['n_ceo'].clip(upper=5) * 3 +
        stats['csuite_breadth'] * 2 +
        stats['n_board'].clip(upper=10) * 1 +
        stats.get('education_quality_score', pd.Series(0, index=stats.index)) +
        (stats['networksize'].fillna(0) / 100).clip(upper=10)
    )

    print(f"   ✓ BoardEx variables: {len(stats):,} directors, {len(stats.columns)} features")
    return stats


def construct_ciq_variables(
    ciq_persons: pd.DataFrame,
    ciq_professionals: pd.DataFrame,
    ciq_compensation: pd.DataFrame
) -> pd.DataFrame:
    """
    Construct Capital IQ-derived CEO variables.

    Args:
        ciq_persons: DataFrame from ciqperson (demographics)
        ciq_professionals: DataFrame from wrds_professional (affiliations)
        ciq_compensation: DataFrame from wrds_compensation (comp with gvkey)

    Returns:
        DataFrame with personid as key
    """
    print("🔧 Constructing CIQ Variables...")

    # Start with persons if available, else professionals
    if not ciq_persons.empty:
        stats = ciq_persons[['personid', 'yearborn']].drop_duplicates().copy()
    elif not ciq_professionals.empty:
        stats = ciq_professionals[['personid', 'yearborn']].drop_duplicates().copy()
    else:
        print("   ✗ No CIQ data available")
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
        aff_agg.columns = [
            'personid', 'n_companies_ciq', 'n_ceo_ciq', 'n_cfo_ciq', 'n_coo_ciq',
            'n_csuite_ciq', 'n_board_ciq', 'n_key_exec_ciq',
            'avg_tenure_ciq', 'career_start_ciq', 'n_roles_ciq', 'n_countries_ciq'
        ]
        stats = stats.merge(aff_agg, on='personid', how='left')

        # Serial CEO
        stats['serial_ceo_ciq'] = (stats['n_ceo_ciq'].fillna(0) >= 2).astype(int)
        stats['international_exp_ciq'] = (stats['n_countries_ciq'].fillna(0) > 1).astype(int)

    # Compensation variables from wrds_compensation
    if not ciq_compensation.empty:
        comp = ciq_compensation.copy()
        for col in ['salary', 'bonus', 'stock_awards', 'option_awards', 'total_comp']:
            if col in comp.columns:
                comp[col] = pd.to_numeric(comp[col], errors='coerce')

        comp['equity_comp_ciq'] = comp['stock_awards'].fillna(0) + comp['option_awards'].fillna(0)
        comp['equity_ratio_ciq'] = comp['equity_comp_ciq'] / comp['total_comp'].replace(0, np.nan)

        comp_agg = comp.groupby('personid').agg({
            'total_comp': ['mean', 'max'],
            'salary': 'mean',
            'equity_ratio_ciq': 'mean',
            'gvkey': 'nunique',
            'fiscalyear': ['min', 'max']
        }).reset_index()
        comp_agg.columns = [
            'personid', 'avg_total_comp_ciq', 'max_total_comp_ciq',
            'avg_salary_ciq', 'equity_orientation_ciq',
            'n_gvkeys_ciq', 'first_year_ciq', 'last_year_ciq'
        ]
        stats = stats.merge(comp_agg, on='personid', how='left')

        # Compensation trajectory
        stats['comp_span_ciq'] = stats['last_year_ciq'] - stats['first_year_ciq']

    print(f"   ✓ CIQ variables: {len(stats):,} persons, {len(stats.columns)} features")
    return stats


def construct_execucomp_variables(
    execucomp_ceos: pd.DataFrame,
    execucomp_full: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Construct ExecuComp-derived compensation variables.

    Args:
        execucomp_ceos: CEO compensation records
        execucomp_full: Optional full executive compensation for peer comparison

    Returns:
        DataFrame with (execid, fiscalyear) as key
    """
    print("🔧 Constructing ExecuComp Variables...")

    if execucomp_ceos.empty:
        print("   ✗ Missing ExecuComp data")
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

    print(f"   ✓ ExecuComp variables: {len(df):,} CEO-years, {len(df.columns)} features")
    return df
