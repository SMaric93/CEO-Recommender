"""
WRDS Data Pull Functions.

Centralized functions for pulling data from WRDS:
- BoardEx: Profiles, employment, education, companies
- Capital IQ: Persons, professionals, compensation
- ExecuComp: CEOs, full compensation
- Crosswalks: ExecuComp-BoardEx, BoardEx-CIQ, People link
"""
from typing import Optional
import pandas as pd


def pull_boardex_profiles(db, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Pull BoardEx director profiles.

    Args:
        db: WRDS connection object
        limit: Optional row limit for testing

    Returns:
        DataFrame with directorid, demographics, and network information
    """
    print("📊 Pulling BoardEx Profiles...")
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
        print(f"   ✓ Retrieved {len(df):,} profiles")
        return df
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return pd.DataFrame()


def pull_boardex_employment(db, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Pull BoardEx employment/directorship history.

    Args:
        db: WRDS connection object
        limit: Optional row limit for testing

    Returns:
        DataFrame with employment records including roles and dates
    """
    print("📊 Pulling BoardEx Employment...")
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
        print(f"   ✓ Retrieved {len(df):,} employment records")
        return df
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return pd.DataFrame()


def pull_boardex_education(db, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Pull BoardEx education records.

    Args:
        db: WRDS connection object
        limit: Optional row limit for testing

    Returns:
        DataFrame with education qualifications and institutions
    """
    print("📊 Pulling BoardEx Education...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT directorid, qualification, institutionname
    FROM boardex.na_dir_profile_edu
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        print(f"   ✓ Retrieved {len(df):,} education records")
        return df
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return pd.DataFrame()


def pull_boardex_companies(db, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Pull BoardEx company profiles.

    Args:
        db: WRDS connection object
        limit: Optional row limit for testing

    Returns:
        DataFrame with company identifiers and sector
    """
    print("📊 Pulling BoardEx Companies...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT boardid as companyid, boardname as companyname, isin, ticker, sector
    FROM boardex.na_wrds_company_profile
    WHERE ticker IS NOT NULL
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        print(f"   ✓ Retrieved {len(df):,} companies")
        return df
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return pd.DataFrame()


def pull_ciq_persons(db, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Pull CIQ person demographics from ciq_pplintel.ciqperson.

    Args:
        db: WRDS connection object
        limit: Optional row limit for testing

    Returns:
        DataFrame with person demographics
    """
    print("📊 Pulling CIQ Persons...")
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
        print(f"   ✓ Retrieved {len(df):,} persons")
        return df
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return pd.DataFrame()


def pull_ciq_professionals(db, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Pull CIQ professional affiliations from ciq_pplintel.wrds_professional.

    Args:
        db: WRDS connection object
        limit: Optional row limit for testing

    Returns:
        DataFrame with professional affiliations and role flags
    """
    print("📊 Pulling CIQ Professional Affiliations...")
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
        print(f"   ✓ Retrieved {len(df):,} affiliations, {df['personid'].nunique():,} persons")
        return df
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return pd.DataFrame()


def pull_ciq_compensation(db, min_year: int = 1990, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Pull CIQ compensation from ciq_pplintel.wrds_compensation.

    Args:
        db: WRDS connection object
        min_year: Minimum fiscal year to include
        limit: Optional row limit for testing

    Returns:
        DataFrame with compensation data including gvkey
    """
    print("📊 Pulling CIQ Compensation...")
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT companyid, gvkey, year as fiscalyear, personid, personname, title,
           profunctionname, yearborn, rank, keyexecflag, topkeyexecflag,
           ctype1 as salary, ctype2 as bonus, ctype3 as stock_awards,
           ctype4 as option_awards, ctype5 as non_equity_incentive,
           ctype7 as all_other_comp, ctype8 as total_comp
    FROM ciq_pplintel.wrds_compensation
    WHERE year >= {min_year}
    {limit_clause}
    """
    try:
        df = db.raw_sql(query)
        for col in ['personid', 'companyid', 'gvkey', 'fiscalyear']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        print(f"   ✓ Retrieved {len(df):,} records, {df['gvkey'].nunique():,} gvkeys")
        return df
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return pd.DataFrame()


def pull_execucomp_ceos(db, min_year: int = 1990) -> pd.DataFrame:
    """
    Pull ExecuComp CEO records.

    Args:
        db: WRDS connection object
        min_year: Minimum fiscal year to include

    Returns:
        DataFrame with CEO compensation by gvkey-year
    """
    print("📊 Pulling ExecuComp CEOs...")
    query = f"""
    SELECT execid, gvkey, year as fiscalyear, ceoann, exec_fullname, titleann,
           tdc1, salary, bonus, stock_awards_fv, option_awards_fv
    FROM execcomp.anncomp
    WHERE ceoann = 'CEO'
    AND year >= {min_year}
    """
    try:
        df = db.raw_sql(query)
        for col in ['execid', 'gvkey', 'fiscalyear']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        print(f"   ✓ Retrieved {len(df):,} CEO-years, {df['execid'].nunique():,} CEOs")
        return df
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return pd.DataFrame()


def pull_execucomp_full(db, min_year: int = 1990) -> pd.DataFrame:
    """
    Pull all ExecuComp records (not just CEO) for additional compensation variables.

    Args:
        db: WRDS connection object
        min_year: Minimum fiscal year to include

    Returns:
        DataFrame with full executive compensation
    """
    print("📊 Pulling ExecuComp Full...")
    query = f"""
    SELECT execid, gvkey, year as fiscalyear, exec_fullname, titleann,
           tdc1, tdc2, salary, bonus, stock_awards_fv, option_awards_fv,
           othcomp, ltip, allothtot, pension_chg
    FROM execcomp.anncomp
    WHERE year >= {min_year}
    """
    try:
        df = db.raw_sql(query)
        for col in ['execid', 'gvkey', 'fiscalyear']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        print(f"   ✓ Retrieved {len(df):,} exec-years")
        return df
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return pd.DataFrame()


def pull_exec_boardex_link(db) -> pd.DataFrame:
    """
    Pull ExecuComp-BoardEx crosswalk.

    Args:
        db: WRDS connection object

    Returns:
        DataFrame mapping execid to directorid
    """
    print("📊 Pulling ExecuComp-BoardEx Link...")
    query = """
    SELECT execid, directorid, exec_fullname, directorname, score
    FROM wrdsapps.exec_boardex_link
    WHERE execid IS NOT NULL AND directorid IS NOT NULL
    """
    try:
        df = db.raw_sql(query)
        df['execid'] = pd.to_numeric(df['execid'], errors='coerce').astype('Int64')
        df['directorid'] = pd.to_numeric(df['directorid'], errors='coerce').astype('Int64')
        print(f"   ✓ Retrieved {len(df):,} links")
        return df
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return pd.DataFrame()


def pull_boardex_ciq_company_match(db) -> pd.DataFrame:
    """
    Pull BoardEx-CIQ company-level match.

    Args:
        db: WRDS connection object

    Returns:
        DataFrame mapping BoardEx company to CIQ company
    """
    print("📊 Pulling BoardEx-CIQ Company Match...")
    query = """
    SELECT boardex_boardid as boardex_companyid,
           ciq_companyid,
           matchscore
    FROM wrdsapps.boardex_ciq_company_match
    WHERE boardex_boardid IS NOT NULL AND ciq_companyid IS NOT NULL
    """
    try:
        df = db.raw_sql(query)
        print(f"   ✓ Retrieved {len(df):,} company matches")
        return df
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return pd.DataFrame()


def pull_wrds_people_link(db) -> pd.DataFrame:
    """
    Pull direct BoardEx-CIQ person link.

    Args:
        db: WRDS connection object

    Returns:
        DataFrame mapping BoardEx directorid to CIQ personid
    """
    print("📊 Pulling WRDS People Link...")
    try:
        query = """
        SELECT boardex_directorid as directorid, ciq_personid as personid
        FROM wrdsapps.peoplelink
        WHERE boardex_directorid IS NOT NULL AND ciq_personid IS NOT NULL
        """
        df = db.raw_sql(query)
        print(f"   ✓ Retrieved {len(df):,} person links")
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
            print(f"   ✓ Retrieved {len(df):,} person links (via link_boardex_ciq)")
            return df
        except:
            return pd.DataFrame()


def pull_ciq_gvkey(db) -> pd.DataFrame:
    """
    Pull CIQ company to GVKEY mapping.

    Args:
        db: WRDS connection object

    Returns:
        DataFrame mapping CIQ companyid to gvkey
    """
    print("📊 Pulling CIQ-GVKEY Mapping...")
    query = """
    SELECT companyid as ciq_companyid, gvkey
    FROM ciq.wrds_gvkey
    WHERE gvkey IS NOT NULL
    """
    try:
        df = db.raw_sql(query)
        df['gvkey'] = pd.to_numeric(df['gvkey'], errors='coerce').astype('Int64')
        print(f"   ✓ Retrieved {len(df):,} mappings")
        return df
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return pd.DataFrame()
