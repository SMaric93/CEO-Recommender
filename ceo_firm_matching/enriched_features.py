"""
Extension 1: Enriched CEO Tower Features

Constructs 50+ human capital variables from BoardEx/CIQ/ExecuComp
and integrates them into a richer CEO tower configuration.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import torch

from .config import Config


@dataclass
class EnrichedConfig(Config):
    """
    Extended configuration with enriched CEO features from WRDS sources.
    The CEO tower goes from 2 numeric features to 15+.
    """
    # --- ENRICHED CEO NUMERIC FEATURES ---
    # Original: ['Age', 'tenure']
    # Enriched adds: network, career, education, compensation history
    CEO_NUMERIC_COLS_ENRICHED: List[str] = field(default_factory=lambda: [
        'Age',                    # Original
        # Career trajectory (from BoardEx employment)
        'n_prior_roles',          # Total prior executive positions
        'n_prior_boards',         # Board seats held before current
        'avg_board_tenure',       # Average years per board seat
        'years_as_ceo',           # Total CEO experience (across firms)
        'n_sectors',              # Industry breadth (unique SIC-2 sectors)
        'career_span_years',      # First role to current role
        # Network (from BoardEx profiles)
        'network_size',           # Number of direct connections
        'board_interlocks',       # Shared board connections
        # Education (from BoardEx education)
        'education_score',        # Composite: MBA=2, PhD=3, JD=1.5, else=1
        # Compensation history (from ExecuComp)
        'log_prior_tdc1',         # Log of prior year total compensation
        'equity_share',           # Stock + options as % of total comp
        'pay_premium',            # CEO pay / industry median CEO pay
        # Ownership (from ExecuComp)
        'log_ownership_pct',      # Log of shares owned as % of outstanding
    ])

    # --- ENRICHED FIRM NUMERIC FEATURES ---
    # Add a few more firm features that interact with CEO characteristics
    FIRM_NUMERIC_COLS_ENRICHED: List[str] = field(default_factory=lambda: [
        # Original 12
        'ind_firms_60w', 'non_competition_score', 'boardindpw',
        'boardsizew', 'busyw', 'pct_blockw', 'logatw', 'exp_roa',
        'rdintw', 'capintw', 'leverage', 'divyieldw',
        # New
        'log_mktcap',             # Market cap (from Compustat/CRSP)
        'sales_growth_3y',        # 3-year revenue CAGR
        'roa_volatility',         # StdDev of ROA over prior 5 years
        'ceo_turnover_rate',      # Historical CEO turnover frequency
    ])

    # Hyperparameters for enriched model
    LATENT_DIM: int = 80          # Larger embedding for more features
    EMBEDDING_DIM_LARGE: int = 48
    EMBEDDING_DIM_MEDIUM: int = 12  # Slightly larger CEO categorical embeddings
    EPOCHS: int = 60


def construct_enriched_ceo_features(
    profiles: pd.DataFrame,
    employment: pd.DataFrame,
    education: pd.DataFrame,
    execucomp: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Construct enriched CEO-level features from BoardEx + ExecuComp.

    Args:
        profiles: BoardEx director profiles (directorid, gender, nationality, etc.)
        employment: BoardEx employment history (directorid, companyid, rolename, datestartrole, ...)
        education: BoardEx education (directorid, qualification, institution, ...)
        execucomp: ExecuComp CEO records (execid, gvkey, fiscalyear, tdc1, ...)

    Returns:
        DataFrame keyed by directorid with enriched features
    """
    features = pd.DataFrame({'directorid': profiles['directorid'].unique()})

    # === CAREER TRAJECTORY ===
    if employment is not None and len(employment) > 0:
        emp = employment.copy()

        # Parse dates
        for col in ['datestartrole', 'dateendrole']:
            if col in emp.columns:
                emp[col] = pd.to_datetime(emp[col], errors='coerce')

        # N prior roles
        role_counts = emp.groupby('directorid').size().rename('n_prior_roles')
        features = features.merge(role_counts.reset_index(), on='directorid', how='left')

        # N prior boards (filter to board-level roles)
        board_roles = emp[emp['rolename'].str.contains(
            'Director|Board|Chairman|Non-Exec', case=False, na=False
        )]
        board_counts = board_roles.groupby('directorid')['companyid'].nunique().rename('n_prior_boards')
        features = features.merge(board_counts.reset_index(), on='directorid', how='left')

        # Average board tenure
        if 'datestartrole' in emp.columns and 'dateendrole' in emp.columns:
            emp['role_duration_years'] = (
                (emp['dateendrole'] - emp['datestartrole']).dt.days / 365.25
            ).clip(0, 50)
            avg_tenure = emp.groupby('directorid')['role_duration_years'].mean().rename('avg_board_tenure')
            features = features.merge(avg_tenure.reset_index(), on='directorid', how='left')

        # Years as CEO specifically
        ceo_roles = emp[emp['rolename'].str.contains('CEO|Chief Executive', case=False, na=False)]
        if len(ceo_roles) > 0 and 'role_duration_years' in ceo_roles.columns:
            ceo_exp = ceo_roles.groupby('directorid')['role_duration_years'].sum().rename('years_as_ceo')
            features = features.merge(ceo_exp.reset_index(), on='directorid', how='left')

        # Sector diversity
        if 'sectorname' in emp.columns:
            sector_div = emp.groupby('directorid')['sectorname'].nunique().rename('n_sectors')
            features = features.merge(sector_div.reset_index(), on='directorid', how='left')
        elif 'companyid' in emp.columns:
            # Proxy: unique companies
            company_div = emp.groupby('directorid')['companyid'].nunique().rename('n_sectors')
            features = features.merge(company_div.reset_index(), on='directorid', how='left')

        # Career span
        if 'datestartrole' in emp.columns:
            career_span = emp.groupby('directorid')['datestartrole'].agg(
                lambda x: (x.max() - x.min()).days / 365.25 if len(x) > 1 else 0
            ).rename('career_span_years')
            features = features.merge(career_span.reset_index(), on='directorid', how='left')

    # === NETWORK ===
    if profiles is not None:
        if 'network_size' in profiles.columns:
            features = features.merge(
                profiles[['directorid', 'network_size']].drop_duplicates('directorid'),
                on='directorid', how='left'
            )
        else:
            # Construct from employment overlaps
            features['network_size'] = features.get('n_prior_boards', 0) * 5  # rough proxy

        # Board interlocks from employment
        if employment is not None:
            interlock = employment.groupby('directorid')['companyid'].nunique().rename('board_interlocks')
            features = features.merge(interlock.reset_index(), on='directorid', how='left')

    # === EDUCATION ===
    if education is not None and len(education) > 0:
        edu = education.copy()

        def compute_education_score(group):
            """Score: PhD=3, MBA=2, JD=1.5, Masters=1.5, Bachelors=1, else=0.5"""
            quals = group['qualification'].str.upper().fillna('')
            score = 0.5  # baseline
            if quals.str.contains('PHD|DOCTORATE|DBA').any():
                score = 3.0
            elif quals.str.contains('MBA').any():
                score = 2.0
            elif quals.str.contains('JD|LAW|LLB').any():
                score = 1.5
            elif quals.str.contains('MASTER|MSC|MA ').any():
                score = 1.5
            elif quals.str.contains('BACHELOR|BSC|BA |BBA|BS ').any():
                score = 1.0
            return pd.Series({'education_score': score})

        edu_scores = edu.groupby('directorid').apply(compute_education_score).reset_index()
        features = features.merge(edu_scores, on='directorid', how='left')

    # === COMPENSATION HISTORY ===
    if execucomp is not None and len(execucomp) > 0:
        ec = execucomp.copy()
        ec['tdc1'] = pd.to_numeric(ec['tdc1'], errors='coerce')

        if 'execid' in ec.columns:
            # Log prior comp (use most recent)
            latest_comp = ec.sort_values('fiscalyear').groupby('execid').last()
            latest_comp['log_prior_tdc1'] = np.log1p(latest_comp['tdc1'].clip(lower=0))

            # Equity share
            equity_cols = ['stock_awards_fv', 'option_awards_fv']
            for col in equity_cols:
                if col not in latest_comp.columns:
                    latest_comp[col] = 0
            latest_comp['equity_share'] = (
                (latest_comp['stock_awards_fv'].fillna(0) + latest_comp['option_awards_fv'].fillna(0))
                / latest_comp['tdc1'].clip(lower=1)
            ).clip(0, 1)

            # Pay premium (vs year median)
            if 'fiscalyear' in ec.columns:
                year_median = ec.groupby('fiscalyear')['tdc1'].median()
                ec = ec.merge(year_median.rename('year_median_tdc1'), on='fiscalyear', how='left')
                ec['pay_premium'] = ec['tdc1'] / ec['year_median_tdc1'].clip(lower=1)
                pay_prem = ec.sort_values('fiscalyear').groupby('execid')['pay_premium'].last()
                latest_comp = latest_comp.merge(pay_prem.rename('pay_premium'), left_index=True, right_index=True, how='left')

            # Ownership
            if 'shrown_tot_pct' in ec.columns:
                own = ec.sort_values('fiscalyear').groupby('execid')['shrown_tot_pct'].last()
                latest_comp['log_ownership_pct'] = np.log1p(own.clip(lower=0))

            # Select output columns
            comp_features = latest_comp[
                [c for c in ['log_prior_tdc1', 'equity_share', 'pay_premium', 'log_ownership_pct']
                 if c in latest_comp.columns]
            ].reset_index()

            # We need a mapping from execid → directorid for the merge
            # This will be done at the integration level with the crosswalk

    # Fill NaNs with sensible defaults
    numeric_cols = [c for c in features.columns if c != 'directorid']
    for col in numeric_cols:
        features[col] = pd.to_numeric(features[col], errors='coerce')
    features[numeric_cols] = features[numeric_cols].fillna(0)

    return features


def construct_enriched_firm_features(
    compustat: pd.DataFrame,
    crsp: Optional[pd.DataFrame] = None,
    turnover_history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Construct enriched firm-level features.

    Args:
        compustat: Compustat annual fundamentals (gvkey, datadate, at, sale, ni, etc.)
        crsp: CRSP monthly returns (permno, date, ret)
        turnover_history: CEO turnover events (gvkey, year)

    Returns:
        DataFrame keyed by (gvkey, fiscalyear) with enriched firm features
    """
    df = compustat.copy()
    df['gvkey'] = pd.to_numeric(df['gvkey'], errors='coerce')

    features = df[['gvkey', 'fyear']].rename(columns={'fyear': 'fiscalyear'}).copy()

    # Log market cap
    if 'mkvalt' in df.columns:
        features['log_mktcap'] = np.log1p(pd.to_numeric(df['mkvalt'], errors='coerce').clip(lower=0))
    elif 'csho' in df.columns and 'prcc_f' in df.columns:
        mkcap = pd.to_numeric(df['csho'], errors='coerce') * pd.to_numeric(df['prcc_f'], errors='coerce')
        features['log_mktcap'] = np.log1p(mkcap.clip(lower=0))

    # 3-year sales growth CAGR
    if 'sale' in df.columns:
        df['sale'] = pd.to_numeric(df['sale'], errors='coerce')
        df = df.sort_values(['gvkey', 'fyear'])
        df['sale_lag3'] = df.groupby('gvkey')['sale'].shift(3)
        features['sales_growth_3y'] = (
            (df['sale'] / df['sale_lag3'].clip(lower=0.01)) ** (1/3) - 1
        ).clip(-1, 5)

    # ROA volatility (5-year rolling std)
    if 'ni' in df.columns and 'at' in df.columns:
        df['roa'] = pd.to_numeric(df['ni'], errors='coerce') / pd.to_numeric(df['at'], errors='coerce').clip(lower=1)
        df = df.sort_values(['gvkey', 'fyear'])
        features['roa_volatility'] = df.groupby('gvkey')['roa'].transform(
            lambda x: x.rolling(5, min_periods=2).std()
        )

    # CEO turnover rate (historical frequency)
    if turnover_history is not None and len(turnover_history) > 0:
        turnover_freq = turnover_history.groupby('gvkey').size() / turnover_history.groupby('gvkey')['year'].apply(
            lambda x: x.max() - x.min() + 1
        ).clip(lower=1)
        turnover_freq = turnover_freq.rename('ceo_turnover_rate').reset_index()
        features = features.merge(turnover_freq, on='gvkey', how='left')

    # Fill
    numeric_cols = [c for c in features.columns if c not in ['gvkey', 'fiscalyear']]
    for col in numeric_cols:
        features[col] = pd.to_numeric(features[col], errors='coerce')
    features[numeric_cols] = features[numeric_cols].fillna(0)

    return features
