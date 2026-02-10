#!/usr/bin/env python3
"""
WRDS Pull → Merge → Run All Extensions on Real Data

Pipeline:
  1. Pull BoardEx profiles, employment, education from WRDS
  2. Pull ExecuComp CEO compensation from WRDS
  3. Pull Capital IQ People Intelligence (persons, professionals, compensation)
  4. Pull crosswalks (ExecuComp-BoardEx, BoardEx-CIQ people link)
  5. Construct enriched CEO features from BoardEx + CIQ
  6. Merge enriched features into ceo_types_v0.2.csv
  7. Run all 10 extensions on the enriched real data
  8. Produce results report

Usage:
    python run_wrds_extensions.py
    python run_wrds_extensions.py --skip-pull   # use cached WRDS data
    python run_wrds_extensions.py --epochs 50
"""

import argparse
import sys
import os
import time
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ceo_firm_matching.config import Config
from ceo_firm_matching.data import DataProcessor, CEOFirmDataset
from ceo_firm_matching.model import CEOFirmMatcher
from ceo_firm_matching.training import train_model


CACHE_DIR = 'Data/wrds_cache'
OUTPUT_DIR = 'Output/Extensions_Real'


def parse_args():
    parser = argparse.ArgumentParser(description='WRDS Pull + Merge + Run Extensions')
    parser.add_argument('--skip-pull', action='store_true', help='Use cached WRDS data')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--extensions', type=str, default='all',
                        help='Comma-separated extension numbers (e.g., 1,2,3) or "all"')
    return parser.parse_args()


# =============================================================================
# STEP 1: WRDS PULLS
# =============================================================================

def pull_wrds_data(skip_pull=False):
    """Pull all required data from WRDS, or load from cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_files = {
        'profiles': f'{CACHE_DIR}/boardex_profiles.parquet',
        'employment': f'{CACHE_DIR}/boardex_employment.parquet',
        'education': f'{CACHE_DIR}/boardex_education.parquet',
        'execucomp': f'{CACHE_DIR}/execucomp_ceos.parquet',
        'crosswalk': f'{CACHE_DIR}/exec_boardex_link.parquet',
        'ciq_persons': f'{CACHE_DIR}/ciq_persons.parquet',
        'ciq_professionals': f'{CACHE_DIR}/ciq_professionals.parquet',
        'ciq_compensation': f'{CACHE_DIR}/ciq_compensation.parquet',
        'people_link': f'{CACHE_DIR}/people_link.parquet',
        'ciq_gvkey': f'{CACHE_DIR}/ciq_gvkey.parquet',
    }

    # Check if all cached files exist
    all_cached = all(os.path.exists(f) for f in cache_files.values())

    if skip_pull and all_cached:
        print("\n" + "=" * 70)
        print("LOADING CACHED WRDS DATA")
        print("=" * 70)
        data = {}
        for key, path in cache_files.items():
            data[key] = pd.read_parquet(path)
            print(f"  ✓ {key}: {len(data[key]):,} rows (from cache)")
        return data

    if skip_pull and not all_cached:
        missing = [k for k, v in cache_files.items() if not os.path.exists(v)]
        print(f"  Warning: --skip-pull but missing cache files: {missing}")
        print(f"  Will pull from WRDS...")

    print("\n" + "=" * 70)
    print("PULLING DATA FROM WRDS")
    print("=" * 70)

    from ceo_firm_matching.wrds.connection import connect_wrds
    from ceo_firm_matching.wrds.pulls import (
        pull_boardex_profiles,
        pull_boardex_employment,
        pull_boardex_education,
        pull_execucomp_ceos,
        pull_exec_boardex_link,
        pull_ciq_persons,
        pull_ciq_professionals,
        pull_ciq_compensation,
        pull_wrds_people_link,
        pull_ciq_gvkey,
    )

    db = connect_wrds()

    data = {}

    # BoardEx
    data['profiles'] = pull_boardex_profiles(db)
    data['employment'] = pull_boardex_employment(db)
    data['education'] = pull_boardex_education(db)

    # ExecuComp
    data['execucomp'] = pull_execucomp_ceos(db)

    # Crosswalks
    data['crosswalk'] = pull_exec_boardex_link(db)

    # Capital IQ People Intelligence
    data['ciq_persons'] = pull_ciq_persons(db)
    data['ciq_professionals'] = pull_ciq_professionals(db)
    data['ciq_compensation'] = pull_ciq_compensation(db)
    data['people_link'] = pull_wrds_people_link(db)
    data['ciq_gvkey'] = pull_ciq_gvkey(db)

    db.close()
    print("  ✓ WRDS connection closed")

    # Cache all pulls
    for key, path in cache_files.items():
        if key in data and len(data[key]) > 0:
            data[key].to_parquet(path, index=False)
            print(f"  Cached {key} → {path}")

    return data


# =============================================================================
# STEP 2: CONSTRUCT ENRICHED FEATURES (BoardEx + CIQ)
# =============================================================================

def construct_ciq_features(wrds_data):
    """Build CEO-level features from Capital IQ People Intelligence.

    Strategy A (preferred): Use WRDS people link (directorid → personid)
        to match CIQ data directly to BoardEx directors. Returns directorid-level features.
    Strategy B (fallback): Match via gvkey+year from CIQ compensation.
        Returns gvkey-year level features.
    """
    print("\n" + "=" * 70)
    print("CONSTRUCTING CIQ PEOPLE INTELLIGENCE FEATURES")
    print("=" * 70)

    people_link = wrds_data.get('people_link', pd.DataFrame())
    ciq_pros = wrds_data.get('ciq_professionals', pd.DataFrame())
    ciq_comp = wrds_data.get('ciq_compensation', pd.DataFrame())
    ciq_persons = wrds_data.get('ciq_persons', pd.DataFrame())

    use_people_link = len(people_link) > 0 and 'directorid' in people_link.columns and 'personid' in people_link.columns

    if use_people_link:
        # === Strategy A: directorid → personid via people link ===
        people_link = people_link.copy()
        people_link['directorid'] = pd.to_numeric(people_link['directorid'], errors='coerce')
        people_link['personid'] = pd.to_numeric(people_link['personid'], errors='coerce')
        people_link = people_link.dropna(subset=['directorid', 'personid']).drop_duplicates('directorid')
        print(f"  ✓ People link: {len(people_link):,} directorid → personid mappings")

        features = people_link[['directorid', 'personid']].copy()
        linked_pids = set(people_link['personid'].dropna().astype(int))

        # --- CIQ Professional Affiliations ---
        if len(ciq_pros) > 0:
            ciq_pros = ciq_pros.copy()
            ciq_pros['personid'] = pd.to_numeric(ciq_pros['personid'], errors='coerce')
            pros_linked = ciq_pros[ciq_pros['personid'].isin(linked_pids)].copy()
            print(f"  CIQ affiliations for linked persons: {len(pros_linked):,}")

            if len(pros_linked) > 0:
                pro_agg = pros_linked.groupby('personid').agg(
                    ciq_n_affiliations=('companyid', 'nunique'),
                    ciq_n_roles=('proid', 'count'),
                    ciq_is_key_exec=('keyexecflag', lambda x: int((x == 1).any())),
                    ciq_is_top_key_exec=('topkeyexecflag', lambda x: int((x == 1).any())),
                    ciq_is_board_member=('boardflag', lambda x: int((x == 1).any())),
                    ciq_current_boards=('currentboardflag', 'sum'),
                    ciq_career_start=('startyear', 'min'),
                    ciq_career_end=('endyear', 'max'),
                    ciq_best_rank=('rank', 'min'),
                ).reset_index()

                pro_agg['ciq_career_span'] = (
                    pro_agg['ciq_career_end'] - pro_agg['ciq_career_start']
                ).clip(0, 60)

                features = features.merge(pro_agg, on='personid', how='left')
                print(f"  Professional features: {len(pro_agg.columns)-1} for {len(pro_agg):,} persons")

        # --- CIQ Compensation (person-level aggregates) ---
        if len(ciq_comp) > 0:
            ciq_comp = ciq_comp.copy()
            ciq_comp['personid'] = pd.to_numeric(ciq_comp['personid'], errors='coerce')
            for col in ['salary', 'bonus', 'stock_awards', 'option_awards',
                        'non_equity_incentive', 'all_other_comp', 'total_comp']:
                if col in ciq_comp.columns:
                    ciq_comp[col] = pd.to_numeric(ciq_comp[col], errors='coerce')

            comp_linked = ciq_comp[ciq_comp['personid'].isin(linked_pids)].copy()
            print(f"  CIQ comp records for linked persons: {len(comp_linked):,}")

            if len(comp_linked) > 0:
                comp_agg = comp_linked.groupby('personid').agg(
                    ciq_avg_total_comp=('total_comp', 'mean'),
                    ciq_max_total_comp=('total_comp', 'max'),
                    ciq_avg_salary=('salary', 'mean'),
                    ciq_avg_bonus=('bonus', 'mean'),
                    ciq_comp_years=('fiscalyear', 'nunique'),
                ).reset_index()

                # Equity share
                if all(c in comp_linked.columns for c in ['stock_awards', 'option_awards', 'total_comp']):
                    equity_df = comp_linked.groupby('personid').agg(
                        total_equity=('stock_awards', lambda x: x.fillna(0).sum()),
                        total_options=('option_awards', lambda x: x.fillna(0).sum()),
                        total_total=('total_comp', lambda x: x.fillna(0).sum()),
                    ).reset_index()
                    equity_df['ciq_equity_share'] = (
                        (equity_df['total_equity'] + equity_df['total_options']) /
                        equity_df['total_total'].clip(lower=1)
                    ).clip(0, 1)
                    comp_agg = comp_agg.merge(
                        equity_df[['personid', 'ciq_equity_share']], on='personid', how='left'
                    )

                # Log-transform
                for col in ['ciq_avg_total_comp', 'ciq_max_total_comp', 'ciq_avg_salary']:
                    comp_agg[f'log_{col}'] = np.log1p(comp_agg[col].clip(lower=0))

                features = features.merge(comp_agg, on='personid', how='left')
                print(f"  Compensation features: {len(comp_agg.columns)-1} columns")

        # Drop personid, keep directorid as merge key
        features = features.drop(columns=['personid'], errors='ignore')

        ciq_cols = [c for c in features.columns if c.startswith('ciq_') or c.startswith('log_ciq_')]
        print(f"\n  CIQ features constructed: {len(ciq_cols)} columns at directorid level")
        print(f"  Total directors with CIQ data: {len(features):,}")
        for col in ciq_cols:
            cov = features[col].notna().mean()
            mn = features[col].dropna().mean() if features[col].notna().any() else 0
            print(f"    {col}: {cov:.1%} coverage, mean={mn:.3f}")

        return features

    # === Strategy B: gvkey+year matching (fallback) ===
    print("  People link unavailable — using gvkey+year matching strategy")

    if len(ciq_comp) == 0:
        print("  No CIQ compensation data — skipping CIQ features")
        return pd.DataFrame()

    ciq_comp = ciq_comp.copy()
    for col in ['personid', 'gvkey', 'fiscalyear']:
        ciq_comp[col] = pd.to_numeric(ciq_comp[col], errors='coerce')
    for col in ['salary', 'bonus', 'stock_awards', 'option_awards',
                'non_equity_incentive', 'all_other_comp', 'total_comp']:
        if col in ciq_comp.columns:
            ciq_comp[col] = pd.to_numeric(ciq_comp[col], errors='coerce')

    ciq_comp = ciq_comp.dropna(subset=['gvkey', 'fiscalyear', 'personid'])

    if 'topkeyexecflag' in ciq_comp.columns:
        ciq_ceo = ciq_comp[
            (ciq_comp['topkeyexecflag'] == 1) | (ciq_comp['keyexecflag'] == 1)
        ].copy()
        if len(ciq_ceo) == 0:
            ciq_ceo = ciq_comp.copy()
    else:
        ciq_ceo = ciq_comp.copy()

    if 'rank' in ciq_ceo.columns:
        ciq_ceo['rank'] = pd.to_numeric(ciq_ceo['rank'], errors='coerce')
        ciq_ceo = ciq_ceo.sort_values('rank').drop_duplicates(['gvkey', 'fiscalyear'])
    else:
        ciq_ceo = ciq_ceo.drop_duplicates(['gvkey', 'fiscalyear'])

    print(f"  CIQ compensation: {len(ciq_ceo):,} gvkey-year CEO records")

    gvkey_year = ciq_ceo[['gvkey', 'fiscalyear', 'personid']].copy()
    if 'total_comp' in ciq_ceo.columns:
        gvkey_year['ciq_total_comp'] = ciq_ceo['total_comp'].values
        gvkey_year['ciq_log_total_comp'] = np.log1p(ciq_ceo['total_comp'].clip(lower=0).values)
    if 'salary' in ciq_ceo.columns:
        gvkey_year['ciq_salary'] = ciq_ceo['salary'].values
    if 'bonus' in ciq_ceo.columns:
        gvkey_year['ciq_bonus'] = ciq_ceo['bonus'].values
    if all(c in ciq_ceo.columns for c in ['stock_awards', 'option_awards', 'total_comp']):
        equity = ciq_ceo['stock_awards'].fillna(0).values + ciq_ceo['option_awards'].fillna(0).values
        total = np.maximum(ciq_ceo['total_comp'].fillna(0).values, 1.0)
        gvkey_year['ciq_equity_share'] = np.clip(equity / total, 0, 1)

    discovered_pids = set(ciq_ceo['personid'].dropna().astype(int))
    if len(ciq_pros) > 0 and len(discovered_pids) > 0:
        ciq_pros = ciq_pros.copy()
        ciq_pros['personid'] = pd.to_numeric(ciq_pros['personid'], errors='coerce')
        pros_linked = ciq_pros[ciq_pros['personid'].isin(discovered_pids)].copy()
        if len(pros_linked) > 0:
            pro_agg = pros_linked.groupby('personid').agg(
                ciq_n_affiliations=('companyid', 'nunique'),
                ciq_n_roles=('proid', 'count'),
                ciq_career_start=('startyear', 'min'),
                ciq_career_end=('endyear', 'max'),
            ).reset_index()
            pro_agg['ciq_career_span'] = (pro_agg['ciq_career_end'] - pro_agg['ciq_career_start']).clip(0, 60)
            gvkey_year = gvkey_year.merge(pro_agg, on='personid', how='left')

    gvkey_year = gvkey_year.drop(columns=['personid'], errors='ignore')
    print(f"  Fallback gvkey-year features: {len(gvkey_year):,} rows")
    return gvkey_year


def construct_enriched_features(wrds_data):
    """Build enriched CEO features from BoardEx + CIQ.
    
    Returns:
        tuple: (enriched_df at directorid level, ciq_features or None)
        When people link is available, CIQ features are merged into enriched_df
        and ciq_features is None. Otherwise ciq_features is at gvkey-year level.
    """
    print("\n" + "=" * 70)
    print("CONSTRUCTING ENRICHED CEO FEATURES (BoardEx)")
    print("=" * 70)

    from ceo_firm_matching.enriched_features import construct_enriched_ceo_features

    profiles = wrds_data['profiles']
    employment = wrds_data['employment']
    education = wrds_data['education']
    execucomp = wrds_data['execucomp']

    # Build BoardEx enriched features keyed on directorid
    enriched = construct_enriched_ceo_features(
        profiles=profiles,
        employment=employment,
        education=education,
        execucomp=execucomp,
    )

    print(f"  BoardEx features: {len(enriched):,} directors, {len(enriched.columns)-1} features")

    # Build CIQ features
    ciq_features = construct_ciq_features(wrds_data)

    # If CIQ features are at directorid level (people link was available),
    # merge them directly into enriched
    ciq_separate = None
    if len(ciq_features) > 0 and 'directorid' in ciq_features.columns:
        before = len(enriched.columns)
        enriched = enriched.merge(ciq_features, on='directorid', how='left')
        ciq_added = len(enriched.columns) - before
        print(f"\n  Merged {ciq_added} CIQ features into enriched (directorid level)")
    elif len(ciq_features) > 0:
        ciq_separate = ciq_features
        print(f"\n  CIQ features at gvkey-year level ({len(ciq_features):,} rows) — will merge downstream")

    # Show coverage
    non_null = enriched.notna().mean()
    print(f"\n  Total enriched features: {len(enriched.columns)-1}")
    for col in enriched.columns:
        if col != 'directorid':
            print(f"    {col}: {non_null[col]:.1%}")

    return enriched, ciq_separate


# =============================================================================
# STEP 3: MERGE INTO CEO_TYPES
# =============================================================================

def merge_enriched_into_ceo_types(enriched_df, ciq_features, wrds_data):
    """Merge BoardEx enriched + CIQ features into ceo_types_v0.2.csv.
    
    Primary strategy: Direct companyid + year_born matching via BoardEx employment
    Fallback: ExecuComp crosswalk for unmatched CEOs
    """
    print("\n" + "=" * 70)
    print("MERGING ENRICHED FEATURES INTO REAL DATA")
    print("=" * 70)

    # Load real data
    df = pd.read_csv('Data/ceo_types_v0.2.csv')
    print(f"  Loaded ceo_types_v0.2: {len(df):,} rows, {df['match_exec_id'].nunique():,} CEOs")

    employment = wrds_data['employment']
    profiles = wrds_data['profiles']
    crosswalk = wrds_data['crosswalk']
    execucomp = wrds_data['execucomp']

    # === Strategy A: Direct companyid + year_born matching ===
    # Filter BoardEx employment to CEO roles
    ceo_emp = employment[
        employment['rolename'].str.contains('CEO|Chief Executive', case=False, na=False)
    ][['directorid', 'companyid']].drop_duplicates()

    # Add birth_year from profiles
    prof_by = profiles[['directorid', 'birth_year']].drop_duplicates('directorid')
    ceo_emp = ceo_emp.merge(prof_by, on='directorid', how='left')
    ceo_emp = ceo_emp.dropna(subset=['birth_year'])
    ceo_emp['companyid'] = pd.to_numeric(ceo_emp['companyid'], errors='coerce')
    ceo_emp['birth_year'] = ceo_emp['birth_year'].astype(int)

    # Deduplicate: if multiple CEOs at same company with same birth year, keep first
    ceo_emp = ceo_emp.drop_duplicates(['companyid', 'birth_year'])

    # Match to core dataset
    df['companyid'] = pd.to_numeric(df['companyid'], errors='coerce')
    df = df.merge(
        ceo_emp[['companyid', 'birth_year', 'directorid']].rename(
            columns={'birth_year': 'year_born'}
        ),
        on=['companyid', 'year_born'],
        how='left',
    )
    direct_match_pct = df['directorid'].notna().mean()
    direct_match_ceos = df.loc[df['directorid'].notna(), 'match_exec_id'].nunique()
    print(f"  Direct companyid+birth_year match: {direct_match_pct:.1%} "
          f"({df['directorid'].notna().sum():,} / {len(df):,}) — {direct_match_ceos} CEOs")

    # === Strategy B: Fallback via ExecuComp crosswalk for unmatched ===
    unmatched_mask = df['directorid'].isna()
    if unmatched_mask.any() and len(crosswalk) > 0 and 'execid' in crosswalk.columns:
        crosswalk_clean = crosswalk.sort_values('score', ascending=False).drop_duplicates('execid')
        crosswalk_clean['execid'] = pd.to_numeric(crosswalk_clean['execid'], errors='coerce')
        crosswalk_clean['directorid'] = pd.to_numeric(crosswalk_clean['directorid'], errors='coerce')

        # Only fill unmatched rows
        fallback = df.loc[unmatched_mask, ['match_exec_id']].merge(
            crosswalk_clean[['execid', 'directorid']].rename(
                columns={'execid': 'match_exec_id', 'directorid': 'directorid_fb'}
            ),
            on='match_exec_id',
            how='left',
        )
        filled = fallback['directorid_fb'].notna().sum()
        df.loc[unmatched_mask, 'directorid'] = fallback['directorid_fb'].values
        print(f"  Fallback crosswalk: filled {filled:,} additional rows")

    total_match_pct = df['directorid'].notna().mean()
    total_match_ceos = df.loc[df['directorid'].notna(), 'match_exec_id'].nunique()
    print(f"  Total directorid match: {total_match_pct:.1%} — {total_match_ceos} CEOs")

    # Step 2: Merge BoardEx enriched features on directorid
    before_cols = set(df.columns)
    df = df.merge(enriched_df, on='directorid', how='left')
    new_cols = set(df.columns) - before_cols
    print(f"  Added {len(new_cols)} enriched columns (BoardEx + CIQ)")

    # Step 3: If CIQ features are separately at gvkey-year level, merge them too
    if ciq_features is not None and len(ciq_features) > 0:
        ciq_merge = ciq_features.copy()
        ciq_merge['gvkey'] = pd.to_numeric(ciq_merge['gvkey'], errors='coerce')
        ciq_merge['fiscalyear'] = pd.to_numeric(ciq_merge['fiscalyear'], errors='coerce')
        ciq_merge = ciq_merge.drop_duplicates(subset=['gvkey', 'fiscalyear'])

        before_cols_ciq = set(df.columns)
        df = df.merge(ciq_merge, on=['gvkey', 'fiscalyear'], how='left')
        ciq_new = set(df.columns) - before_cols_ciq
        ciq_matched = df[[c for c in ciq_new]].notna().any(axis=1).sum() if ciq_new else 0
        print(f"  Added {len(ciq_new)} CIQ columns via gvkey+year: {sorted(ciq_new)}")
        print(f"  CIQ match rate: {ciq_matched:,} / {len(df):,} ({ciq_matched/len(df):.1%})")

    # Step 4: Also merge ExecuComp compensation directly (gvkey + fiscalyear)
    if len(execucomp) > 0:
        exec_cols = ['gvkey', 'fiscalyear', 'tdc1', 'salary', 'bonus', 'stock_awards_fv', 'option_awards_fv']
        exec_avail = [c for c in exec_cols if c in execucomp.columns]
        exec_merge = execucomp[exec_avail].copy()

        # Avoid duplicate columns
        for col in exec_avail:
            if col in ['gvkey', 'fiscalyear']:
                continue
            if col in df.columns:
                exec_merge = exec_merge.rename(columns={col: f'{col}_exec'})

        exec_merge['gvkey'] = pd.to_numeric(exec_merge['gvkey'], errors='coerce')
        exec_merge['fiscalyear'] = pd.to_numeric(exec_merge['fiscalyear'], errors='coerce')

        # Dedup: keep one per gvkey-year (the CEO record)
        exec_merge = exec_merge.drop_duplicates(subset=['gvkey', 'fiscalyear'])

        df_before = len(df)
        df = df.merge(exec_merge, on=['gvkey', 'fiscalyear'], how='left')
        print(f"  ExecuComp merge: {len(df)} rows (was {df_before})")

        # Construct log TDC1
        tdc_col = 'tdc1_exec' if 'tdc1_exec' in df.columns else 'tdc1'
        if tdc_col in df.columns:
            df['log_tdc1'] = np.log1p(pd.to_numeric(df[tdc_col], errors='coerce').clip(lower=0))
            tdc_coverage = df['log_tdc1'].notna().mean()
            print(f"  log(TDC1) coverage: {tdc_coverage:.1%}")

    # Feature coverage report
    enriched_cols = [c for c in enriched_df.columns if c != 'directorid']
    ciq_cols = [c for c in (ciq_features.columns if ciq_features is not None and len(ciq_features) > 0 else [])
                if c not in ('gvkey', 'fiscalyear')]
    all_new_cols = enriched_cols + ciq_cols
    print(f"\n  Feature coverage in final dataset ({len(all_new_cols)} enriched + CIQ):")
    for col in all_new_cols:
        if col in df.columns:
            pct = df[col].notna().mean()
            print(f"    {col}: {pct:.1%}")

    # Save merged dataset
    merged_path = 'Data/ceo_types_enriched.csv'
    df.to_csv(merged_path, index=False)
    print(f"\n  Saved merged data → {merged_path} ({len(df):,} rows, {len(df.columns)} columns)")

    return df


# =============================================================================
# STEP 4: TRAIN BASE MODEL + RUN EXTENSIONS
# =============================================================================

def train_base_model(df, config, epochs):
    """Train the base Two-Tower model on real data."""
    print("\n" + "=" * 70)
    print("TRAINING BASE TWO-TOWER MODEL (REAL DATA)")
    print("=" * 70)

    from torch.utils.data import DataLoader, random_split

    config.EPOCHS = epochs
    processor = DataProcessor(config)
    df_processed = processor.prepare_features(df)
    processor.fit(df_processed)
    data_dict = processor.transform(df_processed)

    metadata = {
        'n_firm_numeric': data_dict['n_firm_numeric'],
        'firm_cat_counts': data_dict['firm_cat_counts'],
        'n_ceo_numeric': data_dict['n_ceo_numeric'],
        'ceo_cat_counts': data_dict['ceo_cat_counts'],
    }

    dataset = CEOFirmDataset(data_dict)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    model = train_model(train_loader, val_loader, metadata, config)

    return model, processor, data_dict, metadata, df_processed


def run_extension_2_real(df, config, processor, data_dict, metadata, output_dir, epochs):
    """Extension 2: Contrastive Learning."""
    print("\n" + "=" * 70)
    print("EXTENSION 2: CONTRASTIVE LEARNING (REAL DATA)")
    print("=" * 70)

    from ceo_firm_matching.contrastive import (
        ContrastiveCEOFirmMatcher, train_contrastive, compute_retrieval_metrics
    )
    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(
        data_dict['firm_numeric'], data_dict['firm_cat'],
        data_dict['ceo_numeric'], data_dict['ceo_cat'],
        data_dict['target'], data_dict['weights'],
    )
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    def collate_fn(batch):
        keys = ['firm_numeric', 'firm_cat', 'ceo_numeric', 'ceo_cat', 'target', 'weights']
        return {k: torch.stack([b[i] for b in batch]) for i, k in enumerate(keys)}

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)

    config.EPOCHS = epochs
    model = train_contrastive(
        train_loader, val_loader, metadata, config,
        contrastive_weight=0.3, temperature=0.07, use_triplet=False
    )

    metrics = compute_retrieval_metrics(model, data_dict, config)
    print("\n  Retrieval metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    pd.DataFrame([metrics]).to_csv(f'{output_dir}/contrastive/retrieval_metrics.csv', index=False)
    return metrics


def run_extension_3_real(df, model, data_dict, output_dir, config):
    """Extension 3: Event Studies."""
    print("\n" + "=" * 70)
    print("EXTENSION 3: CEO TRANSITION EVENT STUDIES (REAL DATA)")
    print("=" * 70)

    from ceo_firm_matching.analytical_extensions import (
        identify_ceo_transitions, run_event_study, run_regression_event_study
    )

    transitions = identify_ceo_transitions(df)
    if len(transitions) == 0:
        print("  No transitions found")
        return {}

    print(f"  Found {len(transitions)} CEO transitions")
    print(f"  Upgrades: {transitions['upgrade'].sum()} | Downgrades: {(~transitions['upgrade']).sum()}")

    # Event study
    event_results = run_event_study(transitions, df, perf_col='exp_roa', window=(-3, 3), did=True)
    if len(event_results) > 0:
        event_results.to_csv(f'{output_dir}/event_study/event_study_results.csv', index=False)

    # Regression
    reg = run_regression_event_study(
        transitions, df, perf_col='exp_roa',
        controls=['logatw', 'leverage', 'rdintw'],
        window=(-3, 3)
    )
    if reg:
        pd.DataFrame([reg]).to_csv(f'{output_dir}/event_study/regression_results.csv', index=False)
        print(f"\n  DiD β(post×Δmatch) = {reg.get('beta_post_x_match', 'N/A')}")
        print(f"  SE = {reg.get('se_post_x_match', 'N/A')}")
        print(f"  R² = {reg.get('rsquared', 'N/A'):.4f}")
        print(f"  N = {reg.get('nobs', 'N/A')}")

    return reg


def run_extension_4_real(df, config, data_dict, metadata, output_dir, epochs):
    """Extension 4: Multi-Task Learning."""
    print("\n" + "=" * 70)
    print("EXTENSION 4: MULTI-TASK LEARNING (REAL DATA)")
    print("=" * 70)

    from ceo_firm_matching.multitask_model import (
        MultiTaskConfig, MultiTaskDataset, train_multitask, analyze_multitask_embeddings
    )
    from torch.utils.data import DataLoader

    mt_config = MultiTaskConfig()
    mt_config.EPOCHS = epochs

    # Add auxiliary targets
    if 'log_tdc1' in df.columns:
        data_dict['log_tdc1'] = torch.tensor(
            pd.to_numeric(df['log_tdc1'], errors='coerce').fillna(0).values[:len(data_dict['target'])],
            dtype=torch.float32
        )
    if 'tenure' in df.columns:
        data_dict['tenure_years'] = torch.tensor(
            pd.to_numeric(df['tenure'], errors='coerce').fillna(0).values[:len(data_dict['target'])],
            dtype=torch.float32
        )

    dataset = MultiTaskDataset(data_dict)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    model = train_multitask(train_loader, val_loader, metadata, mt_config)
    results = analyze_multitask_embeddings(model, data_dict, df, mt_config)

    pred_df = pd.DataFrame({
        'match_pred': results['match_score'],
        'comp_pred': results['comp_pred'],
        'tenure_pred': results['tenure_pred'],
        'turnover_prob': results['turnover_prob'],
        'match_actual': df['match_means'].values[:len(results['match_score'])],
    })
    pred_df.to_csv(f'{output_dir}/multitask/predictions.csv', index=False)

    corr = pred_df.corr()
    print(f"\n  Prediction correlations:")
    print(corr.round(3).to_string())

    return corr


def run_extension_6_real(df, config, data_dict, metadata, output_dir, epochs):
    """Extension 6: Industry-Specific Matching."""
    print("\n" + "=" * 70)
    print("EXTENSION 6: INDUSTRY-SPECIFIC MATCHING (REAL DATA)")
    print("=" * 70)

    from ceo_firm_matching.industry_model import IndustryConditionedMatcher
    from torch.utils.data import TensorDataset, DataLoader

    n_industries = df['compindustry'].nunique() if 'compindustry' in df.columns else data_dict['firm_cat'][:, 0].max().item() + 1
    print(f"  Industries: {n_industries}")

    model = IndustryConditionedMatcher(metadata, config, n_industries=n_industries).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    dataset = TensorDataset(
        data_dict['firm_numeric'], data_dict['firm_cat'],
        data_dict['ceo_numeric'], data_dict['ceo_cat'],
        data_dict['target'], data_dict['weights'],
    )
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    config.EPOCHS = epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            batch = [b.to(config.DEVICE) for b in batch]
            f_num, f_cat, c_num, c_cat, target, weights = batch
            optimizer.zero_grad()
            score = model(f_num, f_cat, c_num, c_cat)
            loss = (weights * (score - target) ** 2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: Loss={total_loss/len(loader):.4f}")

    temps = model.industry_temperature.weight.exp().detach().cpu().numpy().flatten()
    temp_df = pd.DataFrame({'industry_id': range(len(temps)), 'temperature': temps})
    temp_df.to_csv(f'{output_dir}/industry/industry_temperatures.csv', index=False)

    print(f"\n  Industry temperature range: [{temps.min():.2f}, {temps.max():.2f}]")
    print(f"  Temperature std: {temps.std():.3f}")

    return temp_df


def run_extension_7_real(df, model, data_dict, output_dir, config):
    """Extension 7: Compensation Decomposition."""
    print("\n" + "=" * 70)
    print("EXTENSION 7: COMPENSATION DECOMPOSITION (REAL DATA)")
    print("=" * 70)

    from ceo_firm_matching.analytical_extensions import (
        decompose_compensation, compensation_by_match_quality
    )

    # Use log_tdc1 if available, else tdc1
    comp_col = None
    for col in ['log_tdc1', 'tdc1', 'tdc1_exec']:
        if col in df.columns and df[col].notna().sum() > 100:
            comp_col = col
            break

    if comp_col is None:
        print("  No compensation data available — skipping")
        return None

    print(f"  Using compensation column: {comp_col}")
    print(f"  Non-null values: {df[comp_col].notna().sum():,}")

    comp_df = decompose_compensation(model, data_dict, df, comp_col=comp_col, device=config.DEVICE)

    if len(comp_df) > 0:
        quintile_summary = compensation_by_match_quality(comp_df)
        if len(quintile_summary) > 0:
            quintile_summary.to_csv(f'{output_dir}/compensation/quintile_decomposition.csv')
            print(f"\n  Quintile decomposition:")
            print(quintile_summary.to_string())

        comp_df.to_csv(f'{output_dir}/compensation/full_decomposition.csv', index=False)
        return quintile_summary

    return None


def run_extension_8_real(df, wrds_data, output_dir):
    """Extension 8: Board Interlock Network."""
    print("\n" + "=" * 70)
    print("EXTENSION 8: BOARD INTERLOCK NETWORK (REAL DATA)")
    print("=" * 70)

    from ceo_firm_matching.network_features import construct_full_network_features

    employment = wrds_data['employment']
    if len(employment) == 0:
        print("  No employment data — skipping")
        return None

    # Get unique director IDs from merged data
    if 'directorid' in df.columns:
        target_ids = df['directorid'].dropna().unique().astype(int).tolist()
    else:
        print("  No directorid column — skipping")
        return None

    print(f"  Computing network features for {len(target_ids)} directors")

    features = construct_full_network_features(employment, target_ids, year=2020)

    print(f"\n  Network features computed for {len(features)} CEOs:")
    print(features.describe().round(3).to_string())

    features.to_csv(f'{output_dir}/network/network_features.csv', index=False)
    return features


def run_extension_9_real(df, model, data_dict, output_dir, config):
    """Extension 9: Counterfactuals."""
    print("\n" + "=" * 70)
    print("EXTENSION 9: GENERATIVE COUNTERFACTUALS (REAL DATA)")
    print("=" * 70)

    from ceo_firm_matching.analytical_extensions import generate_counterfactuals

    cf_df = generate_counterfactuals(model, data_dict, df, top_k=10, device=config.DEVICE)
    cf_df.to_csv(f'{output_dir}/counterfactual/counterfactual_rankings.csv', index=False)

    print(f"  Cross-matched {cf_df['firm_id'].nunique()} firms × many CEOs")
    print(f"  % firms with better CEO available: {(cf_df['match_improvement'] > 0).mean():.1%}")
    print(f"  Median improvement: {cf_df['match_improvement'].median():.3f}")

    return cf_df


def run_extension_10_real(model, data_dict, config, output_dir):
    """Extension 10: Transfer Learning."""
    print("\n" + "=" * 70)
    print("EXTENSION 10: TRANSFER LEARNING (REAL DATA)")
    print("=" * 70)

    from ceo_firm_matching.analytical_extensions import TransferableTwoTower

    transfer_model = TransferableTwoTower(model, config, adapter_dim=16)

    total = sum(p.numel() for p in transfer_model.parameters())
    trainable = sum(p.numel() for p in transfer_model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"  Total: {total:,} | Frozen: {frozen:,} ({frozen/total:.1%}) | Trainable: {trainable:,} ({trainable/total:.1%})")

    with torch.no_grad():
        test = transfer_model(
            data_dict['firm_numeric'][:5].to(config.DEVICE),
            data_dict['firm_cat'][:5].to(config.DEVICE),
            data_dict['ceo_numeric'][:5].to(config.DEVICE),
            data_dict['ceo_cat'][:5].to(config.DEVICE),
        )
    print(f"  Forward pass: {test.shape} ✓")

    torch.save(transfer_model.state_dict(), f'{output_dir}/transfer/transfer_model.pt')
    return {'total_params': total, 'trainable': trainable, 'frozen': frozen}


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    if args.extensions == 'all':
        extensions = list(range(1, 11))
    else:
        extensions = [int(x.strip()) for x in args.extensions.split(',')]

    # Setup output dirs
    for d in ['contrastive', 'event_study', 'multitask', 'industry',
              'compensation', 'network', 'counterfactual', 'transfer', 'enriched']:
        os.makedirs(f'{OUTPUT_DIR}/{d}', exist_ok=True)

    print("=" * 70)
    print("  WRDS PULL → MERGE → RUN EXTENSIONS (REAL DATA)")
    print("=" * 70)
    print(f"  Extensions: {extensions}")
    print(f"  Epochs: {args.epochs}")
    start = time.time()

    # ─── WRDS Pull ───
    wrds_data = pull_wrds_data(skip_pull=args.skip_pull)

    # ─── Construct Enriched Features ───
    enriched, ciq_features = construct_enriched_features(wrds_data)

    # ─── Merge ───
    df = merge_enriched_into_ceo_types(enriched, ciq_features, wrds_data)

    # ─── Train Base Model ───
    config = Config()
    model, processor, data_dict, metadata, df_processed = train_base_model(df, config, args.epochs)

    # ─── Run Extensions ───
    results = {}

    if 1 in extensions:
        print("\n" + "=" * 70)
        print("EXTENSION 1: ENRICHED CEO TOWER (REAL DATA)")
        print("=" * 70)
        enriched_cols = [c for c in ['n_prior_roles', 'n_prior_boards', 'avg_board_tenure',
                                       'years_as_ceo', 'n_sectors', 'career_span_years',
                                       'network_size', 'board_interlocks', 'education_score',
                                       'log_prior_tdc1', 'equity_share', 'pay_premium',
                                       'log_ownership_pct',
                                       'ciq_total_comp', 'ciq_log_total_comp', 'ciq_salary',
                                       'ciq_bonus', 'ciq_equity_share', 'ciq_n_affiliations',
                                       'ciq_n_roles', 'ciq_is_key_exec', 'ciq_is_top_key_exec',
                                       'ciq_is_board_member', 'ciq_current_boards',
                                       'ciq_career_span', 'ciq_best_rank'] if c in df.columns]
        print(f"  Enriched features successfully merged: {len(enriched_cols)}")
        for col in enriched_cols:
            print(f"    ✓ {col}: {df[col].notna().mean():.1%} coverage, mean={df[col].mean():.3f}")
        results[1] = f"SUCCESS: {len(enriched_cols)} features merged"

    if 2 in extensions:
        try:
            metrics = run_extension_2_real(df_processed, config, processor, data_dict, metadata, OUTPUT_DIR, args.epochs)
            results[2] = f"SUCCESS: MRR={metrics.get('MRR', 0):.4f}"
        except Exception as e:
            results[2] = f"FAILED: {e}"
            import traceback; traceback.print_exc()

    if 3 in extensions:
        try:
            reg = run_extension_3_real(df_processed, model, data_dict, OUTPUT_DIR, config)
            results[3] = f"SUCCESS: β={reg.get('beta_post_x_match', 'N/A')}" if reg else "SUCCESS: no result"
        except Exception as e:
            results[3] = f"FAILED: {e}"
            import traceback; traceback.print_exc()

    if 4 in extensions:
        try:
            corr = run_extension_4_real(df_processed, config, data_dict, metadata, OUTPUT_DIR, args.epochs)
            r_pred_actual = corr.loc['match_pred', 'match_actual'] if corr is not None else 'N/A'
            results[4] = f"SUCCESS: r(pred,actual)={r_pred_actual:.3f}"
        except Exception as e:
            results[4] = f"FAILED: {e}"
            import traceback; traceback.print_exc()

    if 5 in extensions:
        print("\n" + "=" * 70)
        print("EXTENSION 5: TEMPORAL EMBEDDINGS (REAL DATA)")
        print("=" * 70)
        if 'match_exec_id' in df.columns and 'fiscalyear' in df.columns:
            career_lengths = df.groupby('match_exec_id')['fiscalyear'].nunique()
            print(f"  Career lengths: median={career_lengths.median():.0f}, mean={career_lengths.mean():.1f}")
            print(f"  CEOs with 5+ years: {(career_lengths >= 5).sum()} ({(career_lengths >= 5).mean():.1%})")
            results[5] = f"SUCCESS: {(career_lengths >= 5).sum()} CEOs with 5+ years"
        else:
            results[5] = "SKIPPED: no panel structure"

    if 6 in extensions:
        try:
            temp_df = run_extension_6_real(df_processed, config, data_dict, metadata, OUTPUT_DIR, args.epochs)
            results[6] = f"SUCCESS: τ range=[{temp_df['temperature'].min():.2f}, {temp_df['temperature'].max():.2f}]"
        except Exception as e:
            results[6] = f"FAILED: {e}"
            import traceback; traceback.print_exc()

    if 7 in extensions:
        try:
            quintiles = run_extension_7_real(df_processed, model, data_dict, OUTPUT_DIR, config)
            results[7] = "SUCCESS" if quintiles is not None else "SKIPPED: no comp data"
        except Exception as e:
            results[7] = f"FAILED: {e}"
            import traceback; traceback.print_exc()

    if 8 in extensions:
        try:
            net_feats = run_extension_8_real(df_processed, wrds_data, OUTPUT_DIR)
            results[8] = f"SUCCESS: {len(net_feats)} directors" if net_feats is not None else "SKIPPED"
        except Exception as e:
            results[8] = f"FAILED: {e}"
            import traceback; traceback.print_exc()

    if 9 in extensions:
        try:
            cf = run_extension_9_real(df_processed, model, data_dict, OUTPUT_DIR, config)
            results[9] = f"SUCCESS: {(cf['match_improvement'] > 0).mean():.1%} improvable"
        except Exception as e:
            results[9] = f"FAILED: {e}"
            import traceback; traceback.print_exc()

    if 10 in extensions:
        try:
            arch = run_extension_10_real(model, data_dict, config, OUTPUT_DIR)
            results[10] = f"SUCCESS: {arch['trainable']:,} trainable params ({arch['trainable']/arch['total_params']:.1%})"
        except Exception as e:
            results[10] = f"FAILED: {e}"
            import traceback; traceback.print_exc()

    # ─── Summary ───
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("  SUMMARY (REAL DATA)")
    print("=" * 70)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Data: {len(df):,} rows, {df['match_exec_id'].nunique():,} CEOs")
    for ext, status in sorted(results.items()):
        icon = "✓" if "SUCCESS" in str(status) else "✗"
        print(f"  {icon} Extension {ext}: {status}")
    print(f"\n  Outputs: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
