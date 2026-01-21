#!/usr/bin/env python3
"""
CEO Headers Panel Matching Script

Matches CEO_headers_panel.csv with BoardEx-CapitalIQ-ExecuComp data using
a multi-phase strategy:
  Phase 1: Direct gvkey-year + lastname match
  Phase 2: Work history propagation
  Phase 3: BoardEx employment history expansion (requires WRDS)
  Phase 4: Fuzzy name matching for remaining

Usage:
    python match_ceo_headers.py [--phases 1,2] [--wrds] [--fuzzy-threshold 0.85]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Set
import argparse
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MatchConfig:
    """Configuration for matching parameters."""
    headers_path: str = "Data/CEO_headers_panel.csv"
    exec_panel_path: str = "Output/CEO_Pipeline/ceo_panel_serverside.parquet"
    output_dir: str = "Output/CEO_Pipeline"
    
    # Matching thresholds
    fuzzy_threshold: float = 0.85
    
    # Phase control
    run_phases: List[int] = None
    use_wrds: bool = False
    
    def __post_init__(self):
        if self.run_phases is None:
            self.run_phases = [1, 2]


class CEOHeadersMatcher:
    """Multi-phase matcher for CEO headers panel."""
    
    def __init__(self, config: MatchConfig):
        self.config = config
        self.headers: Optional[pd.DataFrame] = None
        self.exec_panel: Optional[pd.DataFrame] = None
        self.matched: Optional[pd.DataFrame] = None
        self.match_stats = {}
        
    def load_data(self):
        """Load source datasets."""
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        # Load headers panel
        self.headers = pd.read_csv(self.config.headers_path)
        print(f"Headers panel: {len(self.headers):,} rows, {self.headers['match_exec_id'].nunique():,} unique CEOs")
        
        # Load exec panel
        self.exec_panel = pd.read_parquet(self.config.exec_panel_path)
        print(f"Exec panel: {len(self.exec_panel):,} rows, {self.exec_panel['gvkey'].nunique():,} unique firms")
        
        # Ensure numeric types
        self.headers['gvkey'] = pd.to_numeric(self.headers['gvkey'], errors='coerce')
        self.headers['year'] = pd.to_numeric(self.headers['year'], errors='coerce')
        self.exec_panel['gvkey'] = pd.to_numeric(self.exec_panel['gvkey'], errors='coerce')
        self.exec_panel['fiscalyear'] = pd.to_numeric(self.exec_panel['fiscalyear'], errors='coerce')
        
        print()
        
    def _normalize_name(self, name: str) -> str:
        """Normalize name for matching."""
        if pd.isna(name):
            return ""
        return str(name).lower().replace(".", "").replace(",", "").replace("-", " ").strip()
    
    def _lastname_in_fullname(self, lastname: str, fullname: str) -> bool:
        """Check if lastname appears in fullname."""
        if pd.isna(lastname) or pd.isna(fullname):
            return False
        ln = str(lastname).lower().strip()
        fn = str(fullname).lower()
        return ln in fn and len(ln) >= 2
    
    def phase1_gvkey_year_name(self) -> pd.DataFrame:
        """
        Phase 1: Direct gvkey-year match with name validation.
        
        Returns matched rows with match metadata.
        """
        print("=" * 60)
        print("PHASE 1: GVKEY-YEAR + NAME MATCH")
        print("=" * 60)
        
        # Merge on gvkey-year
        merged = self.headers.merge(
            self.exec_panel[['gvkey', 'fiscalyear', 'exec_fullname', 'execid', 'directorid']].drop_duplicates(),
            left_on=['gvkey', 'year'],
            right_on=['gvkey', 'fiscalyear'],
            how='left'
        )
        
        # Validate lastname match
        merged['name_match'] = merged.apply(
            lambda row: self._lastname_in_fullname(row['lastname'], row['exec_fullname']),
            axis=1
        )
        
        # Filter matches
        phase1_matched = merged[merged['name_match']].copy()
        phase1_matched['match_phase'] = 1
        phase1_matched['match_confidence'] = 'high'
        phase1_matched['match_source'] = 'gvkey_year_name'
        
        # Stats
        n_matched = len(phase1_matched)
        n_ceos = phase1_matched['match_exec_id'].nunique()
        pct = 100 * n_matched / len(self.headers)
        
        print(f"Matched: {n_matched:,} gvkey-years ({pct:.1f}%)")
        print(f"Unique CEOs: {n_ceos:,}")
        
        self.match_stats['phase1'] = {
            'rows': n_matched,
            'ceos': n_ceos,
            'pct': pct
        }
        
        # Store confirmed linkages
        self.confirmed_links = phase1_matched[['match_exec_id', 'execid', 'directorid']].drop_duplicates()
        self.confirmed_links = self.confirmed_links[self.confirmed_links['execid'].notna()]
        print(f"Confirmed CEO->execid links: {len(self.confirmed_links):,}")
        print()
        
        return phase1_matched
    
    def phase2_work_history(self, phase1_matched: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 2: Propagate matches via work history.
        
        For CEOs matched in Phase 1, apply their execid/directorid
        to other gvkey-years where they appear.
        """
        print("=" * 60)
        print("PHASE 2: WORK HISTORY PROPAGATION")
        print("=" * 60)
        
        # Get matched match_exec_ids
        matched_ids = set(phase1_matched['match_exec_id'].unique())
        
        # Get unmatched rows
        all_header_rows = self.headers.copy()
        phase1_keys = set(zip(phase1_matched['match_exec_id'], 
                               phase1_matched['gvkey'], 
                               phase1_matched['year']))
        
        unmatched_mask = all_header_rows.apply(
            lambda row: (row['match_exec_id'], row['gvkey'], row['year']) not in phase1_keys,
            axis=1
        )
        unmatched = all_header_rows[unmatched_mask].copy()
        
        print(f"Unmatched rows from Phase 1: {len(unmatched):,}")
        
        # Join with confirmed links
        phase2_matched = unmatched.merge(
            self.confirmed_links,
            on='match_exec_id',
            how='inner'
        )
        
        phase2_matched['match_phase'] = 2
        phase2_matched['match_confidence'] = 'medium-high'
        phase2_matched['match_source'] = 'work_history'
        
        # Stats
        n_matched = len(phase2_matched)
        n_ceos = phase2_matched['match_exec_id'].nunique()
        pct = 100 * n_matched / len(self.headers)
        
        print(f"Matched: {n_matched:,} gvkey-years ({pct:.1f}%)")
        print(f"Unique CEOs (via propagation): {n_ceos:,}")
        
        self.match_stats['phase2'] = {
            'rows': n_matched,
            'ceos': n_ceos,
            'pct': pct
        }
        print()
        
        return phase2_matched
    
    def phase3_boardex_employment(self, previous_matched: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 3: Expand using BoardEx employment history.
        
        Query WRDS for full employment history of matched directors.
        """
        print("=" * 60)
        print("PHASE 3: BOARDEX EMPLOYMENT EXPANSION")
        print("=" * 60)
        
        if not self.config.use_wrds:
            print("WRDS connection not enabled. Skipping Phase 3.")
            print("Use --wrds flag to enable BoardEx queries.")
            print()
            return pd.DataFrame()
        
        try:
            import wrds
        except ImportError:
            print("WRDS library not available. Skipping Phase 3.")
            return pd.DataFrame()
        
        # Get unique directorids from matched
        matched_directorids = previous_matched['directorid'].dropna().unique()
        print(f"Querying BoardEx for {len(matched_directorids):,} directors...")
        
        try:
            conn = wrds.Connection()
            
            # Query employment history
            director_list = ','.join([str(int(d)) for d in matched_directorids[:1000]])  # Limit for query
            query = f"""
                SELECT emp.directorid, emp.companyid, emp.companyname, 
                       emp.rolename, emp.datestartrole, emp.dateendrole
                FROM boardex.na_dir_profile_emp emp
                WHERE emp.directorid IN ({director_list})
                  AND (LOWER(emp.rolename) LIKE '%ceo%' 
                       OR LOWER(emp.rolename) LIKE '%chief executive%')
            """
            
            emp_history = conn.raw_sql(query)
            print(f"Retrieved {len(emp_history):,} CEO role records")
            
            # TODO: Link BoardEx companyid to gvkey
            # This requires additional crosswalk tables
            
            conn.close()
            
        except Exception as e:
            print(f"WRDS query failed: {e}")
            return pd.DataFrame()
        
        # Placeholder for Phase 3 results
        phase3_matched = pd.DataFrame()
        self.match_stats['phase3'] = {'rows': 0, 'ceos': 0, 'pct': 0}
        print()
        
        return phase3_matched
    
    def phase4_fuzzy_matching(self, previous_matched: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 4: Fuzzy name matching for remaining unmatched.
        """
        print("=" * 60)
        print("PHASE 4: FUZZY NAME MATCHING")
        print("=" * 60)
        
        try:
            from rapidfuzz import fuzz
        except ImportError:
            print("rapidfuzz not installed. Skipping Phase 4.")
            print("Install with: pip install rapidfuzz")
            return pd.DataFrame()
        
        # Get unmatched gvkey-years
        matched_keys = set(zip(previous_matched['match_exec_id'],
                               previous_matched['gvkey'],
                               previous_matched['year']))
        
        unmatched_mask = self.headers.apply(
            lambda row: (row['match_exec_id'], row['gvkey'], row['year']) not in matched_keys,
            axis=1
        )
        unmatched = self.headers[unmatched_mask].copy()
        print(f"Remaining unmatched: {len(unmatched):,} gvkey-years")
        
        # For each unmatched, try fuzzy matching against exec_panel
        # This is expensive, so we'll do it strategically
        
        # First, get exec_panel records not yet matched
        matched_exec_keys = set(zip(previous_matched['gvkey'], previous_matched['year']))
        
        # Build lookup by gvkey-year
        exec_lookup = self.exec_panel.groupby(['gvkey', 'fiscalyear']).first().reset_index()
        
        phase4_matches = []
        threshold = self.config.fuzzy_threshold * 100  # rapidfuzz uses 0-100
        
        # Sample for performance (limit to subset)
        sample_unmatched = unmatched.head(10000)  # Limit for performance
        
        for idx, row in sample_unmatched.iterrows():
            gvkey = row['gvkey']
            year = row['year']
            name = self._normalize_name(row['newname'])
            
            # Check if this gvkey-year has an exec record
            exec_rows = exec_lookup[(exec_lookup['gvkey'] == gvkey) & 
                                     (exec_lookup['fiscalyear'] == year)]
            
            if len(exec_rows) == 0:
                continue
                
            # Fuzzy match names
            for _, exec_row in exec_rows.iterrows():
                exec_name = self._normalize_name(exec_row['exec_fullname'])
                score = fuzz.ratio(name, exec_name)
                
                if score >= threshold:
                    match_row = row.to_dict()
                    match_row['execid'] = exec_row['execid']
                    match_row['directorid'] = exec_row['directorid']
                    match_row['match_phase'] = 4
                    match_row['match_confidence'] = 'medium'
                    match_row['match_source'] = f'fuzzy_{score:.0f}'
                    phase4_matches.append(match_row)
                    break
        
        phase4_matched = pd.DataFrame(phase4_matches)
        
        n_matched = len(phase4_matched)
        n_ceos = phase4_matched['match_exec_id'].nunique() if n_matched > 0 else 0
        pct = 100 * n_matched / len(self.headers)
        
        print(f"Matched: {n_matched:,} gvkey-years ({pct:.1f}%)")
        print(f"Unique CEOs: {n_ceos:,}")
        print(f"Note: Only searched first 10,000 unmatched for performance")
        
        self.match_stats['phase4'] = {
            'rows': n_matched,
            'ceos': n_ceos,
            'pct': pct
        }
        print()
        
        return phase4_matched
    
    def run(self) -> pd.DataFrame:
        """Execute matching pipeline."""
        self.load_data()
        
        all_matched = []
        
        # Phase 1
        if 1 in self.config.run_phases:
            phase1 = self.phase1_gvkey_year_name()
            all_matched.append(phase1)
        
        # Phase 2
        if 2 in self.config.run_phases and len(all_matched) > 0:
            phase2 = self.phase2_work_history(pd.concat(all_matched))
            all_matched.append(phase2)
        
        # Phase 3
        if 3 in self.config.run_phases:
            combined = pd.concat(all_matched) if all_matched else pd.DataFrame()
            phase3 = self.phase3_boardex_employment(combined)
            if len(phase3) > 0:
                all_matched.append(phase3)
        
        # Phase 4
        if 4 in self.config.run_phases:
            combined = pd.concat(all_matched) if all_matched else pd.DataFrame()
            phase4 = self.phase4_fuzzy_matching(combined)
            if len(phase4) > 0:
                all_matched.append(phase4)
        
        # Combine all matches
        if all_matched:
            self.matched = pd.concat(all_matched, ignore_index=True)
        else:
            self.matched = pd.DataFrame()
        
        return self.matched
    
    def enrich_with_exec_data(self) -> pd.DataFrame:
        """
        Enrich matched panel with BoardEx-CIQ-ExecuComp variables.
        """
        print("=" * 60)
        print("ENRICHING WITH EXEC PANEL VARIABLES")
        print("=" * 60)
        
        if self.matched is None or len(self.matched) == 0:
            print("No matches to enrich.")
            return pd.DataFrame()
        
        # Select enrichment columns from exec_panel
        enrich_cols = [
            'execid', 'gvkey', 'fiscalyear',
            'age', 'gender', 'nationality', 'networksize',
            'n_total_roles', 'n_companies', 'avg_tenure_days',
            'total_comp_exec', 'salary_exec', 'bonus_exec',
            'equity_ratio_exec', 'is_ceo_ciq', 'serial_ceo_ciq'
        ]
        
        available_cols = [c for c in enrich_cols if c in self.exec_panel.columns]
        
        enriched = self.matched.merge(
            self.exec_panel[available_cols].drop_duplicates(['execid', 'gvkey', 'fiscalyear']),
            left_on=['execid', 'gvkey', 'year'],
            right_on=['execid', 'gvkey', 'fiscalyear'],
            how='left',
            suffixes=('', '_exec')
        )
        
        print(f"Enriched columns: {available_cols}")
        print(f"Rows with enrichment: {enriched['age'].notna().sum():,}")
        print()
        
        return enriched
    
    def save_results(self, enriched: pd.DataFrame):
        """Save matched and enriched data."""
        print("=" * 60)
        print("SAVING RESULTS")
        print("=" * 60)
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save matched panel
        matched_path = output_dir / "headers_matched.parquet"
        enriched.to_parquet(matched_path, index=False)
        print(f"Saved matched panel: {matched_path}")
        
        # Save CSV for easy inspection
        csv_path = output_dir / "headers_matched.csv"
        enriched.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
        
        # Save diagnostics
        diag_path = output_dir / "match_diagnostics.csv"
        diag_df = pd.DataFrame([
            {'phase': 'Overall', 'rows': len(enriched), 
             'ceos': enriched['match_exec_id'].nunique(),
             'pct': 100 * len(enriched) / len(self.headers)}
        ] + [
            {'phase': f'Phase {k.replace("phase", "")}', **v}
            for k, v in self.match_stats.items()
        ])
        diag_df.to_csv(diag_path, index=False)
        print(f"Saved diagnostics: {diag_path}")
        
        # Save unmatched CEOs
        matched_ids = set(enriched['match_exec_id'].unique())
        unmatched = self.headers[~self.headers['match_exec_id'].isin(matched_ids)]
        unmatched_path = output_dir / "unmatched_ceos.csv"
        unmatched.to_csv(unmatched_path, index=False)
        print(f"Saved unmatched: {unmatched_path} ({len(unmatched):,} rows)")
        print()
    
    def print_summary(self):
        """Print final summary."""
        print("=" * 60)
        print("MATCHING SUMMARY")
        print("=" * 60)
        
        total_matched = sum(s.get('rows', 0) for s in self.match_stats.values())
        total_pct = 100 * total_matched / len(self.headers)
        
        print(f"\nHeaders panel: {len(self.headers):,} gvkey-years")
        print(f"Total matched: {total_matched:,} ({total_pct:.1f}%)")
        
        print("\nBreakdown by phase:")
        for phase, stats in self.match_stats.items():
            print(f"  {phase}: {stats['rows']:,} rows ({stats['pct']:.1f}%), {stats['ceos']:,} CEOs")
        
        print()


def main():
    parser = argparse.ArgumentParser(description='Match CEO headers panel')
    parser.add_argument('--phases', type=str, default='1,2',
                        help='Comma-separated phases to run (1,2,3,4)')
    parser.add_argument('--wrds', action='store_true',
                        help='Enable WRDS queries for Phase 3')
    parser.add_argument('--fuzzy-threshold', type=float, default=0.85,
                        help='Fuzzy matching threshold (0-1)')
    parser.add_argument('--output-dir', type=str, default='Output/CEO_Pipeline',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Parse phases
    phases = [int(p.strip()) for p in args.phases.split(',')]
    
    config = MatchConfig(
        run_phases=phases,
        use_wrds=args.wrds,
        fuzzy_threshold=args.fuzzy_threshold,
        output_dir=args.output_dir
    )
    
    matcher = CEOHeadersMatcher(config)
    
    # Run matching
    matcher.run()
    
    # Enrich and save
    enriched = matcher.enrich_with_exec_data()
    matcher.save_results(enriched)
    matcher.print_summary()


if __name__ == "__main__":
    main()
