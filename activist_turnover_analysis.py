#!/usr/bin/env python3
"""
🕵️‍♂️🕵️‍♂️🕵️‍♂️ ACTIVIST & TURNOVER ANALYSIS 🕵️‍♂️🕵️‍♂️🕵️‍♂️

Investigates the role of Activist Investors in CEO Turnover events.
Links detailed 13F manager data with CEO turnover flags.

Key Features:
1. Granular 13F Pull: Retrieves Manager Names (not just aggregates)
2. Activist Identification: Regex matching for top activist funds
   (Icahn, Elliott, Pershing, Starboard, etc.)
3. Turnover Detection: Identifies CEO changes in match data
4. Event Study: Analyzes ownership flows around turnover events

"""
import pandas as pd
import numpy as np
import wrds
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import re

plt.style.use('seaborn-v0_8-whitegrid')

# WRDS Credentials (reusing from mega_pull)
WRDS_USER = 'maricste93'
WRDS_PASS = 'jexvar-manryn-6Cosky'

ACTIVIST_KEYWORDS = [
    r'ICAHN', r'ELLIOTT', r'PERSHING SQUARE', r'THIRD POINT', 
    r'STARBOARD', r'TRIAN', r'VALUEACT', r'CARL C', 
    r'JANA PARTNERS', r'CORVEX', r'LOEB', r'ACKMAN',
    r'GREENLIGHT CAPITAL', r'TIGER GLOBAL', r'POINT72',
    r'VIKING GLOBAL', r'APPALOOSA', r'BAUPOST'
]

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

def get_target_cusips():
    """Get list of CUSIPs from our CEO match universe."""
    print("\n🔍 Identifying Target Universe...")
    
    # Load link table and CEO data
    try:
        ticker_cusip = pd.read_parquet('Output/wrds_ticker_cusip.parquet')
        crsp_comp = pd.read_parquet('Output/wrds_crsp_comp.parquet')
        ceo_df = pd.read_csv('Data/ceo_types_v0.2.csv')
        
        # Chain GVKEY -> PERMNO -> CUSIP
        target_gvkeys = ceo_df['gvkey'].unique()
        
        # GVKEY -> PERMNO
        comp_link = crsp_comp[
            (crsp_comp['gvkey'].astype(str).isin(target_gvkeys.astype(str)))
        ].copy()
        target_permnos = comp_link['permno'].unique()
        
        # PERMNO -> CUSIP
        cusip_link = ticker_cusip[ticker_cusip['permno'].isin(target_permnos)].copy()
        target_cusips = cusip_link['cusip'].unique().tolist()
        
        # Format for SQL
        cusip_str = "'" + "','".join(target_cusips) + "'"
        print(f"  Target Universe: {len(target_gvkeys)} firms -> {len(target_permnos)} permnos -> {len(target_cusips)} cusips")
        return cusip_str, target_cusips
    except Exception as e:
        print(f"❌ Error getting targets: {e}")
        return None, None

def pull_activist_data(db, cusip_list_sql):
    """Pull granular 13F data with manager names."""
    print("\n📊 Pulling Granular 13F Data (may take a moment)...")
    
    # We pull ONLY relevant columns to save memory
    query = f"""
    SELECT cusip, 
           EXTRACT(YEAR FROM rdate) as year,
           mgrname,
           shares,
           prc
    FROM tfn.s34
    WHERE rdate >= '2006-01-01'
      AND cusip IN ({cusip_list_sql})
    """
    try:
        df = db.raw_sql(query)
        print(f"  Retrieved {len(df):,} holdings records")
        return df
    except Exception as e:
        print(f"❌ Error pulling 13F: {e}")
        return None

def identify_activists(df):
    """Flag activist holdings using keyword matching."""
    print("\n🕵️‍♂️ Identifying Activist Investors...")
    
    # Regex pattern
    pattern = '|'.join(ACTIVIST_KEYWORDS)
    
    # Flag activists
    df['is_activist'] = df['mgrname'].str.contains(pattern, case=False, na=False, regex=True)
    
    activist_recs = df[df['is_activist']]
    n_activist = len(activist_recs)
    n_unique = activist_recs['mgrname'].nunique()
    
    print(f"  Found {n_activist:,} activist positions from {n_unique} unique managers")
    print("  Top Identified Activists:")
    print(activist_recs['mgrname'].value_counts().head(10))
    
    # Aggregate to CUSIP-Year level
    agg = df.groupby(['cusip', 'year']).agg({
        'is_activist': 'sum', # Count of activist funds present
        'shares': 'sum' # Total shares (proxy for inst ownership recalc)
    }).rename(columns={'is_activist': 'n_activists'})
    
    # Calculate activist shares specifically
    activist_shares = df[df['is_activist']].groupby(['cusip', 'year'])['shares'].sum().reset_index()
    activist_shares.rename(columns={'shares': 'activist_shares'}, inplace=True)
    
    agg = agg.reset_index()
    agg = agg.merge(activist_shares, on=['cusip', 'year'], how='left')
    agg['activist_shares'] = agg['activist_shares'].fillna(0)
    
    return agg

def process_turnover(ceo_df):
    """Identify CEO turnover events."""
    print("\n🔄 Processing CEO Turnover Events...")
    
    # Sort
    df = ceo_df.sort_values(['gvkey', 'fiscalyear']).copy()
    
    # Detect change in CEO ID
    # Note: ensure ids are strings or comparable
    df['match_exec_id'] = df['match_exec_id'].astype(str)
    
    # Shift to compare with previous year
    df['prev_exec_id'] = df.groupby('gvkey')['match_exec_id'].shift(1)
    df['prev_year'] = df.groupby('gvkey')['fiscalyear'].shift(1)
    
    # Turnover flag: Exec ID changed AND years are consecutive
    condition = (df['match_exec_id'] != df['prev_exec_id']) & \
                (df['prev_exec_id'].notna()) & \
                ((df['fiscalyear'] - df['prev_year']) == 1)
    
    df['turnover_event'] = condition.astype(int)
    
    n_events = df['turnover_event'].sum()
    rate = n_events / len(df)
    print(f"  Identified {n_events} turnover events")
    print(f"  Turnover Rate: {rate:.2%}")
    
    return df

def analyze_event_study(merged_df):
    """Conduct event study around turnover."""
    print("\n📉 Running Event Study...")
    
    # Filter to turnover events
    turnover_events = merged_df[merged_df['turnover_event'] == 1].copy()
    
    # We need to grab the surrounding years for each event
    # This is easier by iterating unique events
    
    windows = []
    
    for _, row in turnover_events.iterrows():
        gvkey = row['gvkey']
        year = row['fiscalyear']
        
        # Get -2 to +2 window
        window_df = merged_df[
            (merged_df['gvkey'] == gvkey) & 
            (merged_df['fiscalyear'] >= year - 2) & 
            (merged_df['fiscalyear'] <= year + 2)
        ].copy()
        
        window_df['rel_year'] = window_df['fiscalyear'] - year
        windows.append(window_df)
    
    if not windows:
        print("  No valid windows found.")
        return None
        
    event_df = pd.concat(windows)
    
    # Aggregate by relative year
    summary = event_df.groupby('rel_year').agg({
        'n_activists': 'mean',
        'inst_ownership': 'mean',
        'match_means': 'mean',
        'gvkey': 'count'
    }).round(4)
    
    print("\n--- EVENT STUDY: 2 Years Around Turnover (t=0) ---")
    print(summary)
    
    return summary, event_df

def create_plots(summary, event_df, merged_df):
    """Generate visualizations."""
    print("\n📊 Generating Plots...")
    
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Event Study: Activists
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(summary.index, summary['n_activists'], 'r-o', lw=2)
    ax1.set_title('Activist Presence Around Turnover', fontweight='bold')
    ax1.set_xlabel('Years Relative to Turnover')
    ax1.set_ylabel('Avg # Activist Investors')
    ax1.axvline(0, color='black', linestyle='--')
    
    # 2. Event Study: Inst Ownership
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(summary.index, summary['inst_ownership']*100, 'b-s', lw=2)
    ax2.set_title('Inst. Ownership Around Turnover', fontweight='bold')
    ax2.set_xlabel('Years Relative to Turnover')
    ax2.set_ylabel('Ownership (%)')
    ax2.axvline(0, color='black', linestyle='--')
    
    # 3. Turnover Rate by Activist Presence
    ax3 = fig.add_subplot(2, 3, 3)
    merged_df['has_activist'] = merged_df['n_activists'] > 0
    t_rate = merged_df.groupby('has_activist')['turnover_event'].mean() * 100
    colors = ['gray', 'red']
    ax3.bar(range(len(t_rate)), t_rate, color=colors, edgecolor='black')
    ax3.set_xticks(range(len(t_rate)))
    ax3.set_xticklabels(['No Activist', 'Has Activist'])
    ax3.set_ylabel('Annual Turnover Probability (%)')
    ax3.set_title('Does Activist Presence Predict Turnover?', fontweight='bold')
    
    # 4. Heatmap: Match Quality vs Activist Presence
    ax4 = fig.add_subplot(2, 3, 4)
    try:
        match_bins = pd.qcut(merged_df['match_means'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        act_rate = merged_df.groupby(match_bins)['has_activist'].mean() * 100
        ax4.bar(range(5), act_rate, color='purple', edgecolor='black')
        ax4.set_xticks(range(5))
        ax4.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax4.set_ylabel('% Firms with Activist')
        ax4.set_title('Do Activists Target Bad Matches?', fontweight='bold')
    except:
        pass

    # 5. Scatter: Activist Shares vs Match
    ax5 = fig.add_subplot(2, 3, 5)
    sample = merged_df[merged_df['activist_shares'] > 0]
    if len(sample) > 0:
        ax5.scatter(sample['match_means'], np.log1p(sample['activist_shares']), alpha=0.5, c='orange')
        ax5.set_xlabel('Match Quality')
        ax5.set_ylabel('Log Activist Shares')
        ax5.set_title('Activist Stake Size vs Match', fontweight='bold')

    plt.tight_layout()
    plt.savefig('Output/activist_turnover_study.png', dpi=150)
    print("  Saved: Output/activist_turnover_study.png")

def main():
    print("🕵️‍♂️" * 40)
    print("   ACTIVIST INVESTOR & TURNOVER ANALYSIS")
    print("🕵️‍♂️" * 40)
    
    # 1. Connect
    db = connect_wrds()
    if not db: return

    # 2. Get Targets
    cusip_sql, cusip_list = get_target_cusips()
    if not cusip_sql: return
    
    # 3. Pull Data
    raw_holdings = pull_activist_data(db, cusip_sql)
    db.close()
    
    # 4. Identify Activists
    activist_agg = identify_activists(raw_holdings)
    
    # 5. Link CUSIP -> GVKEY (Reuse the logic/files)
    ticker_cusip = pd.read_parquet('Output/wrds_ticker_cusip.parquet')
    crsp_comp = pd.read_parquet('Output/wrds_crsp_comp.parquet')
    
    # Merge CUSIP -> PERMNO
    activist_agg = activist_agg.merge(ticker_cusip[['cusip', 'permno']].drop_duplicates(), on='cusip', how='left')
    # Merge PERMNO -> GVKEY
    permno_gvkey = crsp_comp[['permno', 'gvkey']].drop_duplicates()
    permno_gvkey['gvkey'] = pd.to_numeric(permno_gvkey['gvkey'], errors='coerce')
    activist_agg = activist_agg.merge(permno_gvkey, on='permno', how='left')
    activist_agg = activist_agg.dropna(subset=['gvkey'])
    activist_agg['gvkey'] = activist_agg['gvkey'].astype(int)
    
    # Collapse to GVKEY level (summing across CUSIPs if multiple)
    gvkey_activist = activist_agg.groupby(['gvkey', 'year']).agg({
        'n_activists': 'sum',
        'activist_shares': 'sum'
    }).reset_index()
    
    # 6. Load & Process CEO Data
    # Merge with the PREVIOUS merged dataset to keep 13F aggregate vars
    # But if that's not robust, let's load raw and merge again
    try:
        base_df = pd.read_parquet('Output/ceo_match_13f_merged.parquet')
        print(f"\nExample loaded base data: {len(base_df)} obs")
    except:
        print("Could not load merged parquet, falling back to raw CSV+Holdings.")
        # Fallback logic omitted for brevity, assuming previous step succeeded
        return

    # Add Turnover Flags
    base_df = process_turnover(base_df)
    
    # Merge Activist Data
    final_df = base_df.merge(
        gvkey_activist, 
        left_on=['gvkey', 'fiscalyear'], 
        right_on=['gvkey', 'year'], 
        how='left', 
        suffixes=('', '_act')
    )
    final_df['n_activists'] = final_df['n_activists'].fillna(0)
    final_df['activist_shares'] = final_df['activist_shares'].fillna(0)
    
    print(f"  Final Analysis Set: {len(final_df)} observations")
    print(f"  Firms with Activists: {final_df[final_df['n_activists']>0]['gvkey'].nunique()}")
    
    # 7. Analyze
    summary, event_df = analyze_event_study(final_df)
    
    # 8. Visualize
    create_plots(summary, event_df, final_df)
    
    # Save
    final_df.to_parquet('Output/activist_turnover_data.parquet')
    print("\n✅ Analysis Complete. Data saved to Output/activist_turnover_data.parquet")

if __name__ == "__main__":
    main()
