#!/usr/bin/env python3
"""
🔍🔍🔍 MATCH-STRATIFIED DIAGNOSTICS 🔍🔍🔍

Deep dive into Activist/Turnover dynamics separated by Match Quality.
Tests hypothesis that "Turnover Forces" differ for Good (Q5) vs Bad (Q1) Matches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid')

def load_data():
    """Load the prepared activist/turnover data."""
    print("\n📦 Loading Data...")
    try:
        df = pd.read_parquet('Output/activist_turnover_data.parquet')
        print(f"  Loaded {len(df):,} observations")
        
        # Force re-creation of match quintiles to ensure consistent labels
        df['match_q'] = pd.qcut(df['match_means'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        # Cast to string to ensure easy filtering
        df['match_q'] = df['match_q'].astype(str)
        
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def stratified_event_study(df):
    """Run event studies split by Match Quality (Q1 vs Q5)."""
    print("\n📉 Running Stratified Event Study...")
    
    turnover_events = df[df['turnover_event'] == 1].copy()
    print(f"  Total Turnover Events: {len(turnover_events)}")
    
    quintiles = ['Q1', 'Q5']
    results = {}
    
    for q in quintiles:
        # Filter events in this quintile
        q_events = turnover_events[turnover_events['match_q'] == q]
        
        windows = []
        for _, row in q_events.iterrows():
            gvkey = row['gvkey']
            year = row['fiscalyear']
            
            # Get -2 to +2 window
            window_df = df[
                (df['gvkey'] == gvkey) & 
                (df['fiscalyear'] >= year - 2) & 
                (df['fiscalyear'] <= year + 2)
            ].copy()
            
            window_df['rel_year'] = window_df['fiscalyear'] - year
            windows.append(window_df)
        
        if windows:
            q_df = pd.concat(windows)
            summary = q_df.groupby('rel_year').agg({
                'n_activists': 'mean',
                'inst_ownership': 'mean',
                'inst_ownership_change': 'mean'
            })
            results[q] = summary
            print(f"  {q} Events: {len(q_events)}")
        else:
            print(f"  No events for {q}")
            
    return results

def turnover_probability_analysis(df):
    """Calculate P(Turnover) | Match, Activist."""
    print("\n🎲 Analyzing Turnover Probabilities...")
    
    df['has_activist'] = df['n_activists'] > 0
    
    # Group by Match Q and Activist Presence
    probs = df.groupby(['match_q', 'has_activist'], observed=False)['turnover_event'].mean().reset_index()
    probs['turnover_pct'] = probs['turnover_event'] * 100
    
    # Pivot for cleaner view
    pivot = probs.pivot(index='match_q', columns='has_activist', values='turnover_pct')
    pivot.columns = ['No Activist', 'Has Activist']
    pivot['Activist Impact (pp)'] = pivot['Has Activist'] - pivot['No Activist']
    pivot['Relative Risk'] = pivot['Has Activist'] / pivot['No Activist']
    
    print("\n--- TURNOVER PROBABILITY (%) ---")
    print(pivot.round(2))
    
    return pivot

def generate_plots(event_results, turnover_pivot):
    """Generate diagnostic plots."""
    print("\n📊 Generating Plots...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Stratified Event Study: Activist Presence
    ax1 = fig.add_subplot(2, 2, 1)
    if 'Q1' in event_results:
        ax1.plot(event_results['Q1'].index, event_results['Q1']['n_activists'], 'r-o', lw=3, label='Q1 (Bad Match)')
    if 'Q5' in event_results:
        ax1.plot(event_results['Q5'].index, event_results['Q5']['n_activists'], 'g-s', lw=3, label='Q5 (Good Match)')
    
    ax1.set_title('Activist Surge Trigger: Q1 vs Q5', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Years Relative to Turnover')
    ax1.set_ylabel('Avg # Activist Investors')
    ax1.axvline(0, color='black', linestyle='--')
    ax1.legend()
    
    # 2. Stratified Event Study: Institutional Ownership
    ax2 = fig.add_subplot(2, 2, 2)
    if 'Q1' in event_results:
        ax2.plot(event_results['Q1'].index, event_results['Q1']['inst_ownership']*100, 'r-o', lw=3, label='Q1')
    if 'Q5' in event_results:
        ax2.plot(event_results['Q5'].index, event_results['Q5']['inst_ownership']*100, 'g-s', lw=3, label='Q5')
        
    ax2.set_title('Institutional "Dip" Signal: Q1 vs Q5', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Years Relative to Turnover')
    ax2.set_ylabel('Institutional Ownership (%)')
    ax2.axvline(0, color='black', linestyle='--')
    ax2.legend()
    
    # 3. Turnover Probability Gap
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Prepare data for grouped bar chart
    x = np.arange(5)
    width = 0.35
    
    no_act = turnover_pivot['No Activist']
    has_act = turnover_pivot['Has Activist']
    
    ax3.bar(x - width/2, no_act, width, label='No Activist', color='gray', edgecolor='black')
    ax3.bar(x + width/2, has_act, width, label='Has Activist', color='#c0392b', edgecolor='black')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax3.set_ylabel('Annual Turnover Probability (%)')
    ax3.set_title('Does Activist Presence Predict Turnover?', fontweight='bold', fontsize=14)
    ax3.legend()
    
    # 4. Impact Magnitude
    ax4 = fig.add_subplot(2, 2, 4)
    impact = turnover_pivot['Activist Impact (pp)']
    colors = ['red' if x > 0 else 'green' for x in impact]
    ax4.bar(x, impact, color=colors, edgecolor='black', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax4.set_ylabel('Increased Turnover Prob (pp)')
    ax4.set_title('Marginal Impact of Activist on Turnover', fontweight='bold', fontsize=14)
    ax4.axhline(0, color='black')
    
    plt.tight_layout()
    plt.savefig('Output/match_stratified_diagnostics.png', dpi=150)
    print("  Saved: Output/match_stratified_diagnostics.png")

def main():
    print("🔍" * 40)
    print("   MATCH STRATIFIED DIAGNOSTICS")
    print("🔍" * 40)
    
    df = load_data()
    if df is None: return
    
    # Run analysis
    event_results = stratified_event_study(df)
    turnover_pivot = turnover_probability_analysis(df)
    
    # Generate plots
    generate_plots(event_results, turnover_pivot)
    
    print("\n🔍 Diagnostics Complete.")

if __name__ == "__main__":
    main()
