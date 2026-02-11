#!/usr/bin/env python3
"""
Enriched CEO-Firm Match Analysis

Loads WRDS-enriched data (BoardEx + CIQ + ExecuComp features already merged),
trains baseline vs enriched Two-Tower models, runs ML comparisons,
and produces publication-quality match quality visualizations.

Usage:
    python run_enriched_analysis.py                # Full pipeline
    python run_enriched_analysis.py --epochs 80    # More training
    python run_enriched_analysis.py --skip-model   # Viz only (loads saved model)
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

# Matplotlib config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ceo_firm_matching.config import Config
from ceo_firm_matching.data import DataProcessor, CEOFirmDataset
from ceo_firm_matching.model import CEOFirmMatcher
from ceo_firm_matching.training import train_model

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR = 'Output/Enriched_Analysis'
DATA_PATH = 'Data/ceo_types_enriched.csv'

# Available enriched CEO features (verified present in enriched CSV)
ENRICHED_CEO_NUMERIC = [
    'Age',
    # Career trajectory (BoardEx)
    'n_prior_roles', 'n_prior_boards', 'avg_board_tenure',
    'years_as_ceo', 'n_sectors', 'career_span_years',
    # Network (BoardEx)
    'network_size', 'board_interlocks',
    # Compensation (ExecuComp)
    'log_tdc1', 'salary', 'stock_awards_fv', 'option_awards_fv',
    # CIQ career/network
    'ciq_n_affiliations', 'ciq_n_roles', 'ciq_current_boards',
    'ciq_career_span', 'ciq_best_rank', 'ciq_equity_share',
]

# Pretty labels for visualizations
FEATURE_LABELS = {
    'Age': 'CEO Age',
    'n_prior_roles': 'Prior Roles',
    'n_prior_boards': 'Prior Boards',
    'avg_board_tenure': 'Avg Board Tenure',
    'years_as_ceo': 'Years as CEO',
    'n_sectors': 'Industry Breadth',
    'career_span_years': 'Career Span',
    'network_size': 'Network Size',
    'board_interlocks': 'Board Interlocks',
    'log_tdc1': 'Log Compensation',
    'salary': 'Salary ($K)',
    'stock_awards_fv': 'Stock Awards ($K)',
    'option_awards_fv': 'Option Awards ($K)',
    'ciq_n_affiliations': 'CIQ Affiliations',
    'ciq_n_roles': 'CIQ Roles',
    'ciq_current_boards': 'Current Boards',
    'ciq_career_span': 'CIQ Career Span',
    'ciq_best_rank': 'CIQ Rank',
    'ciq_equity_share': 'CIQ Equity Share',
    'tenure': 'Tenure',
    'logatw': 'Firm Size (log AT)',
    'exp_roa': 'Expected ROA',
    'rdintw': 'R&D Intensity',
    'capintw': 'Capital Intensity',
    'leverage': 'Leverage',
    'divyieldw': 'Dividend Yield',
    'boardindpw': 'Board Independence',
    'boardsizew': 'Board Size',
    'busyw': 'Busy Directors',
    'pct_blockw': 'Block Ownership',
    'ind_firms_60w': 'Local Competition',
    'non_competition_score': 'Non-Compete Score',
    'match_means': 'Match Quality',
    'compindustry': 'Industry',
}


class EnrichedConfig(Config):
    """Config with enriched CEO features from WRDS/BoardEx/CIQ/ExecuComp."""
    DATA_PATH = DATA_PATH
    CEO_NUMERIC_COLS = ENRICHED_CEO_NUMERIC
    # Firm features stay the same (original 12)
    LATENT_DIM = 80
    EMBEDDING_DIM_MEDIUM = 12
    EPOCHS = 60


def parse_args():
    parser = argparse.ArgumentParser(description='Enriched CEO-Firm Match Analysis')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--skip-model', action='store_true', help='Skip model training')
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# STEP 1: LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────────────

def load_enriched_data(config):
    """Load the WRDS-enriched dataset and prepare features."""
    print(f"\nLoading enriched data from {config.DATA_PATH}...")
    df = pd.read_csv(config.DATA_PATH, on_bad_lines='skip')
    print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")

    # Check enriched feature coverage
    enriched_cols = [c for c in ENRICHED_CEO_NUMERIC if c != 'Age']
    present = [c for c in enriched_cols if c in df.columns]
    missing = [c for c in enriched_cols if c not in df.columns]
    print(f"  Enriched features present: {len(present)}/{len(enriched_cols)}")
    if missing:
        print(f"  ⚠ Missing: {missing}")

    # Fill NaN in enriched features with median
    for col in present:
        if df[col].isna().any():
            median_val = df[col].median()
            n_fill = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            print(f"    Filled {n_fill} NaNs in {col} with median={median_val:.2f}")

    return df


# ─────────────────────────────────────────────────────────────────────
# STEP 2: TRAIN MODELS (BASELINE vs ENRICHED)
# ─────────────────────────────────────────────────────────────────────

def train_two_towers(df, config, tag=''):
    """Train a Two-Tower model and return model + processor + data."""
    print(f"\n{'='*70}")
    print(f"TRAINING TWO-TOWER MODEL{f' ({tag})' if tag else ''}")
    print(f"  CEO numeric: {len(config.CEO_NUMERIC_COLS)} features")
    print(f"  Firm numeric: {len(config.FIRM_NUMERIC_COLS)} features")
    print(f"  Latent dim: {config.LATENT_DIM}, Epochs: {config.EPOCHS}")
    print(f"{'='*70}")

    processor = DataProcessor(config)
    df_proc = processor.prepare_features(df)
    processor.fit(df_proc)
    data_dict = processor.transform(df_proc)

    metadata = {
        'n_firm_numeric': data_dict['n_firm_numeric'],
        'firm_cat_counts': data_dict['firm_cat_counts'],
        'n_ceo_numeric': data_dict['n_ceo_numeric'],
        'ceo_cat_counts': data_dict['ceo_cat_counts'],
    }

    dataset = CEOFirmDataset(data_dict)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    model = train_model(train_loader, val_loader, metadata, config)

    return model, processor, data_dict, metadata, df_proc


def run_ml_comparison(df_proc, enriched_ceo_cols, firm_cols, target_col='match_means'):
    """Run Gradient Boosting and Random Forest on baseline vs enriched features."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import r2_score
    import sklearn

    baseline_ceo = ['Age', 'tenure']
    enriched_ceo = enriched_ceo_cols + ['tenure']

    # Remove any columns not in df
    enriched_ceo = [c for c in enriched_ceo if c in df_proc.columns]
    firm_cols = [c for c in firm_cols if c in df_proc.columns]

    results = {}

    for label, features in [('Baseline', baseline_ceo + firm_cols),
                             ('Enriched', enriched_ceo + firm_cols)]:
        X = df_proc[features].copy()
        y = df_proc[target_col]

        # Handle any remaining NaNs
        X = X.fillna(X.median())
        mask = y.notna()
        X, y = X[mask], y[mask]

        for name, model_cls, params in [
            ('GBM', GradientBoostingRegressor, {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1}),
            ('RF', RandomForestRegressor, {'n_estimators': 200, 'max_depth': 8, 'n_jobs': -1}),
        ]:
            model = model_cls(**params, random_state=42)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

            # Fit on full data for feature importance
            model.fit(X, y)
            importances = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            key = f"{label}_{name}"
            results[key] = {
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'n_features': len(features),
                'importances': importances,
                'features': features,
            }
            print(f"  {key}: CV R²={cv_scores.mean():.4f} ± {cv_scores.std():.4f} ({len(features)} features)")

    return results


# ─────────────────────────────────────────────────────────────────────
# STEP 3: MATCH QUALITY VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────

def setup_style():
    """Publication-ready matplotlib style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def viz_1_match_heatmaps(df, output_dir):
    """
    VIZ 1: 2D heatmaps of match quality across CEO × Firm observable dimensions.
    Three panels: (Size × Career Span), (R&D × Prior Boards), (ROA × Network Size)
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    panels = [
        ('logatw', 'career_span_years', 'Firm Size (log AT)', 'Career Span (years)'),
        ('rdintw', 'n_prior_boards', 'R&D Intensity', 'Prior Board Seats'),
        ('exp_roa', 'network_size', 'Expected ROA', 'Network Size'),
    ]

    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, panels):
        if xcol not in df.columns or ycol not in df.columns:
            ax.text(0.5, 0.5, f'Missing: {xcol} or {ycol}', ha='center', va='center',
                   transform=ax.transAxes)
            continue

        # Bin both features into quintiles
        x_q = pd.qcut(df[xcol], 5, labels=False, duplicates='drop')
        y_q = pd.qcut(df[ycol], 5, labels=False, duplicates='drop')

        heatmap = df.assign(xq=x_q, yq=y_q).groupby(['yq', 'xq'])['match_means'].mean().unstack()

        # Get bin labels (quintile ranges)
        x_edges = df.groupby(x_q)[xcol].agg(['min', 'max']).apply(
            lambda r: f"{r['min']:.1f}–{r['max']:.1f}", axis=1).values
        y_edges = df.groupby(y_q)[ycol].agg(['min', 'max']).apply(
            lambda r: f"{r['min']:.1f}–{r['max']:.1f}", axis=1).values

        im = ax.imshow(heatmap.values, cmap='RdBu_r', aspect='auto',
                       vmin=-0.5, vmax=0.5, origin='lower')

        # Annotate cells
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                val = heatmap.values[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 0.3 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=8, color=color, fontweight='bold')

        ax.set_xticks(range(len(x_edges)))
        ax.set_xticklabels(x_edges, rotation=40, ha='right', fontsize=8)
        ax.set_yticks(range(len(y_edges)))
        ax.set_yticklabels(y_edges, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.suptitle('Match Quality Across CEO × Firm Observable Dimensions',
                fontsize=15, fontweight='bold', y=1.02)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Mean Match Quality')
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    path = f'{output_dir}/match_heatmaps.pdf'
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


def viz_2_feature_importance_comparison(ml_results, output_dir):
    """
    VIZ 2: Side-by-side feature importance — Baseline vs Enriched GBM.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, key, title in zip(axes,
                               ['Baseline_GBM', 'Enriched_GBM'],
                               ['Baseline Model (14 features)', 'Enriched Model (30+ features)']):
        if key not in ml_results:
            ax.text(0.5, 0.5, f'No {key} results', ha='center', va='center',
                   transform=ax.transAxes)
            continue

        imp = ml_results[key]['importances'].head(15).sort_values('importance', ascending=True)
        labels = [FEATURE_LABELS.get(f, f) for f in imp['feature']]
        colors = ['#2196F3' if f in ENRICHED_CEO_NUMERIC else '#FF9800'
                  for f in imp['feature']]

        ax.barh(labels, imp['importance'], color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Feature Importance (Gini)')
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        r2 = ml_results[key]['cv_r2_mean']
        ax.text(0.95, 0.05, f'CV R² = {r2:.3f}',
               transform=ax.transAxes, ha='right', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                         edgecolor='gray', alpha=0.9))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', label='WRDS Enriched Features'),
        Patch(facecolor='#FF9800', label='Original Features'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
              framealpha=0.9, fontsize=11, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    path = f'{output_dir}/feature_importance_comparison.pdf'
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


def viz_3_match_distribution_by_industry(df, output_dir):
    """
    VIZ 3: Violin plots of match quality by industry.
    Shows which sectors have strongest/weakest CEO-firm matching.
    """
    setup_style()

    # Get top industries by count
    top_ind = df['compindustry'].value_counts().head(12).index
    plot_df = df[df['compindustry'].isin(top_ind)].copy()

    # Order by median match quality
    order = plot_df.groupby('compindustry')['match_means'].median().sort_values().index

    fig, ax = plt.subplots(figsize=(14, 6))

    parts = ax.violinplot(
        [plot_df[plot_df['compindustry'] == ind]['match_means'].values for ind in order],
        positions=range(len(order)),
        showmeans=True, showmedians=True, showextrema=False
    )

    # Color bodies
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(order)))
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(colors[i])
        body.set_alpha(0.7)
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('red')

    # Stats annotation
    for i, ind in enumerate(order):
        sub = plot_df[plot_df['compindustry'] == ind]['match_means']
        ax.text(i, sub.quantile(0.95) + 0.05, f'n={len(sub)}',
               ha='center', fontsize=7, color='gray')

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Match Quality (match_means)')
    ax.set_title('Match Quality Distribution by Industry', fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], color='black', lw=2, label='Mean'),
              Line2D([0], [0], color='red', lw=2, label='Median')]
    ax.legend(handles=legend, loc='upper left')

    plt.tight_layout()
    path = f'{output_dir}/match_by_industry.pdf'
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


def viz_4_best_worst_matches(df, model, processor, config, output_dir):
    """
    VIZ 4: Spider/radar charts of top-10 best vs worst CEO-firm matches.
    Each axis = a normalized observable.
    """
    setup_style()

    # Get predictions
    model.eval()
    data_dict = processor.transform(df.copy())
    with torch.no_grad():
        preds = model(
            data_dict['firm_numeric'].to(config.DEVICE),
            data_dict['firm_cat'].to(config.DEVICE),
            data_dict['ceo_numeric'].to(config.DEVICE),
            data_dict['ceo_cat'].to(config.DEVICE)
        ).cpu().numpy().flatten()

    df_pred = df.copy()
    df_pred['predicted_match'] = preds

    # Select radar features (mix of CEO and Firm)
    radar_features = ['Age', 'n_prior_roles', 'network_size', 'log_tdc1',
                      'logatw', 'exp_roa', 'rdintw', 'leverage']
    radar_features = [f for f in radar_features if f in df.columns]
    radar_labels = [FEATURE_LABELS.get(f, f) for f in radar_features]

    # Normalize to [0, 1] for radar
    normed = df_pred[radar_features].copy()
    for col in radar_features:
        mn, mx = normed[col].min(), normed[col].max()
        normed[col] = (normed[col] - mn) / (mx - mn + 1e-8)

    # Top 10 best and worst by predicted match
    best_idx = df_pred['predicted_match'].nlargest(10).index
    worst_idx = df_pred['predicted_match'].nsmallest(10).index

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
    angles += angles[:1]

    for ax, idx, title, color in zip(axes,
                                      [best_idx, worst_idx],
                                      ['Top-10 Best Matches', 'Top-10 Worst Matches'],
                                      ['#2E7D32', '#C62828']):
        # Plot each observation
        for i, row_idx in enumerate(idx):
            values = normed.loc[row_idx, radar_features].values.tolist()
            values += values[:1]
            alpha = 0.4 if i > 0 else 0.8
            lw = 2 if i == 0 else 0.8
            ax.plot(angles, values, linewidth=lw, color=color, alpha=alpha)
            ax.fill(angles, values, color=color, alpha=0.03)

        # Mean profile (thick)
        mean_vals = normed.loc[idx, radar_features].mean().values.tolist()
        mean_vals += mean_vals[:1]
        ax.plot(angles, mean_vals, linewidth=3, color=color, linestyle='--',
               label='Mean profile')
        ax.fill(angles, mean_vals, color=color, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, fontsize=9)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Observable Profiles of Best vs Worst CEO-Firm Matches',
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = f'{output_dir}/best_worst_radar.pdf'
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


def viz_5_counterfactual_improvements(df, model, processor, config, output_dir):
    """
    VIZ 5: Top counterfactual CEO-firm reassignments.
    For each firm, find which CEO in the sample would maximize predicted match quality.
    """
    setup_style()

    model.eval()
    data_dict = processor.transform(df.copy())

    with torch.no_grad():
        # Get embeddings
        # Firm embeddings
        f_embs_list = [emb(data_dict['firm_cat'].to(config.DEVICE)[:, i])
                       for i, emb in enumerate(model.firm_embeddings)]
        f_combined = torch.cat([data_dict['firm_numeric'].to(config.DEVICE)] + f_embs_list, dim=1)
        u_firm = model.firm_tower(f_combined)
        u_firm = u_firm / u_firm.norm(dim=1, keepdim=True)

        # CEO embeddings
        c_embs_list = [emb(data_dict['ceo_cat'].to(config.DEVICE)[:, i])
                       for i, emb in enumerate(model.ceo_embeddings)]
        c_combined = torch.cat([data_dict['ceo_numeric'].to(config.DEVICE)] + c_embs_list, dim=1)
        v_ceo = model.ceo_tower(c_combined)
        v_ceo = v_ceo / v_ceo.norm(dim=1, keepdim=True)

        logit_scale = model.logit_scale.exp()

    # Compute all pairwise match scores (sample 500 firms for speed)
    n = min(500, len(u_firm))
    np.random.seed(42)
    sample_idx = np.random.choice(len(u_firm), n, replace=False)

    u_sample = u_firm[sample_idx]  # (n, d)
    similarity = (u_sample @ v_ceo.T * logit_scale).cpu().numpy()  # (n, N)

    # Current match scores (diagonal of sample)
    current_scores = np.array([similarity[i, sample_idx[i]] for i in range(n)])
    best_ceo_idx = similarity.argmax(axis=1)
    best_scores = similarity.max(axis=1)
    improvements = best_scores - current_scores

    # Build results DataFrame
    cf_df = pd.DataFrame({
        'firm_idx': sample_idx,
        'current_score': current_scores,
        'best_score': best_scores,
        'improvement': improvements,
        'best_ceo_idx': best_ceo_idx,
    })

    # Identify firms with biggest improvement potential
    cf_df = cf_df.sort_values('improvement', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Distribution of improvements
    ax = axes[0]
    ax.hist(cf_df['improvement'], bins=40, color='#1565C0', edgecolor='white',
            alpha=0.8)
    pct_improve = (cf_df['improvement'] > 0).mean() * 100
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Match Quality Improvement')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Counterfactual Improvements', fontweight='bold')
    ax.text(0.95, 0.95, f'{pct_improve:.0f}% of firms\ncould improve',
           transform=ax.transAxes, ha='right', va='top', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))
    ax.grid(axis='y', alpha=0.3)

    # Panel B: Current vs Best — top 25
    ax = axes[1]
    top = cf_df.head(25)
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top['current_score'], height=0.4, color='#EF5350',
           label='Current Match', align='center')
    ax.barh(y_pos - 0.4, top['best_score'], height=0.4, color='#66BB6A',
           label='Optimal Match', align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'Firm {i}' for i in range(1, len(top)+1)], fontsize=8)
    ax.set_xlabel('Predicted Match Quality')
    ax.set_title('Top 25 Firms: Largest Match Improvement', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = f'{output_dir}/counterfactual_improvements.pdf'
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")

    # Save CSV
    cf_path = f'{output_dir}/counterfactual_results.csv'
    cf_df.to_csv(cf_path, index=False)
    print(f"  ✓ Saved: {cf_path}")

    return path, cf_df


def viz_6_embedding_landscape(df, model, processor, config, output_dir):
    """
    VIZ 6: t-SNE of CEO + Firm embeddings colored by match quality.
    """
    setup_style()
    from sklearn.manifold import TSNE

    model.eval()
    data_dict = processor.transform(df.copy())

    with torch.no_grad():
        # Firm embeddings
        f_embs_list = [emb(data_dict['firm_cat'].to(config.DEVICE)[:, i])
                       for i, emb in enumerate(model.firm_embeddings)]
        f_combined = torch.cat([data_dict['firm_numeric'].to(config.DEVICE)] + f_embs_list, dim=1)
        u_firm = model.firm_tower(f_combined)
        u_firm = u_firm / u_firm.norm(dim=1, keepdim=True)

        # CEO embeddings
        c_embs_list = [emb(data_dict['ceo_cat'].to(config.DEVICE)[:, i])
                       for i, emb in enumerate(model.ceo_embeddings)]
        c_combined = torch.cat([data_dict['ceo_numeric'].to(config.DEVICE)] + c_embs_list, dim=1)
        v_ceo = model.ceo_tower(c_combined)
        v_ceo = v_ceo / v_ceo.norm(dim=1, keepdim=True)

    # Subsample for t-SNE speed
    n = min(3000, len(u_firm))
    idx = np.random.RandomState(42).choice(len(u_firm), n, replace=False)
    all_embs = torch.cat([u_firm[idx], v_ceo[idx]], dim=0).cpu().numpy()
    labels = np.concatenate([np.zeros(n), np.ones(n)])  # 0=firm, 1=ceo
    match_vals = np.concatenate([
        data_dict['target'].numpy().flatten()[idx],
        data_dict['target'].numpy().flatten()[idx]
    ])

    print("  Running t-SNE (this may take ~30s)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    embs_2d = tsne.fit_transform(all_embs)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Entity type
    ax = axes[0]
    firm_mask = labels == 0
    ceo_mask = labels == 1
    ax.scatter(embs_2d[firm_mask, 0], embs_2d[firm_mask, 1],
              c='#1565C0', alpha=0.4, s=12, label='Firm')
    ax.scatter(embs_2d[ceo_mask, 0], embs_2d[ceo_mask, 1],
              c='#E65100', alpha=0.4, s=12, label='CEO')
    ax.set_title('Embedding Space by Entity Type', fontweight='bold')
    ax.legend(markerscale=3)
    ax.grid(alpha=0.2)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    # Panel B: Match quality
    ax = axes[1]
    sc = ax.scatter(embs_2d[:n, 0], embs_2d[:n, 1],
                   c=match_vals[:n], cmap='RdBu_r', alpha=0.6, s=15,
                   vmin=-1, vmax=1)
    ax.set_title('Firm Embeddings Colored by Match Quality', fontweight='bold')
    plt.colorbar(sc, ax=ax, label='Match Quality')
    ax.grid(alpha=0.2)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    plt.suptitle('Two-Tower Embedding Landscape (Enriched Model)',
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = f'{output_dir}/embedding_landscape.pdf'
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


def viz_7_model_comparison_summary(ml_results, baseline_corr, enriched_corr, output_dir):
    """
    VIZ 7: Summary comparison of Baseline vs Enriched across all model types.
    """
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: CV R² across ML models
    ax = axes[0]
    models = ['GBM', 'RF']
    x = np.arange(len(models))
    width = 0.35

    baseline_scores = [ml_results.get(f'Baseline_{m}', {}).get('cv_r2_mean', 0) for m in models]
    enriched_scores = [ml_results.get(f'Enriched_{m}', {}).get('cv_r2_mean', 0) for m in models]
    baseline_stds = [ml_results.get(f'Baseline_{m}', {}).get('cv_r2_std', 0) for m in models]
    enriched_stds = [ml_results.get(f'Enriched_{m}', {}).get('cv_r2_std', 0) for m in models]

    bars1 = ax.bar(x - width/2, baseline_scores, width, yerr=baseline_stds,
                   label='Baseline', color='#90CAF9', edgecolor='#1565C0',
                   capsize=5, linewidth=1.2)
    bars2 = ax.bar(x + width/2, enriched_scores, width, yerr=enriched_stds,
                   label='Enriched (WRDS)', color='#A5D6A7', edgecolor='#2E7D32',
                   capsize=5, linewidth=1.2)

    # Annotate bars
    for bars, scores in [(bars1, baseline_scores), (bars2, enriched_scores)]:
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel('5-Fold CV R²')
    ax.set_title('ML Model Performance: Baseline vs Enriched', fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(enriched_scores) * 1.3)

    # Panel B: Two-Tower correlation
    ax = axes[1]
    labels_tt = ['Baseline\nTwo-Towers', 'Enriched\nTwo-Towers']
    values_tt = [baseline_corr, enriched_corr]
    colors_tt = ['#90CAF9', '#A5D6A7']
    edge_colors = ['#1565C0', '#2E7D32']

    bars = ax.bar(labels_tt, values_tt, color=colors_tt, edgecolor=edge_colors,
                 linewidth=1.5, width=0.5)
    for bar, val in zip(bars, values_tt):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Validation Correlation')
    ax.set_title('Two-Tower Neural Network: Baseline vs Enriched', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(values_tt) * 1.3)

    # Improvement annotations
    if len(enriched_scores) >= 2 and len(baseline_scores) >= 2:
        imp_gbm = (enriched_scores[0] - baseline_scores[0]) / max(baseline_scores[0], 1e-6) * 100
        imp_tt = (enriched_corr - baseline_corr) / max(baseline_corr, 1e-6) * 100
        fig.text(0.5, -0.02,
                f'GBM improvement: +{imp_gbm:.1f}%  |  Two-Towers improvement: +{imp_tt:.1f}%',
                ha='center', fontsize=11, color='#2E7D32', fontweight='bold')

    plt.tight_layout()
    path = f'{output_dir}/model_comparison.pdf'
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


def compute_validation_correlation(model, data_dict, config):
    """Compute validation correlation for a trained model."""
    model.eval()
    with torch.no_grad():
        preds = model(
            data_dict['firm_numeric'].to(config.DEVICE),
            data_dict['firm_cat'].to(config.DEVICE),
            data_dict['ceo_numeric'].to(config.DEVICE),
            data_dict['ceo_cat'].to(config.DEVICE)
        ).cpu().numpy().flatten()

    targets = data_dict['target'].numpy().flatten()
    corr = np.corrcoef(preds, targets)[0, 1]
    return corr


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start = time.time()

    print("=" * 70)
    print("  ENRICHED CEO-FIRM MATCH ANALYSIS")
    print("  WRDS Data → Train Models → Visualize Match Quality")
    print("=" * 70)

    # ─── STEP 1: Load Data ───
    enriched_config = EnrichedConfig()
    enriched_config.EPOCHS = args.epochs
    df = load_enriched_data(enriched_config)

    # ─── STEP 2: Train Baseline Two-Towers ───
    if not args.skip_model:
        print("\n" + "=" * 70)
        print("PART A: BASELINE TWO-TOWER MODEL")
        print("=" * 70)
        baseline_config = Config()
        baseline_config.DATA_PATH = DATA_PATH
        baseline_config.EPOCHS = args.epochs

        # Need all required columns for baseline (filter df)
        baseline_model, baseline_proc, baseline_data, baseline_meta, baseline_df = \
            train_two_towers(df, baseline_config, tag='Baseline')
        baseline_corr = compute_validation_correlation(
            baseline_model, baseline_data, baseline_config)
        print(f"  Baseline validation correlation: {baseline_corr:.3f}")

        # ─── STEP 3: Train Enriched Two-Towers ───
        print("\n" + "=" * 70)
        print("PART B: ENRICHED TWO-TOWER MODEL")
        print("=" * 70)
        enriched_model, enriched_proc, enriched_data, enriched_meta, enriched_df = \
            train_two_towers(df, enriched_config, tag='Enriched')
        enriched_corr = compute_validation_correlation(
            enriched_model, enriched_data, enriched_config)
        print(f"  Enriched validation correlation: {enriched_corr:.3f}")
    else:
        print("\n  ⚠ Skipping model training (--skip-model)")
        baseline_corr = 0.613
        enriched_corr = 0.0
        # Still need processor for viz
        enriched_model = None
        enriched_proc = None
        enriched_df = df

    # ─── STEP 4: ML Comparison ───
    print("\n" + "=" * 70)
    print("PART C: ML COMPARISON (BASELINE vs ENRICHED)")
    print("=" * 70)

    # Prepare tenure
    if 'tenure' not in df.columns and 'fiscalyear' in df.columns and 'ceo_year' in df.columns:
        df['tenure'] = (df['fiscalyear'] - df['ceo_year']).clip(lower=0)

    ml_results = run_ml_comparison(
        df,
        enriched_ceo_cols=ENRICHED_CEO_NUMERIC,
        firm_cols=Config.FIRM_NUMERIC_COLS,
        target_col='match_means'
    )

    # ─── STEP 5: Visualizations ───
    print("\n" + "=" * 70)
    print("PART D: MATCH QUALITY VISUALIZATIONS")
    print("=" * 70)

    viz_paths = []

    # VIZ 1: Match quality heatmaps
    print("\n  [1/7] Match Quality Heatmaps...")
    viz_paths.append(viz_1_match_heatmaps(df, OUTPUT_DIR))

    # VIZ 2: Feature importance comparison
    print("\n  [2/7] Feature Importance Comparison...")
    viz_paths.append(viz_2_feature_importance_comparison(ml_results, OUTPUT_DIR))

    # VIZ 3: Industry distribution
    print("\n  [3/7] Match Distribution by Industry...")
    viz_paths.append(viz_3_match_distribution_by_industry(df, OUTPUT_DIR))

    if not args.skip_model and enriched_model is not None:
        # VIZ 4: Best/worst match radar
        print("\n  [4/7] Best vs Worst Match Profiles...")
        viz_paths.append(viz_4_best_worst_matches(
            enriched_df, enriched_model, enriched_proc, enriched_config, OUTPUT_DIR))

        # VIZ 5: Counterfactual improvements
        print("\n  [5/7] Counterfactual Improvements...")
        path, cf_df = viz_5_counterfactual_improvements(
            enriched_df, enriched_model, enriched_proc, enriched_config, OUTPUT_DIR)
        viz_paths.append(path)

        # VIZ 6: Embedding landscape
        print("\n  [6/7] Embedding Landscape (t-SNE)...")
        viz_paths.append(viz_6_embedding_landscape(
            enriched_df, enriched_model, enriched_proc, enriched_config, OUTPUT_DIR))

        # VIZ 7: Model comparison
        print("\n  [7/7] Model Comparison Summary...")
        viz_paths.append(viz_7_model_comparison_summary(
            ml_results, baseline_corr, enriched_corr, OUTPUT_DIR))
    else:
        print("\n  ⚠ Skipping model-dependent visualizations (4-7)")

    # ─── SUMMARY ───
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("  ENRICHED ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Data: {len(df):,} rows")
    print(f"  CEO features — Baseline: {len(Config.CEO_NUMERIC_COLS)} → Enriched: {len(ENRICHED_CEO_NUMERIC)}")
    if not args.skip_model:
        print(f"  Two-Towers correlation — Baseline: {baseline_corr:.3f} → Enriched: {enriched_corr:.3f}")
    for key, res in sorted(ml_results.items()):
        print(f"  {key}: CV R²={res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}")
    print(f"\n  Visualizations saved to: {OUTPUT_DIR}/")
    for p in viz_paths:
        print(f"    ✓ {os.path.basename(p)}")
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
