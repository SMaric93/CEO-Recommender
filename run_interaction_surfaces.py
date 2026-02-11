#!/usr/bin/env python3
"""
CEO × Firm Interaction Surfaces

For each pair of (CEO feature, Firm feature), sweep a grid through the trained
Two-Tower model holding all other features at their sample mean/mode, and plot
the predicted match quality surface.

Usage:
    python run_interaction_surfaces.py              # Full panel
    python run_interaction_surfaces.py --epochs 60  # More training
    python run_interaction_surfaces.py --grid 40    # Finer grid
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ceo_firm_matching.config import Config
from ceo_firm_matching.data import DataProcessor, CEOFirmDataset
from ceo_firm_matching.model import CEOFirmMatcher
from ceo_firm_matching.training import train_model

# ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = 'Output/Interaction_Surfaces'
DATA_PATH  = 'Data/ceo_types_enriched.csv'

# Enriched CEO numeric features available in the data
ENRICHED_CEO_NUMERIC = [
    'Age',
    'n_prior_roles', 'n_prior_boards', 'avg_board_tenure',
    'years_as_ceo', 'n_sectors', 'career_span_years',
    'network_size', 'board_interlocks',
    'log_tdc1', 'salary', 'stock_awards_fv', 'option_awards_fv',
    'ciq_n_affiliations', 'ciq_n_roles', 'ciq_current_boards',
    'ciq_career_span', 'ciq_best_rank', 'ciq_equity_share',
]

# Pretty labels
LABELS = {
    'Age': 'CEO Age', 'tenure': 'CEO Tenure (years)',
    'n_prior_roles': 'Prior Executive Roles',
    'n_prior_boards': 'Prior Board Seats',
    'avg_board_tenure': 'Avg Board Tenure (years)',
    'years_as_ceo': 'Years as CEO',
    'n_sectors': 'Industry Breadth (#sectors)',
    'career_span_years': 'Career Span (years)',
    'network_size': 'Network Size',
    'board_interlocks': 'Board Interlocks',
    'log_tdc1': 'Log Total Compensation',
    'salary': 'Salary ($K)',
    'stock_awards_fv': 'Stock Awards ($K)',
    'option_awards_fv': 'Option Awards ($K)',
    'ciq_n_affiliations': 'CIQ Affiliations',
    'ciq_n_roles': 'CIQ Career Roles',
    'ciq_current_boards': 'Current Board Seats',
    'ciq_career_span': 'CIQ Career Span (years)',
    'ciq_best_rank': 'CIQ Executive Rank',
    'ciq_equity_share': 'Equity Comp Share',
    'logatw': 'Firm Size (log assets)',
    'exp_roa': 'Expected ROA',
    'rdintw': 'R&D Intensity',
    'capintw': 'Capital Intensity',
    'leverage': 'Leverage',
    'divyieldw': 'Dividend Yield',
    'boardindpw': 'Board Independence',
    'boardsizew': 'Board Size',
    'busyw': 'Busy Directors',
    'pct_blockw': 'Block Ownership %',
    'ind_firms_60w': 'Local Industry Competition',
    'non_competition_score': 'Non-Compete Score',
}

# ── KEY CEO FEATURES (all 20 numeric: 19 enriched + tenure) ──
CEO_FEATURES = [
    'tenure', 'Age', 'n_prior_roles', 'n_prior_boards', 'avg_board_tenure',
    'years_as_ceo', 'n_sectors', 'career_span_years',
    'network_size', 'board_interlocks',
    'log_tdc1', 'salary', 'stock_awards_fv', 'option_awards_fv',
    'ciq_n_affiliations', 'ciq_n_roles', 'ciq_current_boards',
    'ciq_career_span', 'ciq_best_rank', 'ciq_equity_share',
]

# ── KEY FIRM DIMENSIONS ──
FIRM_DIMENSIONS = [
    ('logatw',       'Firm Size'),
    ('exp_roa',      'Expected ROA'),
    ('rdintw',       'R&D Intensity'),
    ('leverage',     'Leverage'),
    ('capintw',      'Capital Intensity'),
    ('boardindpw',   'Board Independence'),
]

# Build full combinatorial pairs: 20 CEO × 6 Firm = 120 surfaces
INTERACTION_PAIRS = []
for ceo_feat in CEO_FEATURES:
    for firm_feat, firm_label in FIRM_DIMENSIONS:
        ceo_label = LABELS.get(ceo_feat, ceo_feat).replace('CEO ', '')
        title = f'{ceo_label} × {firm_label}'
        INTERACTION_PAIRS.append((ceo_feat, firm_feat, title))


class EnrichedConfig(Config):
    """Config with enriched CEO features."""
    DATA_PATH = DATA_PATH
    CEO_NUMERIC_COLS = ENRICHED_CEO_NUMERIC
    LATENT_DIM = 80
    EMBEDDING_DIM_MEDIUM = 12
    EPOCHS = 60


def parse_args():
    p = argparse.ArgumentParser(description='CEO × Firm Interaction Surfaces')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--grid',   type=int, default=30, help='Grid resolution per axis')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────

def load_and_train(config):
    """Load enriched data, fill NaNs, train the Two-Tower model."""
    print(f"\nLoading {config.DATA_PATH}...")
    df = pd.read_csv(config.DATA_PATH, on_bad_lines='skip')
    print(f"  {len(df):,} rows × {len(df.columns)} cols")

    # Fill enriched NaNs with median
    for col in ENRICHED_CEO_NUMERIC:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    processor = DataProcessor(config)
    df_proc = processor.prepare_features(df)
    processor.fit(df_proc)
    data_dict = processor.transform(df_proc)

    metadata = {k: data_dict[k] for k in
                ['n_firm_numeric', 'firm_cat_counts', 'n_ceo_numeric', 'ceo_cat_counts']}

    dataset = CEOFirmDataset(data_dict)
    n_train = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train],
                                     generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE)

    model = train_model(train_loader, val_loader, metadata, config)

    # Compute validation correlation
    model.eval()
    with torch.no_grad():
        preds = model(
            data_dict['firm_numeric'].to(config.DEVICE),
            data_dict['firm_cat'].to(config.DEVICE),
            data_dict['ceo_numeric'].to(config.DEVICE),
            data_dict['ceo_cat'].to(config.DEVICE),
        ).cpu().numpy().flatten()
    targets = data_dict['target'].numpy().flatten()
    corr = np.corrcoef(preds, targets)[0, 1]
    print(f"\n  Validation correlation: {corr:.3f}")

    return model, processor, data_dict, df_proc


# ─────────────────────────────────────────────────────────────────────
# INTERACTION SURFACE COMPUTATION
# ─────────────────────────────────────────────────────────────────────

def compute_interaction_surface(model, processor, data_dict, config,
                                 ceo_feature, firm_feature, grid_n=30):
    """
    Sweep a 2D grid over (ceo_feature, firm_feature) and predict match quality
    at each point, holding all other features at their sample mean (numeric)
    or mode (categorical).

    Returns:
        ceo_vals:  1D array of CEO feature values (in original scale)
        firm_vals: 1D array of Firm feature values (in original scale)
        surface:   2D array of predicted match quality, shape (len(ceo_vals), len(firm_vals))
    """
    device = config.DEVICE

    # ── Locate features in the tensor layout ──
    ceo_numeric_cols = processor.final_ceo_numeric   # e.g. ['Age', 'tenure', 'n_prior_roles', ...]
    firm_numeric_cols = processor.final_firm_numeric  # e.g. ['ind_firms_60w', ...]

    if ceo_feature not in ceo_numeric_cols:
        raise ValueError(f"CEO feature '{ceo_feature}' not in {ceo_numeric_cols}")
    if firm_feature not in firm_numeric_cols:
        raise ValueError(f"Firm feature '{firm_feature}' not in {firm_numeric_cols}")

    ceo_idx  = ceo_numeric_cols.index(ceo_feature)
    firm_idx = firm_numeric_cols.index(firm_feature)

    # ── Build baseline inputs (mean numeric, mode categorical) ──
    firm_numeric_base = data_dict['firm_numeric'].mean(dim=0, keepdim=True).to(device)
    ceo_numeric_base  = data_dict['ceo_numeric'].mean(dim=0, keepdim=True).to(device)

    # Mode on CPU for MPS compatibility
    firm_cat_base = torch.mode(data_dict['firm_cat'], dim=0)[0].unsqueeze(0).to(device)
    ceo_cat_base  = torch.mode(data_dict['ceo_cat'],  dim=0)[0].unsqueeze(0).to(device)

    # ── Grid in *scaled* space (what the model sees) ──
    ceo_col_data  = data_dict['ceo_numeric'][:, ceo_idx]
    firm_col_data = data_dict['firm_numeric'][:, firm_idx]

    # Use 5th-95th percentile range for cleaner surfaces
    ceo_lo,  ceo_hi  = ceo_col_data.quantile(0.05).item(),  ceo_col_data.quantile(0.95).item()
    firm_lo, firm_hi = firm_col_data.quantile(0.05).item(), firm_col_data.quantile(0.95).item()

    ceo_grid_scaled  = np.linspace(ceo_lo,  ceo_hi,  grid_n)
    firm_grid_scaled = np.linspace(firm_lo, firm_hi, grid_n)

    # ── Invert scaling to get original-scale tick labels ──
    ceo_scaler  = processor.scalers['ceo']
    firm_scaler = processor.scalers['firm']

    # Build dummy arrays for inverse transform
    ceo_dummy = np.zeros((grid_n, len(ceo_numeric_cols)))
    ceo_dummy[:, ceo_idx] = ceo_grid_scaled
    ceo_vals_orig = ceo_scaler.inverse_transform(ceo_dummy)[:, ceo_idx]

    firm_dummy = np.zeros((grid_n, len(firm_numeric_cols)))
    firm_dummy[:, firm_idx] = firm_grid_scaled
    firm_vals_orig = firm_scaler.inverse_transform(firm_dummy)[:, firm_idx]

    # ── Sweep the grid ──
    surface = np.zeros((grid_n, grid_n))  # (ceo, firm)

    model.eval()
    with torch.no_grad():
        for i, cv in enumerate(ceo_grid_scaled):
            # Batch: repeat baseline for all firm values at once
            f_num = firm_numeric_base.expand(grid_n, -1).clone()
            c_num = ceo_numeric_base.expand(grid_n, -1).clone()
            f_cat = firm_cat_base.expand(grid_n, -1).clone()
            c_cat = ceo_cat_base.expand(grid_n, -1).clone()

            # Set the swept features
            c_num[:, ceo_idx]  = cv
            f_num[:, firm_idx] = torch.tensor(firm_grid_scaled, dtype=torch.float32, device=device)

            scores = model(f_num, f_cat, c_num, c_cat).cpu().numpy().flatten()
            surface[i, :] = scores

    return ceo_vals_orig, firm_vals_orig, surface


# ─────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────

def plot_single_surface(ceo_vals, firm_vals, surface, ceo_name, firm_name,
                         title, output_path):
    """Plot one interaction surface as a heatmap with contours."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Diverging colormap centered on the median of the surface
    vmin, vmax = surface.min(), surface.max()
    vmid = np.median(surface)

    # Use TwoSlopeNorm if range spans both sides of the median
    if vmin < vmid < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax)
    else:
        norm = None

    im = ax.imshow(surface, aspect='auto', cmap='RdBu_r', origin='lower',
                   extent=[firm_vals[0], firm_vals[-1], ceo_vals[0], ceo_vals[-1]],
                   norm=norm, interpolation='bicubic')

    # Add contour lines
    X, Y = np.meshgrid(firm_vals, ceo_vals)
    cs = ax.contour(X, Y, surface, levels=8, colors='black', linewidths=0.6, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')

    ax.set_xlabel(LABELS.get(firm_name, firm_name), fontsize=12)
    ax.set_ylabel(LABELS.get(ceo_name, ceo_name), fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Predicted Match Quality', fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def plot_panel(all_surfaces, output_path):
    """Create a multi-panel (4×3) figure of all interaction surfaces."""
    n = len(all_surfaces)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    for i, (ceo_vals, firm_vals, surface, ceo_name, firm_name, title) in enumerate(all_surfaces):
        ax = axes[i]

        vmin, vmax = surface.min(), surface.max()
        vmid = np.median(surface)
        if vmin < vmid < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax)
        else:
            norm = None

        im = ax.imshow(surface, aspect='auto', cmap='RdBu_r', origin='lower',
                       extent=[firm_vals[0], firm_vals[-1],
                               ceo_vals[0], ceo_vals[-1]],
                       norm=norm, interpolation='bicubic')

        # Contours
        X, Y = np.meshgrid(firm_vals, ceo_vals)
        cs = ax.contour(X, Y, surface, levels=6, colors='black',
                        linewidths=0.4, alpha=0.4)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')

        ax.set_xlabel(LABELS.get(firm_name, firm_name), fontsize=9)
        ax.set_ylabel(LABELS.get(ceo_name, ceo_name), fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=7)

        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    # Hide empty axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Predicted Match Quality: CEO × Firm Observable Interactions\n'
                 '(Two-Tower model, other features held at sample mean/mode)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def plot_grouped_panel(surfaces_group, firm_label, output_path):
    """Create a panel figure with all CEO features for one firm dimension (5x4)."""
    n = len(surfaces_group)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i, (ceo_vals, firm_vals, surface, ceo_name, firm_name, title) in enumerate(surfaces_group):
        ax = axes[i]
        vmin, vmax = surface.min(), surface.max()
        vmid = np.median(surface)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax) if vmin < vmid < vmax else None

        im = ax.imshow(surface, aspect='auto', cmap='RdBu_r', origin='lower',
                       extent=[firm_vals[0], firm_vals[-1], ceo_vals[0], ceo_vals[-1]],
                       norm=norm, interpolation='bicubic')
        X, Y = np.meshgrid(firm_vals, ceo_vals)
        cs = ax.contour(X, Y, surface, levels=6, colors='black', linewidths=0.35, alpha=0.35)
        ax.clabel(cs, inline=True, fontsize=5.5, fmt='%.2f')
        ax.set_ylabel(LABELS.get(ceo_name, ceo_name), fontsize=8)
        ax.set_xlabel(LABELS.get(firm_name, firm_name), fontsize=7)
        ax.set_title(LABELS.get(ceo_name, ceo_name), fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Predicted Match Quality vs {firm_label}\n'
                 f'(All CEO dimensions, other features at sample mean/mode)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def compute_interaction_stats(surface, grid_n=30, k=5):
    """Compute interpretable statistics from a surface."""
    HH = np.mean(surface[-k:, -k:])
    HL = np.mean(surface[-k:, :k])
    LH = np.mean(surface[:k, -k:])
    LL = np.mean(surface[:k, :k])
    interaction = (HH - HL) - (LH - LL)
    ceo_gradient = np.mean(surface[-k:, :]) - np.mean(surface[:k, :])
    firm_gradient = np.mean(surface[:, -k:]) - np.mean(surface[:, :k])
    if interaction > 0.01:
        pattern = 'Complementary'
    elif interaction < -0.01:
        pattern = 'Substitutes'
    elif abs(ceo_gradient) > 0.02 or abs(firm_gradient) > 0.02:
        pattern = 'Additive'
    else:
        pattern = 'Flat'
    return {
        'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH,
        'interaction': interaction, 'ceo_gradient': ceo_gradient,
        'firm_gradient': firm_gradient, 'spread': surface.max() - surface.min(),
        'pattern': pattern,
    }


# ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start = time.time()

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    print("=" * 70)
    print("  CEO x FIRM INTERACTION SURFACES (FULL SWEEP)")
    print(f"  {len(CEO_FEATURES)} CEO features x {len(FIRM_DIMENSIONS)} firm dimensions")
    print(f"  = {len(INTERACTION_PAIRS)} total surfaces")
    print("=" * 70)

    config = EnrichedConfig()
    config.EPOCHS = args.epochs
    model, processor, data_dict, df_proc = load_and_train(config)

    print(f"\n{'='*70}")
    print(f"  Computing {len(INTERACTION_PAIRS)} surfaces (grid={args.grid})")
    print(f"{'='*70}")

    all_surfaces = []
    grouped = {}
    stats_rows = []

    for ceo_feat, firm_feat, title in INTERACTION_PAIRS:
        try:
            ceo_vals, firm_vals, surface = compute_interaction_surface(
                model, processor, data_dict, config,
                ceo_feat, firm_feat, grid_n=args.grid)
            entry = (ceo_vals, firm_vals, surface, ceo_feat, firm_feat, title)
            all_surfaces.append(entry)
            grouped.setdefault(firm_feat, []).append(entry)

            path = plot_single_surface(
                ceo_vals, firm_vals, surface, ceo_feat, firm_feat, title,
                f'{OUTPUT_DIR}/{ceo_feat}_x_{firm_feat}.pdf')

            s = compute_interaction_stats(surface, grid_n=args.grid)
            s['ceo_feature'] = ceo_feat
            s['firm_feature'] = firm_feat
            s['title'] = title
            s['min'] = surface.min()
            s['max'] = surface.max()
            stats_rows.append(s)

            marker = {'Complementary': '+', 'Substitutes': '-',
                      'Additive': '~', 'Flat': '.'}.get(s['pattern'], '?')
            print(f"  [{marker}] {title:40s}  {s['pattern']:14s}  "
                  f"D={s['interaction']:+.4f}  spread={s['spread']:.3f}")

        except Exception as e:
            print(f"  [!] {title:40s}  ERROR: {e}")

    # Grouped panels (one per firm dimension)
    print(f"\n{'='*70}")
    print("  Generating grouped panel figures")
    print(f"{'='*70}")
    for firm_feat, firm_label in FIRM_DIMENSIONS:
        if firm_feat in grouped:
            path = plot_grouped_panel(
                grouped[firm_feat], firm_label,
                f'{OUTPUT_DIR}/panel_{firm_feat}.pdf')
            print(f"  -> {firm_label:25s}  panel_{firm_feat}.pdf  ({len(grouped[firm_feat])} panels)")

    # Summary CSV
    stats_df = pd.DataFrame(stats_rows).sort_values('interaction', ascending=False)
    csv_path = f'{OUTPUT_DIR}/interaction_statistics.csv'
    stats_df.to_csv(csv_path, index=False)
    print(f"\n  -> Summary statistics: {csv_path}")

    # Print top/bottom
    print(f"\n{'='*70}")
    print("  TOP 10 COMPLEMENTARY PAIRS (supermodular)")
    print(f"{'='*70}")
    for _, r in stats_df.head(10).iterrows():
        print(f"  {r['title']:40s}  D={r['interaction']:+.4f}  [{r['min']:.3f}, {r['max']:.3f}]")

    print(f"\n{'='*70}")
    print("  TOP 10 SUBSTITUTE PAIRS (submodular)")
    print(f"{'='*70}")
    for _, r in stats_df.tail(10).iterrows():
        print(f"  {r['title']:40s}  D={r['interaction']:+.4f}  [{r['min']:.3f}, {r['max']:.3f}]")

    # Top-12 panel
    top_surfaces = []
    top_12 = pd.concat([stats_df.head(6), stats_df.tail(6)])
    for _, r in top_12.iterrows():
        for entry in all_surfaces:
            if entry[3] == r['ceo_feature'] and entry[4] == r['firm_feature']:
                top_surfaces.append(entry)
                break
    if top_surfaces:
        panel_path = plot_panel(top_surfaces, f'{OUTPUT_DIR}/top_interactions_panel.pdf')
        print(f"\n  -> Top-12 interactions panel: {panel_path}")

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"  Done in {elapsed:.1f}s -- {len(all_surfaces)} surfaces + "
          f"{len(FIRM_DIMENSIONS)} grouped panels saved to {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

