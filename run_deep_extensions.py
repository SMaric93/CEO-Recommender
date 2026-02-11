#!/usr/bin/env python3
"""
Deep Learning Extensions for CEO-Firm Matching

Three extensions beyond the base Two-Tower model:
  1. Cross-Attention Model  — CEO features attend to Firm features;
     the attention matrix is a learned complementarity map.
  2. Contrastive Learning   — InfoNCE loss for embedding-based retrieval;
     "who is the best CEO for this firm?"
  3. Integrated Gradients   — per-observation feature attributions;
     "why is THIS CEO a good match for THIS firm?"

Usage:
    python run_deep_extensions.py                   # all three + 50 epochs
    python run_deep_extensions.py --epochs 80       # more training
    python run_deep_extensions.py --skip-attention   # skip cross-attention
"""

import argparse, os, sys, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

# ═════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════

OUTPUT_DIR = 'Output/Deep_Extensions'
DATA_PATH  = 'Data/ceo_types_enriched.csv'

ENRICHED_CEO_NUMERIC = [
    'Age',
    'n_prior_roles', 'n_prior_boards', 'avg_board_tenure',
    'years_as_ceo', 'n_sectors', 'career_span_years',
    'network_size', 'board_interlocks',
    'log_tdc1', 'salary', 'stock_awards_fv', 'option_awards_fv',
    'ciq_n_affiliations', 'ciq_n_roles', 'ciq_current_boards',
    'ciq_career_span', 'ciq_best_rank', 'ciq_equity_share',
]

LABELS = {
    'Age': 'CEO Age', 'tenure': 'Tenure',
    'n_prior_roles': 'Prior Roles', 'n_prior_boards': 'Prior Boards',
    'avg_board_tenure': 'Avg Board Tenure', 'years_as_ceo': 'Years as CEO',
    'n_sectors': 'Industry Breadth', 'career_span_years': 'Career Span',
    'network_size': 'Network Size', 'board_interlocks': 'Board Interlocks',
    'log_tdc1': 'Log Comp', 'salary': 'Salary',
    'stock_awards_fv': 'Stock Awards', 'option_awards_fv': 'Option Awards',
    'ciq_n_affiliations': 'CIQ Affiliations', 'ciq_n_roles': 'CIQ Roles',
    'ciq_current_boards': 'Current Boards', 'ciq_career_span': 'CIQ Career',
    'ciq_best_rank': 'CIQ Rank', 'ciq_equity_share': 'Equity Share',
    'logatw': 'Firm Size', 'exp_roa': 'Exp ROA',
    'rdintw': 'R&D Intensity', 'capintw': 'Capital Intensity',
    'leverage': 'Leverage', 'divyieldw': 'Div Yield',
    'boardindpw': 'Board Indep', 'boardsizew': 'Board Size',
    'busyw': 'Busy Directors', 'pct_blockw': 'Block Own',
    'ind_firms_60w': 'Local Competition', 'non_competition_score': 'Non-Compete',
}


class EnrichedConfig(Config):
    DATA_PATH = DATA_PATH
    CEO_NUMERIC_COLS = ENRICHED_CEO_NUMERIC
    LATENT_DIM = 80
    EMBEDDING_DIM_MEDIUM = 12
    EPOCHS = 60


def parse_args():
    p = argparse.ArgumentParser(description='Deep Learning Extensions')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--skip-attention', action='store_true')
    p.add_argument('--skip-contrastive', action='store_true')
    p.add_argument('--skip-ig', action='store_true')
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════
#  DATA LOADING (shared across all extensions)
# ═════════════════════════════════════════════════════════════════════

def load_data(config):
    """Load enriched data and prepare for training."""
    print(f"\nLoading {config.DATA_PATH}...")
    df = pd.read_csv(config.DATA_PATH, on_bad_lines='skip')
    print(f"  {len(df):,} rows x {len(df.columns)} cols")

    for col in ENRICHED_CEO_NUMERIC:
        if col in df.columns and df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)

    processor = DataProcessor(config)
    df_proc = processor.prepare_features(df)
    processor.fit(df_proc)
    data_dict = processor.transform(df_proc)

    metadata = {k: data_dict[k] for k in
                ['n_firm_numeric', 'firm_cat_counts', 'n_ceo_numeric', 'ceo_cat_counts']}

    dataset = CEOFirmDataset(data_dict)
    n_train = int(0.8 * len(dataset))
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE)

    return processor, data_dict, metadata, df_proc, train_loader, val_loader


# ═════════════════════════════════════════════════════════════════════
#  EXTENSION 1: CROSS-ATTENTION MODEL
# ═════════════════════════════════════════════════════════════════════

class CrossAttentionMatcher(nn.Module):
    """
    Cross-Attention CEO-Firm Matcher.

    Treats each numeric feature as a token, projects to d-dim,
    then CEO tokens attend to Firm tokens via multi-head cross-attention.
    The attention weights form a learned complementarity map.
    """
    def __init__(self, metadata, config, d_model=32, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        n_ceo = metadata['n_ceo_numeric']
        n_firm = metadata['n_firm_numeric']

        # Per-feature projection to d_model
        self.ceo_proj = nn.Linear(1, d_model)
        self.firm_proj = nn.Linear(1, d_model)

        # Learnable positional embeddings (one per feature)
        self.ceo_pos = nn.Parameter(torch.randn(1, n_ceo, d_model) * 0.02)
        self.firm_pos = nn.Parameter(torch.randn(1, n_firm, d_model) * 0.02)

        # Categorical embedding layers
        self.firm_embeddings = nn.ModuleList([
            nn.Embedding(n, d_model) for n in metadata['firm_cat_counts']
        ])
        self.ceo_embeddings = nn.ModuleList([
            nn.Embedding(n, d_model) for n in metadata['ceo_cat_counts']
        ])

        # Cross-attention: CEO queries attend to Firm keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=0.1, batch_first=True
        )

        # Self-attention for processing attended representation
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=0.1, batch_first=True
        )

        # LayerNorms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Prediction head
        total_tokens = n_ceo + len(metadata['ceo_cat_counts'])
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        # For storing attention weights
        self.last_attn_weights = None

    def forward(self, f_numeric, f_cat, c_numeric, c_cat, return_attn=False):
        B = f_numeric.size(0)

        # Project each numeric feature: [B, n_features] → [B, n_features, d_model]
        firm_tokens = self.firm_proj(f_numeric.unsqueeze(-1)) + self.firm_pos
        ceo_tokens  = self.ceo_proj(c_numeric.unsqueeze(-1))  + self.ceo_pos

        # Add categorical tokens
        for i, emb in enumerate(self.firm_embeddings):
            cat_tok = emb(f_cat[:, i]).unsqueeze(1)  # [B, 1, d]
            firm_tokens = torch.cat([firm_tokens, cat_tok], dim=1)

        for i, emb in enumerate(self.ceo_embeddings):
            cat_tok = emb(c_cat[:, i]).unsqueeze(1)
            ceo_tokens = torch.cat([ceo_tokens, cat_tok], dim=1)

        # Cross-attention: CEO attends to Firm
        attended, attn_weights = self.cross_attn(
            query=ceo_tokens, key=firm_tokens, value=firm_tokens,
            need_weights=True
        )
        # attn_weights: [B, n_ceo_tokens, n_firm_tokens] (averaged over heads)

        self.last_attn_weights = attn_weights.detach()

        attended = self.ln1(ceo_tokens + attended)

        # Self-attention refinement
        refined, _ = self.self_attn(attended, attended, attended)
        refined = self.ln2(attended + refined)

        # Pool: mean over CEO tokens
        pooled = refined.mean(dim=1)  # [B, d_model]

        score = self.head(pooled)

        if return_attn:
            return score, attn_weights
        return score


def train_cross_attention(train_loader, val_loader, metadata, config):
    """Train the cross-attention model."""
    model = CrossAttentionMatcher(metadata, config).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    print(f"\n  Training CrossAttentionMatcher on {config.DEVICE} for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            preds = model(batch['firm_numeric'], batch['firm_cat'],
                          batch['ceo_numeric'], batch['ceo_cat'])
            loss = (batch['weights'] * (preds - batch['target'])**2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")

    return model


def visualize_attention(model, data_dict, processor, config, output_dir):
    """Extract and visualize the average cross-attention map."""
    model.eval()
    device = config.DEVICE

    # Forward pass on full data to get attention weights
    with torch.no_grad():
        preds, attn = model(
            data_dict['firm_numeric'].to(device),
            data_dict['firm_cat'].to(device),
            data_dict['ceo_numeric'].to(device),
            data_dict['ceo_cat'].to(device),
            return_attn=True
        )

    # Average attention across all observations
    # attn shape: [N, n_ceo_tokens, n_firm_tokens]
    avg_attn = attn.mean(dim=0).cpu().numpy()  # [n_ceo_tokens, n_firm_tokens]

    # Feature names
    ceo_features = processor.final_ceo_numeric + config.CEO_CAT_COLS
    firm_features = processor.final_firm_numeric + config.FIRM_CAT_COLS
    n_ceo_num = len(processor.final_ceo_numeric)
    n_firm_num = len(processor.final_firm_numeric)

    # Crop to just numeric × numeric for cleaner visualization
    attn_numeric = avg_attn[:n_ceo_num, :n_firm_num]
    ceo_labels = [LABELS.get(f, f) for f in processor.final_ceo_numeric]
    firm_labels = [LABELS.get(f, f) for f in processor.final_firm_numeric]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(attn_numeric, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    ax.set_xticks(range(len(firm_labels)))
    ax.set_xticklabels(firm_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(ceo_labels)))
    ax.set_yticklabels(ceo_labels, fontsize=9)
    ax.set_xlabel('Firm Features (Keys)', fontsize=12)
    ax.set_ylabel('CEO Features (Queries)', fontsize=12)
    ax.set_title('Cross-Attention Weights: Which Firm Features\n'
                 'Does Each CEO Feature Attend To?',
                 fontsize=13, fontweight='bold')

    # Annotate cells
    for i in range(attn_numeric.shape[0]):
        for j in range(attn_numeric.shape[1]):
            val = attn_numeric[i, j]
            color = 'white' if val > attn_numeric.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=6, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight', fontsize=10)
    plt.tight_layout()
    path = f'{output_dir}/cross_attention_map.pdf'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")

    # Validation correlation
    targets = data_dict['target'].numpy().flatten()
    pred_np = preds.cpu().numpy().flatten()
    corr = np.corrcoef(pred_np, targets)[0, 1]
    print(f"  Cross-attention validation correlation: {corr:.3f}")

    return corr, avg_attn


# ═════════════════════════════════════════════════════════════════════
#  EXTENSION 2: CONTRASTIVE LEARNING
# ═════════════════════════════════════════════════════════════════════

class ContrastiveTwoTower(nn.Module):
    """
    Two-Tower model trained with combined InfoNCE + MSE loss.
    Produces L2-normalized embeddings for retrieval.
    """
    def __init__(self, metadata, config):
        super().__init__()

        # Firm embeddings
        self.firm_embeddings = nn.ModuleList([
            nn.Embedding(n, config.EMBEDDING_DIM_LARGE)
            for n in metadata['firm_cat_counts']
        ])
        # CEO embeddings
        self.ceo_embeddings = nn.ModuleList([
            nn.Embedding(n, config.EMBEDDING_DIM_MEDIUM)
            for n in metadata['ceo_cat_counts']
        ])

        # Firm tower
        firm_in = metadata['n_firm_numeric'] + len(metadata['firm_cat_counts']) * config.EMBEDDING_DIM_LARGE
        self.firm_tower = nn.Sequential(
            nn.Linear(firm_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, config.LATENT_DIM),
        )

        # CEO tower
        ceo_in = metadata['n_ceo_numeric'] + len(metadata['ceo_cat_counts']) * config.EMBEDDING_DIM_MEDIUM
        self.ceo_tower = nn.Sequential(
            nn.Linear(ceo_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, config.LATENT_DIM),
        )

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_firm(self, f_numeric, f_cat):
        f_embs = [emb(f_cat[:, i]) for i, emb in enumerate(self.firm_embeddings)]
        f_combined = torch.cat([f_numeric] + f_embs, dim=1)
        z = self.firm_tower(f_combined)
        return F.normalize(z, dim=1)

    def encode_ceo(self, c_numeric, c_cat):
        c_embs = [emb(c_cat[:, i]) for i, emb in enumerate(self.ceo_embeddings)]
        c_combined = torch.cat([c_numeric] + c_embs, dim=1)
        z = self.ceo_tower(c_combined)
        return F.normalize(z, dim=1)

    def forward(self, f_numeric, f_cat, c_numeric, c_cat):
        z_firm = self.encode_firm(f_numeric, f_cat)
        z_ceo  = self.encode_ceo(c_numeric, c_cat)
        logit_scale = self.logit_scale.exp()
        score = (z_firm * z_ceo).sum(dim=1, keepdim=True) * logit_scale
        return score, z_firm, z_ceo


def info_nce_loss(z_firm, z_ceo, temperature=0.07):
    """
    InfoNCE contrastive loss.
    Treats diagonal as positive pairs, off-diagonal as negatives.
    """
    B = z_firm.size(0)
    # Similarity matrix: [B, B]
    logits = z_firm @ z_ceo.T / temperature
    labels = torch.arange(B, device=z_firm.device)
    # Symmetric loss (CEO→Firm and Firm→CEO)
    loss_cf = F.cross_entropy(logits, labels)
    loss_fc = F.cross_entropy(logits.T, labels)
    return (loss_cf + loss_fc) / 2


def train_contrastive(train_loader, val_loader, metadata, config, alpha=0.3):
    """Train with combined InfoNCE + MSE loss."""
    model = ContrastiveTwoTower(metadata, config).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    print(f"\n  Training ContrastiveTwoTower (alpha={alpha}) "
          f"on {config.DEVICE} for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()

            score, z_firm, z_ceo = model(
                batch['firm_numeric'], batch['firm_cat'],
                batch['ceo_numeric'], batch['ceo_cat'])

            # Combined loss
            mse = (batch['weights'] * (score - batch['target'])**2).mean()
            nce = info_nce_loss(z_firm, z_ceo, temperature=0.07)
            loss = alpha * nce + (1 - alpha) * mse

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if epoch % 10 == 0:
            avg = total_loss / len(train_loader)
            print(f"    Epoch {epoch}: Loss = {avg:.4f}")

    return model


def retrieval_analysis(model, data_dict, processor, config, df_proc, output_dir, top_k=10):
    """
    For each firm, find the top-K best-matching CEOs by embedding similarity.
    Compare to actual match_means.
    """
    model.eval()
    device = config.DEVICE

    with torch.no_grad():
        z_firm = model.encode_firm(
            data_dict['firm_numeric'].to(device),
            data_dict['firm_cat'].to(device)).cpu()
        z_ceo = model.encode_ceo(
            data_dict['ceo_numeric'].to(device),
            data_dict['ceo_cat'].to(device)).cpu()

    # Full similarity matrix [N_firms, N_ceos]
    sim_matrix = z_firm @ z_ceo.T  # [N, N]

    targets = data_dict['target'].numpy().flatten()
    preds_diag = sim_matrix.diag().numpy()

    # Correlation between model similarity and actual match quality
    corr = np.corrcoef(preds_diag, targets)[0, 1]
    print(f"  Contrastive model correlation: {corr:.3f}")

    N = len(targets)

    # For each firm, rank all CEOs by similarity
    ranks = []
    for i in range(N):
        sims = sim_matrix[i].numpy()
        rank = (sims >= sims[i]).sum()  # rank of the actual CEO
        ranks.append(rank)
    ranks = np.array(ranks)

    hit_at_1 = (ranks == 1).mean()
    hit_at_5 = (ranks <= 5).mean()
    hit_at_10 = (ranks <= 10).mean()
    mrr = (1.0 / ranks).mean()

    print(f"  Retrieval metrics:")
    print(f"    Hit@1:  {hit_at_1:.3f}")
    print(f"    Hit@5:  {hit_at_5:.3f}")
    print(f"    Hit@10: {hit_at_10:.3f}")
    print(f"    MRR:    {mrr:.3f}")

    # Find best counterfactual matches (biggest improvement over actual)
    improvements = []
    for i in range(N):
        actual_sim = sim_matrix[i, i].item()
        best_ceo_idx = sim_matrix[i].argmax().item()
        best_sim = sim_matrix[i, best_ceo_idx].item()
        improvement = best_sim - actual_sim
        if best_ceo_idx != i:
            improvements.append({
                'firm_idx': i,
                'actual_ceo_idx': i,
                'best_ceo_idx': best_ceo_idx,
                'actual_sim': actual_sim,
                'best_sim': best_sim,
                'improvement': improvement,
                'actual_match_means': targets[i],
            })

    imp_df = pd.DataFrame(improvements).sort_values('improvement', ascending=False)

    # Visualization: rank distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Rank distribution
    ax = axes[0]
    ax.hist(np.minimum(ranks, 50), bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(1, color='red', linewidth=2, label=f'Rank 1 ({hit_at_1:.1%})')
    ax.set_xlabel('Rank of Actual CEO', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('CEO Retrieval Rank Distribution', fontsize=12, fontweight='bold')
    ax.legend()

    # 2. Similarity vs match_means
    ax = axes[1]
    ax.scatter(preds_diag, targets, alpha=0.2, s=8, color='navy')
    ax.set_xlabel('Contrastive Similarity', fontsize=11)
    ax.set_ylabel('Actual Match Quality', fontsize=11)
    ax.set_title(f'Similarity vs Match Quality (r={corr:.3f})', fontsize=12, fontweight='bold')

    # 3. Improvement distribution
    ax = axes[2]
    ax.hist(imp_df['improvement'].values[:200], bins=40, color='coral', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Potential Improvement (best - actual similarity)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Counterfactual CEO Reassignment Gains', fontsize=12, fontweight='bold')

    plt.tight_layout()
    path = f'{output_dir}/contrastive_retrieval.pdf'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")

    return corr, {'hit@1': hit_at_1, 'hit@5': hit_at_5, 'hit@10': hit_at_10, 'mrr': mrr}


# ═════════════════════════════════════════════════════════════════════
#  EXTENSION 3: INTEGRATED GRADIENTS
# ═════════════════════════════════════════════════════════════════════

def integrated_gradients(model, inputs, baselines, n_steps=50):
    """
    Compute Integrated Gradients for each input feature.

    Args:
        model: trained model (forward takes f_num, f_cat, c_num, c_cat)
        inputs: dict with 'firm_numeric', 'firm_cat', 'ceo_numeric', 'ceo_cat'
                each of shape [1, n_features]
        baselines: dict with same structure (typically zeros or means)
        n_steps: number of interpolation steps

    Returns:
        attributions: dict with 'firm_numeric', 'ceo_numeric' arrays
    """
    model.eval()

    # Interpolate along the path
    alphas = torch.linspace(0, 1, n_steps + 1, device=inputs['firm_numeric'].device)

    # Accumulate gradients
    firm_grads = []
    ceo_grads = []

    for alpha in alphas:
        # Interpolated inputs
        f_num = baselines['firm_numeric'] + alpha * (inputs['firm_numeric'] - baselines['firm_numeric'])
        c_num = baselines['ceo_numeric'] + alpha * (inputs['ceo_numeric'] - baselines['ceo_numeric'])

        f_num = f_num.clone().requires_grad_(True)
        c_num = c_num.clone().requires_grad_(True)

        # Forward (handle both base model returning tensor and contrastive returning tuple)
        out = model(f_num, inputs['firm_cat'], c_num, inputs['ceo_cat'])
        score = out[0] if isinstance(out, tuple) else out

        # Backward
        model.zero_grad()
        score.backward()

        firm_grads.append(f_num.grad.detach().clone())
        ceo_grads.append(c_num.grad.detach().clone())

    # Average gradients and multiply by (input - baseline)
    firm_grads = torch.stack(firm_grads).mean(dim=0)
    ceo_grads = torch.stack(ceo_grads).mean(dim=0)

    firm_ig = firm_grads * (inputs['firm_numeric'].detach() - baselines['firm_numeric'])
    ceo_ig  = ceo_grads  * (inputs['ceo_numeric'].detach()  - baselines['ceo_numeric'])

    return {
        'firm_numeric': firm_ig.squeeze(0).cpu().numpy(),
        'ceo_numeric':  ceo_ig.squeeze(0).cpu().numpy()
    }


def run_integrated_gradients(model, data_dict, processor, config, df_proc, output_dir,
                              n_cases=10, n_steps=100):
    """
    Compute Integrated Gradients for top/bottom matches and aggregate.
    """
    device = config.DEVICE

    # Get predictions for all observations
    model.eval()
    with torch.no_grad():
        all_preds = model(
            data_dict['firm_numeric'].to(device),
            data_dict['firm_cat'].to(device),
            data_dict['ceo_numeric'].to(device),
            data_dict['ceo_cat'].to(device),
        ).cpu().numpy().flatten()

    targets = data_dict['target'].numpy().flatten()

    # Baseline: sample mean (move mode computation to CPU for MPS compat)
    baselines = {
        'firm_numeric': data_dict['firm_numeric'].mean(dim=0, keepdim=True).to(device),
        'firm_cat': torch.mode(data_dict['firm_cat'].cpu(), dim=0)[0].unsqueeze(0).to(device),
        'ceo_numeric': data_dict['ceo_numeric'].mean(dim=0, keepdim=True).to(device),
        'ceo_cat': torch.mode(data_dict['ceo_cat'].cpu(), dim=0)[0].unsqueeze(0).to(device),
    }

    # Sort by match quality to find best and worst
    sorted_idx = np.argsort(targets)
    best_idx = sorted_idx[-n_cases:][::-1]
    worst_idx = sorted_idx[:n_cases]

    ceo_features = processor.final_ceo_numeric
    firm_features = processor.final_firm_numeric
    all_features = firm_features + ceo_features
    all_labels = [LABELS.get(f, f) for f in all_features]

    # Compute IG for best and worst cases
    print(f"\n  Computing Integrated Gradients ({n_steps} steps, {n_cases} best + {n_cases} worst)...")
    best_igs = []
    worst_igs = []

    for idx in best_idx:
        inputs = {
            'firm_numeric': data_dict['firm_numeric'][idx:idx+1].to(device),
            'firm_cat': data_dict['firm_cat'][idx:idx+1].to(device),
            'ceo_numeric': data_dict['ceo_numeric'][idx:idx+1].to(device),
            'ceo_cat': data_dict['ceo_cat'][idx:idx+1].to(device),
        }
        ig = integrated_gradients(model, inputs, baselines, n_steps=n_steps)
        best_igs.append(np.concatenate([ig['firm_numeric'], ig['ceo_numeric']]))

    for idx in worst_idx:
        inputs = {
            'firm_numeric': data_dict['firm_numeric'][idx:idx+1].to(device),
            'firm_cat': data_dict['firm_cat'][idx:idx+1].to(device),
            'ceo_numeric': data_dict['ceo_numeric'][idx:idx+1].to(device),
            'ceo_cat': data_dict['ceo_cat'][idx:idx+1].to(device),
        }
        ig = integrated_gradients(model, inputs, baselines, n_steps=n_steps)
        worst_igs.append(np.concatenate([ig['firm_numeric'], ig['ceo_numeric']]))

    best_igs = np.array(best_igs)   # [n_cases, n_features]
    worst_igs = np.array(worst_igs)

    # ── Also compute aggregate IG over a larger random sample ──
    np.random.seed(42)
    sample_idx = np.random.choice(len(targets), size=min(200, len(targets)), replace=False)
    all_igs = []
    for idx in sample_idx:
        inputs = {
            'firm_numeric': data_dict['firm_numeric'][idx:idx+1].to(device),
            'firm_cat': data_dict['firm_cat'][idx:idx+1].to(device),
            'ceo_numeric': data_dict['ceo_numeric'][idx:idx+1].to(device),
            'ceo_cat': data_dict['ceo_cat'][idx:idx+1].to(device),
        }
        ig = integrated_gradients(model, inputs, baselines, n_steps=n_steps)
        all_igs.append(np.concatenate([ig['firm_numeric'], ig['ceo_numeric']]))

    all_igs = np.array(all_igs)

    # ── VIZ 1: Average IG for best vs worst matches ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Best matches
    ax = axes[0]
    avg_best = best_igs.mean(axis=0)
    order = np.argsort(np.abs(avg_best))[::-1]
    top_n = 20
    idx_top = order[:top_n][::-1]
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in avg_best[idx_top]]
    ax.barh(range(top_n), avg_best[idx_top], color=colors, edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([all_labels[i] for i in idx_top], fontsize=8)
    ax.set_xlabel('Attribution (IG)', fontsize=10)
    ax.set_title(f'Best Matches (top {n_cases})\nFeature Attributions',
                 fontsize=11, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.5)

    # Worst matches
    ax = axes[1]
    avg_worst = worst_igs.mean(axis=0)
    order_w = np.argsort(np.abs(avg_worst))[::-1]
    idx_top_w = order_w[:top_n][::-1]
    colors_w = ['#2ecc71' if v > 0 else '#e74c3c' for v in avg_worst[idx_top_w]]
    ax.barh(range(top_n), avg_worst[idx_top_w], color=colors_w, edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([all_labels[i] for i in idx_top_w], fontsize=8)
    ax.set_xlabel('Attribution (IG)', fontsize=10)
    ax.set_title(f'Worst Matches (bottom {n_cases})\nFeature Attributions',
                 fontsize=11, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    path = f'{output_dir}/ig_best_worst.pdf'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")

    # ── VIZ 2: Aggregate feature importance (mean |IG|) ──
    fig, ax = plt.subplots(figsize=(10, 8))
    mean_abs_ig = np.abs(all_igs).mean(axis=0)
    order_agg = np.argsort(mean_abs_ig)[::-1]
    idx_show = order_agg[:25][::-1]

    # Color by CEO vs Firm
    n_firm = len(firm_features)
    colors_agg = ['#3498db' if i < n_firm else '#e67e22' for i in idx_show]
    ax.barh(range(len(idx_show)), mean_abs_ig[idx_show], color=colors_agg, edgecolor='white')
    ax.set_yticks(range(len(idx_show)))
    ax.set_yticklabels([all_labels[i] for i in idx_show], fontsize=9)
    ax.set_xlabel('Mean |Integrated Gradient|', fontsize=11)
    ax.set_title('Aggregate Feature Importance via Integrated Gradients\n'
                 '(Blue=Firm, Orange=CEO)',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    path = f'{output_dir}/ig_aggregate_importance.pdf'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")

    # ── VIZ 3: Individual case studies (waterfall) ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for panel_i, (label, case_idx, case_igs) in enumerate([
        ('Best #1', best_idx[0], best_igs[0]),
        ('Best #2', best_idx[1], best_igs[1]),
        ('Best #3', best_idx[2], best_igs[2]),
        ('Worst #1', worst_idx[0], worst_igs[0]),
        ('Worst #2', worst_idx[1], worst_igs[1]),
        ('Worst #3', worst_idx[2], worst_igs[2]),
    ]):
        ax = axes.flatten()[panel_i]
        order_case = np.argsort(np.abs(case_igs))[::-1]
        top_15 = order_case[:15][::-1]
        colors_case = ['#2ecc71' if v > 0 else '#e74c3c' for v in case_igs[top_15]]
        ax.barh(range(15), case_igs[top_15], color=colors_case, edgecolor='white')
        ax.set_yticks(range(15))
        ax.set_yticklabels([all_labels[i] for i in top_15], fontsize=7)
        ax.set_title(f'{label} (match={targets[case_idx]:.2f}, pred={all_preds[case_idx]:.2f})',
                     fontsize=9, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.5)

    fig.suptitle('Integrated Gradients: Individual Case Studies\n'
                 'Green=positive contribution, Red=negative contribution',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = f'{output_dir}/ig_case_studies.pdf'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")

    return mean_abs_ig, all_labels


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start = time.time()

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    print("=" * 70)
    print("  DEEP LEARNING EXTENSIONS FOR CEO-FIRM MATCHING")
    print("  1. Cross-Attention  2. Contrastive Learning  3. Integrated Gradients")
    print("=" * 70)

    config = EnrichedConfig()
    config.EPOCHS = args.epochs

    processor, data_dict, metadata, df_proc, train_loader, val_loader = load_data(config)

    results = {}

    # ── Extension 1: Cross-Attention ──
    if not args.skip_attention:
        print(f"\n{'='*70}")
        print("  EXTENSION 1: CROSS-ATTENTION MODEL")
        print(f"{'='*70}")

        attn_model = train_cross_attention(train_loader, val_loader, metadata, config)
        attn_corr, attn_map = visualize_attention(
            attn_model, data_dict, processor, config, OUTPUT_DIR)
        results['cross_attention_corr'] = attn_corr

    # ── Extension 2: Contrastive Learning ──
    if not args.skip_contrastive:
        print(f"\n{'='*70}")
        print("  EXTENSION 2: CONTRASTIVE LEARNING (InfoNCE + MSE)")
        print(f"{'='*70}")

        contrastive_model = train_contrastive(
            train_loader, val_loader, metadata, config, alpha=0.3)
        cont_corr, retrieval_metrics = retrieval_analysis(
            contrastive_model, data_dict, processor, config, df_proc, OUTPUT_DIR)
        results['contrastive_corr'] = cont_corr
        results.update({f'retrieval_{k}': v for k, v in retrieval_metrics.items()})

    # ── Extension 3: Integrated Gradients ──
    if not args.skip_ig:
        print(f"\n{'='*70}")
        print("  EXTENSION 3: INTEGRATED GRADIENTS")
        print(f"{'='*70}")

        # Train a standard Two-Tower model first, then attribute
        print("  Training base Two-Tower for IG analysis...")
        base_model = train_model(train_loader, val_loader, metadata, config)
        ig_importance, ig_labels = run_integrated_gradients(
            base_model, data_dict, processor, config, df_proc, OUTPUT_DIR,
            n_cases=10, n_steps=100)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    # Also train baseline for comparison
    print("  Training baseline Two-Tower for comparison...")
    baseline_model = train_model(train_loader, val_loader, metadata, config)
    baseline_model.eval()
    with torch.no_grad():
        bp = baseline_model(
            data_dict['firm_numeric'].to(config.DEVICE),
            data_dict['firm_cat'].to(config.DEVICE),
            data_dict['ceo_numeric'].to(config.DEVICE),
            data_dict['ceo_cat'].to(config.DEVICE),
        ).cpu().numpy().flatten()
    baseline_corr = np.corrcoef(bp, data_dict['target'].numpy().flatten())[0, 1]
    results['baseline_corr'] = baseline_corr

    print(f"\n  Model Performance Comparison:")
    print(f"  {'='*50}")
    print(f"  Baseline Two-Tower:     r = {results.get('baseline_corr', 'N/A'):.3f}")
    if 'cross_attention_corr' in results:
        print(f"  Cross-Attention:        r = {results['cross_attention_corr']:.3f}")
    if 'contrastive_corr' in results:
        print(f"  Contrastive (InfoNCE):  r = {results['contrastive_corr']:.3f}")
        print(f"    Hit@1:  {results.get('retrieval_hit@1', 0):.3f}")
        print(f"    Hit@10: {results.get('retrieval_hit@10', 0):.3f}")
        print(f"    MRR:    {results.get('retrieval_mrr', 0):.3f}")

    # Summary visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = []
    model_corrs = []
    for name, key in [('Baseline\nTwo-Tower', 'baseline_corr'),
                       ('Cross-\nAttention', 'cross_attention_corr'),
                       ('Contrastive\n(InfoNCE+MSE)', 'contrastive_corr')]:
        if key in results:
            model_names.append(name)
            model_corrs.append(results[key])

    colors = ['#95a5a6', '#e74c3c', '#3498db'][:len(model_names)]
    bars = ax.bar(model_names, model_corrs, color=colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, model_corrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Validation Correlation', fontsize=12)
    ax.set_title('Model Architecture Comparison', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(model_corrs) * 1.15)
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/model_comparison.pdf'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"  Done in {elapsed:.1f}s — outputs saved to {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
