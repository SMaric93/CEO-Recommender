"""
Extension 5: Time-Varying CEO Embeddings

Models CEO career trajectories using temporal attention over
sequential CEO-year observations. Instead of treating each year
independently, this captures how CEO capabilities evolve over time.

Architecture:
    CEO at t=1 → embed_1 ─┐
    CEO at t=2 → embed_2 ──├─→ Temporal Attention → CEO_embedding(t)
    CEO at t=3 → embed_3 ─┘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional

from .config import Config


class TemporalAttention(nn.Module):
    """
    Self-attention over a CEO's career sequence.

    Given a sequence of yearly CEO embeddings, produces a context-aware
    representation that weights recent years more heavily while preserving
    long-range career patterns.
    """

    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.positional_decay = nn.Parameter(torch.ones(1))  # Learnable recency bias

    def forward(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence: [B, T, D] — CEO embeddings over T years
            mask: [B, T] — True for padded positions

        Returns:
            [B, D] — Temporally-aware CEO embedding
        """
        B, T, D = sequence.shape

        # Add positional recency bias (more recent = higher weight)
        positions = torch.arange(T, device=sequence.device).float()
        recency_weight = torch.exp(-self.positional_decay * (T - 1 - positions) / T)
        recency_weight = recency_weight.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        sequence = sequence * recency_weight

        # Self-attention
        attn_output, _ = self.attention(sequence, sequence, sequence, key_padding_mask=mask)
        attn_output = self.layer_norm(attn_output + sequence)  # Residual

        # Pool: take the last (most recent) position
        if mask is not None:
            # Find the last valid position for each sequence
            lengths = (~mask).sum(dim=1)  # [B]
            idx = (lengths - 1).clamp(min=0).long()
            output = attn_output[torch.arange(B, device=sequence.device), idx]
        else:
            output = attn_output[:, -1]  # Last position

        return output


class TemporalCEOEncoder(nn.Module):
    """
    CEO tower that processes career sequences instead of single snapshots.

    Given a padded sequence of CEO yearly features, produces a career-aware
    embedding that captures trajectory, growth, and evolution patterns.
    """

    def __init__(self, metadata: Dict[str, int], config: Config, max_seq_len: int = 30):
        super().__init__()
        self.config = config
        self.max_seq_len = max_seq_len

        # Per-year feature encoder (same as base CEO tower minus final projection)
        ceo_input_dim = metadata['n_ceo_numeric'] + len(metadata['ceo_cat_counts']) * config.EMBEDDING_DIM_MEDIUM
        self.yearly_encoder = nn.Sequential(
            nn.Linear(ceo_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # CEO Embeddings (shared across timesteps)
        self.ceo_embeddings = nn.ModuleList([
            nn.Embedding(n_classes, config.EMBEDDING_DIM_MEDIUM)
            for n_classes in metadata['ceo_cat_counts']
        ])

        # Temporal aggregation
        self.temporal_attention = TemporalAttention(
            embed_dim=32,
            n_heads=4,
            dropout=0.1
        )

        # Final projection to latent space
        self.projector = nn.Linear(32, config.LATENT_DIM)

        # Year embedding (positional encoding for calendar year)
        self.year_embedding = nn.Embedding(max_seq_len, 32)

    def forward(
        self,
        c_numeric_seq: torch.Tensor,   # [B, T, n_ceo_numeric]
        c_cat_seq: torch.Tensor,        # [B, T, n_ceo_cat]
        seq_mask: torch.Tensor,         # [B, T] — True for padding
    ) -> torch.Tensor:
        """
        Args:
            c_numeric_seq: Numeric features over T years
            c_cat_seq: Categorical features over T years
            seq_mask: Padding mask

        Returns:
            [B, D] — career-aware CEO embedding
        """
        B, T, _ = c_numeric_seq.shape

        # Encode each year independently
        yearly_embeds = []
        for t in range(T):
            c_embs = [emb(c_cat_seq[:, t, i]) for i, emb in enumerate(self.ceo_embeddings)]
            c_combined = torch.cat([c_numeric_seq[:, t]] + c_embs, dim=1)
            yearly_embeds.append(self.yearly_encoder(c_combined))

        # Stack: [B, T, 32]
        sequence = torch.stack(yearly_embeds, dim=1)

        # Add year positional encoding
        year_pos = torch.arange(T, device=sequence.device).unsqueeze(0).expand(B, -1)
        year_pos = year_pos.clamp(max=self.max_seq_len - 1)
        sequence = sequence + self.year_embedding(year_pos)

        # Temporal attention
        career_embed = self.temporal_attention(sequence, mask=seq_mask)

        # Project to latent space
        career_embed = self.projector(career_embed)
        career_embed = F.normalize(career_embed, dim=1)

        return career_embed


class TemporalTwoTower(nn.Module):
    """
    Two-Tower model where the CEO tower processes career sequences.

    The Firm tower remains snapshot-based (firm characteristics at time t),
    while the CEO tower aggregates the CEO's full career history up to t.
    """

    def __init__(self, metadata: Dict[str, int], config: Config, max_seq_len: int = 30):
        super().__init__()
        self.config = config

        # Firm Tower (standard — snapshot at time t)
        self.firm_embeddings = nn.ModuleList([
            nn.Embedding(n_classes, config.EMBEDDING_DIM_LARGE)
            for n_classes in metadata['firm_cat_counts']
        ])
        firm_input_dim = metadata['n_firm_numeric'] + len(metadata['firm_cat_counts']) * config.EMBEDDING_DIM_LARGE
        self.firm_tower = nn.Sequential(
            nn.Linear(firm_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, config.LATENT_DIM)
        )

        # CEO Tower (temporal)
        self.ceo_temporal = TemporalCEOEncoder(metadata, config, max_seq_len)

        # Temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        f_numeric: torch.Tensor,
        f_cat: torch.Tensor,
        c_numeric_seq: torch.Tensor,
        c_cat_seq: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            f_numeric: [B, n_firm_num]
            f_cat: [B, n_firm_cat]
            c_numeric_seq: [B, T, n_ceo_num]
            c_cat_seq: [B, T, n_ceo_cat]
            seq_mask: [B, T]

        Returns:
            match_score: [B, 1]
        """
        # Firm embedding (snapshot)
        f_embs = [emb(f_cat[:, i]) for i, emb in enumerate(self.firm_embeddings)]
        f_combined = torch.cat([f_numeric] + f_embs, dim=1)
        u_firm = self.firm_tower(f_combined)
        u_firm = F.normalize(u_firm, dim=1)

        # CEO embedding (temporal)
        v_ceo = self.ceo_temporal(c_numeric_seq, c_cat_seq, seq_mask)

        # Match
        logit_scale = self.logit_scale.exp()
        match_score = (u_firm * v_ceo).sum(dim=1, keepdim=True) * logit_scale

        return match_score


class CareerSequenceDataset(Dataset):
    """
    Dataset that constructs CEO career sequences from panel data.

    For each (CEO, year_t) observation, looks back up to max_seq_len years
    and pads shorter sequences.
    """

    def __init__(
        self,
        df: 'pd.DataFrame',
        data_dict: Dict,
        ceo_id_col: str = 'match_exec_id',
        year_col: str = 'fiscalyear',
        max_seq_len: int = 15,
    ):
        import pandas as pd
        self.max_seq_len = max_seq_len
        self.data_dict = data_dict

        # Build CEO career index: for each (ceo_id, year), store row indices
        df = df.reset_index(drop=True)
        self.career_index = {}
        for ceo_id, group in df.groupby(ceo_id_col):
            sorted_group = group.sort_values(year_col)
            years = sorted_group[year_col].values
            indices = sorted_group.index.values
            for i, (year, idx) in enumerate(zip(years, indices)):
                # Look back up to max_seq_len years
                start = max(0, i - max_seq_len + 1)
                seq_indices = indices[start:i + 1]
                self.career_index[(ceo_id, year)] = seq_indices

        self.keys = list(self.career_index.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        seq_indices = self.career_index[key]
        current_idx = seq_indices[-1]  # Most recent year
        T = len(seq_indices)

        # Pad to max_seq_len
        pad_len = self.max_seq_len - T

        # CEO sequence
        c_num = self.data_dict['ceo_numeric'][seq_indices]  # [T, n_ceo_num]
        c_cat = self.data_dict['ceo_cat'][seq_indices]      # [T, n_ceo_cat]

        if pad_len > 0:
            c_num = torch.cat([torch.zeros(pad_len, c_num.size(1)), c_num], dim=0)
            c_cat = torch.cat([torch.zeros(pad_len, c_cat.size(1), dtype=torch.long), c_cat], dim=0)

        # Mask: True for padded positions
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        mask[:pad_len] = True

        return {
            'firm_numeric': self.data_dict['firm_numeric'][current_idx],
            'firm_cat': self.data_dict['firm_cat'][current_idx],
            'ceo_numeric_seq': c_num,
            'ceo_cat_seq': c_cat,
            'seq_mask': mask,
            'target': self.data_dict['target'][current_idx],
            'weights': self.data_dict['weights'][current_idx],
        }
