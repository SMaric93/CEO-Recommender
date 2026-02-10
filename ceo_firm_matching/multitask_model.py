"""
Extension 4: Multi-Task Learning

Joint prediction of match quality + compensation + tenure + turnover probability.
Auxiliary tasks regularize the embedding space, forcing towers to encode richer information.

Architecture:
    CEO Tower ──→ CEO Embedding ─┐
                                  ├─→ Head 1: Match Score (cosine sim, regression)
    Firm Tower ──→ Firm Embedding ─┤
                                  ├─→ Head 2: Expected TDC1 (regression)
                                  ├─→ Head 3: Expected Tenure (regression)
                                  └─→ Head 4: P(Turnover in 3y) (binary classification)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

from .config import Config
from .model import CEOFirmMatcher


@dataclass
class MultiTaskConfig(Config):
    """Configuration for multi-task training."""
    # Task weights in the combined loss
    MATCH_WEIGHT: float = 1.0       # Primary task
    COMP_WEIGHT: float = 0.3        # Compensation prediction
    TENURE_WEIGHT: float = 0.2      # Tenure prediction
    TURNOVER_WEIGHT: float = 0.2    # Turnover classification

    # Architecture
    HEAD_HIDDEN_DIM: int = 32       # Hidden dim for auxiliary heads
    EPOCHS: int = 60


class MultiTaskCEOFirmMatcher(nn.Module):
    """
    Multi-Task Two-Tower model with auxiliary prediction heads.

    The base towers are shared across all tasks. Each task has its own
    prediction head that either operates on:
    - Combined embeddings (match, comp, tenure)
    - Individual tower embeddings (CEO-specific predictions)
    """

    def __init__(self, metadata: Dict[str, int], config: MultiTaskConfig):
        super().__init__()
        self.config = config

        # === SHARED TOWERS (same architecture as base model) ===
        # Firm Embeddings
        self.firm_embeddings = nn.ModuleList([
            nn.Embedding(n_classes, config.EMBEDDING_DIM_LARGE)
            for n_classes in metadata['firm_cat_counts']
        ])
        # CEO Embeddings
        self.ceo_embeddings = nn.ModuleList([
            nn.Embedding(n_classes, config.EMBEDDING_DIM_MEDIUM)
            for n_classes in metadata['ceo_cat_counts']
        ])

        # Firm Tower
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

        # CEO Tower
        ceo_input_dim = metadata['n_ceo_numeric'] + len(metadata['ceo_cat_counts']) * config.EMBEDDING_DIM_MEDIUM
        self.ceo_tower = nn.Sequential(
            nn.Linear(ceo_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, config.LATENT_DIM)
        )

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # === AUXILIARY HEADS ===
        combined_dim = config.LATENT_DIM * 2  # concatenation of firm + CEO embeddings

        # Head 2: Compensation prediction (log TDC1)
        self.comp_head = nn.Sequential(
            nn.Linear(combined_dim, config.HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.HEAD_HIDDEN_DIM, 1)
        )

        # Head 3: Tenure prediction (years)
        self.tenure_head = nn.Sequential(
            nn.Linear(combined_dim, config.HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.HEAD_HIDDEN_DIM, 1)
        )

        # Head 4: Turnover classification (P(turnover in 3y))
        self.turnover_head = nn.Sequential(
            nn.Linear(combined_dim, config.HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.HEAD_HIDDEN_DIM, 1),
            nn.Sigmoid()
        )

    def encode(
        self,
        f_numeric: torch.Tensor,
        f_cat: torch.Tensor,
        c_numeric: torch.Tensor,
        c_cat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get L2-normalized firm and CEO embeddings."""
        # Firm
        f_embs = [emb(f_cat[:, i]) for i, emb in enumerate(self.firm_embeddings)]
        f_combined = torch.cat([f_numeric] + f_embs, dim=1)
        u_firm = self.firm_tower(f_combined)
        u_firm = F.normalize(u_firm, dim=1)

        # CEO
        c_embs = [emb(c_cat[:, i]) for i, emb in enumerate(self.ceo_embeddings)]
        c_combined = torch.cat([c_numeric] + c_embs, dim=1)
        v_ceo = self.ceo_tower(c_combined)
        v_ceo = F.normalize(v_ceo, dim=1)

        return u_firm, v_ceo

    def forward(
        self,
        f_numeric: torch.Tensor,
        f_cat: torch.Tensor,
        c_numeric: torch.Tensor,
        c_cat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through all heads.

        Returns dict with:
            'match_score': [B, 1] — primary match prediction
            'comp_pred': [B, 1] — predicted log(TDC1)
            'tenure_pred': [B, 1] — predicted tenure in years
            'turnover_prob': [B, 1] — P(turnover within 3 years)
            'firm_embedding': [B, D] — firm embedding
            'ceo_embedding': [B, D] — CEO embedding
        """
        u_firm, v_ceo = self.encode(f_numeric, f_cat, c_numeric, c_cat)

        # Head 1: Match score (scaled cosine similarity)
        logit_scale = self.logit_scale.exp()
        match_score = (u_firm * v_ceo).sum(dim=1, keepdim=True) * logit_scale

        # Concatenated embedding for auxiliary heads
        combined = torch.cat([u_firm, v_ceo], dim=1)

        # Auxiliary predictions
        comp_pred = self.comp_head(combined)
        tenure_pred = self.tenure_head(combined)
        turnover_prob = self.turnover_head(combined)

        return {
            'match_score': match_score,
            'comp_pred': comp_pred,
            'tenure_pred': tenure_pred,
            'turnover_prob': turnover_prob,
            'firm_embedding': u_firm,
            'ceo_embedding': v_ceo,
        }


class MultiTaskDataset(Dataset):
    """Dataset that includes auxiliary targets for multi-task learning."""

    def __init__(self, data_dict: Dict[str, torch.Tensor]):
        self.data = data_dict
        self.length = len(data_dict['target'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {
            'firm_numeric': self.data['firm_numeric'][idx],
            'firm_cat': self.data['firm_cat'][idx],
            'ceo_numeric': self.data['ceo_numeric'][idx],
            'ceo_cat': self.data['ceo_cat'][idx],
            'target': self.data['target'][idx],
            'weights': self.data['weights'][idx],
        }

        # Auxiliary targets (may be NaN for some samples)
        for key in ['log_tdc1', 'tenure_years', 'turnover_3y']:
            if key in self.data:
                item[key] = self.data[key][idx]
            else:
                item[key] = torch.tensor(float('nan'), dtype=torch.float32)

        return item


def compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: MultiTaskConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute weighted multi-task loss.

    Handles missing targets gracefully by masking NaN values.

    Returns:
        total_loss: scalar combined loss
        loss_dict: per-task loss values for logging
    """
    loss_dict = {}

    # Task 1: Match quality (MSE, always present)
    match_loss = (batch['weights'] * (outputs['match_score'] - batch['target']) ** 2).mean()
    loss_dict['match'] = match_loss.item()
    total_loss = config.MATCH_WEIGHT * match_loss

    # Task 2: Compensation (MSE on log scale)
    if 'log_tdc1' in batch:
        comp_target = batch['log_tdc1']
        valid_comp = ~torch.isnan(comp_target)
        if valid_comp.any():
            comp_loss = F.mse_loss(
                outputs['comp_pred'][valid_comp],
                comp_target[valid_comp].unsqueeze(1)
            )
            loss_dict['comp'] = comp_loss.item()
            total_loss = total_loss + config.COMP_WEIGHT * comp_loss

    # Task 3: Tenure (MSE)
    if 'tenure_years' in batch:
        tenure_target = batch['tenure_years']
        valid_tenure = ~torch.isnan(tenure_target)
        if valid_tenure.any():
            tenure_loss = F.mse_loss(
                outputs['tenure_pred'][valid_tenure],
                tenure_target[valid_tenure].unsqueeze(1)
            )
            loss_dict['tenure'] = tenure_loss.item()
            total_loss = total_loss + config.TENURE_WEIGHT * tenure_loss

    # Task 4: Turnover (BCE)
    if 'turnover_3y' in batch:
        turnover_target = batch['turnover_3y']
        valid_turnover = ~torch.isnan(turnover_target)
        if valid_turnover.any():
            turnover_loss = F.binary_cross_entropy(
                outputs['turnover_prob'][valid_turnover].squeeze(),
                turnover_target[valid_turnover]
            )
            loss_dict['turnover'] = turnover_loss.item()
            total_loss = total_loss + config.TURNOVER_WEIGHT * turnover_loss

    return total_loss, loss_dict


def train_multitask(
    train_loader: DataLoader,
    val_loader: DataLoader,
    metadata: Dict[str, int],
    config: MultiTaskConfig,
) -> MultiTaskCEOFirmMatcher:
    """
    Train multi-task Two-Tower model.

    Prints per-task loss breakdown at each logging epoch.
    """
    model = MultiTaskCEOFirmMatcher(metadata, config).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"Training Multi-Task Two-Tower on {config.DEVICE}")
    print(f"  Tasks: Match({config.MATCH_WEIGHT}) + Comp({config.COMP_WEIGHT}) "
          f"+ Tenure({config.TENURE_WEIGHT}) + Turnover({config.TURNOVER_WEIGHT})")

    for epoch in range(config.EPOCHS):
        model.train()
        epoch_losses = {}

        for batch in train_loader:
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = model(
                batch['firm_numeric'], batch['firm_cat'],
                batch['ceo_numeric'], batch['ceo_cat']
            )

            loss, task_losses = compute_multitask_loss(outputs, batch, config)

            loss.backward()
            optimizer.step()

            for k, v in task_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v

        if epoch % 5 == 0:
            n = len(train_loader)
            parts = ' | '.join(f"{k}={v/n:.4f}" for k, v in epoch_losses.items())
            print(f"  Epoch {epoch}: {parts}")

    return model


def analyze_multitask_embeddings(
    model: MultiTaskCEOFirmMatcher,
    data_dict: Dict,
    df: 'pd.DataFrame',
    config: MultiTaskConfig,
) -> Dict[str, 'np.ndarray']:
    """
    Extract and analyze embeddings from the multi-task model.

    Returns predictions from all heads for downstream analysis.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(
            data_dict['firm_numeric'].to(config.DEVICE),
            data_dict['firm_cat'].to(config.DEVICE),
            data_dict['ceo_numeric'].to(config.DEVICE),
            data_dict['ceo_cat'].to(config.DEVICE),
        )

    return {
        'match_score': outputs['match_score'].cpu().numpy().flatten(),
        'comp_pred': outputs['comp_pred'].cpu().numpy().flatten(),
        'tenure_pred': outputs['tenure_pred'].cpu().numpy().flatten(),
        'turnover_prob': outputs['turnover_prob'].cpu().numpy().flatten(),
        'firm_embedding': outputs['firm_embedding'].cpu().numpy(),
        'ceo_embedding': outputs['ceo_embedding'].cpu().numpy(),
    }
