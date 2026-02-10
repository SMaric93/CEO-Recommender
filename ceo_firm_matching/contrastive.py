"""
Extension 2: Contrastive Learning with Hard Negatives

Implements InfoNCE / NT-Xent contrastive loss for learning discriminative
CEO-Firm embeddings. Instead of only learning from observed matches,
the model also learns what makes a BAD match by contrasting each
CEO-Firm pair against all other firms (and CEOs) in the batch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple

from .config import Config
from .model import CEOFirmMatcher


class ContrastiveCEOFirmMatcher(nn.Module):
    """
    Two-Tower model with contrastive learning support.

    In addition to the standard regression loss (MSE on match_means),
    this model adds an InfoNCE contrastive loss that:
    1. Treats each CEO-Firm pair as a positive
    2. Uses in-batch negatives (CEO_i vs Firm_j for i != j)
    3. Learns to push mismatched pairs apart

    The combined loss balances regression accuracy with embedding structure.
    """

    def __init__(self, metadata: Dict[str, int], config: Config):
        super().__init__()
        self.base_model = CEOFirmMatcher(metadata, config)
        self.config = config

        # Projection head for contrastive learning
        # Maps from embedding space to contrastive space
        self.firm_projector = nn.Sequential(
            nn.Linear(config.LATENT_DIM, config.LATENT_DIM),
            nn.ReLU(),
            nn.Linear(config.LATENT_DIM, config.LATENT_DIM // 2),
        )
        self.ceo_projector = nn.Sequential(
            nn.Linear(config.LATENT_DIM, config.LATENT_DIM),
            nn.ReLU(),
            nn.Linear(config.LATENT_DIM, config.LATENT_DIM // 2),
        )

    def get_embeddings(
        self,
        f_numeric: torch.Tensor,
        f_cat: torch.Tensor,
        c_numeric: torch.Tensor,
        c_cat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract L2-normalized embeddings from both towers."""
        # Firm Tower
        f_embs = [emb(f_cat[:, i]) for i, emb in enumerate(self.base_model.firm_embeddings)]
        f_combined = torch.cat([f_numeric] + f_embs, dim=1)
        u_firm = self.base_model.firm_tower(f_combined)
        u_firm = F.normalize(u_firm, dim=1)

        # CEO Tower
        c_embs = [emb(c_cat[:, i]) for i, emb in enumerate(self.base_model.ceo_embeddings)]
        c_combined = torch.cat([c_numeric] + c_embs, dim=1)
        v_ceo = self.base_model.ceo_tower(c_combined)
        v_ceo = F.normalize(v_ceo, dim=1)

        return u_firm, v_ceo

    def forward(
        self,
        f_numeric: torch.Tensor,
        f_cat: torch.Tensor,
        c_numeric: torch.Tensor,
        c_cat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning match scores + projected embeddings.

        Returns:
            match_score: [batch, 1] — scaled cosine similarity (same as base)
            firm_proj: [batch, dim//2] — projected firm embeddings for contrastive loss
            ceo_proj: [batch, dim//2] — projected CEO embeddings for contrastive loss
        """
        u_firm, v_ceo = self.get_embeddings(f_numeric, f_cat, c_numeric, c_cat)

        # Match score (same as base model)
        logit_scale = self.base_model.logit_scale.exp()
        match_score = (u_firm * v_ceo).sum(dim=1, keepdim=True) * logit_scale

        # Projected embeddings for contrastive loss
        firm_proj = F.normalize(self.firm_projector(u_firm), dim=1)
        ceo_proj = F.normalize(self.ceo_projector(v_ceo), dim=1)

        return match_score, firm_proj, ceo_proj


def info_nce_loss(
    firm_proj: torch.Tensor,
    ceo_proj: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss (symmetric).

    For batch size B:
    - Positive pairs: (firm_i, ceo_i) for i = 1..B
    - Negative pairs: (firm_i, ceo_j) for i != j

    Loss = -log(exp(sim(firm_i, ceo_i)/τ) / Σ_j exp(sim(firm_i, ceo_j)/τ))

    Args:
        firm_proj: [B, D] normalized firm projections
        ceo_proj: [B, D] normalized CEO projections
        temperature: scaling temperature (lower = harder)

    Returns:
        Scalar InfoNCE loss
    """
    B = firm_proj.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=firm_proj.device)

    # Similarity matrix: [B, B]
    sim_matrix = torch.mm(firm_proj, ceo_proj.t()) / temperature

    # Labels: positive pairs are on the diagonal
    labels = torch.arange(B, device=firm_proj.device)

    # Symmetric loss: firm→ceo and ceo→firm
    loss_f2c = F.cross_entropy(sim_matrix, labels)
    loss_c2f = F.cross_entropy(sim_matrix.t(), labels)

    return (loss_f2c + loss_c2f) / 2


def semi_hard_negative_mining(
    firm_proj: torch.Tensor,
    ceo_proj: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Triplet loss with semi-hard negative mining.

    For each anchor (firm_i), finds the hardest negative CEO_j that is
    farther than the positive CEO_i but within a margin.

    Args:
        firm_proj: [B, D] firm embeddings
        ceo_proj: [B, D] CEO embeddings
        margin: triplet margin

    Returns:
        Scalar triplet loss
    """
    B = firm_proj.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=firm_proj.device)

    # All pairwise distances
    dist_matrix = 1 - torch.mm(firm_proj, ceo_proj.t())  # [B, B], cosine distance

    # Positive distances: diagonal
    pos_dist = torch.diag(dist_matrix)  # [B]

    # For each anchor, find semi-hard negatives:
    # negatives where pos_dist < neg_dist < pos_dist + margin
    losses = []
    for i in range(B):
        neg_dists = dist_matrix[i]  # [B]
        # Mask out the positive
        mask = torch.ones(B, dtype=torch.bool, device=firm_proj.device)
        mask[i] = False
        neg_dists_masked = neg_dists[mask]

        # Semi-hard: neg_dist > pos_dist AND neg_dist < pos_dist + margin
        semi_hard_mask = (neg_dists_masked > pos_dist[i]) & (neg_dists_masked < pos_dist[i] + margin)

        if semi_hard_mask.any():
            # Take the hardest semi-hard negative (smallest distance among semi-hards)
            hardest = neg_dists_masked[semi_hard_mask].min()
            loss = F.relu(pos_dist[i] - hardest + margin)
        else:
            # Fall back to hardest negative overall
            hardest = neg_dists_masked.min()
            loss = F.relu(pos_dist[i] - hardest + margin)

        losses.append(loss)

    return torch.stack(losses).mean()


def train_contrastive(
    train_loader: DataLoader,
    val_loader: DataLoader,
    metadata: Dict[str, int],
    config: Config,
    contrastive_weight: float = 0.3,
    temperature: float = 0.07,
    use_triplet: bool = False,
) -> ContrastiveCEOFirmMatcher:
    """
    Train Two-Tower model with combined regression + contrastive loss.

    Total Loss = (1 - α) * MSE_loss + α * contrastive_loss

    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        metadata: Model architecture metadata
        config: Configuration
        contrastive_weight: α — weight for contrastive loss (0 = pure regression)
        temperature: InfoNCE temperature
        use_triplet: If True, use triplet loss instead of InfoNCE

    Returns:
        Trained ContrastiveCEOFirmMatcher
    """
    model = ContrastiveCEOFirmMatcher(metadata, config).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"Training Contrastive Two-Tower on {config.DEVICE}")
    print(f"  Contrastive weight: {contrastive_weight}")
    print(f"  Temperature: {temperature}")
    print(f"  Loss type: {'Triplet' if use_triplet else 'InfoNCE'}")

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        total_mse = 0
        total_cl = 0

        for batch in train_loader:
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()

            # Forward pass
            match_score, firm_proj, ceo_proj = model(
                batch['firm_numeric'], batch['firm_cat'],
                batch['ceo_numeric'], batch['ceo_cat']
            )

            # Regression loss (weighted MSE)
            mse_loss = (batch['weights'] * (match_score - batch['target']) ** 2).mean()

            # Contrastive loss
            if use_triplet:
                cl_loss = semi_hard_negative_mining(firm_proj, ceo_proj)
            else:
                cl_loss = info_nce_loss(firm_proj, ceo_proj, temperature)

            # Combined
            loss = (1 - contrastive_weight) * mse_loss + contrastive_weight * cl_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_cl += cl_loss.item()

        if epoch % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            avg_mse = total_mse / len(train_loader)
            avg_cl = total_cl / len(train_loader)
            print(f"  Epoch {epoch}: Loss={avg_loss:.4f} (MSE={avg_mse:.4f}, CL={avg_cl:.4f})")

    return model


def compute_retrieval_metrics(
    model: ContrastiveCEOFirmMatcher,
    data_dict: Dict,
    config: Config,
    top_k: int = 10,
) -> Dict[str, float]:
    """
    Evaluate embedding quality via retrieval metrics.

    For each firm, rank all CEOs by embedding similarity and check
    if the actual CEO is in the top-K.

    Args:
        model: Trained contrastive model
        data_dict: Transformed data dictionary
        config: Configuration
        top_k: K for recall@K

    Returns:
        Dict with recall@1, recall@5, recall@10, MRR
    """
    model.eval()
    with torch.no_grad():
        f_num = data_dict['firm_numeric'].to(config.DEVICE)
        f_cat = data_dict['firm_cat'].to(config.DEVICE)
        c_num = data_dict['ceo_numeric'].to(config.DEVICE)
        c_cat = data_dict['ceo_cat'].to(config.DEVICE)

        firm_emb, ceo_emb = model.get_embeddings(f_num, f_cat, c_num, c_cat)

        # Similarity matrix: [N, N]
        N = min(firm_emb.size(0), 5000)  # Cap for memory
        sim = torch.mm(firm_emb[:N], ceo_emb[:N].t())

        # For each firm (row), rank CEOs (columns)
        _, rankings = sim.sort(dim=1, descending=True)

        # True match is on the diagonal
        true_positions = torch.arange(N, device=config.DEVICE)

        # Find rank of true match for each firm
        ranks = []
        for i in range(N):
            rank = (rankings[i] == true_positions[i]).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                ranks.append(rank[0].item() + 1)  # 1-indexed
            else:
                ranks.append(N)

        ranks = np.array(ranks)

    return {
        'recall@1': float(np.mean(ranks <= 1)),
        'recall@5': float(np.mean(ranks <= 5)),
        'recall@10': float(np.mean(ranks <= top_k)),
        'MRR': float(np.mean(1.0 / ranks)),
        'median_rank': float(np.median(ranks)),
    }
