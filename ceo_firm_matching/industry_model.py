"""
Extension 6: Industry-Specific Match Functions

Instead of one global match function, learns industry-conditioned
similarity metrics. Different industries value different CEO traits:
- Tech: innovation, risk tolerance, technical depth
- Banking: regulatory knowledge, risk management, network
- Manufacturing: ops experience, cost discipline, supply chain

Two approaches implemented:
1. Industry-conditioned temperature (lightweight)
2. Industry-specific attention weights on embedding dimensions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

from .config import Config
from .model import CEOFirmMatcher


class IndustryConditionedMatcher(nn.Module):
    """
    Two-Tower model with industry-conditioned similarity.

    The match score is computed as:
        score = (u_firm · v_ceo) * τ(industry)

    where τ is a learnable temperature per industry, allowing
    different industries to have sharper or flatter match distributions.
    """

    def __init__(
        self,
        metadata: Dict[str, int],
        config: Config,
        n_industries: int = 50,
    ):
        super().__init__()

        # Base towers
        self.firm_embeddings = nn.ModuleList([
            nn.Embedding(n_classes, config.EMBEDDING_DIM_LARGE)
            for n_classes in metadata['firm_cat_counts']
        ])
        self.ceo_embeddings = nn.ModuleList([
            nn.Embedding(n_classes, config.EMBEDDING_DIM_MEDIUM)
            for n_classes in metadata['ceo_cat_counts']
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

        # Industry-conditioned temperature
        # Each industry gets its own learned scale factor
        self.industry_temperature = nn.Embedding(n_industries, 1)
        nn.init.constant_(self.industry_temperature.weight, np.log(1 / 0.07))

        # Industry-specific attention over embedding dimensions
        # Learns which dimensions matter more for each industry
        self.industry_attention = nn.Embedding(n_industries, config.LATENT_DIM)
        nn.init.ones_(self.industry_attention.weight)  # Start uniform

    def forward(
        self,
        f_numeric: torch.Tensor,
        f_cat: torch.Tensor,
        c_numeric: torch.Tensor,
        c_cat: torch.Tensor,
        industry_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            f_numeric, f_cat, c_numeric, c_cat: Standard tower inputs
            industry_idx: [B] — industry index for each sample.
                         If None, uses f_cat[:, 0] (first categorical = compindustry).
        """
        # Encode
        f_embs = [emb(f_cat[:, i]) for i, emb in enumerate(self.firm_embeddings)]
        f_combined = torch.cat([f_numeric] + f_embs, dim=1)
        u_firm = self.firm_tower(f_combined)
        u_firm = F.normalize(u_firm, dim=1)

        c_embs = [emb(c_cat[:, i]) for i, emb in enumerate(self.ceo_embeddings)]
        c_combined = torch.cat([c_numeric] + c_embs, dim=1)
        v_ceo = self.ceo_tower(c_combined)
        v_ceo = F.normalize(v_ceo, dim=1)

        # Industry conditioning
        if industry_idx is None:
            industry_idx = f_cat[:, 0]  # First categorical = compindustry

        # Industry-specific dimension attention
        attn_weights = torch.sigmoid(self.industry_attention(industry_idx))  # [B, D]
        u_firm = u_firm * attn_weights
        v_ceo = v_ceo * attn_weights

        # Industry-specific temperature
        temperature = self.industry_temperature(industry_idx).exp()  # [B, 1]

        # Match score
        match_score = (u_firm * v_ceo).sum(dim=1, keepdim=True) * temperature

        return match_score

    def get_industry_importance(self, industry_idx: int) -> torch.Tensor:
        """
        Get the learned importance weights for a specific industry.

        Returns:
            [D] tensor of dimension importance scores (0-1)
        """
        with torch.no_grad():
            idx = torch.tensor([industry_idx], device=self.industry_attention.weight.device)
            return torch.sigmoid(self.industry_attention(idx)).squeeze()

    def compare_industries(self, idx_a: int, idx_b: int) -> Dict[str, float]:
        """Compare what two industries prioritize in CEO-Firm matching."""
        weights_a = self.get_industry_importance(idx_a)
        weights_b = self.get_industry_importance(idx_b)

        diff = weights_a - weights_b
        top_dims_a = torch.argsort(diff, descending=True)[:5]
        top_dims_b = torch.argsort(diff, descending=False)[:5]

        return {
            'correlation': float(F.cosine_similarity(weights_a.unsqueeze(0), weights_b.unsqueeze(0))),
            'industry_a_emphasis_dims': top_dims_a.tolist(),
            'industry_b_emphasis_dims': top_dims_b.tolist(),
            'temperature_a': float(self.industry_temperature.weight[idx_a].exp()),
            'temperature_b': float(self.industry_temperature.weight[idx_b].exp()),
        }


class IndustryExpertMixture(nn.Module):
    """
    Mixture-of-Experts model with industry-specialized expert towers.

    Instead of one set of towers, maintains K expert sub-towers
    and uses a gating mechanism conditioned on industry to select
    which experts to activate for each sample.

    This is the heavier-weight version of industry specialization.
    """

    def __init__(
        self,
        metadata: Dict[str, int],
        config: Config,
        n_experts: int = 4,
        n_industries: int = 50,
    ):
        super().__init__()
        self.n_experts = n_experts

        # Shared embeddings
        self.firm_embeddings = nn.ModuleList([
            nn.Embedding(n_classes, config.EMBEDDING_DIM_LARGE)
            for n_classes in metadata['firm_cat_counts']
        ])
        self.ceo_embeddings = nn.ModuleList([
            nn.Embedding(n_classes, config.EMBEDDING_DIM_MEDIUM)
            for n_classes in metadata['ceo_cat_counts']
        ])

        # Expert towers
        firm_input_dim = metadata['n_firm_numeric'] + len(metadata['firm_cat_counts']) * config.EMBEDDING_DIM_LARGE
        ceo_input_dim = metadata['n_ceo_numeric'] + len(metadata['ceo_cat_counts']) * config.EMBEDDING_DIM_MEDIUM

        self.firm_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(firm_input_dim, 48),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(48, config.LATENT_DIM)
            ) for _ in range(n_experts)
        ])

        self.ceo_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ceo_input_dim, 48),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(48, config.LATENT_DIM)
            ) for _ in range(n_experts)
        ])

        # Gating network: industry → expert weights
        self.gate = nn.Sequential(
            nn.Embedding(n_industries, 32),
        )
        self.gate_proj = nn.Linear(32, n_experts)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        f_numeric: torch.Tensor,
        f_cat: torch.Tensor,
        c_numeric: torch.Tensor,
        c_cat: torch.Tensor,
        industry_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if industry_idx is None:
            industry_idx = f_cat[:, 0]

        # Prepare inputs
        f_embs = [emb(f_cat[:, i]) for i, emb in enumerate(self.firm_embeddings)]
        f_combined = torch.cat([f_numeric] + f_embs, dim=1)

        c_embs = [emb(c_cat[:, i]) for i, emb in enumerate(self.ceo_embeddings)]
        c_combined = torch.cat([c_numeric] + c_embs, dim=1)

        # Expert outputs
        firm_expert_out = torch.stack([expert(f_combined) for expert in self.firm_experts], dim=1)  # [B, K, D]
        ceo_expert_out = torch.stack([expert(c_combined) for expert in self.ceo_experts], dim=1)    # [B, K, D]

        # Gating weights
        gate_emb = self.gate[0](industry_idx)  # [B, 32]
        gate_weights = F.softmax(self.gate_proj(gate_emb), dim=1)  # [B, K]

        # Weighted combination of experts
        u_firm = (firm_expert_out * gate_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        v_ceo = (ceo_expert_out * gate_weights.unsqueeze(-1)).sum(dim=1)   # [B, D]

        u_firm = F.normalize(u_firm, dim=1)
        v_ceo = F.normalize(v_ceo, dim=1)

        logit_scale = self.logit_scale.exp()
        match_score = (u_firm * v_ceo).sum(dim=1, keepdim=True) * logit_scale

        return match_score
