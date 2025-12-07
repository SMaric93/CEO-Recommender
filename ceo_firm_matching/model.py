"""
CEO-Firm Matching: Model Architecture Module

Two-Tower Neural Network for CEO-Firm matching.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict

from .config import Config


class CEOFirmMatcher(nn.Module):
    """
    Two-Tower Neural Network.
    Encodes CEO and Firm features separately and computes similarity.
    """
    def __init__(self, metadata: Dict[str, int], config: Config):
        super(CEOFirmMatcher, self).__init__()
        
        # --- Embeddings ---
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
        
        # --- TOWER A: FIRM ENCODER ---
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
        
        # --- TOWER B: CEO ENCODER ---
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
        
        # Learnable temperature for cosine similarity scaling
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, f_numeric, f_cat, c_numeric, c_cat):
        # Firm Tower
        f_embs = [emb(f_cat[:, i]) for i, emb in enumerate(self.firm_embeddings)]
        f_combined = torch.cat([f_numeric] + f_embs, dim=1)
        u_firm = self.firm_tower(f_combined)
        
        # CEO Tower
        c_embs = [emb(c_cat[:, i]) for i, emb in enumerate(self.ceo_embeddings)]
        c_combined = torch.cat([c_numeric] + c_embs, dim=1)
        v_ceo = self.ceo_tower(c_combined)
        
        # L2 Normalization
        u_firm = u_firm / u_firm.norm(dim=1, keepdim=True)
        v_ceo = v_ceo / v_ceo.norm(dim=1, keepdim=True)
        
        # Match Score (Scaled Dot Product)
        # We use a learnable scale because match_means are not bounded to [-1, 1]
        # Actually, match_means are N(0,1), so they can be > 1. 
        # Standard cosine sim is [-1, 1]. We need to scale it.
        logit_scale = self.logit_scale.exp()
        match_score = (u_firm * v_ceo).sum(dim=1, keepdim=True) * logit_scale
        
        return match_score
