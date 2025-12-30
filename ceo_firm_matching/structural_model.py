"""
CEO-Firm Matching: Structural Distillation Network Model

Two-Tower architecture with frozen BLM interaction matrix constraint.
Observables -> Latent Types -> Frozen Interaction -> Match Value
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .structural_config import StructuralConfig


class StructuralDistillationNet(nn.Module):
    """
    Structural Distillation Network.
    
    Architecture: Observables -> Latent Types -> Frozen Interaction -> Match Value
    
    The model learns to predict CEO and Firm type probabilities from observables,
    then uses the frozen BLM interaction matrix to compute expected match values.
    This enforces the economic structure learned from the BLM estimation.
    
    Attributes:
        A (torch.Tensor): Frozen 5x5 interaction matrix from BLM estimation
        firm_tower (nn.Sequential): Encoder mapping firm features to type logits
        ceo_tower (nn.Sequential): Encoder mapping CEO features to type logits
    """
    
    def __init__(self, metadata: Dict, config: StructuralConfig):
        """
        Initialize the Structural Distillation Network.
        
        Args:
            metadata: Dict containing:
                - n_firm_num: Number of firm numeric features
                - n_ceo_num: Number of CEO numeric features  
                - firm_cat_cards: List of cardinalities for firm categorical features
                - ceo_cat_cards: List of cardinalities for CEO categorical features
            config: StructuralConfig instance with hyperparameters and interaction matrix
        """
        super().__init__()
        
        self.config = config
        
        # --- 1. THE CONSTRAINT (Frozen Interaction Matrix) ---
        # Freezing 'A' ensures the model respects the economic structure.
        # This is a buffer (not a parameter) so it won't be updated by optimizer.
        self.register_buffer(
            'A', 
            torch.tensor(config.BLM_INTERACTION_MATRIX, dtype=torch.float32)
        )
        
        # --- 2. EMBEDDINGS ---
        self.firm_embeddings = nn.ModuleList([
            nn.Embedding(n_classes, config.EMBEDDING_DIM) 
            for n_classes in metadata['firm_cat_cards']
        ])
        
        self.ceo_embeddings = nn.ModuleList([
            nn.Embedding(n_classes, config.EMBEDDING_DIM) 
            for n_classes in metadata['ceo_cat_cards']
        ])
        
        # --- 3. ENCODERS (Towers) ---
        # Firm Tower
        firm_input_dim = (
            metadata['n_firm_num'] + 
            len(metadata['firm_cat_cards']) * config.EMBEDDING_DIM
        )
        self.firm_tower = nn.Sequential(
            nn.Linear(firm_input_dim, config.LATENT_DIM),
            nn.BatchNorm1d(config.LATENT_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # Logits for 5 Firm Classes
        )
        
        # CEO Tower
        ceo_input_dim = (
            metadata['n_ceo_num'] + 
            len(metadata['ceo_cat_cards']) * config.EMBEDDING_DIM
        )
        self.ceo_tower = nn.Sequential(
            nn.Linear(ceo_input_dim, config.LATENT_DIM),
            nn.BatchNorm1d(config.LATENT_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # Logits for 5 CEO Types
        )

    def forward(
        self, 
        f_num: torch.Tensor, 
        f_cat: torch.Tensor, 
        c_num: torch.Tensor, 
        c_cat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            f_num: Firm numeric features [batch, n_firm_num]
            f_cat: Firm categorical features [batch, n_firm_cat]
            c_num: CEO numeric features [batch, n_ceo_num]
            c_cat: CEO categorical features [batch, n_ceo_cat]
            
        Returns:
            Tuple of:
                - ceo_logits: CEO type logits [batch, 5]
                - firm_logits: Firm class logits [batch, 5]
                - expected_match: Expected match value [batch, 1]
        """
        # --- Encode Firm ---
        f_emb_list = [emb(f_cat[:, i]) for i, emb in enumerate(self.firm_embeddings)]
        f_combined = torch.cat([f_num] + f_emb_list, dim=1)
        f_logits = self.firm_tower(f_combined)
        
        # --- Encode CEO ---
        c_emb_list = [emb(c_cat[:, i]) for i, emb in enumerate(self.ceo_embeddings)]
        c_combined = torch.cat([c_num] + c_emb_list, dim=1)
        c_logits = self.ceo_tower(c_combined)
        
        # --- Probabilities ---
        q_firm = F.softmax(f_logits, dim=1)   # P(firm class | observables)
        pi_ceo = F.softmax(c_logits, dim=1)   # P(ceo type | observables)
        
        # --- Interaction (Bilinear Form) ---
        # Expected Match = sum_i sum_j pi_ceo[i] * A[i,j] * q_firm[j]
        # = pi_ceo @ A @ q_firm^T (for batch)
        # 
        # Implementation:
        # weighted_A = pi_ceo @ A  -> [batch, 5] x [5, 5] -> [batch, 5]
        # expected_match = (weighted_A * q_firm).sum()
        weighted_A = torch.matmul(pi_ceo, self.A)
        expected_match = torch.sum(weighted_A * q_firm, dim=1, keepdim=True)
        
        return c_logits, f_logits, expected_match
    
    def get_type_probabilities(
        self, 
        f_num: torch.Tensor, 
        f_cat: torch.Tensor, 
        c_num: torch.Tensor, 
        c_cat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get softmax probabilities for CEO and Firm types.
        
        Useful for interpretation and visualization.
        
        Returns:
            Tuple of (ceo_probs, firm_probs), each [batch, 5]
        """
        c_logits, f_logits, _ = self.forward(f_num, f_cat, c_num, c_cat)
        return F.softmax(c_logits, dim=1), F.softmax(f_logits, dim=1)
