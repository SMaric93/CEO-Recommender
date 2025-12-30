"""
Two Tower neural network architecture for CEO-Firm matching.

This module contains:
- CEOFirmMatcher: The Two-Tower neural network model
- ModelWrapper: sklearn-like wrapper for explainability tools
- train_model: Training loop with weighted MSE loss
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional

from ceo_firm_matching.config import Config
from data_processing import DataProcessor

__all__ = ['CEOFirmMatcher', 'ModelWrapper', 'train_model']


class CEOFirmMatcher(nn.Module):
    """
    Two-Tower Neural Network for CEO-Firm matching.
    
    Encodes CEO and Firm features separately via independent towers,
    then computes a scaled cosine similarity for match prediction.
    
    Architecture:
        - Firm Tower: Numeric features + Categorical embeddings → 64 → 32 → LATENT_DIM
        - CEO Tower: Numeric features + Categorical embeddings → 64 → 32 → LATENT_DIM
        - Output: Scaled dot product of L2-normalized tower outputs
    
    Args:
        metadata: Dictionary with feature counts (from DataProcessor.transform)
        config: Config instance with hyperparameters
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
        """
        Forward pass through both towers.
        
        Args:
            f_numeric: Firm numeric features (N, n_firm_numeric)
            f_cat: Firm categorical indices (N, n_firm_cat)
            c_numeric: CEO numeric features (N, n_ceo_numeric)
            c_cat: CEO categorical indices (N, n_ceo_cat)
            
        Returns:
            Match scores (N, 1)
        """
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
        logit_scale = self.logit_scale.exp()
        match_score = (u_firm * v_ceo).sum(dim=1, keepdim=True) * logit_scale
        
        return match_score


class ModelWrapper:
    """
    Wraps the PyTorch model to expose a sklearn-like API.
    
    Needed for SHAP, PDP, and general interpretation tools that expect
    a simple predict(X) interface with numpy arrays.
    
    Input Layout (Flattened):
    [Firm Numeric...] [Firm Cat...] [CEO Numeric...] [CEO Cat...]
    
    Args:
        model: Trained CEOFirmMatcher instance
        processor: Fitted DataProcessor with feature metadata
    """
    def __init__(self, model: CEOFirmMatcher, processor: DataProcessor):
        self.model = model
        self.processor = processor
        self.device = processor.cfg.DEVICE
        
        # Calculate indices for slicing the flattened input
        self.n_firm_num = len(processor.final_firm_numeric)
        self.n_firm_cat = len(processor.cfg.FIRM_CAT_COLS)
        self.n_ceo_num = len(processor.final_ceo_numeric)
        self.n_ceo_cat = len(processor.cfg.CEO_CAT_COLS)
        
        # Slice Ranges
        self.idx_firm_num_end = self.n_firm_num
        self.idx_firm_cat_end = self.idx_firm_num_end + self.n_firm_cat
        self.idx_ceo_num_end = self.idx_firm_cat_end + self.n_ceo_num
        self.idx_ceo_cat_end = self.idx_ceo_num_end + self.n_ceo_cat
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on a flattened feature array.
        
        Args:
            X: 2D numpy array of shape (n_samples, n_features)
            
        Returns:
            1D numpy array of predictions (n_samples,)
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Slice and Type Cast
        f_num = X_tensor[:, :self.idx_firm_num_end]
        f_cat = X_tensor[:, self.idx_firm_num_end:self.idx_firm_cat_end].long()
        c_num = X_tensor[:, self.idx_firm_cat_end:self.idx_ceo_num_end]
        c_cat = X_tensor[:, self.idx_ceo_num_end:].long()
        
        with torch.no_grad():
            preds = self.model(f_num, f_cat, c_num, c_cat)
            
        return preds.cpu().numpy().flatten()


def train_model(train_loader: DataLoader, val_loader: DataLoader, 
                metadata: Dict[str, int], config: Config) -> Optional[CEOFirmMatcher]:
    """
    Train the Two-Tower model with weighted MSE loss.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        metadata: Feature counts from DataProcessor.transform
        config: Config with hyperparameters
        
    Returns:
        Trained model, or None on failure
    """
    # Initialize Model
    model = CEOFirmMatcher(metadata, config).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    print(f"Starting training on {config.DEVICE} for {config.EPOCHS} epochs...")
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            preds = model(
                batch['firm_numeric'], batch['firm_cat'],
                batch['ceo_numeric'], batch['ceo_cat']
            )
            
            # Weighted MSE Loss
            loss = (batch['weights'] * (preds - batch['target'])**2).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Avg Train Loss = {avg_loss:.4f}")
            
    return model
