"""
CEO-Firm Matching: Training Module

Training loop and utilities for the Two-Tower model.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional

from .config import Config
from .model import CEOFirmMatcher


def train_model(train_loader: DataLoader, val_loader: DataLoader, 
                metadata: Dict[str, int], config: Config) -> Optional[CEOFirmMatcher]:
    """
    Train the CEOFirmMatcher model.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        metadata: Dictionary containing model architecture metadata
        config: Configuration object
        
    Returns:
        Trained CEOFirmMatcher model or None if training fails
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
