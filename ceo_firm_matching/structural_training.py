"""
CEO-Firm Matching: Structural Distillation Training Module

Training loop for the Structural Distillation Network using KL Divergence loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional

from .structural_config import StructuralConfig
from .structural_model import StructuralDistillationNet


def train_structural_model(
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    metadata: dict, 
    config: StructuralConfig
) -> Optional[StructuralDistillationNet]:
    """
    Train the Structural Distillation Network.
    
    Uses KL Divergence loss to match model predictions to BLM posterior probabilities.
    The interaction matrix is frozen, so only the tower parameters are updated.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        metadata: Dictionary containing model architecture metadata
        config: StructuralConfig object
        
    Returns:
        Trained StructuralDistillationNet model or None if training fails
    """
    # Initialize Model
    model = StructuralDistillationNet(metadata, config).to(config.DEVICE)
    
    print(f"Initialized Structural Distillation Network.")
    print(f"  Device: {config.DEVICE}")
    print(f"  Interaction Matrix Frozen: {not model.A.requires_grad}")
    print(f"  CEO Tower Input: {metadata['n_ceo_num']} numeric + {len(metadata['ceo_cat_cards'])} categorical")
    print(f"  Firm Tower Input: {metadata['n_firm_num']} numeric + {len(metadata['firm_cat_cards'])} categorical")
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    print(f"\nStarting Distillation Training for {config.EPOCHS} epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # Move to device
            f_num = batch['firm_num'].to(config.DEVICE)
            f_cat = batch['firm_cat'].to(config.DEVICE)
            c_num = batch['ceo_num'].to(config.DEVICE)
            c_cat = batch['ceo_cat'].to(config.DEVICE)
            target_ceo = batch['target_ceo'].to(config.DEVICE)
            target_firm = batch['target_firm'].to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            c_logits, f_logits, _ = model(f_num, f_cat, c_num, c_cat)
            
            # KL Divergence Loss (log_softmax of predictions vs true posteriors)
            # KL(P || Q) where P is target, Q is prediction
            loss_ceo = criterion(F.log_softmax(c_logits, dim=1), target_ceo)
            loss_firm = criterion(F.log_softmax(f_logits, dim=1), target_firm)
            loss = loss_ceo + loss_firm
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                f_num = batch['firm_num'].to(config.DEVICE)
                f_cat = batch['firm_cat'].to(config.DEVICE)
                c_num = batch['ceo_num'].to(config.DEVICE)
                c_cat = batch['ceo_cat'].to(config.DEVICE)
                target_ceo = batch['target_ceo'].to(config.DEVICE)
                target_firm = batch['target_firm'].to(config.DEVICE)
                
                c_logits, f_logits, _ = model(f_num, f_cat, c_num, c_cat)
                
                loss_ceo = criterion(F.log_softmax(c_logits, dim=1), target_ceo)
                loss_firm = criterion(F.log_softmax(f_logits, dim=1), target_firm)
                val_loss += (loss_ceo + loss_firm).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        # Logging
        if epoch % 10 == 0 or epoch == config.EPOCHS - 1:
            print(f"  Epoch {epoch:3d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.4f}")
    
    return model
