"""
Unit tests for the Structural Distillation Network configuration.
"""
import pytest
import torch

from ceo_firm_matching import StructuralConfig


class TestStructuralConfig:
    """Tests for StructuralConfig dataclass."""
    
    def test_config_instantiation(self):
        """Test that StructuralConfig can be instantiated."""
        config = StructuralConfig()
        
        assert config is not None
        assert isinstance(config.EPOCHS, int)
        assert isinstance(config.LEARNING_RATE, float)
        assert isinstance(config.BATCH_SIZE, int)
    
    def test_device_detection(self):
        """Test that device is properly detected."""
        config = StructuralConfig()
        
        assert config.DEVICE is not None
        assert isinstance(config.DEVICE, torch.device)
        # Should be one of cpu, cuda, or mps
        assert config.DEVICE.type in ['cpu', 'cuda', 'mps']
    
    def test_blm_interaction_matrix_shape(self):
        """Test that BLM interaction matrix has correct shape (5x5)."""
        config = StructuralConfig()
        
        matrix = config.BLM_INTERACTION_MATRIX
        assert len(matrix) == 5, "Matrix should have 5 rows"
        for row in matrix:
            assert len(row) == 5, "Each row should have 5 columns"
    
    def test_blm_interaction_matrix_values(self):
        """Test that BLM interaction matrix values are reasonable."""
        config = StructuralConfig()
        
        matrix = config.BLM_INTERACTION_MATRIX
        # Check that matrix is not all zeros
        all_values = [v for row in matrix for v in row]
        assert any(v != 0 for v in all_values), "Matrix should not be all zeros"
        
        # Check that values are within reasonable range
        for v in all_values:
            assert -10 < v < 10, f"Value {v} seems unreasonable"
    
    def test_probability_columns(self):
        """Test that probability column names are correctly generated."""
        config = StructuralConfig()
        
        assert len(config.CEO_PROB_COLS) == 5
        assert len(config.FIRM_PROB_COLS) == 5
        
        # Check naming pattern
        assert config.CEO_PROB_COLS[0] == 'prob_ceo_1'
        assert config.CEO_PROB_COLS[4] == 'prob_ceo_5'
        assert config.FIRM_PROB_COLS[0] == 'prob_firm_1'
        assert config.FIRM_PROB_COLS[4] == 'prob_firm_5'
    
    def test_feature_definitions(self):
        """Test that feature definitions are non-empty."""
        config = StructuralConfig()
        
        assert len(config.CEO_NUMERIC_COLS) > 0
        assert len(config.CEO_CAT_COLS) > 0
        assert len(config.FIRM_NUMERIC_COLS) > 0
        assert len(config.FIRM_CAT_COLS) > 0
    
    def test_all_cols_property(self):
        """Test the all_cols property returns expected columns."""
        config = StructuralConfig()
        
        all_cols = config.all_cols
        assert isinstance(all_cols, list)
        assert len(all_cols) > 0
        
        # Should include probability columns
        for col in config.CEO_PROB_COLS:
            assert col in all_cols
        for col in config.FIRM_PROB_COLS:
            assert col in all_cols
    
    def test_hyperparameters_reasonable(self):
        """Test that hyperparameters have reasonable default values."""
        config = StructuralConfig()
        
        assert config.EPOCHS > 0
        assert 0 < config.LEARNING_RATE < 1
        assert config.BATCH_SIZE > 0
        assert 0 <= config.DROPOUT < 1
        assert config.LATENT_DIM > 0
        assert config.EMBEDDING_DIM > 0
