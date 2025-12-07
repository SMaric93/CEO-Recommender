"""
Tests for the Config class.
"""
import pytest


class TestConfig:
    """Tests for Config class functionality."""
    
    def test_device_is_set(self, config):
        """Verify device is correctly detected."""
        assert config.DEVICE is not None
        assert str(config.DEVICE) in ['cpu', 'cuda', 'mps']
    
    def test_all_required_cols_returns_list(self, config):
        """Verify all_required_cols returns a non-empty list."""
        cols = config.all_required_cols
        assert isinstance(cols, list)
        assert len(cols) > 0
    
    def test_all_required_cols_includes_target(self, config):
        """Verify target column is included in required columns."""
        cols = config.all_required_cols
        assert config.TARGET_COL in cols
        assert config.WEIGHT_COL in cols
    
    def test_feature_columns_defined(self, config):
        """Verify feature column lists are properly defined."""
        assert len(config.CEO_NUMERIC_COLS) > 0
        assert len(config.CEO_CAT_COLS) > 0
        assert len(config.FIRM_NUMERIC_COLS) > 0
        assert len(config.FIRM_CAT_COLS) > 0
    
    def test_hyperparameters_reasonable(self, config):
        """Verify hyperparameters are within reasonable bounds."""
        assert 0 < config.LEARNING_RATE < 1
        assert 0 < config.EPOCHS <= 1000
        assert 0 < config.LATENT_DIM <= 512
