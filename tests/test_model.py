"""
Unit tests for the model architecture.
"""
import pytest
import torch

from ceo_firm_matching import CEOFirmMatcher, Config


class TestCEOFirmMatcher:
    """Tests for the CEOFirmMatcher model."""
    
    @pytest.fixture
    def sample_metadata(self):
        """Provides sample metadata for model initialization."""
        return {
            'n_firm_numeric': 12,
            'firm_cat_counts': [4, 4, 2, 2],  # 4 categorical features
            'n_ceo_numeric': 2,
            'ceo_cat_counts': [2, 4, 2, 2, 2, 2, 2],  # 7 categorical features
        }
    
    def test_model_initialization(self, sample_metadata):
        """Test that model initializes without error."""
        config = Config()
        model = CEOFirmMatcher(sample_metadata, config)
        
        assert model is not None
        assert hasattr(model, 'firm_tower')
        assert hasattr(model, 'ceo_tower')
        assert hasattr(model, 'logit_scale')
    
    def test_model_forward_shape(self, sample_metadata):
        """Test that forward pass returns correct output shape."""
        config = Config()
        model = CEOFirmMatcher(sample_metadata, config)
        model.eval()
        
        batch_size = 32
        
        # Create dummy inputs
        f_numeric = torch.randn(batch_size, sample_metadata['n_firm_numeric'])
        f_cat = torch.randint(0, 2, (batch_size, len(sample_metadata['firm_cat_counts'])))
        c_numeric = torch.randn(batch_size, sample_metadata['n_ceo_numeric'])
        c_cat = torch.randint(0, 2, (batch_size, len(sample_metadata['ceo_cat_counts'])))
        
        with torch.no_grad():
            output = model(f_numeric, f_cat, c_numeric, c_cat)
        
        assert output.shape == (batch_size, 1)
    
    def test_model_gradients_flow(self, sample_metadata):
        """Test that gradients flow through the model."""
        config = Config()
        model = CEOFirmMatcher(sample_metadata, config)
        model.train()
        
        batch_size = 8
        
        # Create dummy inputs
        f_numeric = torch.randn(batch_size, sample_metadata['n_firm_numeric'])
        f_cat = torch.randint(0, 2, (batch_size, len(sample_metadata['firm_cat_counts'])))
        c_numeric = torch.randn(batch_size, sample_metadata['n_ceo_numeric'])
        c_cat = torch.randint(0, 2, (batch_size, len(sample_metadata['ceo_cat_counts'])))
        target = torch.randn(batch_size, 1)
        
        output = model(f_numeric, f_cat, c_numeric, c_cat)
        loss = ((output - target) ** 2).mean()
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_embedding_dimensions(self, sample_metadata):
        """Test that embeddings have correct dimensions."""
        config = Config()
        model = CEOFirmMatcher(sample_metadata, config)
        
        # Firm embeddings should match firm_cat_counts
        assert len(model.firm_embeddings) == len(sample_metadata['firm_cat_counts'])
        
        # CEO embeddings should match ceo_cat_counts
        assert len(model.ceo_embeddings) == len(sample_metadata['ceo_cat_counts'])
