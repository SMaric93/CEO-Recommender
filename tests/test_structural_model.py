"""
Unit tests for the Structural Distillation Network model architecture.
"""
import pytest
import torch
import torch.nn.functional as F

from ceo_firm_matching import StructuralDistillationNet, StructuralConfig


class TestStructuralDistillationNet:
    """Tests for the StructuralDistillationNet model."""
    
    @pytest.fixture
    def sample_metadata(self):
        """Provides sample metadata for model initialization."""
        return {
            'n_firm_num': 12,
            'n_ceo_num': 2,
            'firm_cat_cards': [4, 4, 2, 2],  # 4 categorical features
            'ceo_cat_cards': [2, 4, 2, 2, 2, 2, 2],  # 7 categorical features
        }
    
    @pytest.fixture
    def config(self):
        """Provides a StructuralConfig instance."""
        return StructuralConfig()
    
    @pytest.fixture
    def model(self, sample_metadata, config):
        """Provides an initialized model."""
        return StructuralDistillationNet(sample_metadata, config)
    
    def test_model_initialization(self, sample_metadata, config):
        """Test that model initializes without error."""
        model = StructuralDistillationNet(sample_metadata, config)
        
        assert model is not None
        assert hasattr(model, 'firm_tower')
        assert hasattr(model, 'ceo_tower')
        assert hasattr(model, 'A')
    
    def test_interaction_matrix_frozen(self, model):
        """Test that the interaction matrix A is frozen (not trainable)."""
        # A should be a buffer, not a parameter
        assert 'A' not in dict(model.named_parameters())
        
        # A should be in buffers
        buffer_names = [name for name, _ in model.named_buffers()]
        assert 'A' in buffer_names
        
        # A should not require grad
        assert not model.A.requires_grad
    
    def test_interaction_matrix_shape(self, model):
        """Test that interaction matrix has correct shape."""
        assert model.A.shape == (5, 5)
    
    def test_forward_output_shapes(self, model, sample_metadata):
        """Test that forward pass returns correct output shapes."""
        model.eval()
        batch_size = 32
        
        # Create dummy inputs
        f_num = torch.randn(batch_size, sample_metadata['n_firm_num'])
        f_cat = torch.randint(0, 2, (batch_size, len(sample_metadata['firm_cat_cards'])))
        c_num = torch.randn(batch_size, sample_metadata['n_ceo_num'])
        c_cat = torch.randint(0, 2, (batch_size, len(sample_metadata['ceo_cat_cards'])))
        
        with torch.no_grad():
            c_logits, f_logits, expected_match = model(f_num, f_cat, c_num, c_cat)
        
        # Check shapes
        assert c_logits.shape == (batch_size, 5), "CEO logits should be (batch, 5)"
        assert f_logits.shape == (batch_size, 5), "Firm logits should be (batch, 5)"
        assert expected_match.shape == (batch_size, 1), "Expected match should be (batch, 1)"
    
    def test_logits_produce_valid_probabilities(self, model, sample_metadata):
        """Test that logits can be converted to valid probability distributions."""
        model.eval()
        batch_size = 16
        
        f_num = torch.randn(batch_size, sample_metadata['n_firm_num'])
        f_cat = torch.randint(0, 2, (batch_size, len(sample_metadata['firm_cat_cards'])))
        c_num = torch.randn(batch_size, sample_metadata['n_ceo_num'])
        c_cat = torch.randint(0, 2, (batch_size, len(sample_metadata['ceo_cat_cards'])))
        
        with torch.no_grad():
            c_logits, f_logits, _ = model(f_num, f_cat, c_num, c_cat)
        
        # Convert to probabilities
        c_probs = F.softmax(c_logits, dim=1)
        f_probs = F.softmax(f_logits, dim=1)
        
        # Check they sum to 1
        assert torch.allclose(c_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        assert torch.allclose(f_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        
        # Check all values are in [0, 1]
        assert (c_probs >= 0).all() and (c_probs <= 1).all()
        assert (f_probs >= 0).all() and (f_probs <= 1).all()
    
    def test_gradients_flow_to_towers(self, model, sample_metadata):
        """Test that gradients flow through the model to tower parameters."""
        model.train()
        batch_size = 8
        
        f_num = torch.randn(batch_size, sample_metadata['n_firm_num'])
        f_cat = torch.randint(0, 2, (batch_size, len(sample_metadata['firm_cat_cards'])))
        c_num = torch.randn(batch_size, sample_metadata['n_ceo_num'])
        c_cat = torch.randint(0, 2, (batch_size, len(sample_metadata['ceo_cat_cards'])))
        
        target_ceo = F.softmax(torch.randn(batch_size, 5), dim=1)
        target_firm = F.softmax(torch.randn(batch_size, 5), dim=1)
        
        c_logits, f_logits, _ = model(f_num, f_cat, c_num, c_cat)
        
        # Compute KL divergence loss
        loss = (
            F.kl_div(F.log_softmax(c_logits, dim=1), target_ceo, reduction='batchmean') +
            F.kl_div(F.log_softmax(f_logits, dim=1), target_firm, reduction='batchmean')
        )
        loss.backward()
        
        # Check gradients exist for trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "Should have trainable parameters"
        
        for param in trainable_params:
            assert param.grad is not None, "Gradient should exist"
    
    def test_embedding_counts(self, model, sample_metadata):
        """Test that embeddings have correct counts."""
        assert len(model.firm_embeddings) == len(sample_metadata['firm_cat_cards'])
        assert len(model.ceo_embeddings) == len(sample_metadata['ceo_cat_cards'])
    
    def test_get_type_probabilities(self, model, sample_metadata):
        """Test the get_type_probabilities helper method."""
        model.eval()
        batch_size = 16
        
        f_num = torch.randn(batch_size, sample_metadata['n_firm_num'])
        f_cat = torch.randint(0, 2, (batch_size, len(sample_metadata['firm_cat_cards'])))
        c_num = torch.randn(batch_size, sample_metadata['n_ceo_num'])
        c_cat = torch.randint(0, 2, (batch_size, len(sample_metadata['ceo_cat_cards'])))
        
        with torch.no_grad():
            ceo_probs, firm_probs = model.get_type_probabilities(f_num, f_cat, c_num, c_cat)
        
        # Check shapes
        assert ceo_probs.shape == (batch_size, 5)
        assert firm_probs.shape == (batch_size, 5)
        
        # Check valid probabilities
        assert torch.allclose(ceo_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        assert torch.allclose(firm_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)
    
    def test_interaction_computes_expected_match(self, model, sample_metadata):
        """Test that expected match is computed correctly via bilinear form."""
        model.eval()
        batch_size = 4
        
        f_num = torch.randn(batch_size, sample_metadata['n_firm_num'])
        f_cat = torch.randint(0, 2, (batch_size, len(sample_metadata['firm_cat_cards'])))
        c_num = torch.randn(batch_size, sample_metadata['n_ceo_num'])
        c_cat = torch.randint(0, 2, (batch_size, len(sample_metadata['ceo_cat_cards'])))
        
        with torch.no_grad():
            c_logits, f_logits, expected_match = model(f_num, f_cat, c_num, c_cat)
            
            # Manually compute expected match
            pi_ceo = F.softmax(c_logits, dim=1)
            q_firm = F.softmax(f_logits, dim=1)
            weighted_A = torch.matmul(pi_ceo, model.A)
            manual_match = torch.sum(weighted_A * q_firm, dim=1, keepdim=True)
        
        # Should match the model output
        assert torch.allclose(expected_match, manual_match, atol=1e-5)
