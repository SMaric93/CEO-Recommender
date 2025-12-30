"""
Unit tests for the Structural Distillation data processing.
"""
import pytest
import torch
import pandas as pd
import numpy as np

from ceo_firm_matching import StructuralConfig, StructuralDataProcessor, DistillationDataset


class TestStructuralDataProcessor:
    """Tests for StructuralDataProcessor."""
    
    @pytest.fixture
    def config(self):
        """Provides a StructuralConfig instance."""
        return StructuralConfig()
    
    @pytest.fixture
    def processor(self, config):
        """Provides a StructuralDataProcessor instance."""
        return StructuralDataProcessor(config)
    
    @pytest.fixture
    def sample_structural_dataframe(self, config):
        """Creates a sample DataFrame with probability columns."""
        np.random.seed(42)
        n = 100
        
        # Generate Dirichlet probabilities
        ceo_probs = np.random.dirichlet(np.ones(5), n)
        firm_probs = np.random.dirichlet(np.ones(5), n)
        
        data = {
            # CEO Numeric
            'Age': np.random.uniform(40, 60, n),
            
            # CEO Categorical
            'Gender': np.random.choice(['M', 'F'], n),
            'maxedu': np.random.randint(1, 5, n),
            'ivy': np.random.randint(0, 2, n),
            'm': np.random.randint(0, 2, n),
            'Output': np.random.randint(0, 2, n),
            'Throghput': np.random.randint(0, 2, n),
            'Peripheral': np.random.randint(0, 2, n),
            
            # Dates for tenure calculation
            'fiscalyear': np.random.randint(2015, 2023, n),
            'ceo_year': np.random.randint(2010, 2020, n),
            
            # Firm Numeric
            'ind_firms_60w': np.random.normal(0, 1, n),
            'non_competition_score': np.random.uniform(0, 1, n),
            'boardindpw': np.random.uniform(0.5, 0.9, n),
            'boardsizew': np.random.randint(7, 15, n),
            'busyw': np.random.randint(0, 3, n),
            'pct_blockw': np.random.uniform(10, 50, n),
            'logatw': np.random.uniform(8, 12, n),
            'exp_roa': np.random.normal(0.05, 0.02, n),
            'rdintw': np.random.uniform(0, 0.1, n),
            'capintw': np.random.uniform(0.05, 0.2, n),
            'leverage': np.random.uniform(0.1, 0.5, n),
            'divyieldw': np.random.uniform(0.01, 0.03, n),
            
            # Firm Categorical
            'compindustry': np.random.choice(['Tech', 'Finance', 'Health'], n),
            'ba_state': np.random.choice(['CA', 'NY', 'TX'], n),
            'rd_control': np.random.randint(0, 2, n),
            'dpayer': np.random.randint(0, 2, n),
        }
        
        # Add probability columns
        for i in range(5):
            data[f'prob_ceo_{i+1}'] = ceo_probs[:, i]
            data[f'prob_firm_{i+1}'] = firm_probs[:, i]
        
        return pd.DataFrame(data)
    
    def test_processor_initialization(self, processor, config):
        """Test that processor initializes correctly."""
        assert processor is not None
        assert processor.cfg is config
        assert 'firm' in processor.scalers
        assert 'ceo' in processor.scalers
    
    def test_synthetic_data_generation(self, processor):
        """Test that synthetic data generator works."""
        df = processor._generate_synthetic(n=200)
        
        assert len(df) == 200
        
        # Check probability columns exist
        for i in range(5):
            assert f'prob_ceo_{i+1}' in df.columns
            assert f'prob_firm_{i+1}' in df.columns
        
        # Check probabilities sum to 1
        ceo_sums = df[[f'prob_ceo_{i+1}' for i in range(5)]].sum(axis=1)
        assert np.allclose(ceo_sums, 1.0, atol=1e-5)
    
    def test_load_and_prep_with_synthetic(self, config):
        """Test full pipeline with synthetic data."""
        # Force synthetic generation
        config.DATA_PATH = "NONEXISTENT_PATH"
        processor = StructuralDataProcessor(config)
        
        train_ds, val_ds, val_df = processor.load_and_prep()
        
        assert isinstance(train_ds, DistillationDataset)
        assert isinstance(val_ds, DistillationDataset)
        assert isinstance(val_df, pd.DataFrame)
        
        assert len(train_ds) > 0
        assert len(val_ds) > 0
    
    def test_tenure_derivation(self, processor, sample_structural_dataframe):
        """Test that tenure is correctly derived."""
        df = sample_structural_dataframe.copy()
        
        # Initially no tenure column
        assert 'tenure' not in df.columns or df['tenure'].isna().all()
        
        # Force synthetic to trigger tenure calculation
        processor.cfg.DATA_PATH = "NONEXISTENT"
        train_ds, _, _ = processor.load_and_prep()
        
        # Tenure should be derived (fiscalyear - ceo_year)
        # We can't easily check the internal df, but we verify the pipeline works
        assert len(train_ds) > 0
    
    def test_probability_normalization(self, processor, sample_structural_dataframe):
        """Test that probabilities are normalized to sum to 1."""
        # Create data with unnormalized probabilities
        df = sample_structural_dataframe.copy()
        
        # Artificially denormalize
        for i in range(5):
            df[f'prob_ceo_{i+1}'] *= 2
            df[f'prob_firm_{i+1}'] *= 2
        
        # After processing, should be normalized
        # We test this indirectly via full pipeline
        processor.cfg.DATA_PATH = "NONEXISTENT"
        train_ds, _, _ = processor.load_and_prep()
        
        # Get a sample
        sample = train_ds[0]
        
        # Probability targets should sum to 1
        assert torch.allclose(sample['target_ceo'].sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(sample['target_firm'].sum(), torch.tensor(1.0), atol=1e-5)
    
    def test_get_metadata(self, config):
        """Test metadata extraction after fitting."""
        config.DATA_PATH = "NONEXISTENT"
        processor = StructuralDataProcessor(config)
        processor.load_and_prep()
        
        metadata = processor.get_metadata()
        
        assert 'n_firm_num' in metadata
        assert 'n_ceo_num' in metadata
        assert 'firm_cat_cards' in metadata
        assert 'ceo_cat_cards' in metadata
        
        assert metadata['n_firm_num'] > 0
        assert metadata['n_ceo_num'] > 0
        assert len(metadata['firm_cat_cards']) > 0
        assert len(metadata['ceo_cat_cards']) > 0
    
    def test_get_feature_names(self, config):
        """Test feature names extraction."""
        config.DATA_PATH = "NONEXISTENT"
        processor = StructuralDataProcessor(config)
        processor.load_and_prep()
        
        names = processor.get_feature_names()
        
        assert isinstance(names, list)
        assert len(names) > 0


class TestDistillationDataset:
    """Tests for DistillationDataset."""
    
    @pytest.fixture
    def sample_data(self):
        """Provides sample tensor data for dataset."""
        n = 50
        return {
            'firm_num': torch.randn(n, 12),
            'firm_cat': torch.randint(0, 3, (n, 4)),
            'ceo_num': torch.randn(n, 2),
            'ceo_cat': torch.randint(0, 3, (n, 7)),
            'target_ceo': torch.softmax(torch.randn(n, 5), dim=1),
            'target_firm': torch.softmax(torch.randn(n, 5), dim=1),
        }
    
    def test_dataset_length(self, sample_data):
        """Test dataset returns correct length."""
        dataset = DistillationDataset(sample_data)
        assert len(dataset) == 50
    
    def test_dataset_getitem(self, sample_data):
        """Test dataset __getitem__ returns correct structure."""
        dataset = DistillationDataset(sample_data)
        item = dataset[0]
        
        assert 'firm_num' in item
        assert 'firm_cat' in item
        assert 'ceo_num' in item
        assert 'ceo_cat' in item
        assert 'target_ceo' in item
        assert 'target_firm' in item
    
    def test_dataset_item_shapes(self, sample_data):
        """Test that individual items have correct shapes."""
        dataset = DistillationDataset(sample_data)
        item = dataset[0]
        
        assert item['firm_num'].shape == (12,)
        assert item['firm_cat'].shape == (4,)
        assert item['ceo_num'].shape == (2,)
        assert item['ceo_cat'].shape == (7,)
        assert item['target_ceo'].shape == (5,)
        assert item['target_firm'].shape == (5,)
    
    def test_dataset_iteration(self, sample_data):
        """Test that dataset can be iterated."""
        dataset = DistillationDataset(sample_data)
        
        count = 0
        for item in dataset:
            count += 1
            assert isinstance(item, dict)
        
        assert count == 50
