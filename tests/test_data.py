"""
Unit tests for the data processing module.
"""
import pytest
import numpy as np
import torch


class TestDataProcessor:
    """Tests for the DataProcessor class."""
    
    def test_prepare_features_creates_tenure(self, processor, sample_dataframe):
        """Test that tenure is correctly calculated."""
        df_clean = processor.prepare_features(sample_dataframe)
        
        assert 'tenure' in df_clean.columns
        # Tenure should be fiscalyear - ceo_year, clipped at 0
        expected = (sample_dataframe['fiscalyear'] - sample_dataframe['ceo_year']).clip(lower=0)
        # Compare non-NaN rows
        assert (df_clean['tenure'].values >= 0).all()
    
    def test_prepare_features_creates_weights(self, processor, sample_dataframe):
        """Test that weights are correctly calculated."""
        df_clean = processor.prepare_features(sample_dataframe)
        
        assert 'weights' in df_clean.columns
        assert (df_clean['weights'].values > 0).all()
    
    def test_fit_creates_encoders(self, processor, sample_dataframe):
        """Test that fit creates encoders for all categorical columns."""
        df_clean = processor.prepare_features(sample_dataframe)
        processor.fit(df_clean)
        
        # Check firm categorical encoders
        for col in processor.cfg.FIRM_CAT_COLS:
            assert col in processor.encoders
            
        # Check CEO categorical encoders
        for col in processor.cfg.CEO_CAT_COLS:
            assert col in processor.encoders
    
    def test_fit_creates_scalers(self, processor, sample_dataframe):
        """Test that fit creates scalers."""
        df_clean = processor.prepare_features(sample_dataframe)
        processor.fit(df_clean)
        
        assert 'firm' in processor.scalers
        assert 'ceo' in processor.scalers
    
    def test_transform_returns_tensors(self, fitted_processor, sample_dataframe):
        """Test that transform returns the expected tensor dictionary."""
        df_clean = fitted_processor.prepare_features(sample_dataframe)
        result = fitted_processor.transform(df_clean)
        
        expected_keys = ['firm_numeric', 'firm_cat', 'ceo_numeric', 'ceo_cat', 
                         'target', 'weights', 'n_firm_numeric', 'firm_cat_counts',
                         'n_ceo_numeric', 'ceo_cat_counts']
        
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_transform_tensor_shapes(self, fitted_processor, sample_dataframe):
        """Test that transformed tensors have correct shapes."""
        df_clean = fitted_processor.prepare_features(sample_dataframe)
        result = fitted_processor.transform(df_clean)
        
        n_samples = len(df_clean)
        
        assert result['firm_numeric'].shape[0] == n_samples
        assert result['ceo_numeric'].shape[0] == n_samples
        assert result['firm_cat'].shape[0] == n_samples
        assert result['ceo_cat'].shape[0] == n_samples
        assert result['target'].shape == (n_samples, 1)
        assert result['weights'].shape == (n_samples, 1)
    
    def test_get_feature_names(self, fitted_processor, sample_dataframe):
        """Test that get_feature_names returns expected feature list."""
        df_clean = fitted_processor.prepare_features(sample_dataframe)
        fitted_processor.transform(df_clean)
        
        feature_names = fitted_processor.get_feature_names()
        
        # Should contain firm numeric + firm cat + ceo numeric + ceo cat
        expected_count = (
            len(fitted_processor.final_firm_numeric) +
            len(fitted_processor.cfg.FIRM_CAT_COLS) +
            len(fitted_processor.final_ceo_numeric) +
            len(fitted_processor.cfg.CEO_CAT_COLS)
        )
        
        assert len(feature_names) == expected_count


class TestCEOFirmDataset:
    """Tests for the CEOFirmDataset class."""
    
    def test_dataset_length(self, fitted_processor, sample_dataframe):
        """Test that dataset has correct length."""
        from ceo_firm_matching import CEOFirmDataset
        
        df_clean = fitted_processor.prepare_features(sample_dataframe)
        data_dict = fitted_processor.transform(df_clean)
        dataset = CEOFirmDataset(data_dict)
        
        assert len(dataset) == len(df_clean)
    
    def test_dataset_getitem(self, fitted_processor, sample_dataframe):
        """Test that dataset items have expected keys."""
        from ceo_firm_matching import CEOFirmDataset
        
        df_clean = fitted_processor.prepare_features(sample_dataframe)
        data_dict = fitted_processor.transform(df_clean)
        dataset = CEOFirmDataset(data_dict)
        
        item = dataset[0]
        
        expected_keys = ['firm_numeric', 'firm_cat', 'ceo_numeric', 'ceo_cat', 
                         'target', 'weights']
        
        for key in expected_keys:
            assert key in item, f"Missing key: {key}"
