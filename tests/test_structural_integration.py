"""
Integration tests for the Structural Distillation Network.
Tests the full pipeline from data loading through training and explainability.
"""
import pytest
import torch
import os
import tempfile
from torch.utils.data import DataLoader

from ceo_firm_matching import (
    StructuralConfig,
    StructuralDataProcessor,
    StructuralDistillationNet,
    train_structural_model,
    IlluminationEngine,
)


class TestStructuralIntegration:
    """Integration tests for the complete Structural Distillation pipeline."""
    
    @pytest.fixture
    def config(self):
        """Provides a StructuralConfig with synthetic data and minimal epochs."""
        cfg = StructuralConfig()
        cfg.DATA_PATH = "SYNTHETIC_FOR_TEST"
        cfg.EPOCHS = 2  # Minimal for testing
        cfg.BATCH_SIZE = 32
        # Use temporary output path
        cfg.OUTPUT_PATH = tempfile.mkdtemp()
        return cfg
    
    @pytest.fixture
    def pipeline_artifacts(self, config):
        """Runs the full pipeline and returns all artifacts."""
        processor = StructuralDataProcessor(config)
        train_ds, val_ds, val_df = processor.load_and_prep()
        
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        
        metadata = processor.get_metadata()
        
        return {
            'config': config,
            'processor': processor,
            'train_ds': train_ds,
            'val_ds': val_ds,
            'val_df': val_df,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'metadata': metadata,
        }
    
    def test_full_data_pipeline(self, pipeline_artifacts):
        """Test that full data pipeline produces valid outputs."""
        train_ds = pipeline_artifacts['train_ds']
        val_ds = pipeline_artifacts['val_ds']
        metadata = pipeline_artifacts['metadata']
        
        # Check datasets
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        
        # Check metadata
        assert metadata['n_firm_num'] > 0
        assert metadata['n_ceo_num'] > 0
    
    def test_model_training(self, pipeline_artifacts):
        """Test that model can be trained end-to-end."""
        config = pipeline_artifacts['config']
        train_loader = pipeline_artifacts['train_loader']
        val_loader = pipeline_artifacts['val_loader']
        metadata = pipeline_artifacts['metadata']
        
        # Train model
        model = train_structural_model(train_loader, val_loader, metadata, config)
        
        assert model is not None
        assert isinstance(model, StructuralDistillationNet)
        
        # Model should be on correct device
        assert next(model.parameters()).device.type == config.DEVICE.type
    
    def test_illumination_engine(self, pipeline_artifacts):
        """Test that IlluminationEngine works after training."""
        config = pipeline_artifacts['config']
        processor = pipeline_artifacts['processor']
        train_loader = pipeline_artifacts['train_loader']
        val_loader = pipeline_artifacts['val_loader']
        metadata = pipeline_artifacts['metadata']
        
        # Train model first
        model = train_structural_model(train_loader, val_loader, metadata, config)
        
        # Create illumination engine
        illuminator = IlluminationEngine(model, processor)
        
        # Test sensitivity analysis
        driver_df = illuminator.compute_global_sensitivity(val_loader)
        
        assert len(driver_df) > 0
        assert 'Feature' in driver_df.columns
        assert 'Sensitivity' in driver_df.columns
        assert 'Magnitude' in driver_df.columns
        assert 'Type' in driver_df.columns
    
    def test_type_distribution_analysis(self, pipeline_artifacts):
        """Test type distribution analysis."""
        config = pipeline_artifacts['config']
        processor = pipeline_artifacts['processor']
        train_loader = pipeline_artifacts['train_loader']
        val_loader = pipeline_artifacts['val_loader']
        metadata = pipeline_artifacts['metadata']
        
        model = train_structural_model(train_loader, val_loader, metadata, config)
        illuminator = IlluminationEngine(model, processor)
        
        type_df = illuminator.analyze_type_distributions(val_loader)
        
        assert len(type_df) == 10  # 5 CEO types + 5 Firm classes
        assert 'Type' in type_df.columns
        assert 'Mean Probability' in type_df.columns
    
    def test_visualization_outputs(self, pipeline_artifacts):
        """Test that visualization outputs are created."""
        config = pipeline_artifacts['config']
        processor = pipeline_artifacts['processor']
        train_loader = pipeline_artifacts['train_loader']
        val_loader = pipeline_artifacts['val_loader']
        metadata = pipeline_artifacts['metadata']
        
        model = train_structural_model(train_loader, val_loader, metadata, config)
        illuminator = IlluminationEngine(model, processor)
        
        # Generate plots
        matrix_path = illuminator.plot_interaction_matrix(save=True)
        driver_df = illuminator.compute_global_sensitivity(val_loader)
        driver_path = illuminator.plot_drivers(driver_df, save=True)
        
        # Check files exist
        assert matrix_path is not None
        assert os.path.exists(matrix_path)
        
        assert driver_path is not None
        assert os.path.exists(driver_path)
    
    def test_batch_processing(self, pipeline_artifacts):
        """Test that batches are processed correctly during training."""
        config = pipeline_artifacts['config']
        train_loader = pipeline_artifacts['train_loader']
        metadata = pipeline_artifacts['metadata']
        
        model = StructuralDistillationNet(metadata, config).to(config.DEVICE)
        model.train()
        
        # Process one batch
        batch = next(iter(train_loader))
        
        f_num = batch['firm_num'].to(config.DEVICE)
        f_cat = batch['firm_cat'].to(config.DEVICE)
        c_num = batch['ceo_num'].to(config.DEVICE)
        c_cat = batch['ceo_cat'].to(config.DEVICE)
        
        c_logits, f_logits, expected_match = model(f_num, f_cat, c_num, c_cat)
        
        # Check outputs are valid
        assert not torch.isnan(c_logits).any()
        assert not torch.isnan(f_logits).any()
        assert not torch.isnan(expected_match).any()
    
    def test_model_determinism_with_seed(self, config):
        """Test that model produces consistent results with fixed seed."""
        torch.manual_seed(42)
        
        processor1 = StructuralDataProcessor(config)
        train_ds1, _, _ = processor1.load_and_prep()
        
        torch.manual_seed(42)
        
        processor2 = StructuralDataProcessor(config)
        train_ds2, _, _ = processor2.load_and_prep()
        
        # First items should be identical
        item1 = train_ds1[0]
        item2 = train_ds2[0]
        
        assert torch.allclose(item1['firm_num'], item2['firm_num'])
        assert torch.allclose(item1['target_ceo'], item2['target_ceo'])


class TestStructuralEndToEnd:
    """End-to-end tests simulating real usage patterns."""
    
    def test_minimal_training_run(self):
        """Test a minimal training run completes without error."""
        config = StructuralConfig()
        config.DATA_PATH = "SYNTHETIC_E2E_TEST"
        config.EPOCHS = 1
        config.BATCH_SIZE = 16
        config.OUTPUT_PATH = tempfile.mkdtemp()
        
        # Run pipeline
        processor = StructuralDataProcessor(config)
        train_ds, val_ds, _ = processor.load_and_prep()
        
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        
        metadata = processor.get_metadata()
        model = train_structural_model(train_loader, val_loader, metadata, config)
        
        # Basic assertion that training completed
        assert model is not None
    
    def test_package_imports(self):
        """Test that all structural components can be imported from package."""
        from ceo_firm_matching import (
            StructuralConfig,
            StructuralDataProcessor,
            DistillationDataset,
            StructuralDistillationNet,
            train_structural_model,
            IlluminationEngine,
        )
        
        assert StructuralConfig is not None
        assert StructuralDataProcessor is not None
        assert DistillationDataset is not None
        assert StructuralDistillationNet is not None
        assert train_structural_model is not None
        assert IlluminationEngine is not None
