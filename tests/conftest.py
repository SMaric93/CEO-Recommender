"""
Pytest configuration and fixtures for CEO-Firm Matching tests.
"""
import pytest
import pandas as pd
import numpy as np

from ceo_firm_matching import Config, DataProcessor
from ceo_firm_matching import StructuralConfig, StructuralDataProcessor


# =============================================================================
# Two Tower Model Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Provides a Config instance."""
    return Config()


@pytest.fixture
def sample_dataframe(config):
    """Creates a sample DataFrame matching the required schema."""
    np.random.seed(42)
    n = 100
    
    data = {
        # IDs
        'gvkey': np.random.randint(1000, 9999, n),
        'match_exec_id': np.random.randint(10000, 99999, n),
        
        # CEO Numeric
        'Age': np.random.uniform(40, 60, n),
        'Output': np.random.randint(0, 2, n),
        'Throghput': np.random.randint(0, 2, n),
        'Peripheral': np.random.randint(0, 2, n),
        
        # CEO Categorical
        'Gender': np.random.choice(['M', 'F'], n),
        'maxedu': np.random.randint(1, 5, n),
        'ivy': np.random.randint(0, 2, n),
        'm': np.random.randint(0, 2, n),
        
        # CEO Dates/Years
        'ceo_year': np.random.randint(2010, 2020, n),
        'dep_baby_ceo': np.random.randint(0, 2, n),
        
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
        'fiscalyear': np.random.randint(2015, 2023, n),
        
        # Targets
        'match_means': np.random.normal(0, 1, n),
        'sd_match_means': np.random.uniform(0.2, 0.8, n),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def processor(config):
    """Provides a DataProcessor instance."""
    return DataProcessor(config)


@pytest.fixture
def fitted_processor(processor, sample_dataframe):
    """Provides a fitted DataProcessor with sample data."""
    df_clean = processor.prepare_features(sample_dataframe)
    processor.fit(df_clean)
    return processor


# =============================================================================
# Structural Distillation Network Fixtures
# =============================================================================

@pytest.fixture
def structural_config():
    """Provides a StructuralConfig instance for testing."""
    cfg = StructuralConfig()
    cfg.DATA_PATH = "SYNTHETIC_TEST"  # Force synthetic data
    cfg.EPOCHS = 2  # Minimal for testing
    cfg.BATCH_SIZE = 32
    return cfg


@pytest.fixture
def structural_sample_dataframe():
    """Creates a sample DataFrame with BLM probability columns."""
    np.random.seed(42)
    n = 100
    
    # Generate Dirichlet probabilities (sum to 1)
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
    
    # Add BLM probability columns
    for i in range(5):
        data[f'prob_ceo_{i+1}'] = ceo_probs[:, i]
        data[f'prob_firm_{i+1}'] = firm_probs[:, i]
    
    return pd.DataFrame(data)


@pytest.fixture
def structural_processor(structural_config):
    """Provides a StructuralDataProcessor instance."""
    return StructuralDataProcessor(structural_config)


@pytest.fixture
def fitted_structural_processor(structural_config):
    """Provides a fitted StructuralDataProcessor with synthetic data."""
    processor = StructuralDataProcessor(structural_config)
    processor.load_and_prep()  # This fits the processor
    return processor


@pytest.fixture
def structural_metadata():
    """Provides sample metadata for structural model initialization."""
    return {
        'n_firm_num': 12,
        'n_ceo_num': 2,
        'firm_cat_cards': [4, 4, 2, 2],  # 4 categorical features
        'ceo_cat_cards': [2, 4, 2, 2, 2, 2, 2],  # 7 categorical features
    }

