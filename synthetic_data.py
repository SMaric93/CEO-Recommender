import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=1000):
    """
    Generates a synthetic DataFrame matching the schema required by the Two Towers model.
    """
    np.random.seed(42)
    
    data = {
        # IDs
        'gvkey': np.random.randint(1000, 9999, n_samples),
        'match_exec_id': np.random.randint(10000, 99999, n_samples),
        
        # CEO Numeric
        'Age': np.random.uniform(30, 70, n_samples),
        'Output': np.random.normal(0, 1, n_samples),
        'Throghput': np.random.normal(0, 1, n_samples), # Keeping original spelling
        'Peripheral': np.random.normal(0, 1, n_samples),
        
        # CEO Categorical
        'Gender': np.random.choice(['M', 'F'], n_samples),
        'maxedu': np.random.randint(1, 5, n_samples),
        'ivy': np.random.randint(0, 2, n_samples),
        
        # CEO Dates/Years
        'ceo_year': np.random.randint(2000, 2023, n_samples),
        'year_born': np.random.randint(1950, 1990, n_samples),
        'dep_baby_ceo': np.random.randint(0, 2, n_samples),
        'DOB': pd.date_range(start='1950-01-01', periods=n_samples).strftime('%Y-%m-%d'),
        
        # Firm Numeric
        'ind_firms_60': np.random.normal(0, 1, n_samples),
        'non_competition_score': np.random.uniform(0, 1, n_samples),
        'boardindpw': np.random.uniform(0, 1, n_samples),
        'boardsizew': np.random.randint(5, 20, n_samples),
        'busyw': np.random.randint(0, 5, n_samples),
        'pct_blockw': np.random.uniform(0, 100, n_samples),
        'logat': np.random.uniform(5, 15, n_samples),
        'exp_roa': np.random.normal(0.05, 0.02, n_samples),
        
        # Firm Categorical
        'compindustry': np.random.choice(['Tech', 'Finance', 'Health', 'Energy'], n_samples),
        'fiscalyear': np.random.randint(2000, 2023, n_samples),
        
        # Targets
        'match_means': np.random.normal(0, 1, n_samples),
        'sd_match_means': np.random.uniform(0.1, 1.0, n_samples)
    }
    
    return pd.DataFrame(data)
