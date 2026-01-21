"""
Analysis Diagnostic Functions.

Common analysis patterns for CEO-Firm matching diagnostics:
- Descriptive statistics by match quintile
- OLS regression analysis
- Random Forest feature importance
- Stratified analysis utilities
"""
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


def run_descriptive_analysis(
    df: pd.DataFrame,
    match_col: str = 'match_means',
    numeric_cols: Optional[List[str]] = None,
    n_quantiles: int = 5
) -> pd.DataFrame:
    """
    Descriptive statistics stratified by match quality quintiles.

    Args:
        df: DataFrame with match quality and features
        match_col: Column name for match quality
        numeric_cols: Columns to analyze (auto-detected if None)
        n_quantiles: Number of quantiles for stratification

    Returns:
        DataFrame with means by quintile
    """
    print(f"📊 Descriptive Analysis by {match_col} Quintile...")

    if match_col not in df.columns:
        print(f"   ✗ {match_col} not found in data")
        return pd.DataFrame()

    # Create quintile column
    df = df.copy()
    df['match_quintile'] = pd.qcut(
        df[match_col],
        q=n_quantiles,
        labels=[f'Q{i+1}' for i in range(n_quantiles)],
        duplicates='drop'
    )

    # Auto-detect numeric columns if not provided
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in [match_col, 'match_quintile']]

    # Calculate means by quintile
    stats = df.groupby('match_quintile')[numeric_cols].mean().T
    stats['Q5-Q1'] = stats['Q5'] - stats['Q1']

    print(f"   ✓ Analyzed {len(numeric_cols)} variables across {n_quantiles} quintiles")
    return stats


def run_regression_analysis(
    df: pd.DataFrame,
    y_col: str = 'match_means',
    x_cols: Optional[List[str]] = None,
    controls: Optional[List[str]] = None,
    fe_cols: Optional[List[str]] = None,
    cluster_col: Optional[str] = None
) -> dict:
    """
    OLS regression analysis with optional fixed effects.

    Args:
        df: DataFrame with outcome and predictors
        y_col: Dependent variable column
        x_cols: Predictor columns
        controls: Control variable columns
        fe_cols: Fixed effect columns (creates dummies)
        cluster_col: Column for clustered standard errors

    Returns:
        Dict with regression results and coefficients
    """
    if not STATSMODELS_AVAILABLE:
        print("   ⚠️ statsmodels not available")
        return {}

    print("📈 OLS Regression Analysis...")

    if y_col not in df.columns:
        print(f"   ✗ {y_col} not found")
        return {}

    df = df.copy().dropna(subset=[y_col])

    # Build regressor list
    all_x = []
    if x_cols:
        all_x.extend(x_cols)
    if controls:
        all_x.extend(controls)

    # Filter to available columns
    all_x = [c for c in all_x if c in df.columns]

    if not all_x:
        print("   ✗ No valid predictors")
        return {}

    # Add fixed effects as dummies
    if fe_cols:
        for fe in fe_cols:
            if fe in df.columns:
                dummies = pd.get_dummies(df[fe], prefix=fe, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                all_x.extend(dummies.columns.tolist())

    # Prepare data
    df = df.dropna(subset=all_x)
    X = sm.add_constant(df[all_x])
    y = df[y_col]

    # Fit model
    model = sm.OLS(y, X).fit()

    # Extract results
    results = {
        'n_obs': int(model.nobs),
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'coefficients': model.params.to_dict(),
        'std_errors': model.bse.to_dict(),
        't_stats': model.tvalues.to_dict(),
        'p_values': model.pvalues.to_dict(),
    }

    print(f"   ✓ N={results['n_obs']:,}, R²={results['r_squared']:.4f}")
    return results


def run_random_forest(
    df: pd.DataFrame,
    y_col: str = 'match_means',
    x_cols: Optional[List[str]] = None,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42
) -> Tuple[Optional[object], pd.DataFrame]:
    """
    Random Forest feature importance analysis.

    Args:
        df: DataFrame with outcome and features
        y_col: Target variable column
        x_cols: Feature columns (auto-detected if None)
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed

    Returns:
        Tuple of (fitted model, feature importance DataFrame)
    """
    if not SKLEARN_AVAILABLE:
        print("   ⚠️ scikit-learn not available")
        return None, pd.DataFrame()

    print("🌲 Random Forest Feature Importance...")

    if y_col not in df.columns:
        print(f"   ✗ {y_col} not found")
        return None, pd.DataFrame()

    df = df.copy()

    # Auto-detect numeric columns if not provided
    if x_cols is None:
        x_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        x_cols = [c for c in x_cols if c != y_col]

    # Filter available columns
    x_cols = [c for c in x_cols if c in df.columns]

    if not x_cols:
        print("   ✗ No valid features")
        return None, pd.DataFrame()

    # Prepare data
    df = df.dropna(subset=[y_col] + x_cols)
    X = df[x_cols]
    y = df[y_col]

    if len(df) < 50:
        print(f"   ⚠️ Too few observations ({len(df)})")
        return None, pd.DataFrame()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Fit model
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Feature importance
    importance = pd.DataFrame({
        'feature': x_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # Calculate R² on test set
    r2 = rf.score(X_test, y_test)

    print(f"   ✓ {len(x_cols)} features, Test R²={r2:.4f}")
    print(f"   Top 5: {importance.head()['feature'].tolist()}")

    return rf, importance


def stratified_analysis(
    df: pd.DataFrame,
    stratify_col: str,
    analysis_func: callable,
    n_quantiles: int = 5,
    **kwargs
) -> dict:
    """
    Run an analysis function stratified by quantiles of a column.

    Args:
        df: DataFrame to analyze
        stratify_col: Column to stratify by
        analysis_func: Function to apply to each stratum
        n_quantiles: Number of quantiles
        **kwargs: Additional arguments passed to analysis_func

    Returns:
        Dict mapping quantile label to analysis results
    """
    print(f"📊 Stratified Analysis by {stratify_col}...")

    if stratify_col not in df.columns:
        print(f"   ✗ {stratify_col} not found")
        return {}

    df = df.copy()
    df['_stratum'] = pd.qcut(
        df[stratify_col],
        q=n_quantiles,
        labels=[f'Q{i+1}' for i in range(n_quantiles)],
        duplicates='drop'
    )

    results = {}
    for q in df['_stratum'].unique():
        stratum_df = df[df['_stratum'] == q]
        results[str(q)] = analysis_func(stratum_df, **kwargs)

    print(f"   ✓ Completed {len(results)} strata")
    return results
