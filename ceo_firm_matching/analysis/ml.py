"""
Machine Learning Analysis Module.

Provides ML-based match quality analysis with robustness:
- Random Forest with feature importance
- Gradient Boosting (XGBoost/LightGBM compatible)
- SHAP value explanations
- Cross-validation framework
- Hyperparameter sensitivity analysis
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


@dataclass
class MLResult:
    """Container for ML model results."""
    model_name: str
    n_obs: int
    n_features: int
    train_r2: float
    test_r2: float
    cv_r2_mean: float
    cv_r2_std: float
    rmse: float
    mae: float
    feature_importance: pd.DataFrame
    model: Any = None
    shap_values: Any = None


class MLAnalyzer:
    """
    ML analysis with cross-validation and SHAP explanations.
    
    Example usage:
        analyzer = MLAnalyzer(df)
        results = analyzer.run_all()
        analyzer.plot_feature_importance(results['RandomForest'])
    """
    
    # Default feature groups
    CEO_FEATURES = ['Age', 'tenure', 'Output', 'Throghput', 'Peripheral', 'maxedu', 'ivy', 'm']
    FIRM_FEATURES = ['logatw', 'exp_roa', 'rdintw', 'capintw', 'leverage', 'boardindpw',
                     'boardsizew', 'busyw', 'pct_blockw', 'divyieldw']
    
    def __init__(
        self,
        df: pd.DataFrame,
        y_col: str = 'match_means',
        x_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize ML analyzer.
        
        Args:
            df: DataFrame with features and target
            y_col: Target column name
            x_cols: Feature columns (auto-detect if None)
            test_size: Fraction for test set
            random_state: Random seed
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required. Install: pip install scikit-learn")
        
        self.df = df.copy()
        self.y_col = y_col
        self.test_size = test_size
        self.random_state = random_state
        self.results: Dict[str, MLResult] = {}
        
        # Compute tenure if needed
        if 'tenure' not in self.df.columns and all(c in self.df.columns for c in ['fiscalyear', 'ceo_year']):
            self.df['tenure'] = self.df['fiscalyear'] - self.df['ceo_year']
        
        # Set feature columns
        if x_cols is None:
            self.x_cols = [c for c in self.CEO_FEATURES + self.FIRM_FEATURES if c in self.df.columns]
        else:
            self.x_cols = [c for c in x_cols if c in self.df.columns]
        
        # Prepare train/test split
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare train/test splits."""
        from sklearn.model_selection import train_test_split
        
        df = self.df.dropna(subset=[self.y_col] + self.x_cols)
        
        X = df[self.x_cols]
        y = df[self.y_col]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Scale features for regularized models
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"📊 ML Data: {len(self.X_train)} train, {len(self.X_test)} test, {len(self.x_cols)} features")
    
    def _evaluate_model(
        self,
        model,
        model_name: str,
        use_scaled: bool = False,
        compute_shap: bool = True
    ) -> MLResult:
        """Evaluate a fitted model with CV and metrics."""
        X_train = self.X_train_scaled if use_scaled else self.X_train
        X_test = self.X_test_scaled if use_scaled else self.X_test
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        mae = mean_absolute_error(self.y_test, y_test_pred)
        
        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        X_full = np.vstack([X_train, X_test])
        y_full = pd.concat([self.y_train, self.y_test])
        cv_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring='r2')
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.x_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(model, 'coef_'):
            importance = pd.DataFrame({
                'feature': self.x_cols,
                'importance': np.abs(model.coef_)
            }).sort_values('importance', ascending=False)
        else:
            importance = pd.DataFrame()
        
        # SHAP values
        shap_values = None
        if compute_shap and SHAP_AVAILABLE and not use_scaled:
            try:
                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(self.X_test)
            except Exception:
                pass
        
        return MLResult(
            model_name=model_name,
            n_obs=len(self.X_train) + len(self.X_test),
            n_features=len(self.x_cols),
            train_r2=train_r2,
            test_r2=test_r2,
            cv_r2_mean=cv_scores.mean(),
            cv_r2_std=cv_scores.std(),
            rmse=rmse,
            mae=mae,
            feature_importance=importance,
            model=model,
            shap_values=shap_values
        )
    
    def run_random_forest(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        **kwargs
    ) -> MLResult:
        """Train and evaluate Random Forest."""
        print("🌲 Training Random Forest...")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            **kwargs
        )
        model.fit(self.X_train, self.y_train)
        
        result = self._evaluate_model(model, "RandomForest")
        self.results["RandomForest"] = result
        
        print(f"   ✓ Test R²={result.test_r2:.4f}, CV R²={result.cv_r2_mean:.4f}±{result.cv_r2_std:.3f}")
        return result
    
    def run_gradient_boosting(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        **kwargs
    ) -> MLResult:
        """Train and evaluate Gradient Boosting."""
        print("🚀 Training Gradient Boosting...")
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.random_state,
            **kwargs
        )
        model.fit(self.X_train, self.y_train)
        
        result = self._evaluate_model(model, "GradientBoosting")
        self.results["GradientBoosting"] = result
        
        print(f"   ✓ Test R²={result.test_r2:.4f}, CV R²={result.cv_r2_mean:.4f}±{result.cv_r2_std:.3f}")
        return result
    
    def run_ridge(self, alpha: float = 1.0) -> MLResult:
        """Train and evaluate Ridge regression."""
        print("📏 Training Ridge Regression...")
        
        model = Ridge(alpha=alpha, random_state=self.random_state)
        model.fit(self.X_train_scaled, self.y_train)
        
        result = self._evaluate_model(model, "Ridge", use_scaled=True, compute_shap=False)
        self.results["Ridge"] = result
        
        print(f"   ✓ Test R²={result.test_r2:.4f}, CV R²={result.cv_r2_mean:.4f}±{result.cv_r2_std:.3f}")
        return result
    
    def run_lasso(self, alpha: float = 0.1) -> MLResult:
        """Train and evaluate Lasso regression."""
        print("✂️ Training Lasso Regression...")
        
        model = Lasso(alpha=alpha, random_state=self.random_state, max_iter=10000)
        model.fit(self.X_train_scaled, self.y_train)
        
        result = self._evaluate_model(model, "Lasso", use_scaled=True, compute_shap=False)
        self.results["Lasso"] = result
        
        print(f"   ✓ Test R²={result.test_r2:.4f}, CV R²={result.cv_r2_mean:.4f}±{result.cv_r2_std:.3f}")
        return result
    
    def run_all(self) -> Dict[str, MLResult]:
        """Run all ML models."""
        self.run_random_forest()
        self.run_gradient_boosting()
        self.run_ridge()
        self.run_lasso()
        return self.results
    
    def run_hyperparameter_robustness(self) -> pd.DataFrame:
        """Run hyperparameter sensitivity analysis."""
        print("🔧 Hyperparameter Robustness Analysis...")
        
        results = []
        
        # RF depth sensitivity
        for depth in [5, 10, 15, 20, None]:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=depth,
                random_state=self.random_state, n_jobs=-1
            )
            model.fit(self.X_train, self.y_train)
            test_r2 = r2_score(self.y_test, model.predict(self.X_test))
            results.append({
                'model': 'RF',
                'param': 'max_depth',
                'value': str(depth),
                'test_r2': test_r2
            })
        
        # RF n_estimators sensitivity
        for n_est in [50, 100, 200, 500]:
            model = RandomForestRegressor(
                n_estimators=n_est, max_depth=10,
                random_state=self.random_state, n_jobs=-1
            )
            model.fit(self.X_train, self.y_train)
            test_r2 = r2_score(self.y_test, model.predict(self.X_test))
            results.append({
                'model': 'RF',
                'param': 'n_estimators',
                'value': str(n_est),
                'test_r2': test_r2
            })
        
        # GBM learning rate sensitivity
        for lr in [0.01, 0.05, 0.1, 0.2]:
            model = GradientBoostingRegressor(
                n_estimators=100, learning_rate=lr,
                random_state=self.random_state
            )
            model.fit(self.X_train, self.y_train)
            test_r2 = r2_score(self.y_test, model.predict(self.X_test))
            results.append({
                'model': 'GBM',
                'param': 'learning_rate',
                'value': str(lr),
                'test_r2': test_r2
            })
        
        df = pd.DataFrame(results)
        print(f"   ✓ Tested {len(results)} configurations")
        return df
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Generate model comparison table."""
        rows = []
        for name, r in self.results.items():
            rows.append({
                'Model': name,
                'N': r.n_obs,
                'Features': r.n_features,
                'Train R²': f"{r.train_r2:.4f}",
                'Test R²': f"{r.test_r2:.4f}",
                'CV R² (mean±std)': f"{r.cv_r2_mean:.4f}±{r.cv_r2_std:.3f}",
                'RMSE': f"{r.rmse:.4f}",
                'MAE': f"{r.mae:.4f}",
            })
        return pd.DataFrame(rows)
    
    def to_latex(self, title: str = "ML Model Comparison", label: str = "tab:ml") -> str:
        """Generate LaTeX comparison table."""
        df = self.get_comparison_table()
        
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{title}}}",
            f"\\label{{{label}}}",
            r"\begin{tabular}{lcccccc}",
            r"\hline\hline",
            r"Model & N & Train $R^2$ & Test $R^2$ & CV $R^2$ & RMSE & MAE \\",
            r"\hline",
        ]
        
        for _, row in df.iterrows():
            lines.append(
                f"{row['Model']} & {row['N']} & {row['Train R²']} & {row['Test R²']} & "
                f"{row['CV R² (mean±std)']} & {row['RMSE']} & {row['MAE']} \\\\"
            )
        
        lines.extend([
            r"\hline\hline",
            r"\end{tabular}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
    
    def get_feature_importance_table(self, top_n: int = 15) -> pd.DataFrame:
        """Get combined feature importance across models."""
        dfs = []
        for name, r in self.results.items():
            if not r.feature_importance.empty:
                imp = r.feature_importance.head(top_n).copy()
                imp['model'] = name
                imp['rank'] = range(1, len(imp) + 1)
                dfs.append(imp)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
