"""
OLS Regression Analysis Module.

Provides standard econometric analysis with robustness checks:
- Baseline OLS with CEO and Firm features
- Fixed effects specifications (Year, Industry, Firm)
- Clustering variants for standard errors
- Subsample robustness checks
- LaTeX-ready output generation
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    from statsmodels.iolib.summary2 import summary_col
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


@dataclass
class OLSSpec:
    """Specification for an OLS regression."""
    name: str
    y_col: str = 'match_means'
    x_cols: List[str] = field(default_factory=list)
    controls: List[str] = field(default_factory=list)
    fe_cols: List[str] = field(default_factory=list)
    cluster_col: Optional[str] = None
    weight_col: Optional[str] = None
    sample_filter: Optional[str] = None


@dataclass
class OLSResult:
    """Container for OLS regression results."""
    spec_name: str
    n_obs: int
    r_squared: float
    adj_r_squared: float
    coefficients: Dict[str, float]
    std_errors: Dict[str, float]
    t_stats: Dict[str, float]
    p_values: Dict[str, float]
    conf_int_low: Dict[str, float]
    conf_int_high: Dict[str, float]
    model: Any = None


class OLSAnalyzer:
    """
    OLS regression analyzer with robustness framework.
    
    Example usage:
        analyzer = OLSAnalyzer(df)
        results = analyzer.run_baseline()
        table = analyzer.to_latex(results)
    """
    
    # Default feature groups
    CEO_FEATURES = ['Age', 'tenure', 'Output', 'Throghput', 'Peripheral', 'maxedu', 'ivy', 'm']
    FIRM_FEATURES = ['logatw', 'exp_roa', 'rdintw', 'capintw', 'leverage', 'boardindpw']
    SIZE_CONTROLS = ['logatw', 'boardsizew']
    GOVERNANCE_CONTROLS = ['boardindpw', 'pct_blockw', 'busyw']
    
    def __init__(self, df: pd.DataFrame, y_col: str = 'match_means'):
        """
        Initialize analyzer with data.
        
        Args:
            df: DataFrame with CEO-Firm match data
            y_col: Target variable column name
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required. Install: pip install statsmodels")
        
        self.df = df.copy()
        self.y_col = y_col
        self.results: Dict[str, OLSResult] = {}
        
        # Compute tenure if not present
        if 'tenure' not in self.df.columns and 'fiscalyear' in self.df.columns and 'ceo_year' in self.df.columns:
            self.df['tenure'] = self.df['fiscalyear'] - self.df['ceo_year']
    
    def _prepare_data(self, spec: OLSSpec) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data for regression based on specification."""
        df = self.df.copy()
        
        # Apply sample filter
        if spec.sample_filter:
            df = df.query(spec.sample_filter)
        
        # Build feature list
        all_x = list(spec.x_cols) + list(spec.controls)
        all_x = [c for c in all_x if c in df.columns]
        
        # Add fixed effects as dummies
        fe_cols_added = []
        for fe in spec.fe_cols:
            if fe in df.columns:
                dummies = pd.get_dummies(df[fe], prefix=fe, drop_first=True, dtype=float)
                df = pd.concat([df, dummies], axis=1)
                fe_cols_added.extend(dummies.columns.tolist())
        
        all_x.extend(fe_cols_added)
        
        # Drop missing
        required = [spec.y_col] + all_x
        if spec.weight_col:
            required.append(spec.weight_col)
        df = df.dropna(subset=[c for c in required if c in df.columns])
        
        # Ensure all features are numeric
        all_x = [c for c in all_x if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        
        return df, all_x
    
    def run_spec(self, spec: OLSSpec) -> OLSResult:
        """Run a single OLS specification."""
        df, all_x = self._prepare_data(spec)
        
        if len(df) < 30 or not all_x:
            print(f"   ⚠️ {spec.name}: insufficient data or features")
            return None
        
        X = sm.add_constant(df[all_x])
        y = df[spec.y_col]
        
        # Store column names for later
        var_names = X.columns.tolist()
        
        # Weights
        weights = df[spec.weight_col] if spec.weight_col and spec.weight_col in df.columns else None
        
        # Fit model
        model = sm.WLS(y, X, weights=weights).fit() if weights is not None else sm.OLS(y, X).fit()
        
        # Clustered SEs if specified
        if spec.cluster_col and spec.cluster_col in df.columns:
            model = model.get_robustcov_results(
                cov_type='cluster',
                groups=df[spec.cluster_col]
            )
        
        # Extract results - handle both Series and array returns
        def to_dict(x, names):
            if hasattr(x, 'to_dict'):
                return x.to_dict()
            return dict(zip(names, x))
        
        # Extract confidence intervals
        conf_int = model.conf_int()
        if hasattr(conf_int, 'iloc'):
            ci_low = conf_int.iloc[:, 0]
            ci_high = conf_int.iloc[:, 1]
        else:
            ci_low = conf_int[:, 0]
            ci_high = conf_int[:, 1]
        
        result = OLSResult(
            spec_name=spec.name,
            n_obs=int(model.nobs),
            r_squared=model.rsquared,
            adj_r_squared=model.rsquared_adj,
            coefficients=to_dict(model.params, var_names),
            std_errors=to_dict(model.bse, var_names),
            t_stats=to_dict(model.tvalues, var_names),
            p_values=to_dict(model.pvalues, var_names),
            conf_int_low=to_dict(ci_low, var_names),
            conf_int_high=to_dict(ci_high, var_names),
            model=model
        )
        
        self.results[spec.name] = result
        return result
    
    def run_baseline(self) -> Dict[str, OLSResult]:
        """Run baseline specification battery."""
        print("📈 Running OLS Baseline Specifications...")
        
        specs = [
            OLSSpec(
                name="(1) CEO Only",
                y_col=self.y_col,
                x_cols=self.CEO_FEATURES
            ),
            OLSSpec(
                name="(2) Firm Only",
                y_col=self.y_col,
                x_cols=self.FIRM_FEATURES
            ),
            OLSSpec(
                name="(3) CEO + Firm",
                y_col=self.y_col,
                x_cols=self.CEO_FEATURES + self.FIRM_FEATURES
            ),
            OLSSpec(
                name="(4) + Controls",
                y_col=self.y_col,
                x_cols=self.CEO_FEATURES + self.FIRM_FEATURES,
                controls=self.SIZE_CONTROLS + self.GOVERNANCE_CONTROLS
            ),
        ]
        
        results = {}
        for spec in specs:
            result = self.run_spec(spec)
            if result:
                results[spec.name] = result
                print(f"   ✓ {spec.name}: N={result.n_obs:,}, R²={result.r_squared:.4f}")
        
        return results
    
    def run_fixed_effects(self) -> Dict[str, OLSResult]:
        """Run fixed effects specifications."""
        print("📈 Running Fixed Effects Specifications...")
        
        base_x = self.CEO_FEATURES + self.FIRM_FEATURES
        
        specs = [
            OLSSpec(
                name="(1) Year FE",
                y_col=self.y_col,
                x_cols=base_x,
                fe_cols=['fiscalyear']
            ),
            OLSSpec(
                name="(2) Industry FE",
                y_col=self.y_col,
                x_cols=base_x,
                fe_cols=['compindustry']
            ),
            OLSSpec(
                name="(3) Year + Industry FE",
                y_col=self.y_col,
                x_cols=base_x,
                fe_cols=['fiscalyear', 'compindustry']
            ),
        ]
        
        results = {}
        for spec in specs:
            result = self.run_spec(spec)
            if result:
                results[spec.name] = result
                print(f"   ✓ {spec.name}: N={result.n_obs:,}, R²={result.r_squared:.4f}")
        
        return results
    
    def run_clustering_robustness(self) -> Dict[str, OLSResult]:
        """Run SE clustering robustness checks."""
        print("📈 Running Clustering Robustness...")
        
        base_x = self.CEO_FEATURES + self.FIRM_FEATURES
        cluster_options = [None, 'gvkey', 'compindustry', 'fiscalyear']
        
        results = {}
        for cluster in cluster_options:
            name = f"Cluster: {cluster or 'None'}"
            spec = OLSSpec(
                name=name,
                y_col=self.y_col,
                x_cols=base_x,
                cluster_col=cluster
            )
            result = self.run_spec(spec)
            if result:
                results[name] = result
                print(f"   ✓ {name}: N={result.n_obs:,}")
        
        return results
    
    def run_subsample_robustness(self) -> Dict[str, OLSResult]:
        """Run subsample robustness checks."""
        print("📈 Running Subsample Robustness...")
        
        base_x = self.CEO_FEATURES + self.FIRM_FEATURES
        
        # Define subsample filters
        if 'logatw' in self.df.columns:
            median_size = self.df['logatw'].median()
        else:
            median_size = 0
        
        subsamples = [
            ("Full Sample", None),
            ("Large Firms", f"logatw >= {median_size}"),
            ("Small Firms", f"logatw < {median_size}"),
            ("Post-2010", "fiscalyear >= 2010"),
            ("Pre-2010", "fiscalyear < 2010"),
        ]
        
        results = {}
        for name, filter_expr in subsamples:
            spec = OLSSpec(
                name=name,
                y_col=self.y_col,
                x_cols=base_x,
                sample_filter=filter_expr
            )
            result = self.run_spec(spec)
            if result:
                results[name] = result
                print(f"   ✓ {name}: N={result.n_obs:,}, R²={result.r_squared:.4f}")
        
        return results
    
    def run_all(self) -> Dict[str, Dict[str, OLSResult]]:
        """Run complete OLS analysis battery."""
        return {
            'baseline': self.run_baseline(),
            'fixed_effects': self.run_fixed_effects(),
            'clustering': self.run_clustering_robustness(),
            'subsamples': self.run_subsample_robustness(),
        }
    
    def to_latex(
        self,
        results: Dict[str, OLSResult],
        variables: Optional[List[str]] = None,
        title: str = "OLS Regression Results",
        label: str = "tab:ols",
        stars: bool = True
    ) -> str:
        """
        Generate LaTeX regression table.
        
        Args:
            results: Dict of OLSResult objects
            variables: Subset of variables to display (None = all)
            title: Table title
            label: LaTeX label
            stars: Include significance stars
            
        Returns:
            LaTeX table string
        """
        if not results:
            return ""
        
        # Collect all variables
        all_vars = set()
        for r in results.values():
            all_vars.update(r.coefficients.keys())
        
        if variables:
            display_vars = [v for v in variables if v in all_vars]
        else:
            display_vars = sorted([v for v in all_vars if v != 'const'])
        
        # Build table
        n_cols = len(results)
        col_names = list(results.keys())
        
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{title}}}",
            f"\\label{{{label}}}",
            r"\begin{tabular}{l" + "c" * n_cols + "}",
            r"\hline\hline",
            " & " + " & ".join(col_names) + r" \\",
            r"\hline",
        ]
        
        # Coefficient rows
        for var in display_vars:
            row = [var.replace('_', r'\_')]
            for name in col_names:
                r = results[name]
                if var in r.coefficients:
                    coef = r.coefficients[var]
                    se = r.std_errors[var]
                    pval = r.p_values[var]
                    
                    # Stars
                    star = ""
                    if stars:
                        if pval < 0.01:
                            star = "***"
                        elif pval < 0.05:
                            star = "**"
                        elif pval < 0.10:
                            star = "*"
                    
                    row.append(f"{coef:.3f}{star}")
                else:
                    row.append("")
            lines.append(" & ".join(row) + r" \\")
            
            # SE row
            se_row = [""]
            for name in col_names:
                r = results[name]
                if var in r.std_errors:
                    se_row.append(f"({r.std_errors[var]:.3f})")
                else:
                    se_row.append("")
            lines.append(" & ".join(se_row) + r" \\")
        
        # Footer
        lines.append(r"\hline")
        
        # N row
        n_row = ["N"]
        for name in col_names:
            n_row.append(f"{results[name].n_obs:,}")
        lines.append(" & ".join(n_row) + r" \\")
        
        # R² row
        r2_row = ["$R^2$"]
        for name in col_names:
            r2_row.append(f"{results[name].r_squared:.3f}")
        lines.append(" & ".join(r2_row) + r" \\")
        
        lines.extend([
            r"\hline\hline",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            r"\item Notes: Standard errors in parentheses. * p<0.10, ** p<0.05, *** p<0.01.",
            r"\end{tablenotes}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
    
    def get_coefficient_table(self, results: Dict[str, OLSResult]) -> pd.DataFrame:
        """Extract coefficient table as DataFrame."""
        rows = []
        for name, r in results.items():
            for var, coef in r.coefficients.items():
                rows.append({
                    'spec': name,
                    'variable': var,
                    'coef': coef,
                    'se': r.std_errors.get(var),
                    't': r.t_stats.get(var),
                    'p': r.p_values.get(var),
                    'ci_low': r.conf_int_low.get(var),
                    'ci_high': r.conf_int_high.get(var),
                })
        return pd.DataFrame(rows)
