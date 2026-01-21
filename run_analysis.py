#!/usr/bin/env python3
"""
📊 Unified Analysis Runner

Runs the complete three-part analysis framework:
1. Standard OLS with robustness
2. Machine Learning approaches
3. Two-Towers neural network

Generates LaTeX tables and publication-ready figures.

Usage:
    python run_analysis.py                    # Full analysis
    python run_analysis.py --ols              # OLS only
    python run_analysis.py --ml               # ML only
    python run_analysis.py --two-towers       # Two-Towers only
    python run_analysis.py --synthetic        # Use synthetic data
"""
import argparse
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from ceo_firm_matching import Config, generate_synthetic_data
from ceo_firm_matching.analysis import (
    OLSAnalyzer,
    MLAnalyzer,
    TwoTowersAnalyzer,
    plot_coefficient_forest,
    plot_feature_importance,
    plot_model_comparison,
    create_dashboard,
)


def load_data(synthetic: bool = False) -> pd.DataFrame:
    """Load analysis data."""
    if synthetic:
        print("📊 Using synthetic data...")
        return generate_synthetic_data(2000)
    
    config = Config()
    data_path = config.DATA_PATH
    
    if Path(data_path).exists():
        print(f"📊 Loading: {data_path}")
        return pd.read_csv(data_path)
    else:
        print(f"   ⚠️ Data not found at {data_path}, using synthetic")
        return generate_synthetic_data(2000)


def run_ols_analysis(df: pd.DataFrame, output_dir: Path) -> dict:
    """Run OLS analysis battery."""
    print("\n" + "="*60)
    print("PART 1: STANDARD OLS ANALYSIS")
    print("="*60)
    
    analyzer = OLSAnalyzer(df)
    
    # Run all specifications
    all_results = analyzer.run_all()
    
    # Generate LaTeX tables
    for category, results in all_results.items():
        if results:
            latex = analyzer.to_latex(
                results,
                title=f"OLS Results: {category.title()}",
                label=f"tab:ols_{category}"
            )
            tex_path = output_dir / f"ols_{category}.tex"
            tex_path.write_text(latex)
            print(f"   ✓ LaTeX: {tex_path}")
    
    # Coefficient forest plot
    if all_results.get('baseline'):
        key_vars = ['Age', 'tenure', 'logatw', 'exp_roa', 'rdintw', 'leverage']
        key_vars = [v for v in key_vars if v in list(all_results['baseline'].values())[0].coefficients]
        
        if key_vars:
            plot_coefficient_forest(
                all_results['baseline'],
                key_vars,
                output_path=str(output_dir / 'ols_coefficients.pdf')
            )
    
    return all_results


def run_ml_analysis(df: pd.DataFrame, output_dir: Path) -> dict:
    """Run ML analysis battery."""
    print("\n" + "="*60)
    print("PART 2: MACHINE LEARNING ANALYSIS")
    print("="*60)
    
    analyzer = MLAnalyzer(df)
    
    # Run all models
    results = analyzer.run_all()
    
    # Hyperparameter robustness
    hp_df = analyzer.run_hyperparameter_robustness()
    hp_df.to_csv(output_dir / 'ml_hyperparameter_robustness.csv', index=False)
    print(f"   ✓ CSV: {output_dir / 'ml_hyperparameter_robustness.csv'}")
    
    # Comparison table
    comparison = analyzer.get_comparison_table()
    comparison.to_csv(output_dir / 'ml_comparison.csv', index=False)
    
    # LaTeX table
    latex = analyzer.to_latex()
    (output_dir / 'ml_comparison.tex').write_text(latex)
    print(f"   ✓ LaTeX: {output_dir / 'ml_comparison.tex'}")
    
    # Plots
    plot_model_comparison(
        results,
        metric='test_r2',
        output_path=str(output_dir / 'ml_comparison.pdf')
    )
    
    # Feature importance for best model
    best_model = max(results.items(), key=lambda x: x[1].test_r2)[1]
    if not best_model.feature_importance.empty:
        plot_feature_importance(
            best_model.feature_importance,
            output_path=str(output_dir / 'ml_feature_importance.pdf')
        )
    
    return results


def run_two_towers_analysis(df: pd.DataFrame, output_dir: Path) -> dict:
    """Run Two-Towers analysis."""
    print("\n" + "="*60)
    print("PART 3: TWO-TOWERS NEURAL NETWORK")
    print("="*60)
    
    try:
        analyzer = TwoTowersAnalyzer(df)
        
        # Run baseline training
        result = analyzer.run_training("baseline")
        
        if result:
            # Comparison table
            comparison = analyzer.get_comparison_table()
            comparison.to_csv(output_dir / 'two_towers_comparison.csv', index=False)
            
            # LaTeX table
            latex = analyzer.to_latex()
            (output_dir / 'two_towers_comparison.tex').write_text(latex)
            print(f"   ✓ LaTeX: {output_dir / 'two_towers_comparison.tex'}")
        
        return analyzer.results
        
    except Exception as e:
        print(f"   ⚠️ Two-Towers analysis failed: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Run CEO-Firm Match Analysis")
    parser.add_argument('--ols', action='store_true', help='Run OLS only')
    parser.add_argument('--ml', action='store_true', help='Run ML only')
    parser.add_argument('--two-towers', action='store_true', help='Run Two-Towers only')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--output', type=str, default='Output/Analysis', help='Output directory')
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(args.synthetic)
    print(f"   Loaded {len(df):,} observations")
    
    # Determine what to run
    run_all = not (args.ols or args.ml or args.two_towers)
    
    results = {}
    
    # Part 1: OLS
    if run_all or args.ols:
        results['ols'] = run_ols_analysis(df, output_dir)
    
    # Part 2: ML
    if run_all or args.ml:
        results['ml'] = run_ml_analysis(df, output_dir)
    
    # Part 3: Two-Towers
    if run_all or args.two_towers:
        results['two_towers'] = run_two_towers_analysis(df, output_dir)
    
    # Create dashboard if we have both OLS and ML
    if 'ols' in results and 'ml' in results:
        ols_baseline = results['ols'].get('baseline', {})
        ml_results = results['ml']
        if ols_baseline and ml_results:
            create_dashboard(
                ols_baseline,
                ml_results,
                output_path=str(output_dir / 'analysis_dashboard.pdf')
            )
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Files generated: {len(list(output_dir.glob('*')))}")


if __name__ == "__main__":
    main()
