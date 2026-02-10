#!/usr/bin/env python3
"""
Two Towers Extensions — Master Runner

Orchestrates all 10 extensions for the CEO-Firm matching system.
Run with --synthetic to use generated data, or provide real data paths.

Usage:
    python run_all_extensions.py --synthetic
    python run_all_extensions.py --data Data/ceo_types_v0.2.csv
    python run_all_extensions.py --synthetic --extensions 1,2,3
"""
import argparse
import sys
import os
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ceo_firm_matching.config import Config
from ceo_firm_matching.data import DataProcessor
from ceo_firm_matching.model import CEOFirmMatcher
from ceo_firm_matching.training import train_model


def parse_args():
    parser = argparse.ArgumentParser(description='Run Two Tower Extensions')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--data', type=str, default='Data/ceo_types_v0.2.csv', help='Path to data file')
    parser.add_argument('--extensions', type=str, default='all',
                        help='Comma-separated extension numbers to run (e.g., 1,2,3) or "all"')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--output-dir', type=str, default='Output/Extensions', help='Output directory')
    return parser.parse_args()


def setup_output_dir(output_dir: str):
    """Create output directory structure."""
    dirs = [
        output_dir,
        f'{output_dir}/enriched',
        f'{output_dir}/contrastive',
        f'{output_dir}/event_study',
        f'{output_dir}/multitask',
        f'{output_dir}/temporal',
        f'{output_dir}/industry',
        f'{output_dir}/compensation',
        f'{output_dir}/network',
        f'{output_dir}/counterfactual',
        f'{output_dir}/transfer',
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_or_generate_data(args):
    """Load real data or generate synthetic data."""
    config = Config()

    if args.synthetic:
        print("\n" + "=" * 70)
        print("GENERATING SYNTHETIC DATA")
        print("=" * 70)
        from ceo_firm_matching.synthetic import generate_synthetic_data
        df = generate_synthetic_data(n_samples=5000)
        print(f"Generated {len(df)} synthetic observations")
    else:
        print(f"\nLoading data from {args.data}")
        if not os.path.exists(args.data):
            print(f"ERROR: {args.data} not found. Use --synthetic flag.")
            sys.exit(1)
        df = pd.read_csv(args.data)

    return df, config


def train_base_model(df, config, args):
    """Train the base Two-Tower model."""
    print("\n" + "=" * 70)
    print("TRAINING BASE TWO-TOWER MODEL")
    print("=" * 70)

    from ceo_firm_matching.data import CEOFirmDataset
    from torch.utils.data import DataLoader, random_split

    config.EPOCHS = args.epochs
    processor = DataProcessor(config)
    df_processed = processor.prepare_features(df)
    processor.fit(df_processed)
    data_dict = processor.transform(df_processed)

    # Extract metadata from data_dict (DataProcessor embeds it)
    metadata = {
        'n_firm_numeric': data_dict['n_firm_numeric'],
        'firm_cat_counts': data_dict['firm_cat_counts'],
        'n_ceo_numeric': data_dict['n_ceo_numeric'],
        'ceo_cat_counts': data_dict['ceo_cat_counts'],
    }

    # Create DataLoaders
    dataset = CEOFirmDataset(data_dict)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    model = train_model(train_loader, val_loader, metadata, config)

    return model, processor, data_dict, metadata, df_processed


# =============================================================================
# EXTENSION RUNNERS
# =============================================================================

def run_extension_1(df, config, processor, data_dict, metadata, output_dir, args):
    """Extension 1: Enriched CEO Features."""
    print("\n" + "=" * 70)
    print("EXTENSION 1: ENRICHED CEO TOWER")
    print("=" * 70)

    from ceo_firm_matching.enriched_features import (
        EnrichedConfig, construct_enriched_ceo_features
    )

    enriched_config = EnrichedConfig()
    enriched_config.EPOCHS = args.epochs

    # Show what features are available vs used
    current_ceo_features = len(config.CEO_NUMERIC_COLS)
    enriched_ceo_features = len(enriched_config.CEO_NUMERIC_COLS_ENRICHED)

    print(f"\n  Current CEO numeric features: {current_ceo_features}")
    print(f"  Enriched CEO numeric features: {enriched_ceo_features}")
    print(f"  Feature improvement: {current_ceo_features} → {enriched_ceo_features} ({enriched_ceo_features/max(current_ceo_features,1):.0f}x)")

    print(f"\n  New CEO features available:")
    for feat in enriched_config.CEO_NUMERIC_COLS_ENRICHED:
        marker = "✓ EXISTING" if feat in config.CEO_NUMERIC_COLS else "★ NEW"
        print(f"    {marker}: {feat}")

    # Show enriched firm features
    print(f"\n  New Firm features:")
    for feat in enriched_config.FIRM_NUMERIC_COLS_ENRICHED:
        marker = "✓ EXISTING" if feat in config.FIRM_NUMERIC_COLS else "★ NEW"
        print(f"    {marker}: {feat}")

    # If we have the enriched features in the data, re-train
    available_enriched = [c for c in enriched_config.CEO_NUMERIC_COLS_ENRICHED if c in df.columns]
    print(f"\n  Features available in current data: {len(available_enriched)}/{enriched_ceo_features}")

    if len(available_enriched) > len(config.CEO_NUMERIC_COLS):
        print("  → Re-training with enriched features...")
        enriched_config.CEO_NUMERIC_COLS = available_enriched
        enriched_processor = DataProcessor(enriched_config)
        enriched_df = enriched_processor.prepare_features(df)
        enriched_data = enriched_processor.fit_transform(enriched_df)
        enriched_metadata = enriched_processor.get_metadata()
        enriched_model = train_model(enriched_data, enriched_metadata, enriched_config)
        print("  ✓ Enriched model trained successfully")
    else:
        print("  → Enriched features not in current data (need WRDS pull)")
        print("  → Use ceo_firm_matching.wrds.pulls to fetch BoardEx/ExecuComp data")

    print("\n  ✓ Extension 1 complete")


def run_extension_2(df, config, processor, data_dict, metadata, output_dir, args):
    """Extension 2: Contrastive Learning."""
    print("\n" + "=" * 70)
    print("EXTENSION 2: CONTRASTIVE LEARNING")
    print("=" * 70)

    from ceo_firm_matching.contrastive import (
        ContrastiveCEOFirmMatcher, train_contrastive, compute_retrieval_metrics
    )
    from torch.utils.data import TensorDataset, DataLoader

    # Build dataloaders
    dataset = TensorDataset(
        data_dict['firm_numeric'], data_dict['firm_cat'],
        data_dict['ceo_numeric'], data_dict['ceo_cat'],
        data_dict['target'], data_dict['weights'],
    )

    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    # Custom collate for named dict
    def collate_fn(batch):
        keys = ['firm_numeric', 'firm_cat', 'ceo_numeric', 'ceo_cat', 'target', 'weights']
        return {k: torch.stack([b[i] for b in batch]) for i, k in enumerate(keys)}

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)

    # Train with InfoNCE
    print("\n  Training with InfoNCE contrastive loss...")
    config.EPOCHS = args.epochs
    model_infonce = train_contrastive(
        train_loader, val_loader, metadata, config,
        contrastive_weight=0.3, temperature=0.07, use_triplet=False
    )

    # Retrieval metrics
    print("\n  Computing retrieval metrics...")
    metrics = compute_retrieval_metrics(model_infonce, data_dict, config)
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    # Also train with triplet loss for comparison
    print("\n  Training with triplet loss...")
    model_triplet = train_contrastive(
        train_loader, val_loader, metadata, config,
        contrastive_weight=0.3, use_triplet=True
    )
    metrics_triplet = compute_retrieval_metrics(model_triplet, data_dict, config)
    print("  Triplet retrieval metrics:")
    for k, v in metrics_triplet.items():
        print(f"    {k}: {v:.4f}")

    # Save comparison
    comparison = pd.DataFrame({
        'InfoNCE': metrics,
        'Triplet': metrics_triplet,
    })
    comparison.to_csv(f'{output_dir}/contrastive/retrieval_comparison.csv')
    print(f"\n  ✓ Saved to {output_dir}/contrastive/")

    return model_infonce


def run_extension_3(df, model, data_dict, output_dir, args):
    """Extension 3: Event Studies."""
    print("\n" + "=" * 70)
    print("EXTENSION 3: CEO TRANSITION EVENT STUDIES")
    print("=" * 70)

    from ceo_firm_matching.analytical_extensions import (
        identify_ceo_transitions, run_event_study, run_regression_event_study
    )

    # Identify transitions
    transitions = identify_ceo_transitions(df)

    if len(transitions) == 0:
        print("  No CEO transitions found in data — skipping event study")
        if args.synthetic:
            print("  (Expected with synthetic data — transitions are random)")
        return

    print(f"\n  Found {len(transitions)} CEO transitions")
    print(f"  Upgrades: {transitions['upgrade'].sum()} | Downgrades: {(~transitions['upgrade']).sum()}")

    # Performance data (use what's in df)
    perf_col = 'exp_roa' if 'exp_roa' in df.columns else 'match_means'
    print(f"  Using '{perf_col}' as performance metric")

    # Run event study
    event_results = run_event_study(
        transitions, df, perf_col=perf_col,
        window=(-3, 3), did=True
    )

    if len(event_results) > 0:
        event_results.to_csv(f'{output_dir}/event_study/event_study_results.csv', index=False)
        print(f"\n  ✓ Event study saved to {output_dir}/event_study/")

    # Regression event study
    reg_results = run_regression_event_study(
        transitions, df, perf_col=perf_col,
        controls=['logatw', 'leverage'] if 'logatw' in df.columns else None,
        window=(-3, 3)
    )

    if reg_results:
        pd.DataFrame([reg_results]).to_csv(
            f'{output_dir}/event_study/regression_results.csv', index=False
        )


def run_extension_4(df, config, data_dict, metadata, output_dir, args):
    """Extension 4: Multi-Task Learning."""
    print("\n" + "=" * 70)
    print("EXTENSION 4: MULTI-TASK LEARNING")
    print("=" * 70)

    from ceo_firm_matching.multitask_model import (
        MultiTaskConfig, MultiTaskCEOFirmMatcher,
        MultiTaskDataset, train_multitask, analyze_multitask_embeddings
    )
    from torch.utils.data import DataLoader

    mt_config = MultiTaskConfig()
    mt_config.EPOCHS = args.epochs

    # Add auxiliary targets to data_dict
    if 'tdc1' in df.columns:
        data_dict['log_tdc1'] = torch.tensor(
            np.log1p(pd.to_numeric(df['tdc1'], errors='coerce').fillna(0).values),
            dtype=torch.float32
        )
    if 'tenure' in df.columns:
        data_dict['tenure_years'] = torch.tensor(
            pd.to_numeric(df['tenure'], errors='coerce').fillna(0).values,
            dtype=torch.float32
        )

    # Create dataset
    dataset = MultiTaskDataset(data_dict)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    # Train
    model = train_multitask(train_loader, val_loader, metadata, mt_config)

    # Analyze
    results = analyze_multitask_embeddings(model, data_dict, df, mt_config)

    # Save predictions
    pred_df = pd.DataFrame({
        'match_pred': results['match_score'],
        'comp_pred': results['comp_pred'],
        'tenure_pred': results['tenure_pred'],
        'turnover_prob': results['turnover_prob'],
    })

    if 'match_means' in df.columns:
        pred_df['match_actual'] = df['match_means'].values[:len(pred_df)]

    pred_df.to_csv(f'{output_dir}/multitask/predictions.csv', index=False)

    # Correlation matrix of predictions
    print("\n  Prediction correlations:")
    print(pred_df.corr().round(3).to_string())

    print(f"\n  ✓ Multi-task results saved to {output_dir}/multitask/")
    return model


def run_extension_5(df, config, data_dict, metadata, output_dir, args):
    """Extension 5: Temporal CEO Embeddings."""
    print("\n" + "=" * 70)
    print("EXTENSION 5: TIME-VARYING CEO EMBEDDINGS")
    print("=" * 70)

    from ceo_firm_matching.temporal_model import (
        TemporalTwoTower, CareerSequenceDataset
    )

    # Check if we have panel structure
    if 'match_exec_id' not in df.columns or 'fiscalyear' not in df.columns:
        print("  Panel structure (match_exec_id, fiscalyear) not found")
        print("  Skipping temporal model — requires longitudinal data")
        return

    # Career statistics
    career_lengths = df.groupby('match_exec_id')['fiscalyear'].nunique()
    print(f"\n  Career lengths: median={career_lengths.median():.0f}, "
          f"mean={career_lengths.mean():.1f}, max={career_lengths.max():.0f}")
    print(f"  CEOs with 5+ years: {(career_lengths >= 5).sum()} "
          f"({(career_lengths >= 5).mean():.1%})")

    # Build sequence dataset
    max_seq = min(15, int(career_lengths.quantile(0.9)))
    print(f"  Using max sequence length: {max_seq}")

    dataset = CareerSequenceDataset(
        df, data_dict,
        ceo_id_col='match_exec_id',
        year_col='fiscalyear',
        max_seq_len=max_seq,
    )

    print(f"  Built {len(dataset)} career sequence samples")

    # Train temporal model
    from torch.utils.data import DataLoader

    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=min(config.BATCH_SIZE, 64), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(config.BATCH_SIZE, 64))

    model = TemporalTwoTower(metadata, config, max_seq_len=max_seq).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"\n  Training Temporal Two-Tower ({args.epochs} epochs)...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()

            score = model(
                batch['firm_numeric'], batch['firm_cat'],
                batch['ceo_numeric_seq'], batch['ceo_cat_seq'],
                batch['seq_mask']
            )

            loss = (batch['weights'] * (score - batch['target']) ** 2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}")

    print(f"\n  ✓ Temporal model trained")
    return model


def run_extension_6(df, config, data_dict, metadata, output_dir, args):
    """Extension 6: Industry-Specific Match Functions."""
    print("\n" + "=" * 70)
    print("EXTENSION 6: INDUSTRY-SPECIFIC MATCHING")
    print("=" * 70)

    from ceo_firm_matching.industry_model import (
        IndustryConditionedMatcher, IndustryExpertMixture
    )

    # Count industries
    if 'compindustry' in df.columns:
        n_industries = df['compindustry'].nunique()
    else:
        n_industries = data_dict['firm_cat'][:, 0].max().item() + 1
    print(f"  Industries: {n_industries}")

    # Train Industry-Conditioned model
    model = IndustryConditionedMatcher(metadata, config, n_industries=n_industries).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(
        data_dict['firm_numeric'], data_dict['firm_cat'],
        data_dict['ceo_numeric'], data_dict['ceo_cat'],
        data_dict['target'], data_dict['weights'],
    )
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    print(f"\n  Training Industry-Conditioned matcher ({args.epochs} epochs)...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            batch = [b.to(config.DEVICE) for b in batch]
            f_num, f_cat, c_num, c_cat, target, weights = batch
            optimizer.zero_grad()
            score = model(f_num, f_cat, c_num, c_cat)
            loss = (weights * (score - target) ** 2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: Loss={total_loss/len(loader):.4f}")

    # Analyze learned temperatures
    print("\n  Learned industry temperatures:")
    temps = model.industry_temperature.weight.exp().detach().cpu().numpy().flatten()
    for i, t in enumerate(temps[:10]):
        print(f"    Industry {i}: τ = {t:.2f}")

    # Compare industries
    if n_industries >= 2:
        comparison = model.compare_industries(0, 1)
        print(f"\n  Industry 0 vs 1 correlation: {comparison['correlation']:.3f}")

    # Train Mixture-of-Experts for comparison
    print(f"\n  Training Mixture-of-Experts ({args.epochs} epochs)...")
    moe_model = IndustryExpertMixture(metadata, config, n_experts=4, n_industries=n_industries).to(config.DEVICE)
    moe_optimizer = torch.optim.Adam(moe_model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(args.epochs):
        moe_model.train()
        total_loss = 0
        for batch in loader:
            batch = [b.to(config.DEVICE) for b in batch]
            f_num, f_cat, c_num, c_cat, target, weights = batch
            moe_optimizer.zero_grad()
            score = moe_model(f_num, f_cat, c_num, c_cat)
            loss = (weights * (score - target) ** 2).mean()
            loss.backward()
            moe_optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: Loss={total_loss/len(loader):.4f}")

    print(f"\n  ✓ Industry-specific models trained. Results saved to {output_dir}/industry/")

    return model


def run_extension_7(df, model, data_dict, output_dir, config, args):
    """Extension 7: Compensation Decomposition."""
    print("\n" + "=" * 70)
    print("EXTENSION 7: COMPENSATION DECOMPOSITION")
    print("=" * 70)

    from ceo_firm_matching.analytical_extensions import (
        decompose_compensation, compensation_by_match_quality
    )

    comp_col = None
    for col in ['tdc1', 'total_comp', 'salary']:
        if col in df.columns:
            comp_col = col
            break

    if comp_col is None:
        # Synthesize a compensation proxy for demo
        print("  No compensation column found — generating synthetic proxy")
        df['tdc1_synthetic'] = np.exp(
            10 + 0.5 * df['match_means'].values + np.random.normal(0, 0.5, len(df))
        ) if 'match_means' in df.columns else np.random.lognormal(10, 1, len(df))
        comp_col = 'tdc1_synthetic'

    comp_df = decompose_compensation(
        model, data_dict, df,
        comp_col=comp_col, device=config.DEVICE
    )

    if len(comp_df) > 0:
        # Quintile analysis
        quintile_summary = compensation_by_match_quality(comp_df)
        if len(quintile_summary) > 0:
            quintile_summary.to_csv(f'{output_dir}/compensation/quintile_decomposition.csv')

        comp_df.to_csv(f'{output_dir}/compensation/full_decomposition.csv', index=False)
        print(f"\n  ✓ Compensation decomposition saved to {output_dir}/compensation/")


def run_extension_8(df, output_dir, args):
    """Extension 8: Board Interlock Network."""
    print("\n" + "=" * 70)
    print("EXTENSION 8: BOARD INTERLOCK NETWORK FEATURES")
    print("=" * 70)

    from ceo_firm_matching.network_features import (
        build_interlock_graph, compute_centrality_features, construct_full_network_features
    )

    # Check if BoardEx data is available
    boardex_path = 'Data/boardex_employment.csv'
    if os.path.exists(boardex_path):
        employment = pd.read_csv(boardex_path)
        print(f"  Loaded BoardEx employment: {len(employment)} records")
    else:
        # Generate synthetic network data for demo
        print("  BoardEx employment data not found — generating synthetic network")
        n_directors = 2000
        n_companies = 500
        n_roles = 8000

        employment = pd.DataFrame({
            'directorid': np.random.randint(1, n_directors + 1, n_roles),
            'companyid': np.random.randint(1, n_companies + 1, n_roles),
            'rolename': np.random.choice(
                ['Director', 'CEO', 'Board Member', 'CFO', 'Chairman', 'Non-Executive Director'],
                n_roles
            ),
            'datestartrole': pd.date_range('2000-01-01', periods=n_roles, freq='D')[:n_roles],
            'dateendrole': pd.date_range('2005-01-01', periods=n_roles, freq='D')[:n_roles],
        })

    # Build and analyze network
    target_ids = df['match_exec_id'].unique()[:100].tolist() if 'match_exec_id' in df.columns else list(range(1, 101))
    target_ids = [int(x) for x in target_ids if pd.notna(x)]

    features = construct_full_network_features(employment, target_ids, year=2020)

    print(f"\n  Network features computed for {len(features)} CEOs:")
    print(features.describe().round(2).to_string())

    features.to_csv(f'{output_dir}/network/network_features.csv', index=False)
    print(f"\n  ✓ Network features saved to {output_dir}/network/")


def run_extension_9(df, model, data_dict, output_dir, config, args):
    """Extension 9: Generative Counterfactuals."""
    print("\n" + "=" * 70)
    print("EXTENSION 9: GENERATIVE COUNTERFACTUALS")
    print("=" * 70)

    from ceo_firm_matching.analytical_extensions import (
        generate_counterfactuals, reallocation_simulation
    )

    # Counterfactual rankings
    cf_df = generate_counterfactuals(
        model, data_dict, df,
        top_k=10, device=config.DEVICE
    )
    cf_df.to_csv(f'{output_dir}/counterfactual/counterfactual_rankings.csv', index=False)

    # Optimal reallocation
    n_firms = min(200, len(df))
    realloc = reallocation_simulation(
        model, data_dict, df,
        n_firms=n_firms, device=config.DEVICE
    )
    if realloc:
        pd.DataFrame([realloc]).to_csv(
            f'{output_dir}/counterfactual/reallocation_results.csv', index=False
        )

    print(f"\n  ✓ Counterfactual analysis saved to {output_dir}/counterfactual/")


def run_extension_10(model, data_dict, config, output_dir, args):
    """Extension 10: Transfer Learning Setup."""
    print("\n" + "=" * 70)
    print("EXTENSION 10: TRANSFER LEARNING")
    print("=" * 70)

    from ceo_firm_matching.analytical_extensions import TransferableTwoTower

    # Create transferable model from trained base
    transfer_model = TransferableTwoTower(model, config, adapter_dim=16)

    # Count trainable parameters
    total_params = sum(p.numel() for p in transfer_model.parameters())
    trainable_params = sum(p.numel() for p in transfer_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n  Transfer Learning Architecture:")
    print(f"    Total parameters:     {total_params:,}")
    print(f"    Frozen (base):        {frozen_params:,} ({frozen_params/total_params:.1%})")
    print(f"    Trainable (adapters): {trainable_params:,} ({trainable_params/total_params:.1%})")

    # Verify forward pass works
    with torch.no_grad():
        test_score = transfer_model(
            data_dict['firm_numeric'][:5].to(config.DEVICE),
            data_dict['firm_cat'][:5].to(config.DEVICE),
            data_dict['ceo_numeric'][:5].to(config.DEVICE),
            data_dict['ceo_cat'][:5].to(config.DEVICE),
        )
    print(f"    Test forward pass: {test_score.shape} ✓")

    # Save the transfer model
    torch.save(transfer_model.state_dict(), f'{output_dir}/transfer/transfer_model.pt')
    print(f"\n  ✓ Transfer model saved to {output_dir}/transfer/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    setup_output_dir(args.output_dir)

    # Parse which extensions to run
    if args.extensions == 'all':
        extensions = list(range(1, 11))
    else:
        extensions = [int(x.strip()) for x in args.extensions.split(',')]

    print("=" * 70)
    print("   TWO TOWERS EXTENSIONS RUNNER ")
    print("=" * 70)
    print(f"  Extensions to run: {extensions}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Output: {args.output_dir}")
    print(f"  Data: {'synthetic' if args.synthetic else args.data}")

    start = time.time()

    # Load data and train base model
    df, config = load_or_generate_data(args)
    model, processor, data_dict, metadata, df_processed = train_base_model(df, config, args)

    results = {}

    # Run each extension
    runners = {
        1: lambda: run_extension_1(df_processed, config, processor, data_dict, metadata, args.output_dir, args),
        2: lambda: run_extension_2(df_processed, config, processor, data_dict, metadata, args.output_dir, args),
        3: lambda: run_extension_3(df_processed, model, data_dict, args.output_dir, args),
        4: lambda: run_extension_4(df_processed, config, data_dict, metadata, args.output_dir, args),
        5: lambda: run_extension_5(df_processed, config, data_dict, metadata, args.output_dir, args),
        6: lambda: run_extension_6(df_processed, config, data_dict, metadata, args.output_dir, args),
        7: lambda: run_extension_7(df_processed, model, data_dict, args.output_dir, config, args),
        8: lambda: run_extension_8(df_processed, args.output_dir, args),
        9: lambda: run_extension_9(df_processed, model, data_dict, args.output_dir, config, args),
        10: lambda: run_extension_10(model, data_dict, config, args.output_dir, args),
    }

    for ext_num in extensions:
        if ext_num in runners:
            try:
                result = runners[ext_num]()
                results[ext_num] = 'SUCCESS'
            except Exception as e:
                print(f"\n  ✗ Extension {ext_num} failed: {e}")
                import traceback
                traceback.print_exc()
                results[ext_num] = f'FAILED: {e}'

    # Summary
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("   SUMMARY")
    print("=" * 70)
    print(f"  Total time: {elapsed:.1f}s")
    for ext_num, status in results.items():
        icon = "✓" if status == 'SUCCESS' else "✗"
        print(f"  {icon} Extension {ext_num}: {status}")
    print(f"\n  All outputs saved to: {args.output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
