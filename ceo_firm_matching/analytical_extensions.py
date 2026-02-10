"""
Extension 3: CEO Transition Event Studies (Causal Identification)
Extension 7: Compensation Decomposition via Embeddings
Extension 9: Generative Counterfactuals
Extension 10: Transfer Learning to Private Firms

These analytical extensions operate on trained models and don't require
new architectures — they're downstream analyses that extract economic
insights from the learned embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings


# =============================================================================
# EXTENSION 3: CEO TRANSITION EVENT STUDIES
# =============================================================================

def identify_ceo_transitions(
    df: pd.DataFrame,
    ceo_id_col: str = 'match_exec_id',
    firm_id_col: str = 'gvkey',
    year_col: str = 'fiscalyear',
) -> pd.DataFrame:
    """
    Identify CEO departure events from panel data.

    A transition occurs when a firm's CEO changes between consecutive years.

    Returns:
        DataFrame with columns:
        - gvkey, transition_year
        - departing_ceo_id, successor_ceo_id
        - departing_match, successor_match (if match_means available)
        - match_change (successor - departing)
    """
    df = df.sort_values([firm_id_col, year_col])

    transitions = []
    for gvkey, group in df.groupby(firm_id_col):
        group = group.sort_values(year_col)
        ceos = group[ceo_id_col].values
        years = group[year_col].values
        matches = group['match_means'].values if 'match_means' in group.columns else [None] * len(group)

        for i in range(1, len(ceos)):
            if ceos[i] != ceos[i - 1] and pd.notna(ceos[i]) and pd.notna(ceos[i - 1]):
                transitions.append({
                    'gvkey': gvkey,
                    'transition_year': years[i],
                    'departing_ceo_id': ceos[i - 1],
                    'successor_ceo_id': ceos[i],
                    'departing_match': matches[i - 1],
                    'successor_match': matches[i],
                    'match_change': (matches[i] - matches[i - 1]) if matches[i] is not None else None,
                })

    trans_df = pd.DataFrame(transitions)
    if len(trans_df) > 0:
        trans_df['upgrade'] = trans_df['match_change'] > 0
    print(f"Found {len(trans_df)} CEO transitions")
    return trans_df


def run_event_study(
    transitions: pd.DataFrame,
    performance: pd.DataFrame,
    perf_col: str = 'tobinq',
    firm_id_col: str = 'gvkey',
    year_col: str = 'fiscalyear',
    window: Tuple[int, int] = (-3, 3),
    did: bool = True,
) -> pd.DataFrame:
    """
    Run event study around CEO transitions.

    Args:
        transitions: Output of identify_ceo_transitions
        performance: Panel data with firm performance (gvkey, fiscalyear, tobinq, ...)
        perf_col: Performance variable to track
        window: (pre, post) event window in years
        did: If True, computes DiD (upgrade vs downgrade treatment)

    Returns:
        DataFrame with event-time performance paths for treatment/control
    """
    pre, post = window
    results = []

    for _, event in transitions.iterrows():
        gvkey = event['gvkey']
        t0 = event['transition_year']

        # Get performance around event window
        firm_perf = performance[performance[firm_id_col] == gvkey].copy()
        firm_perf['event_time'] = firm_perf[year_col] - t0

        window_data = firm_perf[
            (firm_perf['event_time'] >= pre) &
            (firm_perf['event_time'] <= post)
        ].copy()

        if len(window_data) < abs(pre) + post:
            continue  # Not enough data around event

        # Normalize to t=-1
        baseline = window_data.loc[window_data['event_time'] == -1, perf_col]
        if len(baseline) == 0:
            continue
        baseline_val = baseline.values[0]

        window_data['perf_normalized'] = window_data[perf_col] - baseline_val

        if 'upgrade' in event and pd.notna(event['upgrade']):
            window_data['treatment'] = 'upgrade' if event['upgrade'] else 'downgrade'
        else:
            window_data['treatment'] = 'all'

        window_data['match_change'] = event.get('match_change', np.nan)
        results.append(window_data)

    if not results:
        print("Warning: No events with sufficient data for event study")
        return pd.DataFrame()

    results_df = pd.concat(results, ignore_index=True)

    # Aggregate: mean performance by event_time and treatment
    agg = results_df.groupby(['event_time', 'treatment']).agg({
        'perf_normalized': ['mean', 'std', 'count'],
        perf_col: 'mean',
    }).reset_index()

    # Flatten column names
    agg.columns = ['event_time', 'treatment', 'perf_mean', 'perf_std', 'n_obs', 'raw_perf']
    agg['perf_se'] = agg['perf_std'] / np.sqrt(agg['n_obs'])
    agg['ci_lower'] = agg['perf_mean'] - 1.96 * agg['perf_se']
    agg['ci_upper'] = agg['perf_mean'] + 1.96 * agg['perf_se']

    if did and 'upgrade' in results_df.columns:
        # DiD: difference between upgrade and downgrade at each event time
        pivot = agg.pivot(index='event_time', columns='treatment', values='perf_mean')
        if 'upgrade' in pivot.columns and 'downgrade' in pivot.columns:
            agg_did = pd.DataFrame({
                'event_time': pivot.index,
                'did_effect': pivot['upgrade'] - pivot['downgrade'],
            })
            print("\n=== Difference-in-Differences ===")
            print(agg_did.to_string(index=False))

    return agg


def run_regression_event_study(
    transitions: pd.DataFrame,
    performance: pd.DataFrame,
    perf_col: str = 'tobinq',
    controls: Optional[List[str]] = None,
    window: Tuple[int, int] = (-3, 3),
) -> Dict[str, Any]:
    """
    Regression-based event study with controls.

    Y_{it} = α_i + γ_t + β₁ Post_{it} + β₂ (Post × ΔMatch)_{it} + X'_{it}δ + ε_{it}

    β₂ is the causal effect: "For each unit increase in match quality change,
    what is the effect on firm performance post-transition?"
    """
    try:
        import statsmodels.api as sm
        from statsmodels.regression.linear_model import OLS
    except ImportError:
        print("statsmodels required for regression event study")
        return {}

    pre, post = window
    # Build regression dataset
    rows = []
    for _, event in transitions.iterrows():
        gvkey = event['gvkey']
        t0 = event['transition_year']
        match_change = event.get('match_change', np.nan)

        firm_data = performance[performance['gvkey'] == gvkey].copy()
        firm_data['event_time'] = firm_data['fiscalyear'] - t0
        firm_data = firm_data[(firm_data['event_time'] >= pre) & (firm_data['event_time'] <= post)]

        firm_data['post'] = (firm_data['event_time'] >= 0).astype(int)
        firm_data['match_change'] = match_change
        firm_data['post_x_match_change'] = firm_data['post'] * match_change
        firm_data['event_id'] = f"{gvkey}_{t0}"

        rows.append(firm_data)

    if not rows:
        return {}

    reg_df = pd.concat(rows, ignore_index=True)

    # Run regression
    y = reg_df[perf_col].dropna()
    X_cols = ['post', 'match_change', 'post_x_match_change']
    if controls:
        X_cols.extend([c for c in controls if c in reg_df.columns])

    X = reg_df.loc[y.index, X_cols].dropna()
    y = y.loc[X.index]
    X = sm.add_constant(X)

    try:
        model = OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': reg_df.loc[X.index, 'event_id']})
        print("\n=== Regression Event Study ===")
        print(f"Y = {perf_col}")
        print(model.summary2().tables[1].to_string())

        return {
            'params': model.params.to_dict(),
            'pvalues': model.pvalues.to_dict(),
            'rsquared': model.rsquared,
            'nobs': model.nobs,
            'beta_post_x_match': model.params.get('post_x_match_change', np.nan),
            'se_post_x_match': model.bse.get('post_x_match_change', np.nan),
        }
    except Exception as e:
        print(f"Regression failed: {e}")
        return {}


# =============================================================================
# EXTENSION 7: COMPENSATION DECOMPOSITION VIA EMBEDDINGS
# =============================================================================

def decompose_compensation(
    model: nn.Module,
    data_dict: Dict[str, torch.Tensor],
    df: pd.DataFrame,
    comp_col: str = 'tdc1',
    device: str = 'cpu',
) -> pd.DataFrame:
    """
    Decompose CEO compensation into CEO premium, firm premium, and match surplus.

    TDC1 = f(CEO_embedding) + g(Firm_embedding) + h(CEO_emb × Firm_emb) + ε

    This separates pay into:
    - CEO premium: what the market pays this person regardless of firm
    - Firm premium: what this firm pays any CEO
    - Match surplus: extra pay from a good match (complementarity rent)

    Args:
        model: Trained Two-Tower model with .encode() or forward method
        data_dict: Transformed data
        df: Original DataFrame with compensation column
        comp_col: Compensation column name
        device: Computation device

    Returns:
        DataFrame with decomposed compensation components
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        print("statsmodels required for compensation decomposition")
        return pd.DataFrame()

    model.eval()
    with torch.no_grad():
        # Get embeddings
        if hasattr(model, 'encode'):
            u_firm, v_ceo = model.encode(
                data_dict['firm_numeric'].to(device),
                data_dict['firm_cat'].to(device),
                data_dict['ceo_numeric'].to(device),
                data_dict['ceo_cat'].to(device),
            )
        else:
            # Base model: extract embeddings manually
            f_embs = [emb(data_dict['firm_cat'][:, i].to(device))
                      for i, emb in enumerate(model.firm_embeddings)]
            f_combined = torch.cat([data_dict['firm_numeric'].to(device)] + f_embs, dim=1)
            u_firm = model.firm_tower(f_combined)
            u_firm = F.normalize(u_firm, dim=1)

            c_embs = [emb(data_dict['ceo_cat'][:, i].to(device))
                      for i, emb in enumerate(model.ceo_embeddings)]
            c_combined = torch.cat([data_dict['ceo_numeric'].to(device)] + c_embs, dim=1)
            v_ceo = model.ceo_tower(c_combined)
            v_ceo = F.normalize(v_ceo, dim=1)

    u_firm = u_firm.cpu().numpy()
    v_ceo = v_ceo.cpu().numpy()
    D = u_firm.shape[1]

    # Elementwise interaction (match-specific features)
    interaction = u_firm * v_ceo  # [N, D]

    # Build regression: TDC1 ~ CEO_emb + Firm_emb + Interaction
    comp = pd.to_numeric(df[comp_col], errors='coerce').astype('float64').values if comp_col in df.columns else None
    if comp is None:
        print(f"Warning: '{comp_col}' not found in DataFrame")
        return pd.DataFrame()

    log_comp = np.log1p(np.maximum(comp, 0))
    valid = ~np.isnan(log_comp)

    X_ceo = v_ceo[valid]
    X_firm = u_firm[valid]
    X_interact = interaction[valid]
    X_full = np.hstack([X_ceo, X_firm, X_interact])

    y = log_comp[valid]

    X_full = sm.add_constant(X_full)
    reg = sm.OLS(y, X_full).fit()

    print("\n=== Compensation Decomposition ===")
    print(f"R² = {reg.rsquared:.4f}")
    print(f"N = {reg.nobs:.0f}")

    # Variance decomposition using fitted values
    # CEO contribution: prediction from CEO embeddings only
    X_ceo_only = np.hstack([X_ceo, np.zeros_like(X_firm), np.zeros_like(X_interact)])
    X_ceo_only = sm.add_constant(X_ceo_only)
    pred_ceo = reg.predict(X_ceo_only)

    # Firm contribution
    X_firm_only = np.hstack([np.zeros_like(X_ceo), X_firm, np.zeros_like(X_interact)])
    X_firm_only = sm.add_constant(X_firm_only)
    pred_firm = reg.predict(X_firm_only)

    # Match surplus (interaction contribution)
    X_int_only = np.hstack([np.zeros_like(X_ceo), np.zeros_like(X_firm), X_interact])
    X_int_only = sm.add_constant(X_int_only)
    pred_match = reg.predict(X_int_only)

    total_var = np.var(y)
    results = {
        'total_var_log_comp': total_var,
        'ceo_premium_var': np.var(pred_ceo) / total_var,
        'firm_premium_var': np.var(pred_firm) / total_var,
        'match_surplus_var': np.var(pred_match) / total_var,
        'residual_var': 1 - reg.rsquared,
        'r_squared': reg.rsquared,
    }

    print(f"\nVariance Decomposition of log(TDC1):")
    print(f"  CEO premium:    {results['ceo_premium_var']:.1%}")
    print(f"  Firm premium:   {results['firm_premium_var']:.1%}")
    print(f"  Match surplus:  {results['match_surplus_var']:.1%}")
    print(f"  Residual:       {results['residual_var']:.1%}")

    # Build output DataFrame
    comp_df = df[valid].copy() if isinstance(valid, np.ndarray) else df.copy()
    comp_df = comp_df.reset_index(drop=True)
    comp_df['log_comp'] = y
    comp_df['pred_ceo_premium'] = pred_ceo
    comp_df['pred_firm_premium'] = pred_firm
    comp_df['pred_match_surplus'] = pred_match
    comp_df['pred_total'] = reg.fittedvalues
    comp_df['residual'] = reg.resid

    return comp_df


def compensation_by_match_quality(
    comp_df: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    Tabulate compensation decomposition by match quality quantile.

    Shows how CEO premium, firm premium, and match surplus vary
    across the match quality distribution.
    """
    if 'match_means' not in comp_df.columns:
        print("match_means not found — skipping quantile analysis")
        return pd.DataFrame()

    comp_df['match_quintile'] = pd.qcut(
        comp_df['match_means'], n_quantiles, labels=False, duplicates='drop'
    ) + 1

    summary = comp_df.groupby('match_quintile').agg({
        'log_comp': 'mean',
        'pred_ceo_premium': 'mean',
        'pred_firm_premium': 'mean',
        'pred_match_surplus': 'mean',
        'match_means': 'mean',
    }).round(3)

    print("\n=== Compensation by Match Quality Quintile ===")
    print(summary.to_string())
    return summary


# =============================================================================
# EXTENSION 9: GENERATIVE COUNTERFACTUALS
# =============================================================================

def generate_counterfactuals(
    model: nn.Module,
    data_dict: Dict[str, torch.Tensor],
    df: pd.DataFrame,
    firm_id_col: str = 'gvkey',
    ceo_id_col: str = 'match_exec_id',
    top_k: int = 10,
    device: str = 'cpu',
) -> pd.DataFrame:
    """
    Generate counterfactual match scores: "What if CEO X was at Firm Y?"

    For each firm, ranks all available CEOs by predicted match quality
    and identifies the optimal CEO and the worst CEO.

    Args:
        model: Trained Two-Tower model
        data_dict: Transformed data
        df: Original DataFrame
        top_k: Number of top/bottom matches to report

    Returns:
        DataFrame with counterfactual rankings
    """
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'encode'):
            u_firm, v_ceo = model.encode(
                data_dict['firm_numeric'].to(device),
                data_dict['firm_cat'].to(device),
                data_dict['ceo_numeric'].to(device),
                data_dict['ceo_cat'].to(device),
            )
        else:
            f_embs = [emb(data_dict['firm_cat'][:, i].to(device))
                      for i, emb in enumerate(model.firm_embeddings)]
            f_combined = torch.cat([data_dict['firm_numeric'].to(device)] + f_embs, dim=1)
            u_firm = model.firm_tower(f_combined)
            u_firm = F.normalize(u_firm, dim=1)

            c_embs = [emb(data_dict['ceo_cat'][:, i].to(device))
                      for i, emb in enumerate(model.ceo_embeddings)]
            c_combined = torch.cat([data_dict['ceo_numeric'].to(device)] + c_embs, dim=1)
            v_ceo = model.ceo_tower(c_combined)
            v_ceo = F.normalize(v_ceo, dim=1)

        logit_scale = model.logit_scale.exp().cpu().item() if hasattr(model, 'logit_scale') else 14.3

    u_firm = u_firm.cpu()
    v_ceo = v_ceo.cpu()

    # Get unique firms and CEOs (use latest year for each)
    df = df.reset_index(drop=True)
    latest_year = df.groupby(firm_id_col)['fiscalyear'].idxmax() if 'fiscalyear' in df.columns else df.index

    # Sample firms (cap at 200 for memory)
    firm_indices = df.loc[latest_year].index.values[:200] if hasattr(latest_year, 'values') else latest_year[:200]

    # Get unique CEO embeddings
    ceo_unique = df.drop_duplicates(subset=ceo_id_col, keep='last') if ceo_id_col in df.columns else df
    ceo_indices = ceo_unique.index.values[:1000]  # Cap

    firm_embs = u_firm[firm_indices]  # [F, D]
    ceo_embs = v_ceo[ceo_indices]     # [C, D]

    # Full cross-match: [F, C]
    cross_scores = torch.mm(firm_embs, ceo_embs.t()) * logit_scale

    print(f"\n=== Counterfactual Analysis ===")
    print(f"Cross-matching {len(firm_indices)} firms × {len(ceo_indices)} CEOs")

    results = []
    for i, firm_idx in enumerate(firm_indices):
        firm_id = df.loc[firm_idx, firm_id_col] if firm_id_col in df.columns else firm_idx
        actual_ceo = df.loc[firm_idx, ceo_id_col] if ceo_id_col in df.columns else None
        actual_match = df.loc[firm_idx, 'match_means'] if 'match_means' in df.columns else None

        scores = cross_scores[i].numpy()
        ranking = np.argsort(-scores)

        # Best match
        best_idx = ranking[0]
        best_ceo_df_idx = ceo_indices[best_idx]
        best_ceo_id = df.loc[best_ceo_df_idx, ceo_id_col] if ceo_id_col in df.columns else best_idx

        # Worst match
        worst_idx = ranking[-1]
        worst_ceo_df_idx = ceo_indices[worst_idx]
        worst_ceo_id = df.loc[worst_ceo_df_idx, ceo_id_col] if ceo_id_col in df.columns else worst_idx

        # Actual CEO's rank
        actual_rank = None
        if actual_ceo is not None:
            actual_in_ceo_list = df.loc[ceo_indices, ceo_id_col] == actual_ceo
            if actual_in_ceo_list.any():
                actual_ceo_pos = actual_in_ceo_list.values.argmax()
                actual_rank = int((ranking == actual_ceo_pos).argmax()) + 1

        results.append({
            'firm_id': firm_id,
            'actual_ceo': actual_ceo,
            'actual_match': actual_match,
            'actual_rank': actual_rank,
            'best_ceo': best_ceo_id,
            'best_score': float(scores[ranking[0]]),
            'worst_ceo': worst_ceo_id,
            'worst_score': float(scores[ranking[-1]]),
            'score_range': float(scores[ranking[0]] - scores[ranking[-1]]),
            'match_improvement': float(scores[ranking[0]]) - (actual_match if actual_match else 0),
        })

    cf_df = pd.DataFrame(results)

    print(f"\nMatch Improvement Statistics:")
    print(f"  Mean actual rank: {cf_df['actual_rank'].mean():.1f} / {len(ceo_indices)}")
    print(f"  Median improvement: {cf_df['match_improvement'].median():.3f}")
    print(f"  % firms with better CEO available: {(cf_df['match_improvement'] > 0).mean():.1%}")

    return cf_df


def reallocation_simulation(
    model: nn.Module,
    data_dict: Dict[str, torch.Tensor],
    df: pd.DataFrame,
    n_firms: int = 100,
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    Hungarian algorithm reallocation: optimally reassign CEOs to firms.

    Answers: "If we could freely reassign all CEOs, how much better
    would total match quality be?"
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        print("scipy required for Hungarian reallocation")
        return {}

    model.eval()
    with torch.no_grad():
        if hasattr(model, 'encode'):
            u_firm, v_ceo = model.encode(
                data_dict['firm_numeric'][:n_firms].to(device),
                data_dict['firm_cat'][:n_firms].to(device),
                data_dict['ceo_numeric'][:n_firms].to(device),
                data_dict['ceo_cat'][:n_firms].to(device),
            )
        else:
            return {}

    u_firm = u_firm.cpu().numpy()
    v_ceo = v_ceo.cpu().numpy()

    # Cost matrix (negative similarity for minimization)
    cost_matrix = -np.dot(u_firm, v_ceo.T)

    # Solve
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Current assignment (diagonal)
    current_scores = np.diag(-cost_matrix)
    optimal_scores = -cost_matrix[row_ind, col_ind]

    results = {
        'n_firms': n_firms,
        'current_total_match': float(current_scores.sum()),
        'optimal_total_match': float(optimal_scores.sum()),
        'improvement': float(optimal_scores.sum() - current_scores.sum()),
        'improvement_pct': float((optimal_scores.mean() - current_scores.mean()) / abs(current_scores.mean()) * 100),
        'n_reassigned': int(np.sum(row_ind != col_ind)),
        'pct_reassigned': float(np.mean(row_ind != col_ind) * 100),
    }

    print(f"\n=== Optimal Reallocation ({n_firms} firms) ===")
    print(f"  Current total match:  {results['current_total_match']:.2f}")
    print(f"  Optimal total match:  {results['optimal_total_match']:.2f}")
    print(f"  Improvement:          {results['improvement']:.2f} ({results['improvement_pct']:.1f}%)")
    print(f"  CEOs reassigned:      {results['n_reassigned']} ({results['pct_reassigned']:.1f}%)")

    return results


# =============================================================================
# EXTENSION 10: TRANSFER LEARNING TO PRIVATE FIRMS
# =============================================================================

class TransferableTwoTower(nn.Module):
    """
    Two-Tower model designed for transfer learning.

    Key differences from base model:
    1. Tower weights can be frozen independently
    2. Includes adapter layers for domain shift
    3. CEO tower is designed to transfer (trained on public firms,
       applied to private firms where we lack match_means but have CEO features)
    """

    def __init__(self, base_model: nn.Module, config, adapter_dim: int = 16):
        super().__init__()
        self.base = base_model
        self._device = next(base_model.parameters()).device

        # Freeze base towers
        for param in self.base.parameters():
            param.requires_grad = False

        # Adapter layers (lightweight fine-tuning for new domain)
        self.firm_adapter = nn.Sequential(
            nn.Linear(config.LATENT_DIM, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, config.LATENT_DIM),
        )
        self.ceo_adapter = nn.Sequential(
            nn.Linear(config.LATENT_DIM, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, config.LATENT_DIM),
        )

        # Skip connections ensure adapters start as identity
        nn.init.zeros_(self.firm_adapter[2].weight)
        nn.init.zeros_(self.firm_adapter[2].bias)
        nn.init.zeros_(self.ceo_adapter[2].weight)
        nn.init.zeros_(self.ceo_adapter[2].bias)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, f_numeric, f_cat, c_numeric, c_cat):
        device = self._device
        f_numeric = f_numeric.to(device)
        f_cat = f_cat.to(device)
        c_numeric = c_numeric.to(device)
        c_cat = c_cat.to(device)

        # Get base embeddings (frozen)
        with torch.no_grad():
            f_embs = [emb(f_cat[:, i]) for i, emb in enumerate(self.base.firm_embeddings)]
            f_combined = torch.cat([f_numeric] + f_embs, dim=1)
            u_firm = self.base.firm_tower(f_combined)
            u_firm = F.normalize(u_firm, dim=1)

            c_embs = [emb(c_cat[:, i]) for i, emb in enumerate(self.base.ceo_embeddings)]
            c_combined = torch.cat([c_numeric] + c_embs, dim=1)
            v_ceo = self.base.ceo_tower(c_combined)
            v_ceo = F.normalize(v_ceo, dim=1)

        # Adapt (only adapter weights are trainable)
        u_firm = u_firm + self.firm_adapter.to(device)(u_firm)  # Skip connection
        v_ceo = v_ceo + self.ceo_adapter.to(device)(v_ceo)

        u_firm = F.normalize(u_firm, dim=1)
        v_ceo = F.normalize(v_ceo, dim=1)

        logit_scale = self.logit_scale.exp()
        match_score = (u_firm * v_ceo).sum(dim=1, keepdim=True) * logit_scale

        return match_score

    def predict_private_firm_match(
        self,
        ceo_features: Dict[str, torch.Tensor],
        firm_features: Dict[str, torch.Tensor],
    ) -> float:
        """
        Predict match quality for a private firm using transferred model.

        CEO features come from BoardEx/CIQ (available for private firms).
        Firm features are limited (no Compustat) — use what's available.
        """
        self.eval()
        with torch.no_grad():
            score = self.forward(
                firm_features['numeric'],
                firm_features['categorical'],
                ceo_features['numeric'],
                ceo_features['categorical'],
            )
        return float(score.item())


def prepare_transfer_data(
    private_df: pd.DataFrame,
    public_processor: Any,
    available_firm_cols: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Prepare private firm data for transfer learning.

    For private firms, we typically have:
    - CEO features from BoardEx/CIQ (same as public)
    - Limited firm features (no quarterly Compustat, no CRSP)
    - Some features from CIQ (revenue, employees, industry)

    Strategy: Use same feature engineering but handle missing firm features
    by imputing with industry medians from public firm training data.
    """
    df = private_df.copy()

    # Use public firm encoders/scalers where possible
    if hasattr(public_processor, 'firm_scaler'):
        # Get feature names from the public processor
        all_firm_cols = public_processor.config.FIRM_NUMERIC_COLS

        # Fill missing firm features with 0 (will be scaled)
        for col in all_firm_cols:
            if col not in df.columns:
                df[col] = 0  # Will be imputed or handled

    # Transform using public processor (shared encoding)
    try:
        data_dict = public_processor.transform(df)
        print(f"Prepared {len(df)} private firm observations for transfer")
        return data_dict
    except Exception as e:
        print(f"Transfer data preparation failed: {e}")
        print("Falling back to zero-imputation for missing features")
        return {}
