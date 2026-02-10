"""
Extension 8: Board Interlock Network Features

Constructs graph-based features from the BoardEx employment data:
1. CEO centrality in the board interlock network
2. Shared-board connections with current firm's directors
3. Network distance to prior CEOs of the same firm
4. Information bridge score (connects otherwise separate clusters)

Uses the bipartite graph Person ↔ Company projected to Person-Person.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import defaultdict


def build_interlock_graph(
    employment: pd.DataFrame,
    year: Optional[int] = None,
    lookback_years: int = 5,
) -> Dict[int, set]:
    """
    Build the board interlock adjacency list from BoardEx employment.

    Two directors are connected if they served on the same board
    within the lookback window.

    Args:
        employment: BoardEx employment records (directorid, companyid, datestartrole, dateendrole)
        year: Reference year (filters to active roles). If None, uses all.
        lookback_years: How far back to look for overlapping tenures.

    Returns:
        Dict mapping directorid → set of connected directorids
    """
    emp = employment.copy()

    # Parse dates
    for col in ['datestartrole', 'dateendrole']:
        if col in emp.columns:
            emp[col] = pd.to_datetime(emp[col], errors='coerce')

    # Filter to relevant timeframe
    if year is not None:
        cutoff_start = pd.Timestamp(f'{year - lookback_years}-01-01')
        cutoff_end = pd.Timestamp(f'{year}-12-31')

        # Active during window: started before end AND (ended after start OR still active)
        mask = (emp['datestartrole'] <= cutoff_end)
        if 'dateendrole' in emp.columns:
            mask = mask & (
                (emp['dateendrole'] >= cutoff_start) |
                emp['dateendrole'].isna()
            )
        emp = emp[mask]

    # Filter to board-level roles
    board_mask = emp['rolename'].str.contains(
        'Director|Board|Chairman|CEO|President|CFO|COO',
        case=False, na=False
    )
    emp = emp[board_mask]

    # Build bipartite: company → set of directors
    company_directors = defaultdict(set)
    for _, row in emp.iterrows():
        if pd.notna(row['directorid']) and pd.notna(row['companyid']):
            company_directors[row['companyid']].add(int(row['directorid']))

    # Project to person-person adjacency
    adjacency = defaultdict(set)
    for company, directors in company_directors.items():
        for d1 in directors:
            for d2 in directors:
                if d1 != d2:
                    adjacency[d1].add(d2)

    return dict(adjacency)


def compute_centrality_features(
    adjacency: Dict[int, set],
    target_ids: List[int],
) -> pd.DataFrame:
    """
    Compute network centrality measures for target directors.

    Args:
        adjacency: Person-to-person adjacency list
        target_ids: Director IDs to compute features for

    Returns:
        DataFrame with directorid and centrality features
    """
    results = []

    for did in target_ids:
        neighbors = adjacency.get(did, set())
        degree = len(neighbors)

        # Degree centrality
        n_nodes = len(adjacency)
        degree_centrality = degree / max(n_nodes - 1, 1)

        # Local clustering coefficient
        if degree >= 2:
            neighbor_list = list(neighbors)
            n_triangles = 0
            n_possible = degree * (degree - 1) / 2
            for i, n1 in enumerate(neighbor_list):
                for n2 in neighbor_list[i + 1:]:
                    if n2 in adjacency.get(n1, set()):
                        n_triangles += 1
            clustering = n_triangles / n_possible if n_possible > 0 else 0
        else:
            clustering = 0

        # 2-hop reach (size of 2-neighborhood)
        two_hop = set()
        for n in neighbors:
            two_hop.update(adjacency.get(n, set()))
        two_hop.discard(did)
        two_hop_reach = len(two_hop)

        # Structural holes (constraint score proxy)
        # Burt's constraint: how much of your network is redundant
        if degree > 0:
            redundancy = []
            for n in neighbors:
                n_neighbors = adjacency.get(n, set())
                overlap = len(neighbors & n_neighbors)
                redundancy.append(overlap / degree)
            constraint = np.mean(redundancy)
        else:
            constraint = 1.0  # Fully constrained (no connections)

        # Information bridge score: 1 - constraint
        bridge_score = 1.0 - constraint

        results.append({
            'directorid': did,
            'degree_centrality': degree_centrality,
            'network_degree': degree,
            'clustering_coeff': clustering,
            'two_hop_reach': two_hop_reach,
            'constraint': constraint,
            'bridge_score': bridge_score,
            'log_network_size': np.log1p(degree),
        })

    return pd.DataFrame(results)


def compute_shared_board_features(
    employment: pd.DataFrame,
    ceo_director_id: int,
    firm_company_id: int,
    year: int,
) -> Dict[str, float]:
    """
    Compute features about a CEO's connection to a firm's existing board.

    Args:
        employment: BoardEx employment records
        ceo_director_id: Director ID of the CEO
        firm_company_id: Company ID of the firm
        year: Reference year

    Returns:
        Dict of connection features
    """
    emp = employment.copy()
    for col in ['datestartrole', 'dateendrole']:
        if col in emp.columns:
            emp[col] = pd.to_datetime(emp[col], errors='coerce')

    # Find current board members of the firm
    cutoff = pd.Timestamp(f'{year}-12-31')
    firm_board = emp[
        (emp['companyid'] == firm_company_id) &
        (emp['datestartrole'] <= cutoff) &
        ((emp['dateendrole'] >= cutoff) | emp['dateendrole'].isna())
    ]['directorid'].unique()

    # Find companies where the CEO has served
    ceo_companies = emp[emp['directorid'] == ceo_director_id]['companyid'].unique()

    # Find companies where board members have served
    board_companies = emp[emp['directorid'].isin(firm_board)]['companyid'].unique()

    # Shared boards: companies where both the CEO and any board member served
    shared = set(ceo_companies) & set(board_companies)

    # Direct connections: board members the CEO has co-served with
    ceo_co_served = set()
    for company in ceo_companies:
        company_people = emp[emp['companyid'] == company]['directorid'].unique()
        ceo_co_served.update(company_people)
    direct_connections = len(set(firm_board) & ceo_co_served)

    return {
        'n_shared_boards': len(shared),
        'n_direct_connections': direct_connections,
        'board_connection_pct': direct_connections / max(len(firm_board), 1),
        'ceo_on_firm_board': int(ceo_director_id in firm_board),
    }


def construct_full_network_features(
    employment: pd.DataFrame,
    target_ceo_ids: List[int],
    year: int = 2020,
    lookback: int = 5,
) -> pd.DataFrame:
    """
    Full pipeline: build graph → compute features for target CEOs.

    Args:
        employment: BoardEx employment records
        target_ceo_ids: Director IDs to compute features for
        year: Reference year
        lookback: Years to look back for overlapping tenures

    Returns:
        DataFrame with network features for each CEO
    """
    print(f"Building interlock graph for year {year} (lookback={lookback})...")
    adjacency = build_interlock_graph(employment, year=year, lookback_years=lookback)
    print(f"  Graph: {len(adjacency)} nodes, {sum(len(v) for v in adjacency.values()) // 2} edges")

    print(f"Computing centrality for {len(target_ceo_ids)} CEOs...")
    features = compute_centrality_features(adjacency, target_ceo_ids)

    return features


def compute_ceo_firm_network_distance(
    adjacency: Dict[int, set],
    ceo_id: int,
    firm_board_ids: List[int],
    max_hops: int = 3,
) -> int:
    """
    BFS shortest path from CEO to any member of the firm's board.

    Returns:
        Minimum number of hops (1 = direct connection, max_hops+1 = unreachable)
    """
    if ceo_id in firm_board_ids:
        return 0

    visited = {ceo_id}
    frontier = {ceo_id}
    target_set = set(firm_board_ids)

    for hop in range(1, max_hops + 1):
        next_frontier = set()
        for node in frontier:
            for neighbor in adjacency.get(node, set()):
                if neighbor in target_set:
                    return hop
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    return max_hops + 1  # Unreachable
