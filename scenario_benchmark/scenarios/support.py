"""Shared helpers used by the easy calibration scenarios.

Kept minimal and consumed by at least one live scenario per the
no-speculative-dead-code rule.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from scenario_benchmark.sample_scaffold import SampleScaffold
from scenario_benchmark.scenarios.result import ScenarioSet
from scenario_benchmark.truth_schemas import InstanceManifestRow


def pair_swap_labels(
    *, gen_roi: np.ndarray, gen_comp: np.ndarray,
    source_positions: np.ndarray, host_positions: np.ndarray,
    moved_to_roi: str,
) -> None:
    """In-place pair-swap of (roi, comp) labels between paired
    source and host positions; hosts additionally get their ROI
    overwritten to ``moved_to_roi``. Ascending-index canonical
    pairing.

    Consumed by scenarios that perform generated-ROI relabelling.
    Not used by the within-anchor-ROI calibration scenarios.
    """
    sources_sorted = np.array(sorted(int(i) for i in source_positions), dtype=int)
    hosts_sorted = np.array(sorted(int(i) for i in host_positions), dtype=int)
    if sources_sorted.size != hosts_sorted.size:
        raise ValueError(
            f"pair_swap_labels: sources ({sources_sorted.size}) and "
            f"hosts ({hosts_sorted.size}) must have equal size."
        )
    src_prev_comp = gen_comp[sources_sorted].copy()
    host_prev_roi = gen_roi[hosts_sorted].copy()
    host_prev_comp = gen_comp[hosts_sorted].copy()
    gen_roi[hosts_sorted] = moved_to_roi
    gen_comp[hosts_sorted] = src_prev_comp
    gen_roi[sources_sorted] = host_prev_roi
    gen_comp[sources_sorted] = host_prev_comp


def deterministic_broad_anchor(
    scaffold: SampleScaffold, roi_indices: np.ndarray, *, k: int,
) -> np.ndarray:
    """Canonical broad connected k-subset of a ROI. Thin wrapper on
    :func:`_grow_connected_k_within_canonical_component` with no
    exclusion."""
    return _grow_connected_k_within_canonical_component(
        scaffold, roi_indices, k=k, excluded=frozenset(),
    )


def deterministic_host_patch(
    scaffold: SampleScaffold, roi_indices: np.ndarray,
    *, k: int, excluded: np.ndarray,
) -> np.ndarray:
    """Canonical broad connected k-subset of ``roi_indices \\ excluded``
    — the exclusion-aware wrapper on the same shared primitive used
    for the A anchor. Empty when no component can hold ``k`` disjoint
    from ``excluded``."""
    return _grow_connected_k_within_canonical_component(
        scaffold, roi_indices, k=k,
        excluded=frozenset(int(x) for x in excluded),
    )


def _grow_connected_k_within_canonical_component(
    scaffold: SampleScaffold, roi_indices: np.ndarray,
    *, k: int, excluded,
) -> np.ndarray:
    """Shared primitive: iterate canonical source-component groups
    (ordered by min ``source_spot_id``). For each, compute
    connected components of the residual after ``excluded`` is
    removed and try to grow a connected k-subset from each residual
    piece's most-interior seed. Return the first successful growth;
    empty array if nothing fits."""
    if roi_indices.size == 0 or k <= 0 or k > roi_indices.size:
        return np.array([], dtype=int)
    spot_ids = scaffold.spot_ids.astype(str)
    components = _components_by_source_component_id(scaffold, roi_indices)
    components.sort(key=lambda c: min(str(spot_ids[i]) for i in c))
    for source_comp in components:
        residual = [int(i) for i in source_comp if int(i) not in excluded]
        if len(residual) < k:
            continue
        residual_pieces = _connected_pieces_within(
            scaffold, np.array(sorted(residual), dtype=int),
        )
        residual_pieces.sort(
            key=lambda piece: min(str(spot_ids[i]) for i in piece)
        )
        for piece in residual_pieces:
            if len(piece) < k:
                continue
            chosen = np.array(sorted(piece), dtype=int)
            grown = _grow_from_interior_seed(
                scaffold, chosen, k=k, spot_ids=spot_ids,
            )
            if grown.size == k:
                return grown
    return np.array([], dtype=int)


def _connected_pieces_within(
    scaffold: SampleScaffold, indices: np.ndarray,
) -> List[List[int]]:
    """Connected components of the subgraph induced on ``indices``.
    Returned as lists of sample-wide indices."""
    idx_set = set(int(i) for i in indices)
    visited: Set[int] = set()
    pieces: List[List[int]] = []
    for start in indices:
        s = int(start)
        if s in visited:
            continue
        stack = [s]
        piece: List[int] = []
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u); piece.append(u)
            for v in scaffold.adj[u]:
                v = int(v)
                if v in idx_set and v not in visited:
                    stack.append(v)
        pieces.append(sorted(piece))
    return pieces


def _grow_from_interior_seed(
    scaffold: SampleScaffold, chosen: np.ndarray,
    *, k: int, spot_ids: np.ndarray,
) -> np.ndarray:
    """Grow a connected k-subset from the induced-graph most-interior
    seed of ``chosen``. Returns empty if the induced subgraph is
    empty or the growth can't reach ``k``."""
    local_adj = scaffold.restricted_adj(chosen)
    if not local_adj:
        return np.array([], dtype=int)
    local_to_global = {i: int(chosen[i]) for i in range(len(local_adj))}
    def _local_id_key(li: int) -> str:
        return spot_ids[local_to_global[li]]
    boundary_local = _induced_graph_boundary(local_adj)
    interior_hops = multi_source_bfs(local_adj, boundary_local)
    reachable = [
        li for li in range(len(local_adj))
        if interior_hops[li] != _UNREACHABLE
    ]
    if not reachable:
        return np.array([], dtype=int)
    max_hops = max(int(interior_hops[li]) for li in reachable)
    seed_candidates = [
        li for li in reachable if int(interior_hops[li]) == max_hops
    ]
    seed_local = min(seed_candidates, key=_local_id_key)
    order = _bfs_grow_by_spot_id(
        local_adj, seed_local, k, spot_id_key=_local_id_key,
    )
    if len(order) != k:
        return np.array([], dtype=int)
    return np.array(
        sorted(int(local_to_global[li]) for li in order), dtype=int,
    )


def _components_by_source_component_id(
    scaffold: SampleScaffold, roi_indices: np.ndarray,
) -> List[List[int]]:
    """Group ``roi_indices`` by the already-materialised
    ``scaffold.source_component_id``. Each returned list holds
    sample-wide indices; overall order is deterministic (by ascending
    minimum ``source_spot_id`` in the caller)."""
    comp_ids = scaffold.source_component_id.astype(str)
    buckets: Dict[str, List[int]] = {}
    for i in roi_indices:
        buckets.setdefault(str(comp_ids[int(i)]), []).append(int(i))
    return [sorted(bucket) for bucket in buckets.values()]


def _induced_graph_boundary(local_adj: List[List[int]]) -> List[int]:
    """Induced-subgraph boundary: nodes whose intra-subgraph degree is
    strictly less than the maximum. Independent of neighbour identity."""
    if not local_adj:
        return []
    degrees = [len(neighs) for neighs in local_adj]
    max_deg = max(degrees)
    if max_deg == 0:
        return list(range(len(local_adj)))
    boundary = [i for i, d in enumerate(degrees) if d < max_deg]
    return boundary or list(range(len(local_adj)))


def _bfs_grow_by_spot_id(
    local_adj: List[List[int]], seed: int, target: int,
    *, spot_id_key,
) -> List[int]:
    """BFS from ``seed`` growing until ``target`` connected nodes.
    Frontier expansion admits new nodes in ascending ``spot_id_key``
    order at each hop (deterministic, geometry-only)."""
    n = len(local_adj)
    if n == 0 or seed < 0 or seed >= n:
        return []
    visited: Set[int] = {int(seed)}
    order: List[int] = [int(seed)]
    frontier: List[int] = [int(seed)]
    while frontier and len(order) < target:
        cands: Set[int] = set()
        for u in frontier:
            for v in local_adj[u]:
                if int(v) not in visited:
                    cands.add(int(v))
        if not cands:
            break
        ordered = sorted(cands, key=spot_id_key)
        nxt: List[int] = []
        for v in ordered:
            if len(order) >= target:
                break
            visited.add(v); order.append(v); nxt.append(v)
        frontier = nxt
    return order


def bfs_patch(
    local_adj: List[List[int]], *, seed: int, target: int,
    excluded: Optional[Set[int]] = None,
) -> List[int]:
    """BFS from ``seed`` growing up to ``target`` connected nodes,
    optionally skipping positions in ``excluded``. Deterministic
    tie-break by ascending local index."""
    excluded = excluded or set()
    n = len(local_adj)
    if n == 0:
        return []
    seed = max(0, min(int(seed), n - 1))
    if seed in excluded:
        for v in local_adj[seed]:
            if int(v) not in excluded:
                seed = int(v); break
        else:
            return []
    visited = {int(seed)}
    order = [int(seed)]
    frontier = [int(seed)]
    while frontier and len(order) < target:
        nxt: List[int] = []
        cands: Set[int] = set()
        for u in frontier:
            for v in local_adj[u]:
                if int(v) in visited or int(v) in excluded:
                    continue
                cands.add(int(v))
        for v in sorted(cands):
            if len(order) >= target:
                break
            visited.add(int(v)); order.append(int(v)); nxt.append(int(v))
        frontier = nxt
    return order


# ---------------------------------------------------------------------------
# Multi-source BFS on a local (restricted) adjacency graph
# ---------------------------------------------------------------------------

_UNREACHABLE = 10 ** 9


def multi_source_bfs(
    local_adj: List[List[int]], sources: Sequence[int],
) -> np.ndarray:
    """Return per-local-node hop distance from the nearest source.
    Non-reachable nodes hold ``_UNREACHABLE``."""
    n = len(local_adj)
    dist = np.full(n, _UNREACHABLE, dtype=np.int64)
    frontier: List[int] = []
    for s in sources:
        s = int(s)
        if 0 <= s < n and dist[s] > 0:
            dist[s] = 0
            frontier.append(s)
    d = 0
    while frontier:
        d += 1
        nxt: List[int] = []
        for u in frontier:
            for v in local_adj[u]:
                if dist[int(v)] == _UNREACHABLE:
                    dist[int(v)] = d
                    nxt.append(int(v))
        frontier = nxt
    return dist


# ---------------------------------------------------------------------------
# All-infeasible scenario set
# ---------------------------------------------------------------------------

def build_all_infeasible_set(
    *, scaffold: SampleScaffold, counterfactual_set_id: str,
    scenario_name: str,
    roles: Sequence[str],
    feature_a_id: str, feature_b_id: str,
    nuisance_id: str, nuisance_parameters: dict,
    requested_parameters: dict,
    rng_seed: int, partition: str, related_sample_group_id: str,
    reason: str,
) -> ScenarioSet:
    """Emit an all-infeasible ScenarioSet (no spot rows)."""
    from scenario_benchmark.scenario_assembly import _build_manifest_row
    instances: List[InstanceManifestRow] = []
    for role in roles:
        iid = f"{counterfactual_set_id}::{role}"
        instances.append(_build_manifest_row(
            scenario_instance_id=iid,
            counterfactual_set_id=counterfactual_set_id,
            counterfactual_role=role, scenario_name=scenario_name,
            feature_a_id=feature_a_id, feature_b_id=feature_b_id,
            nuisance_id=nuisance_id, nuisance_parameters=nuisance_parameters,
            scaffold=scaffold, requested_parameters=requested_parameters,
            realized_parameters={}, role_specific_metadata={},
            generator_infeasible=True, fallback_reason=reason,
            rng_seed=rng_seed, partition=partition,
            related_sample_group_id=related_sample_group_id,
        ))
    return ScenarioSet(
        counterfactual_set_id=counterfactual_set_id,
        instances=instances, spot_truth_rows=[],
    )
