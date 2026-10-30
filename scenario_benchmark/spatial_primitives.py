"""Small graph primitives shared by active scenario generators."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from scenario_benchmark.sample_scaffold import SampleScaffold
from scenario_benchmark.scenario_assembly import StructureAssignment


Components = Tuple[Tuple[int, ...], ...]


def flatten_components(components: Sequence[Sequence[int]]) -> np.ndarray:
    return np.array(
        sorted(int(p) for component in components for p in component),
        dtype=int,
    )


def structures_from_components(
    components: Sequence[Sequence[int]], *, kind: str, prefix: str,
) -> list[StructureAssignment]:
    return [
        StructureAssignment(
            positions=np.array(sorted(int(p) for p in component), dtype=int),
            structure_kind=kind,
            structure_id=f"{prefix}_{i:04d}",
        )
        for i, component in enumerate(components)
    ]


def roi_components(
    scaffold: SampleScaffold, positions: Sequence[int],
) -> Components:
    """Group positions by the scaffold's connected-component labels."""
    buckets: dict[str, list[int]] = {}
    labels = scaffold.source_component_id.astype(str)
    for position in positions:
        p = int(position)
        buckets.setdefault(labels[p], []).append(p)
    ordered = sorted(
        (tuple(sorted(values)) for values in buckets.values()),
        key=lambda component: (-len(component), component),
    )
    return tuple(ordered)


def roi_interior_positions(
    scaffold: SampleScaffold, roi_positions: Sequence[int],
) -> np.ndarray:
    """Positions whose full one-hop neighbourhood remains in the ROI."""
    roi_set = {int(p) for p in roi_positions}
    interior = [
        p for p in roi_set
        if scaffold.adj[p] and all(int(v) in roi_set for v in scaffold.adj[p])
    ]
    return np.array(sorted(interior), dtype=int)


def grow_disjoint_connected_patches(
    scaffold: SampleScaffold,
    *,
    allowed_positions: Sequence[int],
    sizes: Sequence[int],
    rng: np.random.Generator,
) -> Optional[Components]:
    """Grow exact-size connected patches separated by a one-hop buffer."""
    allowed = {int(p) for p in allowed_positions}
    claimed: set[int] = set()
    buffered: set[int] = set()
    patches: list[Tuple[int, ...]] = []
    for size in (int(s) for s in sizes):
        available = allowed - claimed - buffered
        if len(available) < size:
            return None
        seeds = np.array(sorted(available), dtype=int)
        placed: Optional[Tuple[int, ...]] = None
        for seed in rng.permutation(seeds):
            grown = _bfs_grow(
                scaffold, int(seed), target=size, allowed=available,
            )
            if len(grown) == size:
                placed = tuple(sorted(grown))
                break
        if placed is None:
            return None
        patches.append(placed)
        claimed.update(placed)
        for p in placed:
            buffered.update(
                int(v) for v in scaffold.adj[p]
                if int(v) in allowed and int(v) not in claimed
            )
    return tuple(patches)


def pack_connected_patches(
    scaffold: SampleScaffold,
    *,
    host_components: Sequence[Sequence[int]],
    sizes: Sequence[int],
    rng: np.random.Generator,
) -> Optional[Components]:
    """Pack exact-size patches into host components without crossing them."""
    states = [
        {"positions": {int(p) for p in host}, "claimed": set(), "buffered": set()}
        for host in host_components if host
    ]
    patches: list[Tuple[int, ...]] = []
    for size in sorted((int(s) for s in sizes), reverse=True):
        eligible = [
            i for i, state in enumerate(states)
            if len(state["positions"] - state["claimed"] - state["buffered"])
            >= size
        ]
        placed = None
        for state_index in rng.permutation(eligible):
            state = states[int(state_index)]
            available = state["positions"] - state["claimed"] - state["buffered"]
            for seed in rng.permutation(np.array(sorted(available), dtype=int)):
                grown = _bfs_grow(
                    scaffold, int(seed), target=size, allowed=available,
                )
                if len(grown) != size:
                    continue
                placed = tuple(sorted(grown))
                state["claimed"].update(placed)
                for p in placed:
                    state["buffered"].update(
                        int(v) for v in scaffold.adj[p]
                        if int(v) in state["positions"]
                        and int(v) not in state["claimed"]
                    )
                break
            if placed is not None:
                break
        if placed is None:
            return None
        patches.append(placed)
    return tuple(patches)


def _bfs_grow(
    scaffold: SampleScaffold, seed: int, *, target: int, allowed: set[int],
) -> list[int]:
    if seed not in allowed:
        return []
    spot_ids = scaffold.spot_ids.astype(str)
    visited = {seed}
    order = [seed]
    frontier = [seed]
    while frontier and len(order) < target:
        candidates = {
            int(v)
            for u in frontier
            for v in scaffold.adj[u]
            if int(v) in allowed and int(v) not in visited
        }
        frontier = []
        for value in sorted(candidates, key=lambda p: spot_ids[p]):
            if len(order) >= target:
                break
            visited.add(value)
            order.append(value)
            frontier.append(value)
    return order
