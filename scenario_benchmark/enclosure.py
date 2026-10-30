"""Shared geometry for B-labelled components enclosed by A."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Tuple

import numpy as np

from scenario_benchmark.pair_interface import PairInterface
from scenario_benchmark.sample_scaffold import SampleScaffold


ENCLOSURE_ENCLOSED_BY_A = "enclosed_by_a"
ENCLOSURE_NOT_ENCLOSED_BY_A = "not_enclosed_by_a"
ENCLOSURE_INDETERMINATE = "enclosure_indeterminate"
EnclosureState = Literal[
    "enclosed_by_a", "not_enclosed_by_a", "enclosure_indeterminate",
]


@dataclass(frozen=True)
class ComponentEnclosureInfo:
    source_component_id: str
    positions: Tuple[int, ...]
    size: int
    state: EnclosureState
    external_neighbours_A: Tuple[int, ...]
    n_A_contact_edges: int


def classify_component_enclosure_by_a(
    scaffold: SampleScaffold, pair: PairInterface,
) -> Tuple[ComponentEnclosureInfo, ...]:
    """Classify every B component as enclosed, not enclosed, or unknown.

    A component is enclosed only when its observable external neighbours are
    all A-labelled. Degree-deficient components are conservatively unknown
    because they may touch an acquisition boundary.
    """
    max_degree = max((len(neighbours) for neighbours in scaffold.adj), default=0)
    b_indices = np.where(
        scaffold.source_parent_roi_id.astype(str) == pair.b_roi_id,
    )[0]
    if b_indices.size == 0:
        return ()
    component_ids = scaffold.source_component_id[b_indices].astype(str)
    spot_ids = scaffold.spot_ids.astype(str)
    labels = scaffold.source_parent_roi_id.astype(str)
    result: List[ComponentEnclosureInfo] = []
    for component_id in np.unique(component_ids):
        members = b_indices[component_ids == component_id]
        member_set = set(map(int, members))
        a_neighbours: set[int] = set()
        n_a_edges = 0
        any_external = False
        any_non_a_external = False
        degree_deficient = False
        for member in map(int, members):
            if len(scaffold.adj[member]) < max_degree:
                degree_deficient = True
                break
            for neighbour in scaffold.adj[member]:
                neighbour = int(neighbour)
                if neighbour in member_set:
                    continue
                any_external = True
                if labels[neighbour] == pair.a_roi_id:
                    a_neighbours.add(neighbour)
                    n_a_edges += 1
                else:
                    any_non_a_external = True
        if degree_deficient or not any_external:
            state: EnclosureState = ENCLOSURE_INDETERMINATE
        elif any_non_a_external:
            state = ENCLOSURE_NOT_ENCLOSED_BY_A
        else:
            state = ENCLOSURE_ENCLOSED_BY_A
        result.append(ComponentEnclosureInfo(
            source_component_id=str(component_id),
            positions=tuple(sorted(member_set)),
            size=len(member_set),
            state=state,
            external_neighbours_A=tuple(sorted(a_neighbours)),
            n_A_contact_edges=n_a_edges,
        ))
    result.sort(key=lambda item: min(spot_ids[i] for i in item.positions))
    return tuple(result)


def enclosure_ascertainment_counts(
    enclosure: Iterable[ComponentEnclosureInfo],
) -> Dict[str, int]:
    """Report enclosure-classification breakdown across every B component.

    Reported for topology-scenario transparency: the classifier flags
    degree-deficient components as INDETERMINATE (they may touch an
    acquisition boundary and therefore cannot be conservatively enclosed),
    so eligible-domain counts are lower bounds on true enclosure.
    """
    infos = tuple(enclosure)
    def _count(state: EnclosureState) -> Tuple[int, int]:
        selected = tuple(info for info in infos if info.state == state)
        return len(selected), int(sum(info.size for info in selected))
    enc_c, enc_s = _count(ENCLOSURE_ENCLOSED_BY_A)
    not_c, not_s = _count(ENCLOSURE_NOT_ENCLOSED_BY_A)
    ind_c, ind_s = _count(ENCLOSURE_INDETERMINATE)
    return {
        "enclosed_b_component_count": enc_c,
        "enclosed_b_spot_count": enc_s,
        "not_enclosed_b_component_count": not_c,
        "not_enclosed_b_spot_count": not_s,
        "indeterminate_b_component_count": ind_c,
        "indeterminate_b_spot_count": ind_s,
    }


def compute_a_host_support(
    scaffold: SampleScaffold,
    a_indices: np.ndarray,
    b_positions: np.ndarray,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Return complete A components directly surrounding selected B."""
    a_set = set(map(int, a_indices))
    b_set = set(map(int, b_positions))
    ring = {
        int(neighbour)
        for position in b_set
        for neighbour in scaffold.adj[position]
        if int(neighbour) in a_set and int(neighbour) not in b_set
    }
    if not ring:
        return np.array([], dtype=int), ()
    component_ids = scaffold.source_component_id.astype(str)
    host_ids = tuple(sorted({component_ids[position] for position in ring}))
    host_positions = np.array(sorted(
        int(position) for position in a_indices
        if component_ids[int(position)] in host_ids
    ), dtype=int)
    return host_positions, host_ids
