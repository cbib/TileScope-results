"""Pair-scoped geometric authority on top of SampleScaffold.

For a chosen (A, B) pair, exposes:

- A-contact edges (acquisition edges straddling A and B);
- A-side + B-side interface node sets;
- one-hop buffers around each side;
- side-specific interface-distance vectors.

Consumed by inter-ROI scenarios (contact interface + topology
scenarios); the within-ROI expression-field scenarios do not use it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from scenario_benchmark.sample_scaffold import SampleScaffold
from scenario_benchmark.truth_schemas import InterfaceRow


_UNREACHABLE = 10 ** 9
NO_FRAGMENT = -1


@dataclass(frozen=True)
class PairInterface:
    """Pair-scoped interface geometry: sole authority for hop
    distances, canonical fragments, nearest-fragment identity, and
    third-parent-ROI boundary sets."""
    scaffold: SampleScaffold
    a_roi_id: str
    b_roi_id: str
    a_indices: np.ndarray
    b_indices: np.ndarray
    ab_edges: np.ndarray
    interface_a_nodes: np.ndarray
    interface_b_nodes: np.ndarray
    a_side_interface_distance: np.ndarray
    b_side_interface_distance: np.ndarray
    a_boundary_all: np.ndarray
    b_boundary_all: np.ndarray
    interface_fragments: Tuple[Tuple[int, ...], ...]
    nearest_fragment_id_a: np.ndarray
    nearest_fragment_id_b: np.ndarray
    a_boundary_to_third_roi: np.ndarray
    b_boundary_to_third_roi: np.ndarray
    a_third_roi_boundary_distance: np.ndarray
    b_third_roi_boundary_distance: np.ndarray

    @property
    def unreachable(self) -> int:
        return _UNREACHABLE

    def one_hop_buffer(self, side: str) -> np.ndarray:
        """Interface nodes on ``side`` plus their one-hop same-ROI
        neighbours."""
        if side == "a":
            iface = self.interface_a_nodes; roi = self.a_roi_id
        elif side == "b":
            iface = self.interface_b_nodes; roi = self.b_roi_id
        else:
            raise ValueError(f"side must be 'a' or 'b'; got {side!r}.")
        if iface.size == 0:
            return np.array([], dtype=int)
        target = str(roi)
        out = set(int(i) for i in iface)
        for u in list(iface):
            for v in self.scaffold.adj[int(u)]:
                if str(self.scaffold.source_parent_roi_id[v]) == target:
                    out.add(int(v))
        return np.array(sorted(out), dtype=int)


def build_pair_interface(
    scaffold: SampleScaffold, *, a_roi_id: str, b_roi_id: str,
) -> PairInterface:
    """Build a :class:`PairInterface` for one (A, B) pair."""
    roi = scaffold.source_parent_roi_id.astype(str)
    a_indices = np.where(roi == a_roi_id)[0]
    b_indices = np.where(roi == b_roi_id)[0]
    if a_indices.size == 0 or b_indices.size == 0:
        raise ValueError(
            f"build_pair_interface: need spots in both A={a_roi_id!r} "
            f"and B={b_roi_id!r}."
        )
    edges = scaffold.edges
    if edges.shape[0] == 0:
        ab_edges = np.zeros((0, 2), dtype=int)
    else:
        la = roi[edges[:, 0]]; lb = roi[edges[:, 1]]
        mask = (((la == a_roi_id) & (lb == b_roi_id))
                | ((la == b_roi_id) & (lb == a_roi_id)))
        ab_edges = edges[mask]
    interface_a_nodes = _endpoints_with_label(ab_edges, roi, a_roi_id)
    interface_b_nodes = _endpoints_with_label(ab_edges, roi, b_roi_id)
    a_boundary_all = _cross_roi_boundary(scaffold, a_indices, a_roi_id)
    b_boundary_all = _cross_roi_boundary(scaffold, b_indices, b_roi_id)
    fragments = _canonical_interface_fragments(ab_edges, scaffold.spot_ids)
    a_side_dist, nearest_a = _labelled_multi_source_bfs(
        scaffold, fragments, ab_edges, a_indices, own_roi=str(a_roi_id),
    )
    b_side_dist, nearest_b = _labelled_multi_source_bfs(
        scaffold, fragments, ab_edges, b_indices, own_roi=str(b_roi_id),
    )
    a_third = _boundary_to_third_roi(
        scaffold, a_indices, own=a_roi_id, other=b_roi_id,
    )
    b_third = _boundary_to_third_roi(
        scaffold, b_indices, own=b_roi_id, other=a_roi_id,
    )
    a_third_dist = _multi_source_hop_within_roi(scaffold, a_third, a_indices)
    b_third_dist = _multi_source_hop_within_roi(scaffold, b_third, b_indices)
    return PairInterface(
        scaffold=scaffold, a_roi_id=str(a_roi_id), b_roi_id=str(b_roi_id),
        a_indices=a_indices, b_indices=b_indices,
        ab_edges=ab_edges,
        interface_a_nodes=interface_a_nodes,
        interface_b_nodes=interface_b_nodes,
        a_side_interface_distance=a_side_dist,
        b_side_interface_distance=b_side_dist,
        a_boundary_all=a_boundary_all,
        b_boundary_all=b_boundary_all,
        interface_fragments=fragments,
        nearest_fragment_id_a=nearest_a,
        nearest_fragment_id_b=nearest_b,
        a_boundary_to_third_roi=a_third,
        b_boundary_to_third_roi=b_third,
        a_third_roi_boundary_distance=a_third_dist,
        b_third_roi_boundary_distance=b_third_dist,
    )


@dataclass(frozen=True)
class InterfaceTruthBundle:
    """Substrate-derived truth for one scenario instance. Carries
    the owning ``scenario_instance_id`` so the assembler can reject
    accidental cross-role attachment."""
    scenario_instance_id: str
    rows: List["InterfaceRow"]
    spot_map: Dict[int, Tuple[str, int]]


def fragment_side_endpoints(
    pair: "PairInterface", fragment_idx: int, *, side: str,
) -> np.ndarray:
    """Own-ROI-side endpoints of a canonical fragment; sorted asc."""
    if side not in ("a", "b"):
        raise ValueError(f"side must be 'a' or 'b'; got {side!r}.")
    if not (0 <= fragment_idx < len(pair.interface_fragments)):
        raise IndexError(
            f"fragment_idx {fragment_idx} out of range "
            f"(0..{len(pair.interface_fragments)})."
        )
    own = pair.a_roi_id if side == "a" else pair.b_roi_id
    labels = pair.scaffold.source_parent_roi_id.astype(str)
    out = set()
    for k in pair.interface_fragments[fragment_idx]:
        for e in (0, 1):
            n = int(pair.ab_edges[k, e])
            if labels[n] == own:
                out.add(n)
    return np.array(sorted(out), dtype=int)


def active_fragment_ids(
    pair: "PairInterface", anchor_positions: np.ndarray, *, side: str,
) -> Tuple[int, ...]:
    """Canonical fragment indices whose ``side`` endpoints intersect
    ``anchor_positions``. ``side`` must be ``'a'`` or ``'b'``."""
    if side not in ("a", "b"):
        raise ValueError(f"side must be 'a' or 'b'; got {side!r}.")
    anchor_set = set(int(x) for x in anchor_positions)
    active: List[int] = []
    for f_idx in range(len(pair.interface_fragments)):
        endpoints = fragment_side_endpoints(pair, f_idx, side=side)
        if any(int(n) in anchor_set for n in endpoints):
            active.append(f_idx)
    return tuple(active)


def build_interface_truth_bundle(
    pair: "PairInterface", scenario_instance_id: str,
) -> InterfaceTruthBundle:
    """Sole authority for scenario interface truth. Reuses the
    canonical fragments + labelled BFS already computed on ``pair``."""
    rows = _interface_rows_from_pair(pair, scenario_instance_id)
    spot_map = _interface_spot_map_from_pair(pair, scenario_instance_id)
    return InterfaceTruthBundle(
        scenario_instance_id=scenario_instance_id,
        rows=rows, spot_map=spot_map,
    )


def _interface_id(scenario_instance_id: str, fragment_index: int) -> str:
    return f"{scenario_instance_id}::iface_{fragment_index:04d}"


def _interface_rows_from_pair(
    pair: "PairInterface", scenario_instance_id: str,
) -> List["InterfaceRow"]:
    rows: List[InterfaceRow] = []
    fragmentation = len(pair.interface_fragments)
    for f_idx, edge_keys in enumerate(pair.interface_fragments):
        src_ids: List[str] = []; tgt_ids: List[str] = []
        length_sum = 0.0
        for k in edge_keys:
            a, b = int(pair.ab_edges[k, 0]), int(pair.ab_edges[k, 1])
            if str(pair.scaffold.source_parent_roi_id[a]) == pair.a_roi_id:
                src, tgt = a, b
            else:
                src, tgt = b, a
            src_ids.append(str(pair.scaffold.spot_ids[src]))
            tgt_ids.append(str(pair.scaffold.spot_ids[tgt]))
            length_sum += float(np.hypot(
                pair.scaffold.coords[src, 0] - pair.scaffold.coords[tgt, 0],
                pair.scaffold.coords[src, 1] - pair.scaffold.coords[tgt, 1],
            ))
        rows.append(InterfaceRow(
            scenario_instance_id=scenario_instance_id,
            interface_id=_interface_id(scenario_instance_id, f_idx),
            source_parent_roi_id_A=pair.a_roi_id,
            source_parent_roi_id_B=pair.b_roi_id,
            edge_count=len(edge_keys),
            contact_edge_length_sum=length_sum,
            fragmentation_count=fragmentation,
            edge_source_spot_ids=tuple(src_ids),
            edge_target_spot_ids=tuple(tgt_ids),
        ))
    return rows


def _interface_spot_map_from_pair(
    pair: "PairInterface", scenario_instance_id: str,
) -> Dict[int, Tuple[str, int]]:
    out: Dict[int, Tuple[str, int]] = {}
    for i in pair.a_indices:
        fid = int(pair.nearest_fragment_id_a[int(i)])
        if fid != NO_FRAGMENT:
            out[int(i)] = (
                _interface_id(scenario_instance_id, fid),
                int(pair.a_side_interface_distance[int(i)]),
            )
    for i in pair.b_indices:
        fid = int(pair.nearest_fragment_id_b[int(i)])
        if fid != NO_FRAGMENT:
            out[int(i)] = (
                _interface_id(scenario_instance_id, fid),
                int(pair.b_side_interface_distance[int(i)]),
            )
    return out


def interface_rows_from_generated(
    *, scaffold: SampleScaffold, pair: PairInterface,
    generated_roi_id: np.ndarray, scenario_instance_id: str,
) -> List[InterfaceRow]:
    """Emit :class:`InterfaceRow` rows for the (A, B) pair under
    ``generated_roi_id`` labels. Returns an empty list when the
    role has no generated A-B contact edges."""
    a_id = pair.a_roi_id; b_id = pair.b_roi_id
    if scaffold.edges.shape[0] == 0:
        return []
    gen = generated_roi_id.astype(str)
    la = gen[scaffold.edges[:, 0]]; lb = gen[scaffold.edges[:, 1]]
    mask = (((la == a_id) & (lb == b_id))
            | ((la == b_id) & (lb == a_id)))
    ab_edges = scaffold.edges[mask]
    if ab_edges.shape[0] == 0:
        return []
    components = _cluster_edges_by_shared_endpoint(ab_edges)
    fragmentation = len(components)
    rows: List[InterfaceRow] = []
    for cidx, comp_edges in enumerate(components):
        src_ids: List[str] = []; tgt_ids: List[str] = []
        length_sum = 0.0
        for a, b in comp_edges:
            src, tgt = (int(a), int(b)) if gen[a] == a_id else (int(b), int(a))
            src_ids.append(str(scaffold.spot_ids[src]))
            tgt_ids.append(str(scaffold.spot_ids[tgt]))
            length_sum += float(np.hypot(
                scaffold.coords[src, 0] - scaffold.coords[tgt, 0],
                scaffold.coords[src, 1] - scaffold.coords[tgt, 1],
            ))
        rows.append(InterfaceRow(
            scenario_instance_id=scenario_instance_id,
            interface_id=f"{scenario_instance_id}::iface_{cidx:04d}",
            source_parent_roi_id_A=a_id, source_parent_roi_id_B=b_id,
            edge_count=len(comp_edges),
            contact_edge_length_sum=length_sum,
            fragmentation_count=fragmentation,
            edge_source_spot_ids=tuple(src_ids),
            edge_target_spot_ids=tuple(tgt_ids),
        ))
    return rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _endpoints_with_label(
    edges: np.ndarray, roi_labels: np.ndarray, target: str,
) -> np.ndarray:
    if edges.shape[0] == 0:
        return np.array([], dtype=int)
    hits = set()
    for a, b in edges:
        if str(roi_labels[a]) == str(target):
            hits.add(int(a))
        if str(roi_labels[b]) == str(target):
            hits.add(int(b))
    return np.array(sorted(hits), dtype=int)


def _labelled_multi_source_bfs(
    scaffold: SampleScaffold,
    fragments: Tuple[Tuple[int, ...], ...],
    ab_edges: np.ndarray,
    roi_indices: np.ndarray, *, own_roi: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single BFS that returns per-spot (min interface-hop distance,
    nearest fragment id) on ``roi_indices``. Canonical tie-break on
    equidistant arrivals: smaller fragment index wins."""
    n = scaffold.n
    dist = np.full(n, _UNREACHABLE, dtype=np.int64)
    frag = np.full(n, NO_FRAGMENT, dtype=np.int64)
    if roi_indices.size == 0 or len(fragments) == 0 or ab_edges.shape[0] == 0:
        return dist, frag
    roi_set = set(int(i) for i in roi_indices)
    roi_labels = scaffold.source_parent_roi_id.astype(str)
    frontier: List[int] = []
    for f_idx, edge_keys in enumerate(fragments):
        for k in edge_keys:
            a, b = int(ab_edges[k, 0]), int(ab_edges[k, 1])
            if roi_labels[a] == own_roi and a in roi_set:
                _seed(a, f_idx, dist, frag, frontier)
            if roi_labels[b] == own_roi and b in roi_set:
                _seed(b, f_idx, dist, frag, frontier)
    frontier = sorted(set(frontier))
    d = 0
    while frontier:
        d += 1
        best_frag: Dict[int, int] = {}
        for u in frontier:
            fu = int(frag[u])
            for v in scaffold.adj[u]:
                v = int(v)
                if v in roi_set and dist[v] == _UNREACHABLE:
                    if v not in best_frag or fu < best_frag[v]:
                        best_frag[v] = fu
        nxt: List[int] = []
        for v, fid in best_frag.items():
            dist[v] = d; frag[v] = fid; nxt.append(v)
        frontier = sorted(nxt)
    return dist, frag


def _seed(
    node: int, f_idx: int,
    dist: np.ndarray, frag: np.ndarray, frontier: List[int],
) -> None:
    if dist[node] == _UNREACHABLE:
        dist[node] = 0; frag[node] = f_idx; frontier.append(node)
    elif dist[node] == 0 and f_idx < int(frag[node]):
        frag[node] = f_idx


def _multi_source_hop_within_roi(
    scaffold: SampleScaffold, sources: np.ndarray, roi_indices: np.ndarray,
) -> np.ndarray:
    """BFS on the ROI-restricted subgraph from ``sources``. Returns
    per-spot hop distance (``_UNREACHABLE`` outside the ROI or when
    disconnected)."""
    n = scaffold.n
    dist = np.full(n, _UNREACHABLE, dtype=np.int64)
    if sources.size == 0 or roi_indices.size == 0:
        return dist
    roi_set = set(int(i) for i in roi_indices)
    frontier: List[int] = []
    for s in sources:
        s = int(s)
        if s in roi_set:
            dist[s] = 0; frontier.append(s)
    d = 0
    while frontier:
        d += 1
        nxt: List[int] = []
        for u in frontier:
            for v in scaffold.adj[u]:
                v = int(v)
                if v in roi_set and dist[v] == _UNREACHABLE:
                    dist[v] = d; nxt.append(v)
        frontier = nxt
    return dist


def _cross_roi_boundary(
    scaffold: SampleScaffold, roi_indices: np.ndarray, roi_id: str,
) -> np.ndarray:
    target = str(roi_id)
    out = []
    for i in roi_indices:
        for v in scaffold.adj[int(i)]:
            if str(scaffold.source_parent_roi_id[v]) != target:
                out.append(int(i)); break
    return np.array(sorted(set(out)), dtype=int)


def _boundary_to_third_roi(
    scaffold: SampleScaffold, own_indices: np.ndarray,
    *, own: str, other: str,
) -> np.ndarray:
    """Own-ROI spots with at least one neighbour whose parent ROI is
    neither ``own`` nor ``other`` (i.e., a third parent ROI)."""
    own_s, other_s = str(own), str(other)
    out: List[int] = []
    for i in own_indices:
        for v in scaffold.adj[int(i)]:
            n_roi = str(scaffold.source_parent_roi_id[v])
            if n_roi != own_s and n_roi != other_s:
                out.append(int(i)); break
    return np.array(sorted(set(out)), dtype=int)


def _canonical_interface_fragments(
    ab_edges: np.ndarray, spot_ids: np.ndarray,
) -> Tuple[Tuple[int, ...], ...]:
    """Connected components of ``ab_edges`` (edges sharing an endpoint
    are one fragment). Each fragment is the sorted tuple of edge
    indices. Fragments are ordered by their minimum endpoint's
    ``spot_id`` (lexicographic string compare)."""
    if ab_edges.shape[0] == 0:
        return ()
    groups = _cluster_edges_by_shared_endpoint(ab_edges)
    def _min_spot_id(edges_of_fragment: np.ndarray) -> str:
        endpoints = set()
        for a, b in edges_of_fragment:
            endpoints.add(int(a)); endpoints.add(int(b))
        return min(str(spot_ids[i]) for i in endpoints)
    ordered = sorted(
        range(len(groups)), key=lambda k: _min_spot_id(groups[k]),
    )
    edge_index_by_group: List[Tuple[int, ...]] = []
    edge_key_lookup: Dict[Tuple[int, int], int] = {}
    for k, (a, b) in enumerate(ab_edges):
        edge_key_lookup[(int(a), int(b))] = k
    for gk in ordered:
        keys: List[int] = []
        for a, b in groups[gk]:
            k = edge_key_lookup.get((int(a), int(b)))
            if k is None:
                k = edge_key_lookup[(int(b), int(a))]
            keys.append(int(k))
        edge_index_by_group.append(tuple(sorted(keys)))
    return tuple(edge_index_by_group)




def _cluster_edges_by_shared_endpoint(
    edges: np.ndarray,
) -> List[np.ndarray]:
    if edges.shape[0] == 0:
        return []
    parent: Dict[int, int] = {}
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    for a, b in edges:
        a = int(a); b = int(b)
        parent.setdefault(a, a); parent.setdefault(b, b)
        union(a, b)
    groups: Dict[int, List[int]] = {}
    for k, (a, b) in enumerate(edges):
        r = find(int(a))
        groups.setdefault(r, []).append(k)
    order = sorted(
        groups.items(),
        key=lambda kv: min(int(edges[k].min()) for k in kv[1]),
    )
    return [edges[np.array(idx, dtype=int)] for _, idx in order]
