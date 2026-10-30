"""Sample-wide acquisition scaffold — the fixed spatial surface.

Loads a WHOLE sample (all spots, all ROIs) as a single frozen record
carrying:

- fixed acquisition coordinates + per-spot ``source_parent_roi_id``;
- one sample-wide acquisition graph built with Stage-A radius
  (radius_factor=1.5) using ONE sample-wide ``median_nn``;
- per-spot ``source_component_id`` = deterministic label from
  connected components of the acquisition graph restricted to each
  ``source_parent_roi_id`` (components ordered by min ``source_spot_id``,
  singletons and pairs retained);
- helpers for the cross-ROI contact-edge subgraph (interface
  construction uses these edges directly; there is no second
  adjacency authority in v2).

Coordinates never move. Active scenarios preserve source ROI labels.
The scaffold retains one central helper for deriving generated component
labels if a future topology calibration explicitly reassigns ROI identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_RADIUS_FACTOR = 1.5


@dataclass(frozen=True)
class SampleScaffold:
    """Whole-sample acquisition scaffold.

    Attributes:
        sample_id: identifier of the tissue sample.
        coords: ``(N, 2)`` float array; sample-wide spot coordinates.
        n: number of spots.
        spot_ids: ``(N,)`` object array of stable ``source_spot_id``.
        source_parent_roi_id: ``(N,)`` object array; parent ROI per spot.
        source_component_id: ``(N,)`` object array; per-spot component
            derived from the induced subgraph restricted to each
            ``source_parent_roi_id``.
        edges: ``(E, 2)`` int array of undirected acquisition edges
            (Stage-A radius graph on the whole sample; ``i < j`` per
            row). Contact edges are the subset with different
            ``source_parent_roi_id`` values at endpoints.
        adj: sample-wide adjacency list (all edges, no ROI filter).
        median_nn: sample-wide median NN distance used by the radius.
        adjacency_method: ``"radius_multiplier"``.
        adjacency_parameters: ``{"radius_factor": 1.5}`` by default.
    """
    sample_id: str
    coords: np.ndarray
    n: int
    spot_ids: np.ndarray
    source_parent_roi_id: np.ndarray
    source_component_id: np.ndarray
    edges: np.ndarray
    adj: List[List[int]]
    median_nn: float
    adjacency_method: str
    adjacency_parameters: dict

    def contact_edges(self) -> np.ndarray:
        """Cross-ROI acquisition edges (endpoints in different ROIs)."""
        if self.edges.shape[0] == 0:
            return np.zeros((0, 2), dtype=int)
        roi_i = self.source_parent_roi_id[self.edges[:, 0]]
        roi_j = self.source_parent_roi_id[self.edges[:, 1]]
        mask = roi_i != roi_j
        return self.edges[mask]

    def boundary_spots(self, roi_id: str) -> np.ndarray:
        """Indices of spots in ``roi_id`` incident to at least one
        acquisition edge crossing to a non-``roi_id`` spot."""
        target = str(roi_id)
        result = []
        for i in np.where(self.source_parent_roi_id == target)[0]:
            for j in self.adj[i]:
                if str(self.source_parent_roi_id[j]) != target:
                    result.append(int(i))
                    break
        return np.array(sorted(set(result)), dtype=int)

    def derive_generated_components(
        self, generated_roi_id: np.ndarray,
    ) -> np.ndarray:
        """Per-spot component labels from CC of the graph induced by
        ``generated_roi_id``. Labels ``"{generated_roi_id}::c{i:04d}"``
        ordered deterministically by ``min(spot_id)`` per component.

        Reserved for explicit generated-label topology calibrations; the
        current executable scenarios preserve source ROI membership.
        """
        return _derive_components_from_labels(
            adj=self.adj, labels=generated_roi_id.astype(str),
            spot_ids=self.spot_ids,
        )

    def restricted_adj(self, allowed_indices: np.ndarray) -> "List[List[int]]":
        """Local adjacency list restricted to ``allowed_indices``.

        Returns ``local_adj`` where ``local_adj[k]`` are LOCAL indices
        (position within ``allowed_indices``) of neighbours also in
        ``allowed_indices``.
        """
        allowed_set = set(int(i) for i in allowed_indices)
        global_to_local = {int(g): k for k, g in enumerate(allowed_indices)}
        out: List[List[int]] = [[] for _ in range(len(allowed_indices))]
        for k, g in enumerate(allowed_indices):
            for v in self.adj[int(g)]:
                if int(v) in allowed_set:
                    out[k].append(global_to_local[int(v)])
            out[k].sort()
        return out


def load_sample_scaffold(
    df: pd.DataFrame,
    *,
    sample_id: str,
    x_col: str = "x", y_col: str = "y",
    spot_id_col: str = "spot_id",
    roi_id_col: str = "roi_id",
    radius_factor: float = DEFAULT_RADIUS_FACTOR,
) -> SampleScaffold:
    """Build a sample-wide scaffold from a full-sample DataFrame.

    Sample-wide ``median_nn`` and radius-multiplier acquisition graph;
    per-spot components computed by connected components of the
    graph restricted to each ``source_parent_roi_id``. Component ids
    are ``"{roi_id}::c{i:04d}"`` where ``i`` is the ordering by
    minimum ``source_spot_id`` within the ROI (deterministic).
    """
    for col in (x_col, y_col, spot_id_col, roi_id_col):
        if col not in df.columns:
            raise ValueError(
                f"load_sample_scaffold: missing column {col!r}; "
                f"got {list(df.columns)!r}."
            )
    df = df.reset_index(drop=True)
    coords = df[[x_col, y_col]].to_numpy(dtype=float)
    spot_ids = df[spot_id_col].astype(str).to_numpy()
    if len(np.unique(spot_ids)) != len(spot_ids):
        raise ValueError("load_sample_scaffold: source_spot_id must be unique.")
    source_parent_roi_id = df[roi_id_col].astype(str).to_numpy()

    edges, median_nn, adj = _build_radius_graph(coords, radius_factor=radius_factor)
    source_component_id = _derive_source_components(
        adj=adj, source_parent_roi_id=source_parent_roi_id, spot_ids=spot_ids,
    )

    return SampleScaffold(
        sample_id=str(sample_id),
        coords=coords,
        n=int(coords.shape[0]),
        spot_ids=spot_ids,
        source_parent_roi_id=source_parent_roi_id,
        source_component_id=source_component_id,
        edges=edges,
        adj=adj,
        median_nn=float(median_nn),
        adjacency_method="radius_multiplier",
        adjacency_parameters={"radius_factor": float(radius_factor)},
    )


def _build_radius_graph(coords: np.ndarray, *, radius_factor: float):
    """Sample-wide Stage-A radius graph: pairs within
    ``radius_factor * median_nn`` via ``cKDTree.query_pairs``."""
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"_build_radius_graph: coords must be (N, 2); got {coords.shape!r}."
        )
    n = coords.shape[0]
    if n < 2:
        return np.zeros((0, 2), dtype=int), float("nan"), [[] for _ in range(n)]
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    median_nn = float(np.median(dists[:, 1]))
    if not np.isfinite(median_nn) or median_nn <= 0:
        raise ValueError(
            f"_build_radius_graph: median NN spacing not finite/positive ({median_nn})."
        )
    r_max = float(radius_factor * median_nn)
    pairs = tree.query_pairs(r=r_max, output_type="ndarray")
    edges = pairs.astype(int) if pairs.size else np.zeros((0, 2), dtype=int)
    adj: List[List[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    for i in range(n):
        adj[i].sort()
    return edges, median_nn, adj


def _derive_source_components(
    *, adj: Sequence[Sequence[int]],
    source_parent_roi_id: np.ndarray, spot_ids: np.ndarray,
) -> np.ndarray:
    """Per-spot source components via CC of induced subgraph per ROI."""
    return _derive_components_from_labels(
        adj=adj, labels=source_parent_roi_id, spot_ids=spot_ids,
    )


def _derive_components_from_labels(
    *, adj: Sequence[Sequence[int]], labels: np.ndarray,
    spot_ids: np.ndarray,
) -> np.ndarray:
    """Per-spot component labels from CC of the graph induced by
    ``labels`` (any label vector defining ROI membership).

    Components within a label are ordered by min ``spot_id`` (string).
    """
    labels = labels.astype(str)
    n = len(labels)
    out = np.full(n, None, dtype=object)
    for lbl in sorted(set(labels.tolist())):
        idx = np.where(labels == lbl)[0]
        roi_set = set(int(i) for i in idx)
        visited: set = set()
        components: List[List[int]] = []
        for i in sorted(idx.tolist()):
            if i in visited:
                continue
            comp: List[int] = []
            stack = [int(i)]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u); comp.append(u)
                for v in adj[u]:
                    if v in roi_set and v not in visited:
                        stack.append(int(v))
            components.append(sorted(comp))
        components.sort(key=lambda c: min(str(spot_ids[k]) for k in c))
        for ci, comp in enumerate(components):
            label = f"{lbl}::c{ci:04d}"
            for k in comp:
                out[k] = label
    return out
