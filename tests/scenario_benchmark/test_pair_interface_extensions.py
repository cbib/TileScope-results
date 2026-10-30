"""Focused invariants for the PairInterface extensions used by the
contact-based challenge scenarios (canonical interface fragments,
nearest-fragment identity per spot, third-parent-ROI boundary)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scenario_benchmark import load_sample_scaffold
from scenario_benchmark.pair_interface import (
    NO_FRAGMENT, build_pair_interface,
)


def _two_fragment_scaffold():
    """Two A blocks abut a single contiguous B slab: two disjoint A-B
    interface fragments, and B spots midway between them are reachable
    from BOTH fragments — creating equidistant ties for the tie-break
    test. A block 1 at rows 0-2, A block 2 at rows 8-10; B slab spans
    rows 0-10 (contiguous through the middle)."""
    def _block(x0, x1, y0, y1, roi):
        xs, ys = [], []
        for x in range(x0, x1):
            for y in range(y0, y1):
                xs.append(float(x)); ys.append(float(y))
        return np.array(xs), np.array(ys), [roi] * len(xs)
    xa1, ya1, ra1 = _block(0, 3, 0, 3, "A")
    xa2, ya2, ra2 = _block(0, 3, 8, 11, "A")
    xb, yb, rb = _block(3, 6, 0, 11, "B")
    xs = np.concatenate([xa1, xa2, xb])
    ys = np.concatenate([ya1, ya2, yb])
    roi = np.concatenate([ra1, ra2, rb])
    spot_ids = [f"s{i:04d}" for i in range(xs.size)]
    df = pd.DataFrame({"x": xs, "y": ys, "spot_id": spot_ids, "roi_id": roi})
    return load_sample_scaffold(df, sample_id="two_frag", radius_factor=1.5)


def _pair_and_third_roi_scaffold():
    def _block(x0, x1, y0, y1, roi):
        xs, ys = [], []
        for x in range(x0, x1):
            for y in range(y0, y1):
                xs.append(float(x)); ys.append(float(y))
        return np.array(xs), np.array(ys), [roi] * len(xs)
    xa, ya, ra = _block(0, 3, 0, 3, "A")
    xb, yb, rb = _block(3, 6, 0, 3, "B")
    xc, yc, rc = _block(6, 9, 0, 3, "C")
    xs = np.concatenate([xa, xb, xc])
    ys = np.concatenate([ya, yb, yc])
    roi = np.concatenate([ra, rb, rc])
    spot_ids = [f"s{i:04d}" for i in range(xs.size)]
    df = pd.DataFrame({"x": xs, "y": ys, "spot_id": spot_ids, "roi_id": roi})
    return load_sample_scaffold(df, sample_id="third", radius_factor=1.5)


def test_interface_fragments_are_canonical_and_stable():
    scaf = _two_fragment_scaffold()
    pair = build_pair_interface(scaffold=scaf, a_roi_id="A", b_roi_id="B")
    assert len(pair.interface_fragments) == 2
    # Canonical order = ascending by fragment's smallest spot_id.
    for f_idx, edges in enumerate(pair.interface_fragments):
        endpoint_ids = {
            str(scaf.spot_ids[int(pair.ab_edges[k, e])])
            for k in edges for e in (0, 1)
        }
        if f_idx == 0:
            assert min(endpoint_ids) < min(
                str(scaf.spot_ids[int(pair.ab_edges[k, e])])
                for k in pair.interface_fragments[1] for e in (0, 1)
            )


def test_nearest_fragment_id_uses_smaller_index_on_ties():
    """A B spot equidistant from two fragments is assigned to the
    smaller fragment index. On the two-fragment fixture, B spots at
    row 5 (mid-slab) are reachable from both fragment 0 (rows 0-2)
    and fragment 1 (rows 8-10) at the same hop distance; the
    canonical assignment picks fragment 0."""
    scaf = _two_fragment_scaffold()
    pair = build_pair_interface(scaffold=scaf, a_roi_id="A", b_roi_id="B")
    assert len(pair.interface_fragments) == 2
    # Fragment 0 covers the top interface (min spot_id); B interior
    # spots midway between the two fragments must resolve to it on tie.
    mid_row_b = [
        int(i) for i in pair.b_indices
        if abs(float(scaf.coords[int(i), 1]) - 5.0) < 0.5
    ]
    assert mid_row_b, "fixture must produce mid-row B spots for tie test"
    hop_to_frag = pair.b_side_interface_distance
    tied_and_assigned = 0
    for b in mid_row_b:
        assert int(pair.nearest_fragment_id_b[b]) == 0, (
            f"tie at spot {scaf.spot_ids[b]!r} did not resolve to frag 0"
        )
        tied_and_assigned += 1
    assert tied_and_assigned >= 1
    # Determinism across builds.
    pair2 = build_pair_interface(scaffold=scaf, a_roi_id="A", b_roi_id="B")
    assert np.array_equal(pair.nearest_fragment_id_b, pair2.nearest_fragment_id_b)


def test_fragment_side_endpoints_rejects_invalid_side():
    import pytest
    from scenario_benchmark.pair_interface import fragment_side_endpoints
    scaf = _two_fragment_scaffold()
    pair = build_pair_interface(scaffold=scaf, a_roi_id="A", b_roi_id="B")
    with pytest.raises(ValueError):
        fragment_side_endpoints(pair, 0, side="x")


def test_third_roi_boundary_excludes_own_and_paired_neighbors():
    scaf = _pair_and_third_roi_scaffold()
    pair = build_pair_interface(scaffold=scaf, a_roi_id="A", b_roi_id="B")
    # B spots with a C neighbour → in b_boundary_to_third_roi.
    # B spots touching only A or B (no C) → excluded.
    for b in pair.b_indices:
        b = int(b)
        neighbours = [
            str(scaf.source_parent_roi_id[int(v)]) for v in scaf.adj[b]
        ]
        has_c = any(n == "C" for n in neighbours)
        in_boundary = int(b) in set(int(x) for x in pair.b_boundary_to_third_roi)
        assert in_boundary == has_c
    # Symmetry: A has no third-ROI neighbour on this fixture (A only touches B).
    assert pair.a_boundary_to_third_roi.size == 0
