"""Focused tests for the shared component-enclosure authority.

Covers the three states of classify_component_enclosure_by_a.
Embedded-domain truth itself is covered by test_embedded_domains.py.
"""
from __future__ import annotations

import pandas as pd

from scenario_benchmark import (
    SignalModel, build_embedded_domains_set,
    load_sample_scaffold,
)
from scenario_benchmark.pair_interface import build_pair_interface
from scenario_benchmark.enclosure import (
    ENCLOSURE_ENCLOSED_BY_A,
    ENCLOSURE_INDETERMINATE,
    ENCLOSURE_NOT_ENCLOSED_BY_A,
    classify_component_enclosure_by_a,
)


A_ID, B_ID, C_ID = "A", "B", "C"


def _grid_scaffold(labels_by_col_row):
    """Small SampleScaffold from {(col, row): roi_id}."""
    cols = max(c for c, _ in labels_by_col_row) + 1
    rows = max(r for _, r in labels_by_col_row) + 1
    xs, ys, ids, rois = [], [], [], []
    for c in range(cols):
        for r in range(rows):
            xs.append(float(c))
            ys.append(float(r))
            ids.append(f"c{c:02d}r{r:02d}")
            rois.append(labels_by_col_row[(c, r)])
    df = pd.DataFrame({"x": xs, "y": ys, "spot_id": ids, "roi_id": rois})
    return load_sample_scaffold(df, sample_id="s", radius_factor=1.5)


def _classify(layout):
    scaf = _grid_scaffold(layout)
    pair = build_pair_interface(
        scaffold=scaf, a_roi_id=A_ID, b_roi_id=B_ID,
    )
    return scaf, pair, classify_component_enclosure_by_a(scaf, pair)


def test_interior_fully_a_neighboured_component_is_enclosed_by_a():
    """A single-spot B component sitting in the interior of a 6x6 A
    grid, with every external neighbour A-labelled, is
    enclosed_by_a."""
    layout = {(c, r): A_ID for c in range(6) for r in range(6)}
    layout[(3, 3)] = B_ID
    _, _, infos = _classify(layout)
    assert len(infos) == 1
    assert infos[0].state == ENCLOSURE_ENCLOSED_BY_A


def test_interior_component_touching_third_roi_is_not_enclosed_by_a():
    """Same interior B spot, but one external neighbour is C-labelled
    → not_enclosed_by_a (at least one observable non-A external
    neighbour)."""
    layout = {(c, r): A_ID for c in range(6) for r in range(6)}
    layout[(3, 3)] = B_ID
    layout[(3, 4)] = C_ID
    _, _, infos = _classify(layout)
    assert len(infos) == 1
    assert infos[0].state == ENCLOSURE_NOT_ENCLOSED_BY_A


def test_boundary_and_no_external_components_are_indeterminate():
    """The degree rule is a conservative boundary/coverage
    heuristic: any B component with a member below the sample-wide
    max degree — whether from the sample acquisition edge or from
    an isolated component with no external neighbours — is
    classified enclosure_indeterminate. On regular grids the
    boundary branch subsumes the "no external neighbours" branch,
    since a component isolated from all A neighbours can only occur
    at a degree-deficient position."""
    layout = {(c, r): A_ID for c in range(6) for r in range(6)}
    layout[(0, 3)] = B_ID  # column 0 sits on the acquisition edge
    _, _, infos = _classify(layout)
    assert len(infos) == 1
    assert infos[0].state == ENCLOSURE_INDETERMINATE


def test_embedded_positive_does_not_require_a_placement_pool():
    # 20x12 A grid with two enclosed_by_a B components:
    # Both components are enclosed by A; no unrelated placement
    # support exists or is required by the positive archetype.
    # No C anywhere, and no B component with a non-A external
    # neighbour, so there is no not_enclosed_by_a B support to
    # host the null.
    rows, cols = 12, 20
    layout = {}
    for c in range(cols):
        for r in range(rows):
            layout[(c, r)] = "A"
    # Dominant enclosed B block (15 spots).
    for c in range(3, 6):
        for r in range(3, 8):
            layout[(c, r)] = "B"
    # Non-dominant enclosed B blob (5 spots, plus shape).
    for c, r in [(10, 6), (9, 6), (11, 6), (10, 5), (10, 7)]:
        layout[(c, r)] = "B"
    xs, ys, ids, rois = [], [], [], []
    for c in range(cols):
        for r in range(rows):
            xs.append(float(c))
            ys.append(float(r))
            ids.append(f"c{c:02d}r{r:02d}")
            rois.append(layout[(c, r)])
    df = pd.DataFrame({"x": xs, "y": ys, "spot_id": ids, "roi_id": rois})
    scaf = load_sample_scaffold(df, sample_id="null_empty", radius_factor=1.5)
    scenario = build_embedded_domains_set(
        scaffold=scaf, a_roi_id="A", b_roi_id="B",
        counterfactual_set_id="cs",
        feature_a_id="fA", feature_b_id="fB",
        signal_model=SignalModel(effect_delta=1.0),
        base_seed=17,
    )
    by_role = {i.counterfactual_role: i for i in scenario.instances}
    assert set(by_role) == {"positive"}
    assert by_role["positive"].role_disposition == "feasible"
