"""Focused biological invariants for dispersed infiltration."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scenario_benchmark import (
    SignalModel, build_dispersed_infiltration_set, load_sample_scaffold,
)
from scenario_benchmark.scenarios.dispersed_infiltration import (
    MAX_LARGEST_COMPONENT_FRACTION, MIN_DISPERSION_RATIO,
    MIN_ENCLOSED_B_SPOT_COUNT, MIN_ENCLOSED_COMPONENT_COUNT,
    POSITIVE_ROLE, ROLES,
)


A_ID, B_ID = "A", "B"


def _dispersed_scaffold():
    """30x30 grid with well-spread enclosed and non-enclosed B labels."""
    rows, cols = 30, 30
    xs = np.repeat(np.arange(cols), rows).astype(float)
    ys = np.tile(np.arange(rows), cols).astype(float)
    roi = np.array(["A"] * (cols * rows), dtype=object)
    # 13 enclosed B singletons at 4x4 lattice minus 3 corners.
    enclosed = [
        (3, 3), (10, 3), (17, 3), (24, 3),
        (3, 10), (10, 10), (17, 10), (24, 10),
        (3, 17), (10, 17), (17, 17), (24, 17),
        (14, 24),
    ]
    for c, r in enclosed:
        roi[int(c * rows + r)] = "B"
    # Non-enclosed strip: alternating C rows / B rows on gap-3 columns.
    # Each B at (col, y) is adjacent to a C at (col, y±1), non-adjacent
    # to any other B (gap 2 rows / gap 3 cols keeps them all singletons
    # under radius_factor=1.5). Provides ample non-enclosed capacity
    # (>= 13) for the placement-control packing.
    non_enclosed_cols = (3, 6, 9, 12, 15, 18, 21, 24, 27)
    for col in non_enclosed_cols:
        roi[int(col * rows + 26)] = "C"
        roi[int(col * rows + 28)] = "C"
        roi[int(col * rows + 25)] = "B"
        roi[int(col * rows + 27)] = "B"
        roi[int(col * rows + 29)] = "B"
    spot_ids = [f"s{i:04d}" for i in range(xs.size)]
    df = pd.DataFrame({
        "x": xs, "y": ys, "spot_id": spot_ids, "roi_id": roi.tolist(),
    })
    return load_sample_scaffold(df, sample_id="dispersed", radius_factor=1.5)


def _run(scaf=None, **kwargs):
    if scaf is None:
        scaf = _dispersed_scaffold()
    return scaf, build_dispersed_infiltration_set(
        scaffold=scaf, a_roi_id=A_ID, b_roi_id=B_ID,
        counterfactual_set_id="cs",
        feature_a_id="fA", feature_b_id="fB",
        signal_model=SignalModel(effect_delta=1.0), base_seed=42,
        **kwargs,
    )


# 1. Exact roles.


def test_emits_only_the_positive_topology_archetype():
    _, s = _run()
    assert ROLES == (POSITIVE_ROLE,)
    emitted = {i.counterfactual_role for i in s.instances}
    assert emitted == {POSITIVE_ROLE}


# 2. A excluded from B-labelled positions.


def test_a_overlay_absent_from_b_labelled_positions():
    scaf, s = _run()
    id_to_pos = {sid: i for i, sid in enumerate(scaf.spot_ids.astype(str))}
    b_labelled = {
        i for i in range(scaf.n)
        if str(scaf.source_parent_roi_id[i]) == B_ID
    }
    for role in (POSITIVE_ROLE,):
        a_elev = {
            id_to_pos[r.source_spot_id]
            for r in s.spot_truth_rows
            if r.counterfactual_role == role and r.feature_a_is_elevated
        }
        assert not (a_elev & b_labelled), (
            f"role {role}: A elevated on B-labelled positions "
            f"{sorted(a_elev & b_labelled)}"
        )


def test_b_population_baseline_covers_every_b_labelled_spot():
    scaf, result = _run(
        a_host_baseline=0.25,
        b_population_baseline=0.25,
        b_local_increment=0.0,
    )
    selected_ids = {
        row.source_spot_id
        for row in result.spot_truth_rows
        if row.counterfactual_role == POSITIVE_ROLE
        and row.feature_b_is_elevated
    }
    expected_ids = {
        str(scaf.spot_ids[i])
        for i, roi in enumerate(scaf.source_parent_roi_id.astype(str))
        if roi == B_ID
    }
    assert selected_ids == expected_ids


def test_supported_b_component_does_not_disqualify_dispersed_topology():
    """A 15-spot component may coexist with dispersed smaller components.

    TileScope support status is an outcome, not a scenario gate.
    """
    rows = cols = 40
    xs = np.repeat(np.arange(cols), rows).astype(float)
    ys = np.tile(np.arange(rows), cols).astype(float)
    roi = np.array([A_ID] * (cols * rows), dtype=object)
    components = [
        [(c, r) for c in range(4, 7) for r in range(4, 9)],
        [(c, r) for c in range(30, 32) for r in range(4, 7)],
        [(c, r) for c in range(4, 6) for r in range(30, 33)],
        [(c, r) for c in range(30, 32) for r in range(30, 33)],
        [(c, r) for c in range(18, 20) for r in range(20, 23)],
    ]
    for component in components:
        for col, row in component:
            roi[int(col * rows + row)] = B_ID
    scaffold = load_sample_scaffold(pd.DataFrame({
        "x": xs,
        "y": ys,
        "spot_id": [f"s{i:04d}" for i in range(xs.size)],
        "roi_id": roi,
    }), sample_id="supported_b", radius_factor=1.5)
    _, result = _run(scaffold)
    instance = result.instances[0]
    assert instance.role_disposition == "feasible"
    assert max(instance.realized_parameters["observed_component_sizes"]) == 15
    assert instance.realized_parameters["b_spatial_support_class"] == "mixed"


# 3. Dispersion gate (both branches).


def test_gate_rejects_when_largest_component_dominates():
    """One large B block + tiny satellites: largest-fraction exceeds
    the cap → infeasible with the dispersion-cap reason."""
    rows, cols = 20, 20
    xs = np.repeat(np.arange(cols), rows).astype(float)
    ys = np.tile(np.arange(rows), cols).astype(float)
    roi = np.array(["A"] * (cols * rows), dtype=object)
    for c in range(7, 12):
        for r in range(8, 10):
            roi[int(c * rows + r)] = "B"
    for c, r in [(2, 2), (17, 2), (2, 17), (17, 17)]:
        roi[int(c * rows + r)] = "B"
    df = pd.DataFrame({
        "x": xs, "y": ys,
        "spot_id": [f"s{i:04d}" for i in range(xs.size)],
        "roi_id": roi.tolist(),
    })
    scaf = load_sample_scaffold(df, sample_id="dom", radius_factor=1.5)
    _, s = _run(scaf)
    for inst in s.instances:
        assert inst.role_disposition == "infeasible"
        assert (
            inst.fallback_reason
            == "largest_component_fraction_exceeds_dispersion_cap"
        )


def test_gate_rejects_when_dispersion_below_threshold():
    """12 disconnected B singletons packed in a small subregion of the
    A-host bounding-box: count gate clears (12 ≥ 5), spot gate clears
    (12 ≥ 12), largest-fraction 1/12 (cap clears), but pairwise
    centroid mean is below the dispersion threshold against the full
    A-host bounding-box → must reject with exactly
    ``dispersion_ratio_below_threshold``."""
    rows, cols = 20, 20
    xs = np.repeat(np.arange(cols), rows).astype(float)
    ys = np.tile(np.arange(rows), cols).astype(float)
    roi = np.array(["A"] * (cols * rows), dtype=object)
    # 4x3 lattice at gap 2, packed in (6..12, 6..10) — clustered singletons.
    for cx in (6, 8, 10, 12):
        for cy in (6, 8, 10):
            roi[int(cx * rows + cy)] = "B"
    df = pd.DataFrame({
        "x": xs, "y": ys,
        "spot_id": [f"s{i:04d}" for i in range(xs.size)],
        "roi_id": roi.tolist(),
    })
    scaf = load_sample_scaffold(df, sample_id="cluster", radius_factor=1.5)
    _, s = _run(scaf)
    for inst in s.instances:
        assert inst.role_disposition == "infeasible"
        assert inst.fallback_reason == "dispersion_ratio_below_threshold"


# --- Additional thin gate checks ------------------------------------------


def test_gate_rejects_below_component_count():
    """Two enclosed B components fails the count gate (min = 5)."""
    rows, cols = 20, 20
    xs = np.repeat(np.arange(cols), rows).astype(float)
    ys = np.tile(np.arange(rows), cols).astype(float)
    roi = np.array(["A"] * (cols * rows), dtype=object)
    for c, r in [(4, 4), (15, 15)]:
        roi[int(c * rows + r)] = "B"
    df = pd.DataFrame({
        "x": xs, "y": ys,
        "spot_id": [f"s{i:04d}" for i in range(xs.size)],
        "roi_id": roi.tolist(),
    })
    scaf = load_sample_scaffold(df, sample_id="few", radius_factor=1.5)
    _, s = _run(scaf)
    for inst in s.instances:
        assert inst.role_disposition == "infeasible"
        assert inst.fallback_reason == "insufficient_enclosed_component_count"


def test_gate_rejects_when_enclosed_b_spot_count_below_threshold():
    """5 singleton B spots satisfy the component-count gate (5 >= 5)
    but fail the total-spot-count gate (5 < 12) — this is the
    scientifically-marginal cohort the new gate rejects."""
    rows, cols = 30, 30
    xs = np.repeat(np.arange(cols), rows).astype(float)
    ys = np.tile(np.arange(rows), cols).astype(float)
    roi = np.array(["A"] * (cols * rows), dtype=object)
    # 5 singleton B spots, spread out enough to pass the dispersion gate.
    for c, r in [(3, 3), (25, 3), (14, 14), (3, 25), (25, 25)]:
        roi[int(c * rows + r)] = "B"
    df = pd.DataFrame({
        "x": xs, "y": ys,
        "spot_id": [f"s{i:04d}" for i in range(xs.size)],
        "roi_id": roi.tolist(),
    })
    scaf = load_sample_scaffold(df, sample_id="singletons", radius_factor=1.5)
    _, s = _run(scaf)
    for inst in s.instances:
        assert inst.role_disposition == "infeasible"
        assert inst.fallback_reason == "insufficient_enclosed_b_spot_count"


def test_predicate_constants_reachable():
    assert MIN_ENCLOSED_COMPONENT_COUNT == 5
    assert MIN_ENCLOSED_B_SPOT_COUNT == 12
    assert MAX_LARGEST_COMPONENT_FRACTION == 0.40
    assert MIN_DISPERSION_RATIO == 0.25
