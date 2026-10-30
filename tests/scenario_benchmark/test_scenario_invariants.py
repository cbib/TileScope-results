"""Semantic-fixture invariants for the three easy calibration scenarios.

Hand-constructed 20x6 grid scaffold:

    columns 0-11 = anchor ROI ``ANCHOR`` (host, 72 spots)
    columns 12-15 = ``NON_ANCHOR_1`` (byte-identical background)
    columns 16-19 = ``NON_ANCHOR_2`` (byte-identical background)

Every scenario is deterministically feasible on this fixture. Under
the bg + overlay compositional foundation the non-anchor ROIs carry
byte-identical feature vectors across paired positive + null roles
(overlay is zero outside the anchor, bank is feature-independent
per-role).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from scenario_benchmark import (
    FOCAL_PATTERN_MODE_COLOCATED,
    GRADIENT_RELATIONSHIP_CONCORDANT, GRADIENT_RELATIONSHIP_OPPOSED,
    SampleScaffold, SignalModel,
    TILESCOPE_INPUT_COLUMNS,
    build_broad_colocalization_set,
    build_focal_pattern_set,
    build_spatial_gradient_set,
    load_sample_scaffold,
    rows_to_tilescope_dataframe,
    write_tilescope_input_csv,
)
from scenario_benchmark.scenario_assembly import (
    InvariantPolicy, ScenarioSetBuilder, SetInvariantViolation,
    _validate_vector_invariants,
)


ANCHOR = "ANCHOR"


def _scaffold() -> SampleScaffold:
    xs = np.repeat(np.arange(20), 6).astype(float)
    ys = np.tile(np.arange(6), 20).astype(float)
    roi = np.where(
        xs < 12, ANCHOR,
        np.where(xs < 16, "NON_ANCHOR_1", "NON_ANCHOR_2"),
    )
    spot_id = [f"s{i:03d}" for i in range(len(xs))]
    df = pd.DataFrame({
        "x": xs, "y": ys, "spot_id": spot_id, "roi_id": roi,
    })
    return load_sample_scaffold(df, sample_id="hand", radius_factor=1.5)


def _model() -> SignalModel:
    return SignalModel(effect_delta=1.0)


def _rows_by_role(result) -> Dict[str, List]:
    role_of_iid = {i.scenario_instance_id: (i.counterfactual_role,
                                             i.generator_infeasible)
                    for i in result.instances}
    per: Dict[str, List] = {}
    for r in result.spot_truth_rows:
        role, infeas = role_of_iid[r.scenario_instance_id]
        if infeas:
            continue
        per.setdefault(role, []).append(r)
    return per


def _vec_sorted_by_spot(rows, field: str) -> np.ndarray:
    ordered = sorted(rows, key=lambda r: r.source_spot_id)
    return np.array([float(getattr(r, field)) for r in ordered])


def _assert_universal_easy_invariants(result, positive_role: str, null_role: str):
    """Contracts that every easy scenario must satisfy under the
    bg + role_overlay compositional foundation:

    (i)   Coordinate + source-identity byte identity across roles.
    (ii)  A vector byte-identical between positive and null (only B
          overlay changes in these scenarios).
    (iii) Every non-anchor spot byte-identical feature values across
          roles (the bank is feature-independent and the overlay is
          zero off-anchor).

    B-value multiset preservation is NOT part of the contract:
    positive and null overlays share the same increment value but
    populate different positions, so their B multisets naturally
    differ.
    """
    per = _rows_by_role(result)
    assert positive_role in per, f"positive role {positive_role} not emitted"
    assert null_role in per, f"null role {null_role} not emitted"
    pos = sorted(per[positive_role], key=lambda r: r.source_spot_id)
    null = sorted(per[null_role], key=lambda r: r.source_spot_id)
    assert len(pos) == len(null)
    # (i)
    for p, q in zip(pos, null):
        assert p.source_spot_id == q.source_spot_id
        assert p.x == q.x and p.y == q.y
        assert p.source_parent_roi_id == q.source_parent_roi_id
        assert p.source_component_id == q.source_component_id
    # (ii)
    a_pos = _vec_sorted_by_spot(pos, "feature_a_value")
    a_null = _vec_sorted_by_spot(null, "feature_a_value")
    assert np.array_equal(a_pos, a_null), (
        f"feature_a_value drifted between positive and {null_role}"
    )
    # (iii)
    for p, q in zip(pos, null):
        if p.source_parent_roi_id != ANCHOR:
            assert p.feature_a_value == q.feature_a_value, (
                f"non-anchor A drift on {p.source_spot_id}"
            )
            assert p.feature_b_value == q.feature_b_value, (
                f"non-anchor B drift on {p.source_spot_id}"
            )


# ---------------------------------------------------------------------------
# broad_colocalization -- moved to test_broad_colocalization.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# focal_pattern -- moved to test_focal_pattern.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# spatial_gradient (opposed | concordant modes)
# ---------------------------------------------------------------------------

_GRADIENT_MODEL = SignalModel(effect_delta=1.0)


_GRADIENT_MODES = [
    GRADIENT_RELATIONSHIP_OPPOSED,
    GRADIENT_RELATIONSHIP_CONCORDANT,
]


@pytest.mark.parametrize("mode", _GRADIENT_MODES)
def test_spatial_gradient_builder_contract(mode):
    """Both roles emitted; the null permutes B's OVERLAY across
    anchor positions, leaving the bank field byte-identical."""
    scaf = _scaffold()
    result = build_spatial_gradient_set(
        scaffold=scaf, anchor_roi_id=ANCHOR,
        gradient_relationship=mode,
        counterfactual_set_id="cf", feature_a_id="a", feature_b_id="b",
        truth_tail_fraction=0.15,
        gradient_axis=(1.0, 0.0),
        signal_model=_GRADIENT_MODEL, base_seed=1,
    )
    _assert_universal_easy_invariants(
        result, positive_role="positive_spatial_gradient",
        null_role="paired_randomized",
    )
    for inst in result.instances:
        assert inst.realized_parameters["gradient_relationship"] == mode


@pytest.mark.parametrize("mode,expected_sign", [
    (GRADIENT_RELATIONSHIP_OPPOSED, -1.0),
    (GRADIENT_RELATIONSHIP_CONCORDANT, +1.0),
])
def test_spatial_gradient_anchor_pearson_sign(mode, expected_sign):
    """On the anchor, A carries a physical-projection ramp overlay and
    B carries s*ramp overlay on top of the same bank field. Because
    the bank field on A and B are independent per-feature noise
    processes, Pearson r on the anchor is dominated by the ramp's
    slope sign but not saturated to +/-1. Concordant must be
    positive; opposed negative; magnitudes are substantial (>=0.8)
    at effect_delta=1.5 and the frozen bank amplitude 0.20."""
    scaf = _scaffold()
    model = SignalModel(effect_delta=1.5)
    result = build_spatial_gradient_set(
        scaffold=scaf, anchor_roi_id=ANCHOR,
        gradient_relationship=mode,
        counterfactual_set_id="cf", feature_a_id="a", feature_b_id="b",
        truth_tail_fraction=0.15,
        gradient_axis=(1.0, 0.0),
        signal_model=model, base_seed=1,
    )
    per = _rows_by_role(result)
    anchor_rows = [r for r in per["positive_spatial_gradient"]
                   if r.source_parent_roi_id == ANCHOR]
    a = np.array([r.feature_a_value for r in anchor_rows])
    b = np.array([r.feature_b_value for r in anchor_rows])
    r_anchor = float(np.corrcoef(a, b)[0, 1])
    if expected_sign > 0:
        assert r_anchor >= 0.8, f"concordant anchor r={r_anchor} not >= 0.8"
    else:
        assert r_anchor <= -0.8, f"opposed anchor r={r_anchor} not <= -0.8"


def test_spatial_gradient_degenerate_axis_infeasible():
    """gradient_axis orthogonal to the anchor's spatial extent yields
    zero projection range -> infeasibility, not silent fallback."""
    xs = np.arange(20).astype(float)
    ys = np.zeros(20)
    roi = np.array(["ANCHOR"] * 20, dtype=object)
    spot_id = [f"s{i:03d}" for i in range(20)]
    df = pd.DataFrame({"x": xs, "y": ys, "spot_id": spot_id, "roi_id": roi})
    scaf_flat = load_sample_scaffold(df, sample_id="flat", radius_factor=1.5)
    result = build_spatial_gradient_set(
        scaffold=scaf_flat, anchor_roi_id="ANCHOR",
        gradient_relationship=GRADIENT_RELATIONSHIP_OPPOSED,
        counterfactual_set_id="cf", feature_a_id="a", feature_b_id="b",
        truth_tail_fraction=0.15,
        gradient_axis=(0.0, 1.0),
        signal_model=_GRADIENT_MODEL, base_seed=1,
    )
    for inst in result.instances:
        assert inst.generator_infeasible
        assert inst.fallback_reason == "degenerate_projection_range_on_anchor"


# ---------------------------------------------------------------------------
# Subdued independent background bank invariants
# ---------------------------------------------------------------------------

def _run_all_easy_scenarios(scaf):
    """Return {scenario_name: result} for all three easy scenarios on scaf."""
    model = _model()
    grad_model = _GRADIENT_MODEL
    return {
        "broad_colocalization": build_broad_colocalization_set(
            scaffold=scaf, anchor_roi_id=ANCHOR,
            counterfactual_set_id="cf", feature_a_id="a", feature_b_id="b",
            signal_model=model, base_seed=1,
        ),
        "focal_pattern": build_focal_pattern_set(
            scaffold=scaf, anchor_roi_id=ANCHOR,
            mode=FOCAL_PATTERN_MODE_COLOCATED,
            counterfactual_set_id="cf", feature_a_id="a", feature_b_id="b",
            prevalence_fraction=0.10, n_foci=3, min_focus_separation_hops=2,
            signal_model=model, base_seed=1,
        ),
        "spatial_gradient": build_spatial_gradient_set(
            scaffold=scaf, anchor_roi_id=ANCHOR,
            gradient_relationship=GRADIENT_RELATIONSHIP_OPPOSED,
            counterfactual_set_id="cf", feature_a_id="a", feature_b_id="b",
            truth_tail_fraction=0.15,
            gradient_axis=(1.0, 0.0),
            signal_model=grad_model, base_seed=1,
        ),
    }


_POSITIVE_ROLE = {
    "broad_colocalization": "positive_broad",
    "focal_pattern": "positive",
    "spatial_gradient": "positive_spatial_gradient",
}


def test_non_anchor_rois_have_finite_positive_variance():
    scaf = _scaffold()
    results = _run_all_easy_scenarios(scaf)
    non_anchor_rois = sorted(
        set(scaf.source_parent_roi_id.tolist()) - {ANCHOR}
    )
    assert non_anchor_rois, "fixture must contain non-anchor ROIs"
    for scenario_name, result in results.items():
        per = _rows_by_role(result)
        pos_role = _POSITIVE_ROLE[scenario_name]
        assert pos_role in per, f"{scenario_name}: {pos_role} missing"
        pos = per[pos_role]
        for roi_id in non_anchor_rois:
            rows = [r for r in pos if r.source_parent_roi_id == roi_id]
            if len(rows) < 2:
                continue
            va = float(np.var([r.feature_a_value for r in rows]))
            vb = float(np.var([r.feature_b_value for r in rows]))
            assert va > 1e-12, (
                f"{scenario_name} ROI {roi_id}: var(A)={va} not > 1e-12"
            )
            assert vb > 1e-12, (
                f"{scenario_name} ROI {roi_id}: var(B)={vb} not > 1e-12"
            )


def test_non_anchor_A_and_B_backgrounds_are_independent():
    scaf = _scaffold()
    result = build_broad_colocalization_set(
        scaffold=scaf, anchor_roi_id=ANCHOR,
        counterfactual_set_id="cf", feature_a_id="a", feature_b_id="b",
        signal_model=_model(), base_seed=1,
    )
    per = _rows_by_role(result)
    pos = per["positive_broad"]
    non_anchor_rois = sorted(
        set(scaf.source_parent_roi_id.tolist()) - {ANCHOR}
    )
    picked_roi = None
    for roi_id in non_anchor_rois:
        rows = [r for r in pos if r.source_parent_roi_id == roi_id]
        if len(rows) >= 3:
            picked_roi = roi_id
            picked_rows = sorted(rows, key=lambda r: r.source_spot_id)
            break
    assert picked_roi is not None, "need a non-anchor ROI with >= 3 spots"
    a_vec = np.array([r.feature_a_value for r in picked_rows])
    b_vec = np.array([r.feature_b_value for r in picked_rows])
    assert not np.array_equal(a_vec, b_vec), (
        f"non-anchor ROI {picked_roi} has A and B byte-identical — "
        "independent-substream contract violated"
    )


# ---------------------------------------------------------------------------
# Feature-A byte-identity enforcement
# ---------------------------------------------------------------------------

def test_feature_a_byte_identity_violation_raises():
    """A drift between roles listed in ``feature_a_byte_identical_roles``
    must raise :class:`SetInvariantViolation`."""
    scaf = _scaffold()
    builder = ScenarioSetBuilder(
        scaffold=scaf, counterfactual_set_id="cf",
        scenario_name="fixture", feature_a_id="a", feature_b_id="b",
        nuisance_id="default", nuisance_parameters={},
        requested_parameters={}, rng_seed=0,
    )
    a_ref = np.zeros(scaf.n); a_ref[0] = 1.0
    a_drift = np.zeros(scaf.n); a_drift[1] = 1.0
    b = np.zeros(scaf.n)
    builder._feasible_vectors["role_a"] = (a_ref.copy(), b.copy())
    builder._feasible_vectors["role_b"] = (a_drift.copy(), b.copy())
    policy = InvariantPolicy(
        expected_roles=("role_a", "role_b"),
        feature_a_byte_identical_roles=("role_a", "role_b"),
        permitted_role_level_infeasibilities=(),
        require_non_pair_background_identity=False,
    )
    with pytest.raises(SetInvariantViolation, match="feature A drifted"):
        _validate_vector_invariants(builder, policy)


# ---------------------------------------------------------------------------
# Serialization schema tests (retained from prior commit)
# ---------------------------------------------------------------------------

def test_tilescope_input_projection_exact_five_columns(tmp_path):
    scaf = _scaffold()
    result = build_broad_colocalization_set(
        scaffold=scaf, anchor_roi_id=ANCHOR,
        counterfactual_set_id="cf_ts",
        feature_a_id="a", feature_b_id="b",
        signal_model=_model(), base_seed=1,
    )
    per_iid = {}
    for r in result.spot_truth_rows:
        per_iid.setdefault(r.scenario_instance_id, []).append(r)
    iid = sorted(per_iid)[0]
    rows = per_iid[iid]
    df = rows_to_tilescope_dataframe(rows)
    assert tuple(df.columns) == TILESCOPE_INPUT_COLUMNS
    for row_df, row_truth in zip(df.itertuples(index=False), rows):
        assert float(row_df.x) == float(row_truth.x)
        assert float(row_df.gene1) == float(row_truth.feature_a_value)
        assert float(row_df.gene2) == float(row_truth.feature_b_value)
        assert str(row_df.roi_id) == str(row_truth.generated_roi_id)
    path = write_tilescope_input_csv(rows, tmp_path / "ts.csv")
    df_read = pd.read_csv(path)
    assert tuple(df_read.columns) == TILESCOPE_INPUT_COLUMNS


# ---------------------------------------------------------------------------
# Graph helpers used by scenario-specific assertions
# ---------------------------------------------------------------------------

def _elevated_positions(rows, field: str, scaf: SampleScaffold) -> set:
    id_to_pos = {str(scaf.spot_ids[i]): int(i) for i in range(scaf.n)}
    return {id_to_pos[r.source_spot_id] for r in rows if getattr(r, field)}


def _n_ccs_in_anchor(elevated: set, scaf: SampleScaffold) -> int:
    anchor_set = {int(i) for i in
                  np.where(scaf.source_parent_roi_id == ANCHOR)[0]}
    inside = elevated & anchor_set
    visited: set = set()
    ccs = 0
    for seed in sorted(inside):
        if seed in visited:
            continue
        ccs += 1
        stack = [seed]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for v in scaf.adj[u]:
                if int(v) in inside and int(v) not in visited:
                    stack.append(int(v))
    return ccs


def _hop_distance(scaf: SampleScaffold, src: int, dst: int, cutoff: int) -> int | None:
    if src == dst:
        return 0
    visited = {src}
    frontier = [src]
    d = 0
    while frontier and d < cutoff:
        d += 1
        nxt = []
        for u in frontier:
            for v in scaf.adj[u]:
                if int(v) == dst:
                    return d
                if int(v) not in visited:
                    visited.add(int(v))
                    nxt.append(int(v))
        frontier = nxt
    return None
