"""Semantic invariants for nested_foci.

The scenario places focal B patches inside a broadly A-positive
territory; the matched control relocates the same B field into the
B ROI's own support. Expression-only in every role (no ROI
relabelling); feature-A is byte-identical across positive and control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scenario_benchmark import (
    SignalModel, build_nested_foci_set, load_sample_scaffold,
)
from scenario_benchmark.background_bank import SampleBackgroundBank
from scenario_benchmark.signal_model import (
    COMPONENT_EXPRESSION_PROFILES, profiled_component_overlay,
)


A_ID, B_ID, C_ID = "A", "B", "C"


def _scaffold():
    """20x12 grid: A cols 0-9 (broad territory for focal patches),
    B cols 10-16 (B home for matched relocation), C cols 17-19. B has
    no A-B interface — that gate is not required for this scenario."""
    rows, cols = 12, 20
    xs = np.repeat(np.arange(cols), rows).astype(float)
    ys = np.tile(np.arange(rows), cols).astype(float)
    roi = np.array(
        [A_ID] * (10 * rows) + [B_ID] * (7 * rows) + [C_ID] * (3 * rows),
        dtype=object,
    )
    spot_ids = [f"s{i:04d}" for i in range(xs.size)]
    df = pd.DataFrame({
        "x": xs, "y": ys, "spot_id": spot_ids, "roi_id": roi.tolist(),
    })
    return load_sample_scaffold(df, sample_id="cfa", radius_factor=1.5)


def _build(scaf, base_seed=13, profile="core_high"):
    return build_nested_foci_set(
        scaffold=scaf, a_roi_id=A_ID, b_roi_id=B_ID,
        counterfactual_set_id="cs",
        feature_a_id="fA", feature_b_id="fB",
        focus_sizes=(5, 5),
        signal_model=SignalModel(effect_delta=1.0),
        b_expression_profile=profile,
        base_seed=base_seed,
    )


def _rows(spot_rows, role):
    return [r for r in spot_rows if r.counterfactual_role == role]


def _elev_b(scaf, spot_rows, role):
    id_to_pos = {sid: i for i, sid in enumerate(scaf.spot_ids.astype(str))}
    return sorted(
        id_to_pos[r.source_spot_id]
        for r in _rows(spot_rows, role) if r.feature_b_is_elevated
    )


def _elev_a(scaf, spot_rows, role):
    id_to_pos = {sid: i for i, sid in enumerate(scaf.spot_ids.astype(str))}
    return sorted(
        id_to_pos[r.source_spot_id]
        for r in _rows(spot_rows, role) if r.feature_a_is_elevated
    )


def test_no_ab_interface_is_not_a_gate():
    """The scenario estimand only requires focal B inside A + a B home
    for matched relocation; A-B adjacency is not required."""
    scaf = _scaffold()
    scenario = _build(scaf)
    by_role = {i.counterfactual_role: i for i in scenario.instances}
    assert by_role["positive"].role_disposition == "feasible"
    assert by_role["b_foci_relocated_outside_a"].role_disposition == "feasible"
    for inst in scenario.instances:
        if inst.role_disposition == "infeasible":
            assert inst.fallback_reason != "no_ab_interface"


def test_relocation_control_lives_in_b_home_and_a_is_byte_identical():
    """Matched relocation lands inside the B ROI; feature-A elevation
    is identical across positive and the control role."""
    scaf = _scaffold()
    scenario = _build(scaf)
    pos_b = _elev_b(scaf, scenario.spot_truth_rows, "positive")
    control_b = _elev_b(
        scaf, scenario.spot_truth_rows, "b_foci_relocated_outside_a",
    )
    assert control_b and not (set(control_b) & set(pos_b))
    for p in control_b:
        assert scaf.source_parent_roi_id[p] == B_ID
    a_pos = _elev_a(scaf, scenario.spot_truth_rows, "positive")
    a_control = _elev_a(
        scaf, scenario.spot_truth_rows, "b_foci_relocated_outside_a",
    )
    assert a_pos == a_control, (
        f"feature-A elevation must be byte-identical; "
        f"symdiff={sorted(set(a_pos) ^ set(a_control))}"
    )
    for r in scenario.spot_truth_rows:
        assert str(r.generated_roi_id) == str(r.source_parent_roi_id)


@pytest.mark.parametrize("profile", COMPONENT_EXPRESSION_PROFILES)
def test_profiled_b_multiset_is_preserved_by_matched_relocation(profile):
    scaf = _scaffold()
    scenario = _build(scaf, profile=profile)
    bg = SampleBackgroundBank(
        scaffold=scaf, participant_roi_ids=(A_ID, B_ID),
    ).field("fB")
    id_to_pos = {sid: i for i, sid in enumerate(scaf.spot_ids.astype(str))}

    def overlays(role):
        return sorted(
            float(r.feature_b_value) - float(bg[id_to_pos[r.source_spot_id]])
            for r in _rows(scenario.spot_truth_rows, role)
            if r.feature_b_is_elevated
        )

    assert np.allclose(
        overlays("positive"), overlays("b_foci_relocated_outside_a"),
        rtol=0.0, atol=1e-12,
    )


def test_decomposed_mode_basal_only_on_a_host_and_metadata_recorded():
    """When a_host_baseline / b_population_baseline / b_focus_amplitude
    are supplied, per-feature bg banks + A-only basal B overlay apply,
    and resolved amplitudes land in role realized_parameters."""
    from scenario_benchmark.background_bank import BackgroundPolicy
    scaf = _scaffold()
    s = build_nested_foci_set(
        scaffold=scaf, a_roi_id=A_ID, b_roi_id=B_ID,
        counterfactual_set_id="cs_v5",
        feature_a_id="fA", feature_b_id="fB",
        focus_sizes=(5, 5),
        signal_model=SignalModel(effect_delta=1.0),
        a_host_baseline=0.25,
        b_population_baseline=0.08,
        b_focus_amplitude=0.22,
        b_expression_profile="core_high",
        background_policy=BackgroundPolicy(
            non_participant_amplitude=0.12,
            participant_amplitude=0.03,
            latent_abs_clip=1.75,
        ),
        base_seed=17,
    )
    positive = next(i for i in s.instances if i.counterfactual_role == "positive")
    rp = positive.realized_parameters
    assert rp["a_host_baseline_resolved"] == 0.25
    assert rp["b_population_baseline_resolved"] == 0.08
    assert rp["b_focus_amplitude_resolved"] == 0.22
    # A-only basal contract: B-labelled ROI spots see ordinary
    # non-participant B background (no basal overlay).
    pos_rows = sorted(_rows(s.spot_truth_rows, "positive"),
                      key=lambda r: r.source_spot_id)
    id_to_idx = {sid: i for i, sid in enumerate(scaf.spot_ids.astype(str))}
    order = np.argsort([id_to_idx[r.source_spot_id] for r in pos_rows])
    b_vals = np.array([r.feature_b_value for r in pos_rows])[order]
    parents = scaf.source_parent_roi_id.astype(str)
    b_roi_mask = parents == B_ID
    elev = np.array([r.feature_b_is_elevated for r in pos_rows])[order]
    b_roi_unmarked = b_roi_mask & ~elev
    # bg on B_ID with non_participant_amplitude=0.12 * clip=1.75 → |x| ≤ 0.21
    assert np.all(np.abs(b_vals[b_roi_unmarked]) <= 0.21 + 1e-9)
    # A_ID unmarked spots carry baseline 0.08 ± 0.03*1.75 = [0.0275, 0.1325]
    # All A spots carry the basal B field; non-focus spots are those
    # below the smallest possible profiled focus value.
    a_vals = b_vals[parents == A_ID]
    basal_only = a_vals[a_vals < 0.15]
    assert basal_only.size
    assert basal_only.min() >= 0.0275 - 1e-9
    assert basal_only.max() <= 0.1325 + 1e-9


def test_component_profile_directions():
    coords = np.column_stack((np.arange(5, dtype=float), np.zeros(5)))
    ids = np.array([f"p{i}" for i in range(5)])
    kwargs = dict(
        n=5, components=((0, 1, 2, 3, 4),),
        coords=coords, spot_ids=ids, magnitude=1.0,
    )
    core = profiled_component_overlay(profile="core_high", **kwargs)
    rim = profiled_component_overlay(profile="rim_high", **kwargs)
    gradient = profiled_component_overlay(profile="axial_gradient", **kwargs)
    for overlay in (core, rim, gradient):
        assert np.isclose(overlay.mean(), 1.0)
        assert np.isclose(overlay.min(), 0.7)
        assert np.isclose(overlay.max(), 1.3)
    assert core[2] == core.max()
    assert min(rim[0], rim[4]) > rim[2]
    assert gradient[0] == gradient.min()
    assert gradient[4] == gradient.max()
