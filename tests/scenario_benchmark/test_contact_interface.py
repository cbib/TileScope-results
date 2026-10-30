"""Focused invariant tests for contact_interface (both modes).

Hand-built grid: A (columns 0-9), B (columns 10-19), C (columns 20-29).
Grid extended in y to guarantee interface + interior + third-boundary
capacity.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from scenario_benchmark import (
    CONTACT_INTERFACE_MODE_ANCHOR_FADE,
    CONTACT_INTERFACE_MODE_RESPONDER_ACTIVATION,
    CONTACT_INTERFACE_RESPONDER_ACTIVATION_INVARIANTS,
    SampleScaffold, SignalModel,
    build_contact_interface_set,
    load_sample_scaffold,
)
from scenario_benchmark.background_bank import SampleBackgroundBank
from scenario_benchmark.scenario_assembly import (
    ScenarioSetBuilder, SetInvariantViolation, _validate_vector_invariants,
)
from scenario_benchmark.scenarios.contact_interface import PERIPHERY_HOPS


A_ROI = "A"
B_ROI = "B"
C_ROI = "C"


def _rich_scaffold() -> SampleScaffold:
    xs = np.repeat(np.arange(30), 20).astype(float)
    ys = np.tile(np.arange(20), 30).astype(float)
    roi = np.where(xs < 10, A_ROI, np.where(xs < 20, B_ROI, C_ROI))
    spot_id = [f"s{i:04d}" for i in range(len(xs))]
    df = pd.DataFrame({"x": xs, "y": ys, "spot_id": spot_id, "roi_id": roi})
    return load_sample_scaffold(df, sample_id="hand_rich", radius_factor=1.5)


def _detached_scaffold() -> SampleScaffold:
    """A and B far apart with no adjacency; scenario-set-level infeasible."""
    xs = np.concatenate([np.arange(10), np.arange(10) + 50, np.arange(10) + 100]).astype(float)
    ys = np.zeros(30)
    roi = np.array([A_ROI] * 10 + [B_ROI] * 10 + [C_ROI] * 10, dtype=object)
    spot_id = [f"s{i:03d}" for i in range(30)]
    df = pd.DataFrame({"x": xs, "y": ys, "spot_id": spot_id, "roi_id": roi})
    return load_sample_scaffold(df, sample_id="hand_detached", radius_factor=1.5)


def _no_third_scaffold() -> SampleScaffold:
    """A and B only; B has no third-ROI boundary => unrelated_boundary infeasible."""
    xs = np.repeat(np.arange(20), 20).astype(float)
    ys = np.tile(np.arange(20), 20).astype(float)
    roi = np.where(xs < 10, A_ROI, B_ROI)
    spot_id = [f"s{i:04d}" for i in range(len(xs))]
    df = pd.DataFrame({"x": xs, "y": ys, "spot_id": spot_id, "roi_id": roi})
    return load_sample_scaffold(df, sample_id="hand_no_third", radius_factor=1.5)


def _small_interface_scaffold() -> SampleScaffold:
    """A-B interface has 5 endpoints < MIN_INTERFACE_SEGMENT_SPOTS=10.
    B is long enough to satisfy MIN_RESPONDER_INTERIOR_MASS so the
    segment-floor gate is the one that fires, not the interior gate."""
    coords, rois = [], []
    for x in range(0, 10):
        for y in range(0, 20):
            coords.append((x, y)); rois.append(A_ROI)
    # B strip that only overlaps A's y range in 5 rows -> 5 endpoints.
    for x in range(10, 40):
        for y in range(15, 20):
            coords.append((x, y)); rois.append(B_ROI)
    for x in range(0, 5):
        for y in range(25, 30):
            coords.append((x, y)); rois.append(C_ROI)
    xs = np.array([c[0] for c in coords], dtype=float)
    ys = np.array([c[1] for c in coords], dtype=float)
    spot_id = [f"s{i:04d}" for i in range(len(xs))]
    df = pd.DataFrame({"x": xs, "y": ys, "spot_id": spot_id, "roi_id": rois})
    return load_sample_scaffold(df, sample_id="hand_narrow", radius_factor=1.5)


def _model() -> SignalModel:
    return SignalModel(effect_delta=1.0)


def _rows_by_role(result) -> Dict[str, list]:
    by_iid = {i.scenario_instance_id: (i.counterfactual_role, i.generator_infeasible)
              for i in result.instances}
    out: Dict[str, list] = {}
    for r in result.spot_truth_rows:
        role, infeas = by_iid[r.scenario_instance_id]
        if infeas:
            continue
        out.setdefault(role, []).append(r)
    return out


def _vec(rows, field: str) -> np.ndarray:
    ordered = sorted(rows, key=lambda r: r.source_spot_id)
    return np.array([float(getattr(r, field)) for r in ordered])


def _spot_to_pos(scaf: SampleScaffold) -> Dict[str, int]:
    return {str(scaf.spot_ids[i]): i for i in range(scaf.n)}


# ---------------------------------------------------------------------------
# responder_activation mode
# ---------------------------------------------------------------------------

def test_ra_positive_field_equals_bg_plus_expected_overlay():
    scaf = _rich_scaffold()
    res = build_contact_interface_set(
        scaffold=scaf, a_roi_id=A_ROI, b_roi_id=B_ROI,
        counterfactual_set_id="cf",
        feature_a_id="a", feature_b_id="b",
        periphery_hops=2, mode=CONTACT_INTERFACE_MODE_RESPONDER_ACTIVATION,
        signal_model=_model(), base_seed=7,
    )
    per = _rows_by_role(res)
    pos = per["positive"]
    bg = SampleBackgroundBank(scaffold=scaf, participant_roi_ids=(A_ROI, B_ROI))
    bg_a = bg.field("a"); bg_b = bg.field("b")
    a_ids = np.where(scaf.source_parent_roi_id == A_ROI)[0]
    b_ids = np.where(scaf.source_parent_roi_id == B_ROI)[0]
    expected_a = bg_a.copy(); expected_a[a_ids] += 1.0
    a_vec = _vec(pos, "feature_a_value")
    ordered = sorted(pos, key=lambda r: r.source_spot_id)
    spot_to_pos = _spot_to_pos(scaf)
    for i_row, r in enumerate(ordered):
        pos_idx = spot_to_pos[r.source_spot_id]
        assert a_vec[i_row] == pytest.approx(expected_a[pos_idx])
    band_positions = set()
    for r in ordered:
        if r.generated_structure_id == "b_interface_band_positive":
            band_positions.add(spot_to_pos[r.source_spot_id])
    assert len(band_positions) > 0
    for i_row, r in enumerate(ordered):
        pos_idx = spot_to_pos[r.source_spot_id]
        expected = bg_b[pos_idx]
        if pos_idx in set(int(x) for x in b_ids):
            expected += 1.0
        if pos_idx in band_positions:
            expected += 0.5
        assert r.feature_b_value == pytest.approx(expected)


def test_ra_unrelated_boundary_broad_b_identity_and_band_size_match():
    scaf = _rich_scaffold()
    res = build_contact_interface_set(
        scaffold=scaf, a_roi_id=A_ROI, b_roi_id=B_ROI,
        counterfactual_set_id="cf",
        feature_a_id="a", feature_b_id="b",
        periphery_hops=2, mode=CONTACT_INTERFACE_MODE_RESPONDER_ACTIVATION,
        signal_model=_model(), base_seed=7,
    )
    per = _rows_by_role(res)
    assert "unrelated_boundary" in per, "unrelated_boundary expected feasible on rich scaffold"
    b_pos = _vec(per["positive"], "feature_b_value")
    b_ub = _vec(per["unrelated_boundary"], "feature_b_value")
    diff = b_pos - b_ub
    nonzero = diff[np.abs(diff) > 1e-9]
    pos_count = int(np.sum(diff > 1e-9))
    neg_count = int(np.sum(diff < -1e-9))
    assert pos_count == neg_count, (
        f"positive band size ({pos_count}) must equal unrelated band "
        f"size ({neg_count})"
    )
    assert np.all(np.isclose(np.abs(nonzero), 0.5))


def test_ra_unrelated_boundary_infeasible_when_no_third_roi():
    scaf = _no_third_scaffold()
    res = build_contact_interface_set(
        scaffold=scaf, a_roi_id=A_ROI, b_roi_id=B_ROI,
        counterfactual_set_id="cf",
        feature_a_id="a", feature_b_id="b",
        periphery_hops=2, mode=CONTACT_INTERFACE_MODE_RESPONDER_ACTIVATION,
        signal_model=_model(), base_seed=7,
    )
    by_role = {i.counterfactual_role: i for i in res.instances}
    assert by_role["positive"].role_disposition == "feasible"
    assert by_role["no_interface_increment"].role_disposition == "feasible"
    assert by_role["unrelated_boundary"].role_disposition == "infeasible"
    assert by_role["unrelated_boundary"].fallback_reason == "insufficient_unrelated_boundary_pool"


def test_ra_scenario_set_infeasible_when_no_ab_interface_or_narrow_segment():
    scaf = _detached_scaffold()
    res = build_contact_interface_set(
        scaffold=scaf, a_roi_id=A_ROI, b_roi_id=B_ROI,
        counterfactual_set_id="cf",
        feature_a_id="a", feature_b_id="b",
        periphery_hops=2, mode=CONTACT_INTERFACE_MODE_RESPONDER_ACTIVATION,
        signal_model=_model(), base_seed=7,
    )
    reasons = {i.counterfactual_role: i.fallback_reason for i in res.instances}
    assert set(reasons) == {
        "positive", "no_interface_increment",
        "unrelated_boundary",
    }
    assert all(v == "no_ab_interface" for v in reasons.values())
    narrow = _small_interface_scaffold()
    res_narrow = build_contact_interface_set(
        scaffold=narrow, a_roi_id=A_ROI, b_roi_id=B_ROI,
        counterfactual_set_id="cf",
        feature_a_id="a", feature_b_id="b",
        periphery_hops=2, mode=CONTACT_INTERFACE_MODE_RESPONDER_ACTIVATION,
        signal_model=_model(), base_seed=7,
    )
    narrow_reasons = {i.counterfactual_role: i.fallback_reason for i in res_narrow.instances}
    assert all(v == "insufficient_total_contact_support"
               for v in narrow_reasons.values())


def test_ra_invariant_policy_catches_a_field_drift():
    scaf = _rich_scaffold()
    res = build_contact_interface_set(
        scaffold=scaf, a_roi_id=A_ROI, b_roi_id=B_ROI,
        counterfactual_set_id="cf",
        feature_a_id="a", feature_b_id="b",
        periphery_hops=2, mode=CONTACT_INTERFACE_MODE_RESPONDER_ACTIVATION,
        signal_model=_model(), base_seed=7,
    )
    for inst in res.instances:
        if inst.role_disposition == "feasible":
            assert inst.fallback_reason is None
    bg = SampleBackgroundBank(scaffold=scaf, participant_roi_ids=(A_ROI, B_ROI))
    bg_a = bg.field("a"); bg_b = bg.field("b")
    builder = ScenarioSetBuilder(
        scaffold=scaf, counterfactual_set_id="cf-mut",
        scenario_name="contact_interface",
        feature_a_id="a", feature_b_id="b",
        nuisance_id="default", nuisance_parameters={},
        requested_parameters={}, rng_seed=0,
        partition="dev",
        pair_a_roi_id=A_ROI, pair_b_roi_id=B_ROI,
    )
    a_ids = np.where(scaf.source_parent_roi_id == A_ROI)[0]
    empty_arr = np.array([], dtype=int)
    for role_name, mult in [
        ("positive", 1.0),
        ("no_interface_increment", 1.0),
        ("unrelated_boundary", 1.5),  # A drift on last role
    ]:
        role_a = bg_a.copy()
        role_a[a_ids] += mult
        builder.add_instance(
            role=role_name,
            target_selected_a=a_ids,
            target_selected_b=empty_arr,
            target_a_field=role_a,
            target_b_field=bg_b,
        )
    with pytest.raises(SetInvariantViolation, match="feature A drifted"):
        _validate_vector_invariants(
            builder, CONTACT_INTERFACE_RESPONDER_ACTIVATION_INVARIANTS,
        )


# ---------------------------------------------------------------------------
# anchor_fade mode
# ---------------------------------------------------------------------------

def _band_positions_and_hops(res, scaf: SampleScaffold):
    """Return {position_index: hop_distance} for the anchor_fade band."""
    per = _rows_by_role(res)
    ordered = sorted(per["positive"], key=lambda r: r.source_spot_id)
    spot_to_pos = _spot_to_pos(scaf)
    band = {}
    for r in ordered:
        if r.generated_structure_id == "a_anchor_fade_band":
            band[spot_to_pos[r.source_spot_id]] = int(r.source_interface_hops)
    return band


def test_af_positive_a_carries_fade_and_b_matches_null():
    scaf = _rich_scaffold()
    res = build_contact_interface_set(
        scaffold=scaf, a_roi_id=A_ROI, b_roi_id=B_ROI,
        counterfactual_set_id="cf",
        feature_a_id="a", feature_b_id="b",
        periphery_hops=2, mode=CONTACT_INTERFACE_MODE_ANCHOR_FADE,
        signal_model=_model(), base_seed=7,
    )
    per = _rows_by_role(res)
    assert "positive" in per and "no_anchor_extension" in per
    band = _band_positions_and_hops(res, scaf)
    assert len(band) > 0
    # B is byte-identical between positive and null (no interface overlay).
    b_pos = _vec(per["positive"], "feature_b_value")
    b_null = _vec(per["no_anchor_extension"], "feature_b_value")
    assert np.array_equal(b_pos, b_null)
    # A carries broad_a + fade on band positions.
    bg = SampleBackgroundBank(scaffold=scaf, participant_roi_ids=(A_ROI, B_ROI))
    bg_a = bg.field("a")
    a_ids = set(int(x) for x in np.where(scaf.source_parent_roi_id == A_ROI)[0])
    ordered = sorted(per["positive"], key=lambda r: r.source_spot_id)
    spot_to_pos = _spot_to_pos(scaf)
    for r in ordered:
        pos_idx = spot_to_pos[r.source_spot_id]
        expected = bg_a[pos_idx]
        if pos_idx in a_ids:
            expected += 1.0
        if pos_idx in band:
            expected += 0.5 * 1.0 * (1.0 - band[pos_idx] / PERIPHERY_HOPS)
        assert r.feature_a_value == pytest.approx(expected)


def test_af_fade_magnitude_matches_linear_decay():
    scaf = _rich_scaffold()
    res = build_contact_interface_set(
        scaffold=scaf, a_roi_id=A_ROI, b_roi_id=B_ROI,
        counterfactual_set_id="cf",
        feature_a_id="a", feature_b_id="b",
        periphery_hops=2, mode=CONTACT_INTERFACE_MODE_ANCHOR_FADE,
        signal_model=_model(), base_seed=7,
    )
    band = _band_positions_and_hops(res, scaf)
    per = _rows_by_role(res)
    ordered_pos = {r.source_spot_id: r for r in per["positive"]}
    ordered_null = {r.source_spot_id: r for r in per["no_anchor_extension"]}
    hop0 = [sid for sid, r in ordered_pos.items()
            if band.get(_spot_to_pos(scaf)[sid], -1) == 0]
    hop1 = [sid for sid, r in ordered_pos.items()
            if band.get(_spot_to_pos(scaf)[sid], -1) == 1]
    assert hop0, "expected at least one hop=0 band spot"
    assert hop1, "expected at least one hop=1 band spot"
    # positive.A - null.A = fade magnitude at hop h.
    for sid in hop0:
        assert (ordered_pos[sid].feature_a_value
                - ordered_null[sid].feature_a_value) == pytest.approx(0.5)
    for sid in hop1:
        assert (ordered_pos[sid].feature_a_value
                - ordered_null[sid].feature_a_value) == pytest.approx(0.25)


def test_af_null_a_equals_broad_only():
    scaf = _rich_scaffold()
    res = build_contact_interface_set(
        scaffold=scaf, a_roi_id=A_ROI, b_roi_id=B_ROI,
        counterfactual_set_id="cf",
        feature_a_id="a", feature_b_id="b",
        periphery_hops=2, mode=CONTACT_INTERFACE_MODE_ANCHOR_FADE,
        signal_model=_model(), base_seed=7,
    )
    per = _rows_by_role(res)
    bg = SampleBackgroundBank(scaffold=scaf, participant_roi_ids=(A_ROI, B_ROI))
    bg_a = bg.field("a")
    a_ids = set(int(x) for x in np.where(scaf.source_parent_roi_id == A_ROI)[0])
    ordered = sorted(per["no_anchor_extension"], key=lambda r: r.source_spot_id)
    spot_to_pos = _spot_to_pos(scaf)
    for r in ordered:
        pos_idx = spot_to_pos[r.source_spot_id]
        expected = bg_a[pos_idx]
        if pos_idx in a_ids:
            expected += 1.0
        assert r.feature_a_value == pytest.approx(expected)
    # anchor_fade inherits substrate-level gates from responder_activation.
    detached = _detached_scaffold()
    res_af = build_contact_interface_set(
        scaffold=detached, a_roi_id=A_ROI, b_roi_id=B_ROI,
        counterfactual_set_id="cf",
        feature_a_id="a", feature_b_id="b",
        periphery_hops=2, mode=CONTACT_INTERFACE_MODE_ANCHOR_FADE,
        signal_model=_model(), base_seed=7,
    )
    reasons = {i.counterfactual_role: i.fallback_reason for i in res_af.instances}
    assert set(reasons) == {"positive", "no_anchor_extension"}
    assert all(v == "no_ab_interface" for v in reasons.values())


def test_cross_mode_pair_lock_shared_broad_overlays():
    scaf = _rich_scaffold()
    res_ra = build_contact_interface_set(
        scaffold=scaf, a_roi_id=A_ROI, b_roi_id=B_ROI,
        counterfactual_set_id="cf-ra",
        feature_a_id="a", feature_b_id="b",
        periphery_hops=2, mode=CONTACT_INTERFACE_MODE_RESPONDER_ACTIVATION,
        signal_model=_model(), base_seed=7,
    )
    res_af = build_contact_interface_set(
        scaffold=scaf, a_roi_id=A_ROI, b_roi_id=B_ROI,
        counterfactual_set_id="cf-af",
        feature_a_id="a", feature_b_id="b",
        periphery_hops=2, mode=CONTACT_INTERFACE_MODE_ANCHOR_FADE,
        signal_model=_model(), base_seed=7,
    )
    per_ra = _rows_by_role(res_ra)
    per_af = _rows_by_role(res_af)
    # responder_activation.positive.A == anchor_fade.no_anchor_extension.A
    # (both are broad-A only on the same bg_bank).
    a_ra_pos = _vec(per_ra["positive"], "feature_a_value")
    a_af_null = _vec(per_af["no_anchor_extension"], "feature_a_value")
    assert np.array_equal(a_ra_pos, a_af_null)
    # responder_activation.positive.B carries the interface increment on the
    # band; anchor_fade.positive.B does not. They must differ.
    b_ra_pos = _vec(per_ra["positive"], "feature_b_value")
    b_af_pos = _vec(per_af["positive"], "feature_b_value")
    assert not np.array_equal(b_ra_pos, b_af_pos)
    # anchor_fade.positive.B equals broad-B only (== responder_activation
    # positive.B minus the interface_increment on the band).
    diff = b_ra_pos - b_af_pos
    nonzero = diff[np.abs(diff) > 1e-9]
    assert nonzero.size > 0
    assert np.all(np.isclose(nonzero, 0.5))


def test_unrelated_boundary_band_enforces_single_third_and_falls_back_across_components():
    """`_find_unrelated_boundary_band` enforces (a) whole-band single-
    third exclusion (no band spot borders more than one third ROI, not
    only seeds), and (b) deterministic seed fallback across disconnected
    components of the eligibility pool.

    Scaffold: A + B + C (two disconnected top+bottom C segments, top
    too small individually) + D (large mid slab). B spots at column 9
    that border BOTH a C spot and a D spot are the mixed-third
    exclusion target — the returned band must never include them.
    The min-spot-id C-seed sits in a 3-spot top component (< target=5);
    fallback must retry the bottom C-seed whose component supplies 9
    spots and yields a valid band."""
    from scenario_benchmark.pair_interface import build_pair_interface
    from scenario_benchmark.scenarios.contact_interface import (
        _find_unrelated_boundary_band,
    )
    D_ROI = "D"
    xs = np.repeat(np.arange(15), 20).astype(float)
    ys = np.tile(np.arange(20), 15).astype(float)
    # A: x<5. B: 5<=x<10. Third stripe at x>=10:
    #   C at y in {0,1}      (top, only 2 rows -> tiny)
    #   C at y in {15..19}   (bottom, 5 rows   -> large)
    #   D at y in {2..14}    (mid slab, 13 rows)
    def _roi_for(x: float, y: float) -> str:
        if x < 5:
            return A_ROI
        if x < 10:
            return B_ROI
        if y <= 1 or y >= 15:
            return C_ROI
        return D_ROI
    roi = np.array([_roi_for(x, y) for x, y in zip(xs, ys)], dtype=object)
    spot_id = [f"s{i:04d}" for i in range(len(xs))]
    df = pd.DataFrame({"x": xs, "y": ys, "spot_id": spot_id, "roi_id": roi})
    scaf = load_sample_scaffold(df, sample_id="two_thirds_fallback", radius_factor=1.5)
    pair = build_pair_interface(scaffold=scaf, a_roi_id=A_ROI, b_roi_id=B_ROI)
    result = _find_unrelated_boundary_band(
        pair=pair, scaffold=scaf, target_band_size=5, periphery_hops=2,
    )
    assert result is not None, "expected feasible fallback patch on this scaffold"
    band, third_roi_id = result

    # (1) Chosen third and no mixed-third band spots. Interior spots (h=1)
    # may border only B; the contract requires ZERO band spots to border
    # the forbidden third, and at least one seed to border the chosen one.
    forbidden_third = D_ROI if third_roi_id == C_ROI else C_ROI
    borders_chosen = False
    for i in band:
        neighbours = {
            str(scaf.source_parent_roi_id[int(v)]) for v in scaf.adj[int(i)]
        }
        if third_roi_id in neighbours:
            borders_chosen = True
        assert forbidden_third not in neighbours, (
            f"band spot {int(i)} borders forbidden third ROI "
            f"{forbidden_third!r} (chosen was {third_roi_id!r}) — "
            "whole-band single-third exclusion violated"
        )
    assert borders_chosen, (
        "no band spot borders the chosen third ROI — patch was grown from "
        "an unrelated seed"
    )

    # (2) Fallback: the chosen third is C, and the returned band came from
    # the bottom (large) component, not the top (too-small) component.
    # Bottom-component seed range: y in [15..19]; top-component seed range:
    # y in [0..1]. Since the top component has only 3 eligible spots
    # (< target=5), the returned band must sit at high y.
    assert third_roi_id == C_ROI, (
        f"expected fallback to third=C via bottom seed; got {third_roi_id!r}"
    )
    band_ys = [float(scaf.coords[int(i), 1]) for i in band]
    assert min(band_ys) >= 14.0, (
        f"band drawn from top component (small); ys={band_ys!r}. "
        "Seed fallback did not retry the bottom seed."
    )
