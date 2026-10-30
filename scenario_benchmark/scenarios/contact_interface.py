"""Scenario family ``contact_interface`` (two modes).

Compositional formula: ``role_field = bg_bank + broad_overlay(+ interface_overlay)``.
Broad A + broad B across each parent ROI; both modes share the same
substrate-locked interface band, aggregated across every A-B contact
fragment (multiple boundaries between the same pair all contribute).

Modes:

- ``responder_activation`` -- interface increment on the B-side band.
  Roles:
    positive                        primary_positive
    no_interface_increment          mode_ablation
    unrelated_boundary              specificity_control (conditional)
- ``anchor_fade`` -- linear-decay A-side fade extension onto the same
  band. Roles:
    positive                        primary_positive
    no_anchor_extension             mode_ablation
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from scenario_benchmark.pair_interface import (
    PairInterface, _multi_source_hop_within_roi,
    build_interface_truth_bundle, build_pair_interface,
)
from scenario_benchmark.sample_scaffold import SampleScaffold
from scenario_benchmark.scenario_assembly import (
    InvariantPolicy, ScenarioSetBuilder, StructureAssignment,
)
from scenario_benchmark.scenarios.result import ScenarioSet
from scenario_benchmark.scenarios.support import build_all_infeasible_set
from scenario_benchmark.background_bank import SampleBackgroundBank
from scenario_benchmark.signal_model import (
    SignalModel, assert_composition, compose_role_field, constant_overlay,
)


SCENARIO_NAME = "contact_interface"

MODE_RESPONDER_ACTIVATION = "responder_activation"
MODE_ANCHOR_FADE = "anchor_fade"
MODES: Tuple[str, ...] = (MODE_RESPONDER_ACTIVATION, MODE_ANCHOR_FADE)

RESPONDER_ACTIVATION_ROLES: Tuple[str, ...] = (
    "positive", "no_interface_increment", "unrelated_boundary",
)
ANCHOR_FADE_ROLES: Tuple[str, ...] = ("positive", "no_anchor_extension")

# Union retained for discovery/export only. Acceptance is mode-specific.
ROLES: Tuple[str, ...] = tuple(sorted(
    set(RESPONDER_ACTIVATION_ROLES) | set(ANCHOR_FADE_ROLES)
))
ROLE_CONTRACT_BY_MODE = {
    MODE_RESPONDER_ACTIVATION: {
        "required_roles": ("positive", "no_interface_increment"),
        "conditional_roles": ("unrelated_boundary",),
    },
    MODE_ANCHOR_FADE: {
        "required_roles": ("positive", "no_anchor_extension"),
        "conditional_roles": (),
    },
}

# Comparison-kind metadata per role. Persisted via
# ScenarioSetBuilder.add_instance(role_specific_metadata=...) so downstream
# consumers can distinguish primary positives from mode ablations,
# specificity controls, and diagnostic baselines without hard-coding role
# names.
RESPONDER_ACTIVATION_COMPARISON_KIND = {
    "positive": "primary_positive",
    "no_interface_increment": "mode_ablation",
    "unrelated_boundary": "specificity_control",
}
ANCHOR_FADE_COMPARISON_KIND = {
    "positive": "primary_positive",
    "no_anchor_extension": "mode_ablation",
}

# Frozen constants (module top).
PERIPHERY_HOPS = 2
INTERFACE_INCREMENT_RATIO = 0.5      # responder_activation
ANCHOR_FADE_PEAK_RATIO = 0.5         # anchor_fade peak at h=0
MIN_INTERFACE_SEGMENT_SPOTS = 10
MIN_INTERFACE_BAND_MASS = 20
MIN_RESPONDER_INTERIOR_MASS = 20
MAX_BAND_FRACTION_OF_RESPONDER = 0.60
FAR_HOP_MIN = 3

STRUCTURE_KIND_BROAD_A = "broad_region"
STRUCTURE_KIND_BROAD_B = "broad_region"
STRUCTURE_KIND_INTERFACE_BAND = "interface_band"
STRUCTURE_KIND_ANCHOR_FADE = "anchor_fade_band"


CONTACT_INTERFACE_RESPONDER_ACTIVATION_INVARIANTS = InvariantPolicy(
    expected_roles=RESPONDER_ACTIVATION_ROLES,
    feature_a_byte_identical_roles=RESPONDER_ACTIVATION_ROLES,
    # unrelated_boundary depends on a suitable third-ROI B boundary.
    permitted_role_level_infeasibilities=("unrelated_boundary",),
    require_non_pair_background_identity=True,
)


# anchor_fade.positive has an A-side fade overlay on the band;
# no_anchor_extension has broad-A only. A drifts between them by
# construction, so no A byte-identity across roles for this mode.
CONTACT_INTERFACE_ANCHOR_FADE_INVARIANTS = InvariantPolicy(
    expected_roles=ANCHOR_FADE_ROLES,
    feature_a_byte_identical_roles=(),
    permitted_role_level_infeasibilities=(),
    require_non_pair_background_identity=True,
)


def build_contact_interface_set(
    *, scaffold: SampleScaffold,
    a_roi_id: str, b_roi_id: str,
    counterfactual_set_id: str,
    feature_a_id: str, feature_b_id: str,
    periphery_hops: int,
    mode: str,
    signal_model: SignalModel,
    background_policy=None,
    nuisance_id: str = "default",
    nuisance_parameters: Optional[dict] = None,
    base_seed: int = 0,
    partition: str = "dev",
    related_sample_group_id: str = "",
) -> ScenarioSet:
    """Emit one mode of the contact_interface family."""
    if mode not in MODES:
        raise ValueError(
            f"mode must be one of {MODES!r}; got {mode!r}."
        )
    if periphery_hops != PERIPHERY_HOPS:
        raise ValueError(
            f"periphery_hops is frozen at {PERIPHERY_HOPS}; got {periphery_hops!r}."
        )
    effect_delta = float(signal_model.effect_delta)
    broad_a_increment = effect_delta
    broad_b_increment = effect_delta
    interface_increment = INTERFACE_INCREMENT_RATIO * effect_delta

    pair = build_pair_interface(
        scaffold=scaffold, a_roi_id=str(a_roi_id), b_roi_id=str(b_roi_id),
    )
    n_b = int(pair.b_indices.size)

    requested = {
        "a_roi_id": str(a_roi_id), "b_roi_id": str(b_roi_id),
        "periphery_hops": int(periphery_hops),
        "mode": mode,
        "effect_delta": effect_delta,
    }
    nuisance_parameters = dict(nuisance_parameters or {})
    nuisance_parameters.update({
        "periphery_hops": int(periphery_hops),
        "mode": mode,
        "effect_delta": effect_delta,
    })
    mode_roles = (
        RESPONDER_ACTIVATION_ROLES if mode == MODE_RESPONDER_ACTIVATION
        else ANCHOR_FADE_ROLES
    )
    infeas_kwargs = dict(
        scaffold=scaffold, counterfactual_set_id=counterfactual_set_id,
        scenario_name=SCENARIO_NAME, roles=mode_roles,
        feature_a_id=feature_a_id, feature_b_id=feature_b_id,
        nuisance_id=nuisance_id, nuisance_parameters=nuisance_parameters,
        requested_parameters=requested,
        rng_seed=base_seed, partition=partition,
        related_sample_group_id=related_sample_group_id,
    )

    if pair.ab_edges.shape[0] == 0:
        return build_all_infeasible_set(reason="no_ab_interface", **infeas_kwargs)

    segment_result = _select_aggregated_interface_bands(
        pair=pair, periphery_hops=int(periphery_hops), n_b=n_b,
    )
    if isinstance(segment_result, str):
        return build_all_infeasible_set(reason=segment_result, **infeas_kwargs)
    positive_band, n_responder_interior = segment_result

    a_broad_positions = np.array(sorted(int(i) for i in pair.a_indices), dtype=int)
    b_broad_positions = np.array(sorted(int(i) for i in pair.b_indices), dtype=int)

    bg_bank = SampleBackgroundBank(
        scaffold=scaffold,
        participant_roi_ids=(str(a_roi_id), str(b_roi_id)),
        policy=background_policy,
    )
    bg_a = bg_bank.field(feature_a_id)
    bg_b = bg_bank.field(feature_b_id)

    common_realized = {
        "mode": mode,
        "n_a_broad_spots": int(a_broad_positions.size),
        "n_b_broad_spots": int(b_broad_positions.size),
        "n_selected_interface_band_spots": int(positive_band.size),
        "selected_interface_band_fraction_of_b": (
            float(positive_band.size) / float(n_b) if n_b else 0.0
        ),
        "n_responder_interior_spots": n_responder_interior,
        "background_degenerate_rois": list(bg_bank.degenerate_rois),
    }

    if mode == MODE_RESPONDER_ACTIVATION:
        return _build_responder_activation(
            scaffold=scaffold, pair=pair,
            counterfactual_set_id=counterfactual_set_id,
            feature_a_id=feature_a_id, feature_b_id=feature_b_id,
            a_roi_id=str(a_roi_id), b_roi_id=str(b_roi_id),
            periphery_hops=int(periphery_hops),
            bg_a=bg_a, bg_b=bg_b,
            a_broad_positions=a_broad_positions,
            b_broad_positions=b_broad_positions,
            positive_band=positive_band,
            broad_a_increment=broad_a_increment,
            broad_b_increment=broad_b_increment,
            interface_increment=interface_increment,
            n_b=n_b,
            common_realized=common_realized,
            requested=requested, nuisance_id=nuisance_id,
            nuisance_parameters=nuisance_parameters,
            base_seed=base_seed, partition=partition,
            related_sample_group_id=related_sample_group_id,
        )
    return _build_anchor_fade(
        scaffold=scaffold, pair=pair,
        counterfactual_set_id=counterfactual_set_id,
        feature_a_id=feature_a_id, feature_b_id=feature_b_id,
        a_roi_id=str(a_roi_id), b_roi_id=str(b_roi_id),
        bg_a=bg_a, bg_b=bg_b,
        a_broad_positions=a_broad_positions,
        b_broad_positions=b_broad_positions,
        positive_band=positive_band,
        broad_a_increment=broad_a_increment,
        broad_b_increment=broad_b_increment,
        effect_delta=float(signal_model.effect_delta),
        common_realized=common_realized,
        requested=requested, nuisance_id=nuisance_id,
        nuisance_parameters=nuisance_parameters,
        base_seed=base_seed, partition=partition,
        related_sample_group_id=related_sample_group_id,
    )


# ---------------------------------------------------------------------------
# Mode: responder_activation
# ---------------------------------------------------------------------------

def _build_responder_activation(
    *, scaffold: SampleScaffold, pair: PairInterface,
    counterfactual_set_id: str,
    feature_a_id: str, feature_b_id: str,
    a_roi_id: str, b_roi_id: str,
    periphery_hops: int,
    bg_a: np.ndarray, bg_b: np.ndarray,
    a_broad_positions: np.ndarray,
    b_broad_positions: np.ndarray,
    positive_band: np.ndarray,
    broad_a_increment: float,
    broad_b_increment: float,
    interface_increment: float,
    n_b: int,
    common_realized: dict,
    requested: dict, nuisance_id: str,
    nuisance_parameters: dict,
    base_seed: int, partition: str,
    related_sample_group_id: str,
) -> ScenarioSet:
    a_overlay = constant_overlay(
        n=scaffold.n, positions=a_broad_positions, magnitude=broad_a_increment,
    )
    a_field = compose_role_field(bg=bg_a, overlay=a_overlay)

    b_overlay_positive = constant_overlay(
        n=scaffold.n, positions=b_broad_positions, magnitude=broad_b_increment,
    ) + constant_overlay(
        n=scaffold.n, positions=positive_band, magnitude=interface_increment,
    )
    b_field_positive = compose_role_field(bg=bg_b, overlay=b_overlay_positive)

    # Mode ablation: broad B present, interface increment absent.
    b_overlay_no_interface = constant_overlay(
        n=scaffold.n, positions=b_broad_positions, magnitude=broad_b_increment,
    )
    b_field_no_interface = compose_role_field(
        bg=bg_b, overlay=b_overlay_no_interface,
    )

    assert_composition(field=a_field, bg=bg_a, overlay=a_overlay,
                       scenario_name=SCENARIO_NAME, role="positive", feature="a")
    assert_composition(field=b_field_positive, bg=bg_b, overlay=b_overlay_positive,
                       scenario_name=SCENARIO_NAME, role="positive", feature="b")
    assert_composition(
        field=b_field_no_interface, bg=bg_b, overlay=b_overlay_no_interface,
        scenario_name=SCENARIO_NAME, role="no_interface_increment", feature="b",
    )
    positive_realized = dict(common_realized)
    positive_realized["n_b_interface_increment_spots"] = int(positive_band.size)
    no_interface_realized = dict(common_realized)
    no_interface_realized["n_b_interface_increment_spots"] = 0
    builder = ScenarioSetBuilder(
        scaffold=scaffold, counterfactual_set_id=counterfactual_set_id,
        scenario_name=SCENARIO_NAME,
        feature_a_id=feature_a_id, feature_b_id=feature_b_id,
        nuisance_id=nuisance_id, nuisance_parameters=nuisance_parameters,
        requested_parameters=requested, rng_seed=base_seed,
        partition=partition, related_sample_group_id=related_sample_group_id,
        pair_a_roi_id=a_roi_id, pair_b_roi_id=b_roi_id,
    )

    a_structure = [StructureAssignment(
        positions=a_broad_positions,
        structure_kind=STRUCTURE_KIND_BROAD_A, structure_id="a_broad",
    )]
    b_broad_structure = StructureAssignment(
        positions=b_broad_positions,
        structure_kind=STRUCTURE_KIND_BROAD_B, structure_id="b_broad",
    )
    positive_band_structure = StructureAssignment(
        positions=positive_band,
        structure_kind=STRUCTURE_KIND_INTERFACE_BAND,
        structure_id="b_interface_band_positive",
    )

    iid_positive = f"{counterfactual_set_id}::positive"
    positive_truth = build_interface_truth_bundle(pair, iid_positive)
    builder.add_instance(
            role="positive",
            target_selected_a=a_broad_positions,
            target_selected_b=np.array(
                sorted(int(x) for x in
                       set(int(i) for i in b_broad_positions)
                       | set(int(i) for i in positive_band)),
                dtype=int,
            ),
            target_a_field=a_field,
            target_b_field=b_field_positive,
            structures_a=a_structure,
            structures_b=[b_broad_structure, positive_band_structure],
            realized_parameters=positive_realized,
            role_specific_metadata={
                "comparison_kind":
                    RESPONDER_ACTIVATION_COMPARISON_KIND["positive"],
            },
            interface_truth=positive_truth,
    )

    iid_no_interface = f"{counterfactual_set_id}::no_interface_increment"
    no_interface_truth = build_interface_truth_bundle(pair, iid_no_interface)
    builder.add_instance(
            role="no_interface_increment",
            target_selected_a=a_broad_positions,
            target_selected_b=np.array(
                sorted(int(x) for x in set(int(i) for i in b_broad_positions)),
                dtype=int,
            ),
            target_a_field=a_field,
            target_b_field=b_field_no_interface,
            structures_a=a_structure,
            structures_b=[b_broad_structure],
            realized_parameters=no_interface_realized,
            role_specific_metadata={
                "comparison_kind":
                    RESPONDER_ACTIVATION_COMPARISON_KIND["no_interface_increment"],
            },
            interface_truth=no_interface_truth,
    )

    ub_result = _find_unrelated_boundary_band(
        pair=pair, scaffold=scaffold,
        target_band_size=int(positive_band.size),
        periphery_hops=int(periphery_hops),
    )
    iid_ub = f"{counterfactual_set_id}::unrelated_boundary"
    if ub_result is None:
        ub_infeasible_realized = dict(common_realized)
        ub_infeasible_realized["n_b_interface_increment_spots"] = 0
        ub_infeasible_realized["third_boundary_roi_id"] = ""
        builder.add_infeasible_instance(
            role="unrelated_boundary",
            fallback_reason="insufficient_unrelated_boundary_pool",
            realized_parameters=ub_infeasible_realized,
        )
    else:
        ub_band, third_boundary_roi_id = ub_result
        ub_overlay = constant_overlay(
            n=scaffold.n, positions=b_broad_positions, magnitude=broad_b_increment,
        ) + constant_overlay(
            n=scaffold.n, positions=ub_band, magnitude=interface_increment,
        )
        ub_field_b = compose_role_field(bg=bg_b, overlay=ub_overlay)
        assert_composition(
            field=ub_field_b, bg=bg_b, overlay=ub_overlay,
            scenario_name=SCENARIO_NAME, role="unrelated_boundary", feature="b",
        )
        ub_structure = [
            b_broad_structure,
            StructureAssignment(
                positions=ub_band,
                structure_kind=STRUCTURE_KIND_INTERFACE_BAND,
                structure_id="b_interface_band_unrelated",
            ),
        ]
        ub_realized = dict(common_realized)
        ub_realized["n_b_interface_increment_spots"] = int(ub_band.size)
        ub_realized["third_boundary_roi_id"] = str(third_boundary_roi_id)
        ub_truth = build_interface_truth_bundle(pair, iid_ub)
        builder.add_instance(
                role="unrelated_boundary",
                target_selected_a=a_broad_positions,
                target_selected_b=np.array(
                    sorted(int(x) for x in
                           set(int(i) for i in b_broad_positions)
                           | set(int(i) for i in ub_band)),
                    dtype=int,
                ),
                target_a_field=a_field,
                target_b_field=ub_field_b,
                structures_a=a_structure,
                structures_b=ub_structure,
                realized_parameters=ub_realized,
                role_specific_metadata={
                    "comparison_kind":
                        RESPONDER_ACTIVATION_COMPARISON_KIND["unrelated_boundary"],
                },
                interface_truth=ub_truth,
        )

    return builder.finalize(
        invariant_policy=CONTACT_INTERFACE_RESPONDER_ACTIVATION_INVARIANTS,
    )


# ---------------------------------------------------------------------------
# Mode: anchor_fade
# ---------------------------------------------------------------------------

def _build_anchor_fade(
    *, scaffold: SampleScaffold, pair: PairInterface,
    counterfactual_set_id: str,
    feature_a_id: str, feature_b_id: str,
    a_roi_id: str, b_roi_id: str,
    bg_a: np.ndarray, bg_b: np.ndarray,
    a_broad_positions: np.ndarray,
    b_broad_positions: np.ndarray,
    positive_band: np.ndarray,
    broad_a_increment: float,
    broad_b_increment: float,
    effect_delta: float,
    common_realized: dict,
    requested: dict, nuisance_id: str,
    nuisance_parameters: dict,
    base_seed: int, partition: str,
    related_sample_group_id: str,
) -> ScenarioSet:
    # Broad-A + broad-B overlays byte-identical across the two modes'
    # positive roles (both derived from the same substrate + bg_bank).
    a_broad_overlay = constant_overlay(
        n=scaffold.n, positions=a_broad_positions, magnitude=broad_a_increment,
    )
    b_broad_overlay = constant_overlay(
        n=scaffold.n, positions=b_broad_positions, magnitude=broad_b_increment,
    )

    # A-side fade extension on the same band positions: linear decay in
    # hop distance from the A-B interface, peak at h=0.
    b_dist = pair.b_side_interface_distance
    fade_magnitudes = np.array([
        ANCHOR_FADE_PEAK_RATIO * effect_delta
        * (1.0 - float(int(b_dist[int(pos)])) / float(PERIPHERY_HOPS))
        for pos in positive_band
    ], dtype=float)
    overlay_a_fade = np.zeros(scaffold.n, dtype=float)
    overlay_a_fade[positive_band] = fade_magnitudes

    a_overlay_positive = a_broad_overlay + overlay_a_fade
    a_field_positive = compose_role_field(bg=bg_a, overlay=a_overlay_positive)
    b_field_positive = compose_role_field(bg=bg_b, overlay=b_broad_overlay)

    a_field_null = compose_role_field(bg=bg_a, overlay=a_broad_overlay)
    b_field_null = compose_role_field(bg=bg_b, overlay=b_broad_overlay)

    assert_composition(field=a_field_positive, bg=bg_a, overlay=a_overlay_positive,
                       scenario_name=SCENARIO_NAME, role="positive", feature="a")
    assert_composition(field=b_field_positive, bg=bg_b, overlay=b_broad_overlay,
                       scenario_name=SCENARIO_NAME, role="positive", feature="b")
    assert_composition(field=a_field_null, bg=bg_a, overlay=a_broad_overlay,
                       scenario_name=SCENARIO_NAME,
                       role="no_anchor_extension", feature="a")
    assert_composition(field=b_field_null, bg=bg_b, overlay=b_broad_overlay,
                       scenario_name=SCENARIO_NAME,
                       role="no_anchor_extension", feature="b")

    positive_realized = dict(common_realized)
    positive_realized["n_b_interface_increment_spots"] = 0
    positive_realized["n_anchor_fade_band_spots"] = int(positive_band.size)
    null_realized = dict(common_realized)
    null_realized["n_b_interface_increment_spots"] = 0
    null_realized["n_anchor_fade_band_spots"] = 0

    builder = ScenarioSetBuilder(
        scaffold=scaffold, counterfactual_set_id=counterfactual_set_id,
        scenario_name=SCENARIO_NAME,
        feature_a_id=feature_a_id, feature_b_id=feature_b_id,
        nuisance_id=nuisance_id, nuisance_parameters=nuisance_parameters,
        requested_parameters=requested, rng_seed=base_seed,
        partition=partition, related_sample_group_id=related_sample_group_id,
        pair_a_roi_id=a_roi_id, pair_b_roi_id=b_roi_id,
    )

    a_broad_structure = StructureAssignment(
        positions=a_broad_positions,
        structure_kind=STRUCTURE_KIND_BROAD_A, structure_id="a_broad",
    )
    b_broad_structure = StructureAssignment(
        positions=b_broad_positions,
        structure_kind=STRUCTURE_KIND_BROAD_B, structure_id="b_broad",
    )
    a_fade_structure = StructureAssignment(
        positions=positive_band,
        structure_kind=STRUCTURE_KIND_ANCHOR_FADE,
        structure_id="a_anchor_fade_band",
    )

    iid_positive = f"{counterfactual_set_id}::positive"
    positive_truth = build_interface_truth_bundle(pair, iid_positive)
    builder.add_instance(
            role="positive",
            target_selected_a=np.array(
                sorted(int(x) for x in
                       set(int(i) for i in a_broad_positions)
                       | set(int(i) for i in positive_band)),
                dtype=int,
            ),
            target_selected_b=b_broad_positions,
            target_a_field=a_field_positive,
            target_b_field=b_field_positive,
            structures_a=[a_broad_structure],
            # a_fade_structure overrides b_broad on band positions.
            structures_b=[b_broad_structure, a_fade_structure],
            realized_parameters=positive_realized,
            role_specific_metadata={
                "comparison_kind":
                    ANCHOR_FADE_COMPARISON_KIND["positive"],
            },
            interface_truth=positive_truth,
    )

    iid_null = f"{counterfactual_set_id}::no_anchor_extension"
    null_truth = build_interface_truth_bundle(pair, iid_null)
    builder.add_instance(
            role="no_anchor_extension",
            target_selected_a=a_broad_positions,
            target_selected_b=b_broad_positions,
            target_a_field=a_field_null,
            target_b_field=b_field_null,
            structures_a=[a_broad_structure],
            structures_b=[b_broad_structure],
            realized_parameters=null_realized,
            role_specific_metadata={
                "comparison_kind":
                    ANCHOR_FADE_COMPARISON_KIND["no_anchor_extension"],
            },
            interface_truth=null_truth,
    )

    return builder.finalize(
        invariant_policy=CONTACT_INTERFACE_ANCHOR_FADE_INVARIANTS,
    )


# ---------------------------------------------------------------------------
# Interface-segment selection
# ---------------------------------------------------------------------------

def _select_aggregated_interface_bands(
    *, pair: PairInterface, periphery_hops: int, n_b: int,
):
    """Return (positive_band, n_responder_interior) or an infeasibility reason.

    Aggregates every A-B contact fragment: total endpoint count over all
    fragments must clear MIN_INTERFACE_SEGMENT_SPOTS; the returned band
    is the union of B-side hop bands seeded from every contact fragment.
    Two ROIs meeting along multiple boundaries contribute every one to
    the interface increment.
    """
    fragments = pair.interface_fragments
    if not fragments:
        return "no_ab_interface"
    b_dist = pair.b_side_interface_distance
    nearest_b = pair.nearest_fragment_id_b
    b_endpoints_by_fragment: dict = {}
    for i in pair.b_indices:
        idx = int(i)
        if int(b_dist[idx]) == 0:
            fid = int(nearest_b[idx])
            if fid >= 0:
                b_endpoints_by_fragment.setdefault(fid, []).append(idx)
    if not b_endpoints_by_fragment:
        return "insufficient_total_contact_support"

    n_responder_interior = int(np.sum(
        (b_dist[pair.b_indices] >= FAR_HOP_MIN)
        & (b_dist[pair.b_indices] < pair.unreachable)
    ))
    if n_responder_interior < MIN_RESPONDER_INTERIOR_MASS:
        return "insufficient_responder_interior"

    contact_fids = frozenset(int(fid) for fid in b_endpoints_by_fragment)
    total_endpoints = sum(
        len(v) for v in b_endpoints_by_fragment.values()
    )
    if total_endpoints < MIN_INTERFACE_SEGMENT_SPOTS:
        return "insufficient_total_contact_support"

    band_positions: List[int] = []
    for i in pair.b_indices:
        idx = int(i)
        if (int(nearest_b[idx]) in contact_fids
                and 0 <= int(b_dist[idx]) < periphery_hops):
            band_positions.append(idx)
    if len(band_positions) < MIN_INTERFACE_BAND_MASS:
        return "insufficient_band_mass"
    if n_b > 0 and (len(band_positions) / float(n_b)) > MAX_BAND_FRACTION_OF_RESPONDER:
        return "interface_band_dominates_responder"
    band_arr = np.array(sorted(int(x) for x in band_positions), dtype=int)
    return band_arr, n_responder_interior


# ---------------------------------------------------------------------------
# Unrelated-boundary band selection (responder_activation only)
# ---------------------------------------------------------------------------

def _find_unrelated_boundary_band(
    *, pair: PairInterface, scaffold: SampleScaffold,
    target_band_size: int, periphery_hops: int,
) -> Optional[Tuple[np.ndarray, str]]:
    """Matched connected patch on the responder boundary with a SINGLE
    explicit third parent ROI. Iterates candidate third ROIs in sorted
    id order; for each, restricts the eligibility pool to B spots that
    (a) border THIS third ROI (as seeds) OR are within ``periphery_hops``
    of that boundary, AND (b) do not border any OTHER third ROI (whole-
    band exclusion, not just seeds); grows one connected patch of
    ``target_band_size`` spots of width ``periphery_hops``. If the
    canonical-min seed's component is too small, retries the next seed
    deterministically until either one component supplies the requested
    mass or every seed is exhausted. Returns the first
    ``(band, third_roi_id)`` for which the patch can be grown.
    """
    if target_band_size <= 0:
        return None
    a_id, b_id = str(pair.a_roi_id), str(pair.b_roi_id)
    ab_dist = pair.b_side_interface_distance
    # Enumerate candidate third ROIs.
    candidate_third_rois: set = set()
    for i in pair.b_indices:
        for v in scaffold.adj[int(i)]:
            rid = str(scaffold.source_parent_roi_id[int(v)])
            if rid != a_id and rid != b_id:
                candidate_third_rois.add(rid)
    for third_roi_id in sorted(candidate_third_rois):
        # Whole-band single-third safety: a B spot is safe iff none of its
        # non-{A, B} neighbours are OTHER third ROIs. Applied to every
        # spot considered for the band, not only to boundary seeds — an
        # h=1 spot bordering another third would silently violate the
        # single-third contract.
        single_third_safe: List[bool] = []
        borders_this_third: List[bool] = []
        for i in pair.b_indices:
            has_this = False
            has_other = False
            for v in scaffold.adj[int(i)]:
                rid = str(scaffold.source_parent_roi_id[int(v)])
                if rid == a_id or rid == b_id:
                    continue
                if rid == third_roi_id:
                    has_this = True
                else:
                    has_other = True
                    break
            single_third_safe.append(not has_other)
            borders_this_third.append(has_this and not has_other)
        safe_lookup = {
            int(pair.b_indices[k]): single_third_safe[k]
            for k in range(len(pair.b_indices))
        }
        third_boundary_indices = [
            int(pair.b_indices[k]) for k in range(len(pair.b_indices))
            if borders_this_third[k]
        ]
        if not third_boundary_indices:
            continue
        third_boundary = np.array(sorted(third_boundary_indices), dtype=int)
        third_dist = _multi_source_hop_within_roi(
            scaffold, third_boundary, pair.b_indices,
        )
        eligible = [
            int(i) for i in pair.b_indices
            if safe_lookup[int(i)]
            and 0 <= int(third_dist[int(i)]) < periphery_hops
            and int(ab_dist[int(i)]) >= periphery_hops
        ]
        if len(eligible) < target_band_size:
            continue
        allowed = np.array(sorted(eligible), dtype=int)
        local_adj = scaffold.restricted_adj(allowed)
        local_to_global = {k: int(allowed[k]) for k in range(len(allowed))}
        seeds_at_boundary = [
            k for k in range(len(allowed))
            if int(third_dist[local_to_global[k]]) == 0
        ]
        if not seeds_at_boundary:
            seeds_at_boundary = list(range(len(allowed)))
        # Deterministic seed-fallback: try boundary seeds in canonical
        # spot-id order until one connected component grown from that
        # seed supplies target_band_size. Prevents falsely marking a
        # third ROI infeasible when the min-spot-id seed sits in a
        # too-small component while another seed's component is large.
        seeds_sorted = sorted(
            seeds_at_boundary,
            key=lambda k: str(scaffold.spot_ids[local_to_global[k]]),
        )
        for seed_local in seeds_sorted:
            order = [seed_local]
            visited = {seed_local}
            frontier = [seed_local]
            while frontier and len(order) < target_band_size:
                cands = set()
                for u in frontier:
                    for v in local_adj[u]:
                        if int(v) not in visited:
                            cands.add(int(v))
                if not cands:
                    break
                ordered = sorted(
                    cands,
                    key=lambda k: str(scaffold.spot_ids[local_to_global[k]]),
                )
                nxt: List[int] = []
                for v in ordered:
                    if len(order) >= target_band_size:
                        break
                    visited.add(v); order.append(v); nxt.append(v)
                frontier = nxt
            if len(order) >= target_band_size:
                band = np.array(
                    sorted(int(local_to_global[k]) for k in order), dtype=int,
                )
                return band, third_roi_id
    return None
