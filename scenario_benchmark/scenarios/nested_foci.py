"""Broad A field with focal B activity inside the same fixed ROI.

ROI labels never change. The matched relocation keeps A and the B-value
multiset fixed while moving only the focal B patches into a declared
external support ROI.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from scenario_benchmark.background_bank import SampleBackgroundBank
from scenario_benchmark.sample_scaffold import SampleScaffold
from scenario_benchmark.scenario_assembly import (
    InvariantPolicy, ScenarioSetBuilder, StructureAssignment,
)
from scenario_benchmark.scenarios.result import ScenarioSet
from scenario_benchmark.scenarios.support import build_all_infeasible_set
from scenario_benchmark.signal_model import (
    COMPONENT_EXPRESSION_PROFILES, DEFAULT_COMPONENT_PROFILE_CONTRAST,
    SignalModel, assert_composition, compose_role_field, constant_overlay,
    profiled_component_overlay, substream_rng,
)
from scenario_benchmark.spatial_primitives import (
    flatten_components, grow_disjoint_connected_patches,
    pack_connected_patches, roi_components, roi_interior_positions,
    structures_from_components,
)


SCENARIO_NAME = "nested_foci"
MIN_FOCUS_SIZE = 5
DEFAULT_A_HOST_EFFECT_SCALE = 0.3
DEFAULT_B_EXPRESSION_PROFILE = "core_high"

POSITIVE_ROLE = "positive"
MATCHED_RELOCATION_ROLE = "b_foci_relocated_outside_a"
ROLES: Tuple[str, ...] = (POSITIVE_ROLE, MATCHED_RELOCATION_ROLE)
REQUIRED_ROLES: Tuple[str, ...] = ROLES
CONDITIONAL_ROLES: Tuple[str, ...] = ()
COMPARISON_KIND: Dict[str, str] = {
    POSITIVE_ROLE: "primary_positive",
    MATCHED_RELOCATION_ROLE: "matched_relocation",
}
INVARIANTS = InvariantPolicy(
    expected_roles=ROLES,
    feature_a_byte_identical_roles=ROLES,
    permitted_role_level_infeasibilities=(MATCHED_RELOCATION_ROLE,),
    require_non_pair_background_identity=True,
)

STRUCTURE_KIND_FOCUS = "focus"
STRUCTURE_KIND_A_HOST = "a_host_support"


def build_nested_foci_set(
    *, scaffold: SampleScaffold,
    a_roi_id: str, b_roi_id: str,
    counterfactual_set_id: str,
    feature_a_id: str, feature_b_id: str,
    focus_sizes: Sequence[int],
    signal_model: SignalModel,
    a_host_effect_scale: float = DEFAULT_A_HOST_EFFECT_SCALE,
    a_host_baseline: Optional[float] = None,
    b_focus_amplitude: Optional[float] = None,
    b_population_baseline: float = 0.0,
    background_policy=None,
    b_expression_profile: str = DEFAULT_B_EXPRESSION_PROFILE,
    nuisance_id: str = "default",
    nuisance_parameters: Optional[dict] = None,
    base_seed: int = 0,
    partition: str = "dev",
    related_sample_group_id: str = "",
) -> ScenarioSet:
    """Build positive nested foci and its matched relocation."""
    sizes = tuple(int(size) for size in focus_sizes)
    if len(sizes) < 1 or any(size < MIN_FOCUS_SIZE for size in sizes):
        raise ValueError(
            f"focus_sizes must contain values >= {MIN_FOCUS_SIZE}; got {sizes!r}."
        )
    if not (0.0 < float(a_host_effect_scale) <= 1.0):
        raise ValueError("a_host_effect_scale must lie in (0, 1].")
    if b_expression_profile not in COMPONENT_EXPRESSION_PROFILES:
        raise ValueError(
            f"b_expression_profile must be one of "
            f"{COMPONENT_EXPRESSION_PROFILES!r}."
        )

    parents = scaffold.source_parent_roi_id.astype(str)
    a_positions = np.where(parents == str(a_roi_id))[0]
    b_positions = np.where(parents == str(b_roi_id))[0]
    requested = {
        "a_roi_id": str(a_roi_id),
        "b_roi_id": str(b_roi_id),
        "focus_sizes": list(sizes),
        "b_expression_profile": b_expression_profile,
    }
    nuisance_parameters = dict(nuisance_parameters or {})
    nuisance_parameters.update({
        "effect_delta": float(signal_model.effect_delta),
        "b_expression_profile": b_expression_profile,
    })
    infeasible = dict(
        scaffold=scaffold,
        counterfactual_set_id=counterfactual_set_id,
        scenario_name=SCENARIO_NAME,
        roles=ROLES,
        feature_a_id=feature_a_id,
        feature_b_id=feature_b_id,
        nuisance_id=nuisance_id,
        nuisance_parameters=nuisance_parameters,
        requested_parameters=requested,
        rng_seed=base_seed,
        partition=partition,
        related_sample_group_id=related_sample_group_id,
    )
    if a_positions.size == 0:
        return build_all_infeasible_set(reason="a_roi_missing", **infeasible)
    if b_positions.size == 0:
        return build_all_infeasible_set(reason="b_relocation_roi_missing", **infeasible)

    positive_foci = grow_disjoint_connected_patches(
        scaffold,
        allowed_positions=roi_interior_positions(scaffold, a_positions),
        sizes=sizes,
        rng=substream_rng(base_seed=base_seed, tag=f"{SCENARIO_NAME}::positive"),
    )
    if positive_foci is None:
        return build_all_infeasible_set(
            reason="insufficient_a_interior_for_foci", **infeasible,
        )
    positive_focus_positions = flatten_components(positive_foci)
    relocated_foci = pack_connected_patches(
        scaffold,
        host_components=roi_components(scaffold, b_positions),
        sizes=sizes,
        rng=substream_rng(
            base_seed=base_seed, tag=f"{SCENARIO_NAME}::{MATCHED_RELOCATION_ROLE}",
        ),
    )

    resolved_a = (
        float(a_host_baseline) if a_host_baseline is not None
        else float(signal_model.effect_delta) * float(a_host_effect_scale)
    )
    resolved_b_focus = (
        float(b_focus_amplitude) if b_focus_amplitude is not None
        else float(signal_model.effect_delta)
    )
    resolved_b_baseline = float(b_population_baseline)
    decomposed = (
        a_host_baseline is not None
        or b_focus_amplitude is not None
        or b_population_baseline != 0.0
    )
    if decomposed:
        bg_a = SampleBackgroundBank(
            scaffold=scaffold, participant_roi_ids=(str(a_roi_id),),
            policy=background_policy,
        ).field(feature_a_id)
        bg_b = SampleBackgroundBank(
            scaffold=scaffold, participant_roi_ids=(str(a_roi_id),),
            policy=background_policy,
        ).field(feature_b_id)
    else:
        bank = SampleBackgroundBank(
            scaffold=scaffold,
            participant_roi_ids=(str(a_roi_id), str(b_roi_id)),
            policy=background_policy,
        )
        bg_a, bg_b = bank.field(feature_a_id), bank.field(feature_b_id)

    a_overlay = constant_overlay(
        n=scaffold.n, positions=a_positions, magnitude=resolved_a,
    )
    basal_b_overlay = constant_overlay(
        n=scaffold.n, positions=a_positions, magnitude=resolved_b_baseline,
    )
    a_field = compose_role_field(bg=bg_a, overlay=a_overlay)
    positive_focus_overlay = profiled_component_overlay(
        n=scaffold.n,
        components=positive_foci,
        coords=scaffold.coords,
        spot_ids=scaffold.spot_ids,
        magnitude=resolved_b_focus,
        profile=b_expression_profile,
    )
    positive_b_overlay = basal_b_overlay + positive_focus_overlay
    positive_b_field = compose_role_field(bg=bg_b, overlay=positive_b_overlay)
    assert_composition(
        field=a_field, bg=bg_a, overlay=a_overlay,
        scenario_name=SCENARIO_NAME, role=POSITIVE_ROLE, feature="a",
    )
    assert_composition(
        field=positive_b_field, bg=bg_b, overlay=positive_b_overlay,
        scenario_name=SCENARIO_NAME, role=POSITIVE_ROLE, feature="b",
    )

    common_realized = {
        "focus_sizes": list(sizes),
        "n_foci": len(sizes),
        "a_host_support_size": int(a_positions.size),
        "a_host_baseline_resolved": resolved_a,
        "b_population_baseline_resolved": resolved_b_baseline,
        "b_focus_amplitude_resolved": resolved_b_focus,
        "b_expression_profile": b_expression_profile,
        "b_profile_contrast": DEFAULT_COMPONENT_PROFILE_CONTRAST,
    }
    builder = ScenarioSetBuilder(
        scaffold=scaffold,
        counterfactual_set_id=counterfactual_set_id,
        scenario_name=SCENARIO_NAME,
        feature_a_id=feature_a_id,
        feature_b_id=feature_b_id,
        nuisance_id=nuisance_id,
        nuisance_parameters=nuisance_parameters,
        requested_parameters=requested,
        rng_seed=base_seed,
        partition=partition,
        related_sample_group_id=related_sample_group_id,
        pair_a_roi_id=str(a_roi_id),
        pair_b_roi_id=str(b_roi_id),
    )
    a_structure = StructureAssignment(
        positions=a_positions,
        structure_kind=STRUCTURE_KIND_A_HOST,
        structure_id="a_host",
    )
    builder.add_instance(
        role=POSITIVE_ROLE,
        target_selected_a=a_positions,
        target_selected_b=np.flatnonzero(positive_b_overlay),
        target_a_field=a_field,
        target_b_field=positive_b_field,
        structures_a=[a_structure],
        structures_b=structures_from_components(
            positive_foci, kind=STRUCTURE_KIND_FOCUS, prefix="focus",
        ),
        realized_parameters=common_realized,
        role_specific_metadata={
            "comparison_kind": COMPARISON_KIND[POSITIVE_ROLE],
        },
    )

    if relocated_foci is None:
        builder.add_infeasible_instance(
            role=MATCHED_RELOCATION_ROLE,
            fallback_reason="insufficient_b_home_capacity_for_matched_relocation",
            realized_parameters=common_realized,
        )
    else:
        relocated_overlay = profiled_component_overlay(
            n=scaffold.n,
            components=relocated_foci,
            coords=scaffold.coords,
            spot_ids=scaffold.spot_ids,
            magnitude=resolved_b_focus,
            profile=b_expression_profile,
        )
        relocated_b_overlay = basal_b_overlay + relocated_overlay
        relocated_b_field = compose_role_field(
            bg=bg_b, overlay=relocated_b_overlay,
        )
        assert_composition(
            field=relocated_b_field, bg=bg_b, overlay=relocated_b_overlay,
            scenario_name=SCENARIO_NAME,
            role=MATCHED_RELOCATION_ROLE, feature="b",
        )
        builder.add_instance(
            role=MATCHED_RELOCATION_ROLE,
            target_selected_a=a_positions,
            target_selected_b=np.flatnonzero(relocated_b_overlay),
            target_a_field=a_field,
            target_b_field=relocated_b_field,
            structures_a=[a_structure],
            structures_b=structures_from_components(
                relocated_foci, kind=STRUCTURE_KIND_FOCUS,
                prefix="relocated_focus",
            ),
            realized_parameters={
                **common_realized,
                "relocated_focus_sizes": sorted(map(len, relocated_foci)),
            },
            role_specific_metadata={
                "comparison_kind": COMPARISON_KIND[MATCHED_RELOCATION_ROLE],
            },
        )
    return builder.finalize(invariant_policy=INVARIANTS)
