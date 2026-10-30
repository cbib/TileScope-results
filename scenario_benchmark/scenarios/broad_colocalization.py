"""Scenario ``broad_colocalization``.

Whole-ROI co-enrichment: both A and B are elevated on EVERY spot in
the target ROI. Compositional formula:
``role_field = bg_bank.field(feature) + constant_overlay(target_roi, effect_delta)``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from scenario_benchmark.background_bank import SampleBackgroundBank
from scenario_benchmark.sample_scaffold import SampleScaffold
from scenario_benchmark.scenario_assembly import (
    InvariantPolicy, ScenarioSetBuilder, StructureAssignment,
)
from scenario_benchmark.scenarios.result import ScenarioSet
from scenario_benchmark.scenarios.support import build_all_infeasible_set
from scenario_benchmark.signal_model import (
    SignalModel, assert_composition, compose_role_field, constant_overlay,
    substream_rng,
)


SCENARIO_NAME = "broad_colocalization"
POSITIVE_ROLE = "positive_broad"
INDEPENDENT_BACKGROUND_ROLE = "b_independent_background"
ROLES: Tuple[str, ...] = (POSITIVE_ROLE, INDEPENDENT_BACKGROUND_ROLE)
REQUIRED_ROLES: Tuple[str, ...] = ROLES
CONDITIONAL_ROLES: Tuple[str, ...] = ()
COMPARISON_KIND = {
    POSITIVE_ROLE: "primary_positive",
    INDEPENDENT_BACKGROUND_ROLE: "primary_null",
}

# Below this the target ROI cannot carry a meaningful within-ROI signal.
MIN_TARGET_ROI_SPOTS = 5

STRUCTURE_KIND_BROAD = "broad_region"

# A and B share both a broad increment and a modest zero-mean latent
# modulation. The latent is independent of either feature's nuisance
# background, avoiding a directional A-background -> B construction.
BROAD_COLOCALIZATION_SHARED_LATENT_RATIO = 1.5


BROAD_COLOCALIZATION_INVARIANTS = InvariantPolicy(
    expected_roles=ROLES,
    feature_a_byte_identical_roles=ROLES,
    permitted_role_level_infeasibilities=(),
    require_non_pair_background_identity=True,
)


def build_broad_colocalization_set(
    *, scaffold: SampleScaffold, anchor_roi_id: str,
    counterfactual_set_id: str,
    feature_a_id: str, feature_b_id: str,
    signal_model: SignalModel,
    background_policy=None,
    nuisance_id: str = "default",
    nuisance_parameters: Optional[dict] = None,
    base_seed: int = 0,
    partition: str = "dev",
    related_sample_group_id: str = "",
) -> ScenarioSet:
    """Emit the two-role broad_colocalization counterfactual set."""
    effect_delta = float(signal_model.effect_delta)
    broad_a_increment = effect_delta
    broad_b_increment = effect_delta

    source_roi = scaffold.source_parent_roi_id.astype(str)
    target_positions = np.where(source_roi == str(anchor_roi_id))[0]
    n_target = int(target_positions.size)

    requested = {
        "anchor_roi_id": str(anchor_roi_id),
        "effect_delta": effect_delta,
    }
    nuisance_parameters = dict(nuisance_parameters or {})
    nuisance_parameters.setdefault("effect_delta", effect_delta)
    infeas_kwargs = dict(
        scaffold=scaffold, counterfactual_set_id=counterfactual_set_id,
        scenario_name=SCENARIO_NAME, roles=ROLES,
        feature_a_id=feature_a_id, feature_b_id=feature_b_id,
        nuisance_id=nuisance_id, nuisance_parameters=nuisance_parameters,
        requested_parameters=requested,
        rng_seed=base_seed, partition=partition,
        related_sample_group_id=related_sample_group_id,
    )

    if n_target == 0:
        return build_all_infeasible_set(reason="target_roi_missing", **infeas_kwargs)
    if n_target < MIN_TARGET_ROI_SPOTS:
        return build_all_infeasible_set(reason="target_roi_too_small", **infeas_kwargs)

    target_positions = np.array(sorted(int(i) for i in target_positions), dtype=int)

    bg_bank = SampleBackgroundBank(
        scaffold=scaffold, participant_roi_ids=(str(anchor_roi_id),),
        policy=background_policy,
    )
    bg_a = bg_bank.field(feature_a_id)
    bg_b = bg_bank.field(feature_b_id)

    a_overlay = constant_overlay(
        n=scaffold.n, positions=target_positions, magnitude=broad_a_increment,
    )
    latent = substream_rng(
        base_seed=base_seed, tag=f"{SCENARIO_NAME}::shared_latent",
    ).standard_normal(target_positions.size)
    latent -= float(np.mean(latent))
    latent_sd = float(np.std(latent))
    if latent_sd > 0.0:
        latent /= latent_sd
    background_scale = float(np.sqrt(
        0.5 * (
            np.var(bg_a[target_positions]) + np.var(bg_b[target_positions])
        )
    ))
    shared_modulation = (
        BROAD_COLOCALIZATION_SHARED_LATENT_RATIO
        * background_scale
        * latent
    )
    a_overlay[target_positions] += shared_modulation
    b_overlay_positive = np.zeros(scaffold.n, dtype=float)
    b_overlay_positive[target_positions] = (
        broad_b_increment + shared_modulation
    )
    b_overlay_null = np.zeros(scaffold.n, dtype=float)

    a_field = compose_role_field(bg=bg_a, overlay=a_overlay)
    b_field_positive = compose_role_field(bg=bg_b, overlay=b_overlay_positive)
    b_field_null = compose_role_field(bg=bg_b, overlay=b_overlay_null)

    assert_composition(field=a_field, bg=bg_a, overlay=a_overlay,
                       scenario_name=SCENARIO_NAME, role=POSITIVE_ROLE, feature="a")
    assert_composition(field=b_field_positive, bg=bg_b, overlay=b_overlay_positive,
                       scenario_name=SCENARIO_NAME, role=POSITIVE_ROLE, feature="b")
    assert_composition(field=b_field_null, bg=bg_b, overlay=b_overlay_null,
                       scenario_name=SCENARIO_NAME,
                       role=INDEPENDENT_BACKGROUND_ROLE, feature="b")

    non_target_mask = source_roi != str(anchor_roi_id)
    target_mean_a_positive = float(np.mean(a_field[target_positions]))
    target_mean_b_positive = float(np.mean(b_field_positive[target_positions]))
    non_target_mean_a = (
        float(np.mean(a_field[non_target_mask])) if non_target_mask.any() else 0.0
    )
    non_target_mean_b_positive = (
        float(np.mean(b_field_positive[non_target_mask])) if non_target_mask.any() else 0.0
    )
    target_mean_b_null = float(np.mean(b_field_null[target_positions]))
    non_target_mean_b_null = (
        float(np.mean(b_field_null[non_target_mask])) if non_target_mask.any() else 0.0
    )

    common_realized = {
        "n_target_roi_spots": n_target,
        "target_roi_mean_a": target_mean_a_positive,
        "target_roi_mean_b": target_mean_b_positive,
        "non_target_mean_a": non_target_mean_a,
        "non_target_mean_b": non_target_mean_b_positive,
        "background_degenerate_rois": list(bg_bank.degenerate_rois),
    }

    builder = ScenarioSetBuilder(
        scaffold=scaffold, counterfactual_set_id=counterfactual_set_id,
        scenario_name=SCENARIO_NAME,
        feature_a_id=feature_a_id, feature_b_id=feature_b_id,
        nuisance_id=nuisance_id, nuisance_parameters=nuisance_parameters,
        requested_parameters=requested, rng_seed=base_seed,
        partition=partition, related_sample_group_id=related_sample_group_id,
        pair_a_roi_id=str(anchor_roi_id), pair_b_roi_id=str(anchor_roi_id),
    )

    broad_structure = [StructureAssignment(
        positions=target_positions,
        structure_kind=STRUCTURE_KIND_BROAD, structure_id="target_roi",
    )]
    empty_positions = np.array([], dtype=int)

    builder.add_instance(
        role=POSITIVE_ROLE,
        target_selected_a=target_positions,
        target_selected_b=target_positions,
        target_a_field=a_field,
        target_b_field=b_field_positive,
        structures_a=broad_structure,
        structures_b=broad_structure,
        realized_parameters=common_realized,
        role_specific_metadata={
            "comparison_kind": COMPARISON_KIND[POSITIVE_ROLE],
        },
    )

    null_realized = dict(common_realized)
    null_realized["target_roi_mean_b"] = target_mean_b_null
    null_realized["non_target_mean_b"] = non_target_mean_b_null
    builder.add_instance(
        role=INDEPENDENT_BACKGROUND_ROLE,
        target_selected_a=target_positions,
        target_selected_b=empty_positions,
        target_a_field=a_field,
        target_b_field=b_field_null,
        structures_a=broad_structure,
        structures_b=None,
        realized_parameters=null_realized,
        role_specific_metadata={
            "comparison_kind": COMPARISON_KIND[
                INDEPENDENT_BACKGROUND_ROLE
            ],
        },
    )

    return builder.finalize(invariant_policy=BROAD_COLOCALIZATION_INVARIANTS)
