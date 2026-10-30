"""Topology-defined dispersed B infiltration through an A territory.

The biological truth is independent of TileScope construction support.
Eligible cases contain many enclosed B-labelled components spread through a
coherent A-labelled host, with sufficient total B mass and no dominant
component. TileScope support class is recorded only as an evaluation outcome.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scenario_benchmark.background_bank import SampleBackgroundBank
from scenario_benchmark.pair_interface import (
    build_interface_truth_bundle, build_pair_interface,
)
from scenario_benchmark.sample_scaffold import SampleScaffold
from scenario_benchmark.scenario_assembly import (
    InvariantPolicy, ScenarioSetBuilder, StructureAssignment,
)
from scenario_benchmark.enclosure import (
    ENCLOSURE_ENCLOSED_BY_A,
    classify_component_enclosure_by_a,
    compute_a_host_support,
    enclosure_ascertainment_counts,
)
from scenario_benchmark.scenarios.result import ScenarioSet
from scenario_benchmark.scenarios.support import build_all_infeasible_set
from scenario_benchmark.signal_model import (
    SignalModel, assert_composition, compose_role_field, constant_overlay,
)
from scenario_benchmark.spatial_primitives import (
    flatten_components, structures_from_components,
)
from scenario_benchmark.support_class import (
    support_class_fields, support_policy_fields,
)


SCENARIO_NAME = "dispersed_infiltration"

# Biological/geometric archetype gates. None is a TileScope construction
# threshold.
MIN_ENCLOSED_COMPONENT_COUNT = 5
MIN_ENCLOSED_B_SPOT_COUNT = 12
MIN_A_HOST_SPOT_COUNT = 20
MAX_LARGEST_COMPONENT_FRACTION = 0.40
MIN_DISPERSION_RATIO = 0.25

POSITIVE_ROLE = "positive"
ROLES: Tuple[str, ...] = (POSITIVE_ROLE,)
REQUIRED_ROLES: Tuple[str, ...] = ROLES
CONDITIONAL_ROLES: Tuple[str, ...] = ()
COMPARISON_KIND: Dict[str, str] = {POSITIVE_ROLE: "primary_positive"}

INVARIANTS = InvariantPolicy(
    expected_roles=ROLES,
    feature_a_byte_identical_roles=ROLES,
    permitted_role_level_infeasibilities=(),
    require_non_pair_background_identity=True,
)

STRUCTURE_KIND_COMPONENT = "dispersed_b_component"
STRUCTURE_KIND_A_HOST = "a_host_support"
DEFAULT_A_HOST_EFFECT_SCALE = 0.3


def build_dispersed_infiltration_set(
    *, scaffold: SampleScaffold,
    a_roi_id: str, b_roi_id: str,
    counterfactual_set_id: str,
    feature_a_id: str, feature_b_id: str,
    signal_model: SignalModel,
    a_host_effect_scale: float = DEFAULT_A_HOST_EFFECT_SCALE,
    a_host_baseline: Optional[float] = None,
    b_population_baseline: float = 0.0,
    b_local_increment: Optional[float] = None,
    background_policy=None,
    nuisance_id: str = "default",
    nuisance_parameters: Optional[dict] = None,
    base_seed: int = 0,
    partition: str = "dev",
    related_sample_group_id: str = "",
) -> ScenarioSet:
    """Emit one positive topology archetype."""
    if not (0.0 < float(a_host_effect_scale) <= 1.0):
        raise ValueError("a_host_effect_scale must lie in (0, 1].")
    pair = build_pair_interface(
        scaffold=scaffold, a_roi_id=str(a_roi_id), b_roi_id=str(b_roi_id),
    )
    resolved_a = (
        float(a_host_baseline) if a_host_baseline is not None
        else float(signal_model.effect_delta) * float(a_host_effect_scale)
    )
    resolved_b_baseline = float(b_population_baseline)
    resolved_b_increment = (
        float(b_local_increment) if b_local_increment is not None
        else float(signal_model.effect_delta)
    )
    requested = {
        "a_roi_id": str(a_roi_id),
        "b_roi_id": str(b_roi_id),
        "min_enclosed_component_count": MIN_ENCLOSED_COMPONENT_COUNT,
        "min_enclosed_b_spot_count": MIN_ENCLOSED_B_SPOT_COUNT,
        "min_a_host_spot_count": MIN_A_HOST_SPOT_COUNT,
        "max_largest_component_fraction": MAX_LARGEST_COMPONENT_FRACTION,
        "min_dispersion_ratio": MIN_DISPERSION_RATIO,
    }
    support_metadata = {
        **support_class_fields(scaffold, str(a_roi_id), prefix="a"),
        **support_class_fields(scaffold, str(b_roi_id), prefix="b"),
        **support_policy_fields(),
    }
    nuisance_parameters = dict(nuisance_parameters or {})
    nuisance_parameters.update({
        "effect_delta": float(signal_model.effect_delta),
        **support_metadata,
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
    if pair.ab_edges.shape[0] == 0:
        return build_all_infeasible_set(reason="no_ab_interface", **infeasible)

    enclosure = classify_component_enclosure_by_a(scaffold, pair)
    candidates = tuple(
        info for info in enclosure if info.state == ENCLOSURE_ENCLOSED_BY_A
    )
    if len(candidates) < MIN_ENCLOSED_COMPONENT_COUNT:
        return build_all_infeasible_set(
            reason="insufficient_enclosed_component_count", **infeasible,
        )
    components = tuple(info.positions for info in candidates)
    sizes = tuple(sorted(info.size for info in candidates))
    total_b = int(sum(sizes))
    if total_b < MIN_ENCLOSED_B_SPOT_COUNT:
        return build_all_infeasible_set(
            reason="insufficient_enclosed_b_spot_count", **infeasible,
        )
    largest_fraction = float(max(sizes)) / float(total_b)
    if largest_fraction > MAX_LARGEST_COMPONENT_FRACTION:
        return build_all_infeasible_set(
            reason="largest_component_fraction_exceeds_dispersion_cap",
            **infeasible,
        )

    infiltrating_positions = flatten_components(components)
    a_host_positions, a_host_component_ids = compute_a_host_support(
        scaffold, pair.a_indices, infiltrating_positions,
    )
    if a_host_positions.size < MIN_A_HOST_SPOT_COUNT:
        return build_all_infeasible_set(
            reason="insufficient_a_host_mass", **infeasible,
        )
    dispersion_ratio = _dispersion_ratio(
        scaffold, components, a_host_positions,
    )
    if dispersion_ratio < MIN_DISPERSION_RATIO:
        return build_all_infeasible_set(
            reason="dispersion_ratio_below_threshold", **infeasible,
        )

    decomposed = (
        a_host_baseline is not None
        or b_population_baseline != 0.0
        or b_local_increment is not None
    )
    if decomposed:
        bg_a = SampleBackgroundBank(
            scaffold=scaffold, participant_roi_ids=(str(a_roi_id),),
            policy=background_policy,
        ).field(feature_a_id)
        bg_b = SampleBackgroundBank(
            scaffold=scaffold, participant_roi_ids=(str(b_roi_id),),
            policy=background_policy,
        ).field(feature_b_id)
    else:
        bank = SampleBackgroundBank(
            scaffold=scaffold,
            participant_roi_ids=(str(a_roi_id), str(b_roi_id)),
            policy=background_policy,
        )
        bg_a, bg_b = bank.field(feature_a_id), bank.field(feature_b_id)

    parents = scaffold.source_parent_roi_id.astype(str)
    b_labelled_positions = np.where(parents == str(b_roi_id))[0]
    a_overlay = constant_overlay(
        n=scaffold.n, positions=a_host_positions, magnitude=resolved_a,
    )
    b_overlay = constant_overlay(
        n=scaffold.n,
        positions=b_labelled_positions,
        magnitude=resolved_b_baseline,
    ) + constant_overlay(
        n=scaffold.n,
        positions=infiltrating_positions,
        magnitude=resolved_b_increment,
    )
    a_field = compose_role_field(bg=bg_a, overlay=a_overlay)
    b_field = compose_role_field(bg=bg_b, overlay=b_overlay)
    assert_composition(
        field=a_field, bg=bg_a, overlay=a_overlay,
        scenario_name=SCENARIO_NAME, role=POSITIVE_ROLE, feature="a",
    )
    assert_composition(
        field=b_field, bg=bg_b, overlay=b_overlay,
        scenario_name=SCENARIO_NAME, role=POSITIVE_ROLE, feature="b",
    )

    realized = {
        "n_A": int(pair.a_indices.size),
        "n_B": int(pair.b_indices.size),
        "observed_component_ids": [info.source_component_id for info in candidates],
        "observed_component_sizes": list(sizes),
        "observed_component_count": len(candidates),
        "largest_component_fraction": largest_fraction,
        "dispersion_ratio": dispersion_ratio,
        "a_host_support_size": int(a_host_positions.size),
        "selected_a_host_component_ids": list(a_host_component_ids),
        "a_host_baseline_resolved": resolved_a,
        "b_population_baseline_resolved": resolved_b_baseline,
        "b_local_increment_resolved": resolved_b_increment,
        **enclosure_ascertainment_counts(enclosure),
        **support_metadata,
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
    iid = f"{counterfactual_set_id}::{POSITIVE_ROLE}"
    builder.add_instance(
        role=POSITIVE_ROLE,
        target_selected_a=np.flatnonzero(a_overlay),
        target_selected_b=np.flatnonzero(b_overlay),
        target_a_field=a_field,
        target_b_field=b_field,
        structures_a=[StructureAssignment(
            positions=a_host_positions,
            structure_kind=STRUCTURE_KIND_A_HOST,
            structure_id="a_host",
        )],
        structures_b=structures_from_components(
            components, kind=STRUCTURE_KIND_COMPONENT, prefix="component",
        ),
        realized_parameters=realized,
        role_specific_metadata={
            "comparison_kind": COMPARISON_KIND[POSITIVE_ROLE],
        },
        interface_truth=build_interface_truth_bundle(pair, iid),
    )
    return builder.finalize(invariant_policy=INVARIANTS)


def _dispersion_ratio(
    scaffold: SampleScaffold,
    components: Sequence[Sequence[int]],
    a_host_positions: np.ndarray,
) -> float:
    if len(components) < 2:
        return 0.0
    centroids = np.array([
        scaffold.coords[np.asarray(component, dtype=int)].mean(axis=0)
        for component in components
    ])
    host_coords = scaffold.coords[np.asarray(a_host_positions, dtype=int)]
    diagonal = float(np.linalg.norm(
        host_coords.max(axis=0) - host_coords.min(axis=0)
    ))
    if diagonal <= 0.0:
        return 0.0
    distances: List[float] = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            distances.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    return float(np.mean(distances)) / diagonal
