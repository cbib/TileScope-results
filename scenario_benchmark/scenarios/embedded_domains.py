"""Observed B-labelled domain(s) geometrically embedded in A.

Domain identity comes from the fixed source ROI partition. Expression is
attached for visibility and measurement but does not define eligibility.
TileScope construction eligibility is recorded as diagnostic metadata only.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

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


SCENARIO_NAME = "embedded_domains"

# Biological/geometric archetype gates. The five-spot floor rejects isolated
# labels/noise; it is deliberately unrelated to TileScope's 15-spot floor.
MIN_EMBEDDED_DOMAIN_SPOTS = 5
MAX_EMBEDDED_DOMAIN_COUNT = 3
MIN_A_HOST_SPOT_COUNT = 20
DEFAULT_A_HOST_EFFECT_SCALE = 0.3

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

STRUCTURE_KIND_DOMAIN = "embedded_b_domain"
STRUCTURE_KIND_A_HOST = "a_host_support"


def build_embedded_domains_set(
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
    """Emit the observed embedded-domain positive archetype."""
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
        "min_embedded_domain_spots": MIN_EMBEDDED_DOMAIN_SPOTS,
        "max_embedded_domain_count": MAX_EMBEDDED_DOMAIN_COUNT,
        "min_a_host_spot_count": MIN_A_HOST_SPOT_COUNT,
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
    domains = tuple(
        info for info in enclosure
        if info.state == ENCLOSURE_ENCLOSED_BY_A
        and info.size >= MIN_EMBEDDED_DOMAIN_SPOTS
    )
    if not domains:
        return build_all_infeasible_set(
            reason="no_observed_embedded_domains", **infeasible,
        )
    if len(domains) > MAX_EMBEDDED_DOMAIN_COUNT:
        return build_all_infeasible_set(
            reason="too_many_embedded_domains_use_dispersed_infiltration",
            **infeasible,
        )
    components = tuple(domain.positions for domain in domains)
    domain_positions = flatten_components(components)
    a_host_positions, a_host_component_ids = compute_a_host_support(
        scaffold, pair.a_indices, domain_positions,
    )
    if a_host_positions.size < MIN_A_HOST_SPOT_COUNT:
        return build_all_infeasible_set(
            reason="insufficient_a_host_mass", **infeasible,
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
    b_baseline_overlay = constant_overlay(
        n=scaffold.n,
        positions=b_labelled_positions,
        magnitude=resolved_b_baseline,
    )
    b_increment_overlay = (
        constant_overlay(
            n=scaffold.n,
            positions=domain_positions,
            magnitude=resolved_b_increment,
        )
        if resolved_b_increment > 0.0
        else np.zeros(scaffold.n, dtype=float)
    )
    b_overlay = b_baseline_overlay + b_increment_overlay
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
        "observed_component_ids": [domain.source_component_id for domain in domains],
        "observed_component_sizes": [domain.size for domain in domains],
        "observed_component_count": len(domains),
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
            components, kind=STRUCTURE_KIND_DOMAIN, prefix="domain",
        ),
        realized_parameters=realized,
        role_specific_metadata={
            "comparison_kind": COMPARISON_KIND[POSITIVE_ROLE],
        },
        interface_truth=build_interface_truth_bundle(pair, iid),
    )
    return builder.finalize(invariant_policy=INVARIANTS)
