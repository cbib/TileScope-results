"""Truth-schema shape + PK + vocabulary contract."""
from __future__ import annotations

import pytest

from scenario_benchmark.truth_schemas import (
    CHALLENGE_SCENARIOS,
    EASY_CALIBRATION_SCENARIOS,
    EXECUTABLE_SCENARIO_NAMES,
    PHASE1_SCENARIO_NAMES,
    ROLE_DISPOSITION_FEASIBLE,
    ROLE_DISPOSITION_INFEASIBLE,
    ROLE_DISPOSITION_NOT_APPLICABLE,
    SPOT_TRUTH_PK,
    InstanceManifestRow,
    SpotTruthRow,
)


def _mk_row(**overrides):
    kwargs = dict(
        scenario_instance_id="cf_0::positive_broad",
        counterfactual_set_id="cf_0",
        counterfactual_role="positive_broad",
        scenario_name="broad_colocalization",
        feature_a_id="A", feature_b_id="B",
        nuisance_id="default",
    )
    kwargs.update(overrides)
    return InstanceManifestRow(**kwargs)


@pytest.mark.parametrize("disposition,generator_infeasible,reason,ok", [
    # feasible: generator_infeasible=False, no reason
    (ROLE_DISPOSITION_FEASIBLE,        False, None,   True),
    (ROLE_DISPOSITION_FEASIBLE,        True,  None,   False),
    (ROLE_DISPOSITION_FEASIBLE,        False, "why",  False),
    # infeasible: generator_infeasible=True, reason present
    (ROLE_DISPOSITION_INFEASIBLE,      True,  "why",  True),
    (ROLE_DISPOSITION_INFEASIBLE,      False, "why",  False),
    (ROLE_DISPOSITION_INFEASIBLE,      True,  None,   False),
    # not_applicable: generator_infeasible=False, reason present
    (ROLE_DISPOSITION_NOT_APPLICABLE,  False, "why",  True),
    (ROLE_DISPOSITION_NOT_APPLICABLE,  True,  "why",  False),
    (ROLE_DISPOSITION_NOT_APPLICABLE,  False, None,   False),
])
def test_manifest_three_state_contract(
    disposition, generator_infeasible, reason, ok,
):
    """InstanceManifestRow enforces the three-state disposition
    contract in __post_init__: (disposition, generator_infeasible,
    reason) triples are consistent by construction."""
    kwargs = dict(
        role_disposition=disposition,
        generator_infeasible=generator_infeasible,
        fallback_reason=reason,
    )
    if ok:
        row = _mk_row(**kwargs)
        assert row.role_disposition == disposition
    else:
        with pytest.raises(ValueError):
            _mk_row(**kwargs)


def test_spot_truth_pk_matches_contract():
    assert SPOT_TRUTH_PK == (
        "scenario_instance_id", "counterfactual_role", "source_spot_id",
    )


def test_phase1_scenario_names_are_the_active_ab_scenarios():
    assert PHASE1_SCENARIO_NAMES == (
        "broad_colocalization",
        "focal_pattern",
        "spatial_gradient",
        "dispersed_infiltration",
        "embedded_domains",
        "nested_foci",
        "contact_interface",
    )
    assert EASY_CALIBRATION_SCENARIOS == PHASE1_SCENARIO_NAMES[:3]
    assert CHALLENGE_SCENARIOS == PHASE1_SCENARIO_NAMES[3:]


def test_executable_registry_covers_all_phase1_scenarios():
    """Every Phase-1 scenario now has an implemented builder.
    Regression: if a new scenario is added spec-locked, it belongs
    in PHASE1_SCENARIO_NAMES but NOT EXECUTABLE_SCENARIO_NAMES."""
    assert EXECUTABLE_SCENARIO_NAMES == PHASE1_SCENARIO_NAMES


def test_spot_truth_row_easy_stratum_shape():
    row = SpotTruthRow(
        scenario_instance_id="i0",
        counterfactual_role="positive_broad",
        source_spot_id="s0",
        x=1.5, y=2.5,
        source_parent_roi_id="A",
        source_component_id="A::c0000",
        generated_roi_id="A",
        generated_component_id="A::c0000",
        feature_a_value=0.75,
        feature_b_value=0.10,
        feature_a_is_elevated=True,
        feature_b_is_elevated=True,
        participates_in_structure=True,
        generated_structure_id="region_00",
        generated_structure_kind="broad_region",
    )
    assert row.feature_a_is_elevated is True
    with pytest.raises((AttributeError, TypeError)):
        row.feature_a_value = 1.0  # type: ignore


def test_instance_manifest_uses_scenario_name():
    row = InstanceManifestRow(
        scenario_instance_id="i0",
        counterfactual_set_id="cf_0",
        counterfactual_role="positive_broad",
        scenario_name="broad_colocalization",
        feature_a_id="A",
        feature_b_id="B",
        nuisance_id="default",
    )
    assert row.scenario_name == "broad_colocalization"
    assert row.adjacency_method == "radius_multiplier"
    assert row.generator_infeasible is False
