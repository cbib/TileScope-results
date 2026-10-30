"""Analysis obtains positive-role identity from live scenario declarations."""
from __future__ import annotations

import pytest

from scenario_benchmark.materialize import role_contract_for_case
from scenario_benchmark.registry import (
    SCENARIO_REGISTRY, positive_role_for_scenario, registry_fields_for_case,
)


POSITIVE_CASES = [
    ("broad_colocalization", None, "positive_broad"),
    ("focal_pattern", "colocated", "positive"),
    ("focal_pattern", "separated", "positive"),
    ("spatial_gradient", "concordant", "positive_spatial_gradient"),
    ("spatial_gradient", "opposed", "positive_spatial_gradient"),
    ("nested_foci", None, "positive"),
    ("embedded_domains", None, "positive"),
    ("dispersed_infiltration", None, "positive"),
    ("contact_interface", "responder_activation", "positive"),
    ("contact_interface", "anchor_fade", "positive"),
]


@pytest.mark.parametrize(
    ("scenario", "mode", "expected_role"), POSITIVE_CASES,
)
def test_analysis_positive_role_is_required_by_generator(scenario, mode, expected_role):
    case = {"scenario_name": scenario}
    if mode is not None:
        case["mode"] = mode
    contract = role_contract_for_case(case)
    assert positive_role_for_scenario(scenario) == expected_role
    assert expected_role in contract["required_roles"]
    # Adding analysis lookup metadata must not change generated manifests.
    assert set(registry_fields_for_case(case)) == {
        "geometry_source", "a_field_morphology", "b_field_morphology", "roi_topology",
    }


def test_positive_role_cases_cover_every_scenario():
    assert {scenario for scenario, _, _ in POSITIVE_CASES} == set(SCENARIO_REGISTRY)


def test_unknown_scenario_has_no_guessed_positive_role():
    with pytest.raises(KeyError):
        positive_role_for_scenario("unknown_scenario")
