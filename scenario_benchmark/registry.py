"""Scenario registry — geometry metadata and positive-role lookup.

Per-scenario declaration of the ontology axes fixed by the spec:

- ``geometry_source``: expression_field | roi_topology | hybrid
- ``a_field_morphology`` / ``b_field_morphology``: per-field Axis A
  vocabulary (list; ``["not_defining"]`` for topology-driven scenarios).
- ``roi_topology``: within_roi | adjacent_rois | embedded_domains |
  dispersed_components.

Contact interface declares mode-specific morphology under ``by_mode``.
Every entry keys off the CURRENT scenario name; the materializer
propagates these fields into each case manifest. Role contracts live
with the scenario modules and are also resolved into output manifests.
The positive-role lookup lets analysis identify cases without depending on
an analysis-specific contrast table. It does not change generated metadata.
"""
from __future__ import annotations

from typing import Any, Dict


SCENARIO_REGISTRY: Dict[str, Dict[str, Any]] = {
    "broad_colocalization": {
        "module_path": "scenario_benchmark.scenarios.broad_colocalization",
        "positive_role": "positive_broad",
        "geometry_source": "expression_field",
        "a_field_morphology": ["broad_territorial"],
        "b_field_morphology": ["broad_territorial"],
        "roi_topology": "within_roi",
    },
    "focal_pattern": {
        "module_path": "scenario_benchmark.scenarios.focal_pattern",
        "positive_role": "positive",
        "geometry_source": "expression_field",
        "a_field_morphology": ["focal"],
        "b_field_morphology": ["focal"],
        "roi_topology": "within_roi",
    },
    "spatial_gradient": {
        "module_path": "scenario_benchmark.scenarios.spatial_gradient",
        "positive_role": "positive_spatial_gradient",
        "geometry_source": "expression_field",
        "a_field_morphology": ["gradient"],
        "b_field_morphology": ["gradient"],
        "roi_topology": "within_roi",
    },
    "nested_foci": {
        "module_path": "scenario_benchmark.scenarios.nested_foci",
        "positive_role": "positive",
        "geometry_source": "expression_field",
        "a_field_morphology": ["broad_territorial"],
        "b_field_morphology": ["focal"],
        "roi_topology": "within_roi",
    },
    "embedded_domains": {
        "module_path": "scenario_benchmark.scenarios.embedded_domains",
        "positive_role": "positive",
        "geometry_source": "roi_topology",
        "a_field_morphology": ["not_defining"],
        "b_field_morphology": ["not_defining"],
        "roi_topology": "embedded_domains",
    },
    "dispersed_infiltration": {
        "module_path": "scenario_benchmark.scenarios.dispersed_infiltration",
        "positive_role": "positive",
        "geometry_source": "roi_topology",
        "a_field_morphology": ["not_defining"],
        "b_field_morphology": ["not_defining"],
        "roi_topology": "dispersed_components",
    },
    "contact_interface": {
        "module_path": "scenario_benchmark.scenarios.contact_interface",
        "positive_role": "positive",
        "geometry_source": "hybrid",
        "roi_topology": "adjacent_rois",
        "by_mode": {
            "responder_activation": {
                "a_field_morphology": ["broad_territorial"],
                "b_field_morphology": [
                    "broad_territorial", "interface_localized",
                ],
            },
            "anchor_fade": {
                "a_field_morphology": [
                    "broad_territorial", "interface_localized",
                ],
                "b_field_morphology": ["broad_territorial"],
            },
        },
    },
}


def positive_role_for_scenario(scenario_name: str) -> str:
    """Return the declared positive role, shared by a scenario's modes."""
    return SCENARIO_REGISTRY[scenario_name]["positive_role"]


def registry_fields_for_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Return the flat registry-owned fields for a case's manifest."""
    scenario = case["scenario_name"]
    entry = SCENARIO_REGISTRY[scenario]
    by_mode = entry.get("by_mode")
    if by_mode is None:
        return {
            "geometry_source": entry["geometry_source"],
            "a_field_morphology": list(entry["a_field_morphology"]),
            "b_field_morphology": list(entry["b_field_morphology"]),
            "roi_topology": entry["roi_topology"],
        }
    mode = case["mode"]
    if mode not in by_mode:
        raise ValueError(
            f"registry_fields_for_case: scenario {scenario!r} case "
            f"{case.get('case_id')!r} has mode {mode!r} not in "
            f"{sorted(by_mode)!r}"
        )
    return {
        "geometry_source": entry["geometry_source"],
        "a_field_morphology": list(by_mode[mode]["a_field_morphology"]),
        "b_field_morphology": list(by_mode[mode]["b_field_morphology"]),
        "roi_topology": entry["roi_topology"],
    }
