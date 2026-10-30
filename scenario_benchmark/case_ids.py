"""Case-id derivation for the case-oriented Rung-0 corpus."""
from __future__ import annotations

from typing import Any, Dict


def derive_case_id(scenario_name: str, args: Dict[str, Any], *,
                   gradient_relationship: str | None = None,
                   contact_interface_mode: str | None = None,
                   focal_pattern_mode: str | None = None,
                   source_sample_id: str) -> str:
    """Return the case_id for a case on ``source_sample_id``.

    Deterministic, single authority. Raises for unknown scenarios."""
    if scenario_name == "broad_colocalization":
        anchor = args.get("anchor_roi_id")
        return f"{source_sample_id}__roi{anchor}"
    if scenario_name == "focal_pattern":
        if focal_pattern_mode is None:
            raise ValueError(
                "focal_pattern case_id requires focal_pattern_mode"
            )
        return (
            f"{source_sample_id}__roi{args['anchor_roi_id']}"
            f"__{focal_pattern_mode}"
        )
    if scenario_name == "spatial_gradient":
        if gradient_relationship is None:
            raise ValueError(
                "spatial_gradient case_id requires gradient_relationship"
            )
        return (
            f"{source_sample_id}__roi{args['anchor_roi_id']}"
            f"__{gradient_relationship}"
        )
    if scenario_name in {
        "dispersed_infiltration", "embedded_domains", "nested_foci",
    }:
        return f"{source_sample_id}__A{args['a_roi_id']}_B{args['b_roi_id']}"
    if scenario_name == "contact_interface":
        if contact_interface_mode is None:
            raise ValueError(
                "contact_interface case_id requires contact_interface_mode"
            )
        return (
            f"{source_sample_id}__A{args['a_roi_id']}_B{args['b_roi_id']}"
            f"__{contact_interface_mode}"
        )
    raise ValueError(f"derive_case_id: unknown scenario_name {scenario_name!r}")
