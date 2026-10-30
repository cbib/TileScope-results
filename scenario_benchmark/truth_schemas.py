"""Truth-table records for scenario_benchmark v2.

Three records:

- ``SpotTruthRow`` — per-spot ledger row, keyed by
  ``(scenario_instance_id, counterfactual_role, source_spot_id)``.
  Carries the canonical TileScope-input projection plus the enriched
  ground-truth surface (elevation flags, structure identity, chain
  step order, lineage, source-interface hops). Chain / lineage /
  source-interface fields are populated by challenge scenarios that
  perform generated-ROI relabelling; the easy calibration scenarios
  leave them ``None``.
- ``InterfaceRow`` — per-generated-interface geometry record. Used
  by challenge scenarios; not populated by the easy calibration set.
- ``InstanceManifestRow`` — one row per ``scenario_instance_id``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Per-spot truth ledger
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpotTruthRow:
    """Per-spot ledger row.

    First 11 fields are the canonical TileScope-input projection (the
    scenario-projection CSV surface). Remaining fields form the
    enriched ground truth (truth-ledger CSV surface).
    """
    scenario_instance_id: str
    counterfactual_role: str
    source_spot_id: str
    x: float
    y: float
    source_parent_roi_id: str
    source_component_id: str
    generated_roi_id: str  # == source_parent_roi_id in easy scenarios
    generated_component_id: str  # == source_component_id in easy scenarios
    feature_a_value: float
    feature_b_value: float
    feature_a_is_elevated: bool = False
    feature_b_is_elevated: bool = False
    participates_in_structure: bool = False
    generated_structure_id: Optional[str] = None
    generated_structure_kind: Optional[str] = None
    structure_step_index: Optional[int] = None
    lineage_source_component_id: Optional[str] = None
    source_interface_id: Optional[str] = None
    source_interface_hops: Optional[int] = None


# ---------------------------------------------------------------------------
# Per-generated-interface geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterfaceRow:
    """One row per generated A-B contact interface fragment. Derived
    from acquisition edges under ``generated_roi_id``. Populated by
    challenge scenarios; empty for easy calibration scenarios."""
    scenario_instance_id: str
    interface_id: str
    source_parent_roi_id_A: str
    source_parent_roi_id_B: str
    edge_count: int
    contact_edge_length_sum: float
    fragmentation_count: int
    edge_source_spot_ids: Tuple[str, ...]
    edge_target_spot_ids: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Instance manifest
# ---------------------------------------------------------------------------

# Three-state per-role disposition. Recorded on every instance
# manifest. Downstream selection + availability counts must exclude
# ``not_applicable`` from both feasible and infeasible denominators.
ROLE_DISPOSITION_FEASIBLE = "feasible"
ROLE_DISPOSITION_INFEASIBLE = "infeasible"
ROLE_DISPOSITION_NOT_APPLICABLE = "not_applicable"
ROLE_DISPOSITION_VALUES = (
    ROLE_DISPOSITION_FEASIBLE,
    ROLE_DISPOSITION_INFEASIBLE,
    ROLE_DISPOSITION_NOT_APPLICABLE,
)


@dataclass(frozen=True)
class InstanceManifestRow:
    """One row per scenario_instance_id."""
    scenario_instance_id: str
    counterfactual_set_id: str
    counterfactual_role: str
    scenario_name: str
    feature_a_id: str
    feature_b_id: str
    nuisance_id: str
    nuisance_parameters: Dict[str, Any] = field(default_factory=dict)
    adjacency_method: str = "radius_multiplier"
    adjacency_parameters: Dict[str, Any] = field(
        default_factory=lambda: {"radius_factor": 1.5},
    )
    requested_parameters: Dict[str, Any] = field(default_factory=dict)
    realized_parameters: Dict[str, Any] = field(default_factory=dict)
    generator_infeasible: bool = False
    fallback_reason: Optional[str] = None
    role_disposition: str = ROLE_DISPOSITION_FEASIBLE
    rng_seed: int = 0
    manifest_schema_version: int = 1
    generator_commit_sha: str = ""
    generator_source_state: str = ""
    source_data_sha256: str = ""
    source_sample_id: str = ""
    source_parent_roi_ids: Tuple[str, ...] = ()
    partition: str = "dev"
    related_sample_group_id: str = ""

    def __post_init__(self) -> None:
        if self.role_disposition not in ROLE_DISPOSITION_VALUES:
            raise ValueError(
                f"role_disposition must be one of "
                f"{ROLE_DISPOSITION_VALUES!r}; got "
                f"{self.role_disposition!r}."
            )
        # Three-state contract:
        # feasible       -> generator_infeasible=False, no reason
        # infeasible     -> generator_infeasible=True,  reason present
        # not_applicable -> generator_infeasible=False, reason present
        if self.role_disposition == ROLE_DISPOSITION_FEASIBLE:
            if self.generator_infeasible:
                raise ValueError(
                    "role_disposition=feasible requires "
                    "generator_infeasible=False."
                )
            if self.fallback_reason is not None:
                raise ValueError(
                    "role_disposition=feasible must not carry a "
                    "fallback_reason."
                )
        elif self.role_disposition == ROLE_DISPOSITION_INFEASIBLE:
            if not self.generator_infeasible:
                raise ValueError(
                    "role_disposition=infeasible requires "
                    "generator_infeasible=True."
                )
            if not self.fallback_reason:
                raise ValueError(
                    "role_disposition=infeasible requires a "
                    "fallback_reason."
                )
        else:  # not_applicable
            if self.generator_infeasible:
                raise ValueError(
                    "role_disposition=not_applicable requires "
                    "generator_infeasible=False (not a generator "
                    "failure)."
                )
            if not self.fallback_reason:
                raise ValueError(
                    "role_disposition=not_applicable requires a "
                    "reason describing why the estimand is undefined."
                )


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

PHASE1_SCENARIO_NAMES: Tuple[str, ...] = (
    # Easy calibration stratum
    "broad_colocalization",
    "focal_pattern",
    "spatial_gradient",
    # Biologically-specific challenge stratum
    "dispersed_infiltration",
    "embedded_domains",
    "nested_foci",
    "contact_interface",
)

# Executable registry: scenarios with a callable public builder.
EXECUTABLE_SCENARIO_NAMES: Tuple[str, ...] = PHASE1_SCENARIO_NAMES

EASY_CALIBRATION_SCENARIOS: Tuple[str, ...] = PHASE1_SCENARIO_NAMES[:3]
CHALLENGE_SCENARIOS: Tuple[str, ...] = PHASE1_SCENARIO_NAMES[3:]

GENERATED_STRUCTURE_KINDS: Tuple[str, ...] = (
    "broad_region", "focus",
    "a_host_support", "embedded_b_domain", "dispersed_b_component",
    "interface_band", "anchor_fade_band",
)

SPOT_TRUTH_PK: Tuple[str, ...] = (
    "scenario_instance_id",
    "counterfactual_role",
    "source_spot_id",
)
