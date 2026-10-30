"""Scenario-instance assembly authority.

One :class:`ScenarioSetBuilder` per counterfactual set. Every role
adds one instance through :meth:`add_instance`, which accepts the
already-composed per-role feature-A + feature-B fields (both from
``bg_bank.field(feature) + role_overlay``) and emits enriched
per-spot ledger rows plus the :class:`InstanceManifestRow`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scenario_benchmark.pair_interface import InterfaceTruthBundle
from scenario_benchmark.sample_scaffold import SampleScaffold
from scenario_benchmark.truth_schemas import (
    InstanceManifestRow, InterfaceRow, SpotTruthRow,
)


MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StructureAssignment:
    """One structure-membership record for a set of positions.

    Applied inside a role. Position lists must be sample-wide indices.
    """
    positions: np.ndarray
    structure_kind: str  # see truth_schemas.GENERATED_STRUCTURE_KINDS
    structure_id: str


@dataclass(frozen=True)
class InvariantPolicy:
    """Declarative counterfactual-set invariants checked at finalisation.

    Consumed by :meth:`ScenarioSetBuilder.finalize` in the challenge
    scenarios; not applied to the easy calibration scenarios (which
    keep their own contract).
    """
    expected_roles: Tuple[str, ...]
    feature_a_byte_identical_roles: Tuple[str, ...]
    permitted_role_level_infeasibilities: Tuple[str, ...]
    require_non_pair_background_identity: bool
    # Roles permitted to declare their estimand undefined for a
    # substrate (role_disposition="not_applicable"). Kept separate
    # from permitted_role_level_infeasibilities so a role permitted
    # to fail is NOT implicitly permitted to declare its estimand
    # undefined.
    permitted_not_applicable_roles: Tuple[str, ...] = ()


class SetInvariantViolation(RuntimeError):
    """Raised when :meth:`ScenarioSetBuilder.finalize` detects a
    counterfactual-set invariant break — a scenario-code bug, not a
    role-level infeasibility."""


@dataclass(frozen=True)
class GeneratedLabelAssignment:
    """Per-spot generated-ROI relocation payload for a challenge role.

    ``generated_roi_id`` is the sample-wide ``(n,)`` string array of
    generated ROI labels (post-relocation). ``lineage_source_component_id``
    is an optional ``(n,)`` string array recording the origin
    component of each relocated spot; positions with no lineage
    (unchanged spots) hold ``""`` — the row will then carry
    ``None``.

    :class:`ScenarioSetBuilder` is the sole authority for deriving
    ``generated_component_id`` from ``generated_roi_id``; callers
    never precompute component ids.
    """
    generated_roi_id: np.ndarray
    lineage_source_component_id: Optional[np.ndarray] = None


@dataclass
class ScenarioSetBuilder:
    """Accumulator for one counterfactual set."""
    scaffold: SampleScaffold
    counterfactual_set_id: str
    scenario_name: str
    feature_a_id: str
    feature_b_id: str
    nuisance_id: str
    nuisance_parameters: Dict[str, Any]
    requested_parameters: Dict[str, Any]
    rng_seed: int
    partition: str = "dev"
    related_sample_group_id: str = ""
    pair_a_roi_id: str = ""
    pair_b_roi_id: str = ""
    instances: List[InstanceManifestRow] = field(default_factory=list)
    spot_rows: List[SpotTruthRow] = field(default_factory=list)
    interface_rows: List[InterfaceRow] = field(default_factory=list)
    _feasible_vectors: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
    )

    def add_instance(
        self, *,
        role: str,
        target_selected_a: np.ndarray,
        target_selected_b: np.ndarray,
        target_a_field: np.ndarray,
        target_b_field: np.ndarray,
        structures_a: Optional[List[StructureAssignment]] = None,
        structures_b: Optional[List[StructureAssignment]] = None,
        realized_parameters: Optional[Dict[str, Any]] = None,
        role_specific_metadata: Optional[Dict[str, Any]] = None,
        interface_truth: Optional["InterfaceTruthBundle"] = None,
        generated_labels: Optional[GeneratedLabelAssignment] = None,
    ) -> None:
        """Assemble one instance from the caller-composed role fields.

        ``target_a_field`` / ``target_b_field`` are the sample-wide
        ``bg + role_overlay`` arrays for this role. ``interface_truth``
        is the single authority for per-spot interface identity + hops
        AND for the emitted :class:`InterfaceRow` records.
        ``generated_labels`` supplies the sample-wide
        ``generated_roi_id`` array; the builder derives
        ``generated_component_id`` centrally from the connected
        components of the generated-ROI subgraph.
        """
        feat_a = np.asarray(target_a_field, dtype=float)
        feat_b = np.asarray(target_b_field, dtype=float)
        if feat_a.shape != (self.scaffold.n,) or feat_b.shape != (self.scaffold.n,):
            raise ValueError(
                f"add_instance: role {role!r} target field shape mismatch "
                f"(a={feat_a.shape}, b={feat_b.shape}, "
                f"scaffold.n={self.scaffold.n})."
            )
        iid = f"{self.counterfactual_set_id}::{role}"
        elev_a = _boolean_mask_from_selected(target_selected_a, self.scaffold.n)
        elev_b = _boolean_mask_from_selected(target_selected_b, self.scaffold.n)
        structure_map = _flatten_structures(
            (structures_a or []) + (structures_b or []),
        )
        gen_roi, gen_comp, gen_lineage = _resolve_generated_labels(
            scaffold=self.scaffold, generated_labels=generated_labels,
        )
        self.spot_rows.extend(
            _assemble_rows(
                scaffold=self.scaffold, scenario_instance_id=iid,
                counterfactual_role=role,
                feature_a=feat_a, feature_b=feat_b,
                feature_a_elev=elev_a, feature_b_elev=elev_b,
                structure_map=structure_map,
                spot_interface_truth=(
                    interface_truth.spot_map if interface_truth else None
                ),
                generated_roi_id=gen_roi,
                generated_component_id=gen_comp,
                lineage_source_component_id=gen_lineage,
            )
        )
        if interface_truth is not None:
            if interface_truth.scenario_instance_id != iid:
                raise SetInvariantViolation(
                    f"interface_truth belongs to "
                    f"{interface_truth.scenario_instance_id!r} but "
                    f"add_instance is emitting {iid!r}."
                )
            self.interface_rows.extend(interface_truth.rows)
        self._feasible_vectors[str(role)] = (feat_a.copy(), feat_b.copy())
        self.instances.append(
            _build_manifest_row(
                scenario_instance_id=iid,
                counterfactual_set_id=self.counterfactual_set_id,
                counterfactual_role=role,
                scenario_name=self.scenario_name,
                feature_a_id=self.feature_a_id, feature_b_id=self.feature_b_id,
                nuisance_id=self.nuisance_id,
                nuisance_parameters=self.nuisance_parameters,
                scaffold=self.scaffold,
                requested_parameters=self.requested_parameters,
                realized_parameters=realized_parameters or {},
                role_specific_metadata=role_specific_metadata or {},
                generator_infeasible=False, fallback_reason=None,
                rng_seed=self.rng_seed,
                partition=self.partition,
                related_sample_group_id=self.related_sample_group_id,
            )
        )

    def add_infeasible_instance(
        self, *, role: str, fallback_reason: str,
        realized_parameters: Optional[Dict[str, Any]] = None,
        role_specific_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit an infeasibility record (no spot rows). Transactional:
        an infeasible role yields no analyzable spot table."""
        iid = f"{self.counterfactual_set_id}::{role}"
        self.instances.append(
            _build_manifest_row(
                scenario_instance_id=iid,
                counterfactual_set_id=self.counterfactual_set_id,
                counterfactual_role=role,
                scenario_name=self.scenario_name,
                feature_a_id=self.feature_a_id, feature_b_id=self.feature_b_id,
                nuisance_id=self.nuisance_id,
                nuisance_parameters=self.nuisance_parameters,
                scaffold=self.scaffold,
                requested_parameters=self.requested_parameters,
                realized_parameters=realized_parameters or {},
                role_specific_metadata=role_specific_metadata or {},
                generator_infeasible=True, fallback_reason=fallback_reason,
                rng_seed=self.rng_seed,
                partition=self.partition,
                related_sample_group_id=self.related_sample_group_id,
            )
        )

    def add_not_applicable_instance(
        self, *, role: str, reason: str,
        realized_parameters: Optional[Dict[str, Any]] = None,
        role_specific_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a not-applicable record (no spot rows) for a role
        whose estimand is undefined on this substrate.

        Distinct from ``add_infeasible_instance``: not-applicable is
        not a generator failure. Sets ``generator_infeasible=False``
        and ``role_disposition="not_applicable"``. Downstream
        selection + availability counts must exclude
        ``not_applicable`` from both feasible and infeasible
        denominators."""
        from scenario_benchmark.truth_schemas import (
            ROLE_DISPOSITION_NOT_APPLICABLE,
        )
        iid = f"{self.counterfactual_set_id}::{role}"
        self.instances.append(
            _build_manifest_row(
                scenario_instance_id=iid,
                counterfactual_set_id=self.counterfactual_set_id,
                counterfactual_role=role,
                scenario_name=self.scenario_name,
                feature_a_id=self.feature_a_id, feature_b_id=self.feature_b_id,
                nuisance_id=self.nuisance_id,
                nuisance_parameters=self.nuisance_parameters,
                scaffold=self.scaffold,
                requested_parameters=self.requested_parameters,
                realized_parameters=realized_parameters or {},
                role_specific_metadata=role_specific_metadata or {},
                generator_infeasible=False, fallback_reason=reason,
                rng_seed=self.rng_seed,
                partition=self.partition,
                related_sample_group_id=self.related_sample_group_id,
                role_disposition=ROLE_DISPOSITION_NOT_APPLICABLE,
            )
        )


    def finalize(
        self, *, invariant_policy: InvariantPolicy,
    ) -> "ScenarioSet":
        """Assert declarative invariants + return the assembled
        :class:`ScenarioSet`.

        Raises :class:`SetInvariantViolation` on any breach. Only
        feasible roles participate in vector comparisons; infeasible
        roles must have exactly one manifest row and zero truth rows.
        """
        from scenario_benchmark.scenarios.result import ScenarioSet
        _validate_role_emission(self, invariant_policy)
        _validate_infeasibility_structure(self, invariant_policy)
        _validate_vector_invariants(self, invariant_policy)
        return ScenarioSet(
            counterfactual_set_id=self.counterfactual_set_id,
            instances=self.instances,
            spot_truth_rows=self.spot_rows,
            interface_rows=self.interface_rows,
        )


# ---------------------------------------------------------------------------
# Finalisation-time validators
# ---------------------------------------------------------------------------

def _validate_role_emission(
    builder: "ScenarioSetBuilder", policy: InvariantPolicy,
) -> None:
    emitted = tuple(inst.counterfactual_role for inst in builder.instances)
    if len(emitted) != len(set(emitted)):
        raise SetInvariantViolation(
            f"duplicate role emission: {emitted!r}"
        )
    expected = set(policy.expected_roles)
    if set(emitted) != expected:
        missing = expected - set(emitted)
        extra = set(emitted) - expected
        raise SetInvariantViolation(
            f"role emission mismatch: missing={sorted(missing)!r}, "
            f"extra={sorted(extra)!r}, expected={policy.expected_roles!r}."
        )


def _validate_infeasibility_structure(
    builder: "ScenarioSetBuilder", policy: InvariantPolicy,
) -> None:
    from scenario_benchmark.truth_schemas import (
        ROLE_DISPOSITION_FEASIBLE,
        ROLE_DISPOSITION_INFEASIBLE,
        ROLE_DISPOSITION_NOT_APPLICABLE,
    )
    rows_by_role: Dict[str, int] = {}
    for row in builder.spot_rows:
        role = str(row.counterfactual_role)
        rows_by_role[role] = rows_by_role.get(role, 0) + 1
    permitted = set(policy.permitted_role_level_infeasibilities)
    for inst in builder.instances:
        role = str(inst.counterfactual_role)
        n_rows = rows_by_role.get(role, 0)
        disp = inst.role_disposition
        if disp == ROLE_DISPOSITION_INFEASIBLE:
            if role not in permitted:
                raise SetInvariantViolation(
                    f"role {role!r} emitted infeasible but not listed "
                    f"in permitted_role_level_infeasibilities="
                    f"{policy.permitted_role_level_infeasibilities!r}."
                )
            if n_rows != 0:
                raise SetInvariantViolation(
                    f"infeasible role {role!r} has {n_rows} truth rows "
                    f"(expected 0)."
                )
        elif disp == ROLE_DISPOSITION_NOT_APPLICABLE:
            na_permitted = set(policy.permitted_not_applicable_roles)
            if role not in na_permitted:
                raise SetInvariantViolation(
                    f"role {role!r} emitted not_applicable but not "
                    f"listed in permitted_not_applicable_roles="
                    f"{policy.permitted_not_applicable_roles!r}."
                )
            if n_rows != 0:
                raise SetInvariantViolation(
                    f"not_applicable role {role!r} has {n_rows} truth "
                    f"rows (expected 0)."
                )
        else:  # feasible
            if n_rows != builder.scaffold.n:
                raise SetInvariantViolation(
                    f"feasible role {role!r} has {n_rows} truth rows "
                    f"(expected scaffold.n={builder.scaffold.n})."
                )


def _validate_vector_invariants(
    builder: "ScenarioSetBuilder", policy: InvariantPolicy,
) -> None:
    """Byte-identity checks under the bg + overlay contract.

    Two checks:

    - Feature-A byte identity across the roles listed in
      ``feature_a_byte_identical_roles`` — non-swap roles share the
      same A overlay so their composed A fields must match exactly.
    - Non-pair background identity — on every spot outside the paired
      ROIs the composed field is ``bg_bank.field(feature)``, so every
      role's value at that spot must match the anchor role's
      byte-for-byte.
    """
    feasible = list(builder._feasible_vectors.keys())
    if not feasible:
        return
    anchor_role = feasible[0]
    anchor_a, anchor_b = builder._feasible_vectors[anchor_role]
    a_identity_roles = [
        r for r in policy.feature_a_byte_identical_roles
        if r in builder._feasible_vectors
    ]
    if len(a_identity_roles) >= 2:
        reference_role = a_identity_roles[0]
        reference_a, _ = builder._feasible_vectors[reference_role]
        for role in a_identity_roles[1:]:
            vec_a, _ = builder._feasible_vectors[role]
            if not np.array_equal(vec_a, reference_a):
                raise SetInvariantViolation(
                    f"feature A drifted between roles {reference_role!r} "
                    f"and {role!r}; every non-swap role in "
                    f"feature_a_byte_identical_roles must share the same "
                    f"A field byte-for-byte."
                )
    if policy.require_non_pair_background_identity:
        if not (builder.pair_a_roi_id and builder.pair_b_roi_id):
            raise SetInvariantViolation(
                "require_non_pair_background_identity=True but "
                "pair_a_roi_id / pair_b_roi_id are unset on builder."
            )
        roi = builder.scaffold.source_parent_roi_id.astype(str)
        non_pair_mask = (
            (roi != builder.pair_a_roi_id) & (roi != builder.pair_b_roi_id)
        )
        if non_pair_mask.any():
            ref_a_bg = anchor_a[non_pair_mask]
            ref_b_bg = anchor_b[non_pair_mask]
            for role in feasible[1:]:
                vec_a, vec_b = builder._feasible_vectors[role]
                if not np.array_equal(vec_a[non_pair_mask], ref_a_bg):
                    raise SetInvariantViolation(
                        f"non-pair background feature A drifted on role "
                        f"{role!r} vs {anchor_role!r}."
                    )
                if not np.array_equal(vec_b[non_pair_mask], ref_b_bg):
                    raise SetInvariantViolation(
                        f"non-pair background feature B drifted on role "
                        f"{role!r} vs {anchor_role!r}."
                    )


# ---------------------------------------------------------------------------
# Enriched-row assembly
# ---------------------------------------------------------------------------

def _assemble_rows(
    *, scaffold: SampleScaffold, scenario_instance_id: str,
    counterfactual_role: str,
    feature_a: np.ndarray, feature_b: np.ndarray,
    feature_a_elev: np.ndarray, feature_b_elev: np.ndarray,
    structure_map,
    spot_interface_truth: Optional[Dict[int, Tuple[str, int]]] = None,
    generated_roi_id: Optional[np.ndarray] = None,
    generated_component_id: Optional[np.ndarray] = None,
    lineage_source_component_id: Optional[np.ndarray] = None,
) -> List[SpotTruthRow]:
    source_roi = scaffold.source_parent_roi_id.astype(str)
    source_comp = scaffold.source_component_id.astype(str)
    truth = spot_interface_truth or {}
    rows: List[SpotTruthRow] = []
    for i in range(scaffold.n):
        struct = structure_map.get(i)
        if struct is not None:
            kind, sid = struct
            participates = True
        else:
            kind, sid = None, None
            participates = False
        iface = truth.get(i)
        if iface is not None:
            iface_id: Optional[str] = str(iface[0])
            iface_hops: Optional[int] = int(iface[1])
        else:
            iface_id = None
            iface_hops = None
        lineage_val = (
            str(lineage_source_component_id[i])
            if lineage_source_component_id is not None
            and str(lineage_source_component_id[i]) != ""
            else None
        )
        rows.append(SpotTruthRow(
            scenario_instance_id=scenario_instance_id,
            counterfactual_role=counterfactual_role,
            source_spot_id=str(scaffold.spot_ids[i]),
            x=float(scaffold.coords[i, 0]),
            y=float(scaffold.coords[i, 1]),
            source_parent_roi_id=str(source_roi[i]),
            source_component_id=str(source_comp[i]),
            generated_roi_id=str(generated_roi_id[i]),
            generated_component_id=str(generated_component_id[i]),
            feature_a_value=float(feature_a[i]),
            feature_b_value=float(feature_b[i]),
            feature_a_is_elevated=bool(feature_a_elev[i]),
            feature_b_is_elevated=bool(feature_b_elev[i]),
            participates_in_structure=participates,
            generated_structure_id=sid,
            generated_structure_kind=kind,
            structure_step_index=None,
            lineage_source_component_id=lineage_val,
            source_interface_id=iface_id,
            source_interface_hops=iface_hops,
        ))
    return rows


def _resolve_generated_labels(
    *, scaffold: SampleScaffold,
    generated_labels: Optional[GeneratedLabelAssignment],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (generated_roi_id, generated_component_id,
    lineage_source_component_id) sample-wide string arrays. When
    ``generated_labels`` is ``None`` the defaults mirror the source
    identity (backwards-compatible for scenarios that don't relocate).
    Otherwise ``generated_component_id`` is derived centrally from
    connected components of the generated-ROI subgraph."""
    n = scaffold.n
    empty_lineage = np.array([""] * n, dtype=object)
    if generated_labels is None:
        return (
            scaffold.source_parent_roi_id.astype(str).copy(),
            scaffold.source_component_id.astype(str).copy(),
            empty_lineage,
        )
    gen_roi = np.asarray(generated_labels.generated_roi_id, dtype=object)
    if gen_roi.shape != (n,):
        raise ValueError(
            f"GeneratedLabelAssignment.generated_roi_id shape "
            f"{gen_roi.shape!r} != (scaffold.n={n},)."
        )
    gen_comp = _derive_generated_component_ids(scaffold, gen_roi)
    lineage = (
        np.asarray(generated_labels.lineage_source_component_id, dtype=object)
        if generated_labels.lineage_source_component_id is not None
        else empty_lineage
    )
    if lineage.shape != (n,):
        raise ValueError(
            f"GeneratedLabelAssignment.lineage_source_component_id shape "
            f"{lineage.shape!r} != (scaffold.n={n},)."
        )
    return gen_roi.astype(str), gen_comp.astype(str), lineage


def _derive_generated_component_ids(
    scaffold: SampleScaffold, generated_roi_id: np.ndarray,
) -> np.ndarray:
    """Delegate to :meth:`SampleScaffold.derive_generated_components`
    — the SoLE authority for CC on generated ROI labels."""
    return scaffold.derive_generated_components(generated_roi_id.astype(str))


def _boolean_mask_from_selected(selected: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    if selected.size:
        m[np.asarray(selected, dtype=int)] = True
    return m


def _flatten_structures(structures: List[StructureAssignment]):
    """Map position → (kind, id). Later entries override earlier ones on
    collision; callers ensure no unintended overlap (e.g., positive_broad
    A and B share the same broad region, both map to identical kind/id)."""
    out = {}
    for s in structures:
        for pos in s.positions:
            out[int(pos)] = (s.structure_kind, s.structure_id)
    return out


# ---------------------------------------------------------------------------
# Manifest row
# ---------------------------------------------------------------------------

def _build_manifest_row(
    *, scenario_instance_id: str, counterfactual_set_id: str,
    counterfactual_role: str, scenario_name: str,
    feature_a_id: str, feature_b_id: str,
    nuisance_id: str, nuisance_parameters: Dict[str, Any],
    scaffold: SampleScaffold,
    requested_parameters: Dict[str, Any],
    realized_parameters: Dict[str, Any],
    role_specific_metadata: Dict[str, Any],
    generator_infeasible: bool, fallback_reason: Optional[str],
    rng_seed: int, partition: str, related_sample_group_id: str,
    role_disposition: Optional[str] = None,
) -> InstanceManifestRow:
    from scenario_benchmark.truth_schemas import (
        ROLE_DISPOSITION_FEASIBLE,
        ROLE_DISPOSITION_INFEASIBLE,
    )
    realized = dict(realized_parameters)
    if role_specific_metadata:
        realized.setdefault("role_specific", role_specific_metadata)
    if role_disposition is None:
        role_disposition = (
            ROLE_DISPOSITION_INFEASIBLE if generator_infeasible
            else ROLE_DISPOSITION_FEASIBLE
        )
    return InstanceManifestRow(
        scenario_instance_id=scenario_instance_id,
        counterfactual_set_id=counterfactual_set_id,
        counterfactual_role=counterfactual_role,
        scenario_name=scenario_name,
        feature_a_id=feature_a_id, feature_b_id=feature_b_id,
        nuisance_id=nuisance_id,
        nuisance_parameters=dict(nuisance_parameters),
        adjacency_method=scaffold.adjacency_method,
        adjacency_parameters=dict(scaffold.adjacency_parameters),
        requested_parameters=dict(requested_parameters),
        realized_parameters=realized,
        generator_infeasible=bool(generator_infeasible),
        fallback_reason=fallback_reason,
        role_disposition=role_disposition,
        rng_seed=int(rng_seed),
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        source_sample_id=scaffold.sample_id,
        source_parent_roi_ids=tuple(
            sorted(set(scaffold.source_parent_roi_id.tolist()))
        ),
        partition=partition,
        related_sample_group_id=related_sample_group_id,
    )
