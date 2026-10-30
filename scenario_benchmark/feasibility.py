"""Geometry and scenario feasibility audit for scenario_benchmark.

Enumerates candidate anchors / ordered A-B pairs across every
declared substrate on disk (Greenwald + HCC + mouse), invokes
each scenario builder against every candidate, and records
per-role feasibility, typed infeasibility reasons, and simple
substrate-headroom diagnostics to an untracked audit CSV.

The audit never touches TileScope statistics; it only calls
the independent generators and inspects the ``ScenarioSet``
they return (roles, ``generator_infeasible``, ``fallback_reason``,
per-instance spot-row counts).

Candidate parameters are the calibration-locked defaults captured
in ``AUDIT_DEFAULTS`` below, including contact prevalence 0.30 / 0.03.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from scenario_benchmark import (
    CONTACT_INTERFACE_MODE_ANCHOR_FADE,
    CONTACT_INTERFACE_MODE_RESPONDER_ACTIVATION,
    FOCAL_PATTERN_MODE_COLOCATED,
    FOCAL_PATTERN_MODE_SEPARATED,
    GRADIENT_RELATIONSHIP_CONCORDANT, GRADIENT_RELATIONSHIP_OPPOSED,
)
from scenario_benchmark.adapters.greenwald import (
    load_greenwald_sample_scaffold,
)
from scenario_benchmark.datasets import (
    DATASET_REGISTRY,
    enumerate_samples,
    resolve_sample_csv,
)


AUDITED_SCENARIOS: Sequence[str] = (
    "broad_colocalization",
    "focal_pattern_colocated",
    "focal_pattern_separated",
    "spatial_gradient_opposed",
    "spatial_gradient_concordant",
    "dispersed_infiltration",
    "embedded_domains",
    "nested_foci",
    "contact_interface_responder_activation",
    "contact_interface_anchor_fade",
)


# Geometry gates described in scenario_benchmark/CONTRACT.md.
# canonical calibration values: gradient truth_tail_fraction = 0.15;
# AUDIT_DEFAULTS values are propagated verbatim into the frozen
# benchmark_plan.json cases.

_GRADIENT_SIGNAL_MODEL = {"effect_delta": 1.0}

# Single gradient defaults object; opposed + concordant share
# every parameter except gradient_relationship, which is set on
# the invocation. Per-scenario mapping below emits two audit-side
# scenario names that both dispatch to _run_gradient.
_GRADIENT_DEFAULTS: Dict[str, Any] = {
    "truth_tail_fraction": 0.15,
    "gradient_axis": (1.0, 0.0),
    "signal_model": _GRADIENT_SIGNAL_MODEL,
    "primary_positive_role": "positive_spatial_gradient",
    "comparison_role": "paired_randomized",
}

# Both contact_interface modes share every geometric gate and
# scenario-level parameter; they differ only in mode selector.
_CONTACT_INTERFACE_DEFAULTS: Dict[str, Any] = {
    "periphery_hops": 2,
    "signal_model": {"effect_delta": 1.0},
    "primary_positive_role": "positive",
}

# Both focal_pattern modes share every A-side gate and scenario-level
# parameter; they differ only in mode selector.
_FOCAL_PATTERN_DEFAULTS: Dict[str, Any] = {
    "prevalence_fraction": 0.2,
    "n_foci": 5,
    "min_focus_separation_hops": 4,
    "signal_model": {"effect_delta": 1.0},
    "primary_positive_role": "positive",
    "comparison_role": "b_independent_background",
}

AUDIT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "broad_colocalization": {
        "signal_model": {"effect_delta": 1.0},
        "primary_positive_role": "positive_broad",
        "comparison_role": "b_independent_background",
    },
    "focal_pattern_colocated": {
        **_FOCAL_PATTERN_DEFAULTS,
        "mode": FOCAL_PATTERN_MODE_COLOCATED,
    },
    "focal_pattern_separated": {
        **_FOCAL_PATTERN_DEFAULTS,
        "mode": FOCAL_PATTERN_MODE_SEPARATED,
    },
    "spatial_gradient_opposed": {
        **_GRADIENT_DEFAULTS,
        "gradient_relationship": GRADIENT_RELATIONSHIP_OPPOSED,
    },
    "spatial_gradient_concordant": {
        **_GRADIENT_DEFAULTS,
        "gradient_relationship": GRADIENT_RELATIONSHIP_CONCORDANT,
    },
    "dispersed_infiltration": {
        "a_host_effect_scale": 0.2,
        "a_host_baseline": 0.25,
        "b_population_baseline": 0.25,
        "b_local_increment": 0.0,
        "signal_model": {"effect_delta": 1.5},
        "primary_positive_role": "positive",
        "comparison_role": None,
    },
    "embedded_domains": {
        "a_host_effect_scale": 0.2,
        "a_host_baseline": 0.25,
        "b_population_baseline": 0.25,
        "b_local_increment": 0.0,
        "signal_model": {"effect_delta": 1.5},
        "primary_positive_role": "positive",
        "comparison_role": None,
    },
    "nested_foci": {
        "a_host_effect_scale": 0.2,
        "a_host_baseline": 0.25,
        "b_population_baseline": 0.08,
        "b_focus_amplitude": 0.22,
        "focus_sizes": (8, 8, 12, 12),
        "b_expression_profile": "core_high",
        "signal_model": {"effect_delta": 1.5},
        "primary_positive_role": "positive",
        "comparison_role": "b_foci_relocated_outside_a",
    },
    "contact_interface_responder_activation": {
        **_CONTACT_INTERFACE_DEFAULTS,
        "mode": CONTACT_INTERFACE_MODE_RESPONDER_ACTIVATION,
        "comparison_role": "no_interface_increment",
    },
    "contact_interface_anchor_fade": {
        **_CONTACT_INTERFACE_DEFAULTS,
        "mode": CONTACT_INTERFACE_MODE_ANCHOR_FADE,
        "comparison_role": "no_anchor_extension",
    },
}


ANCHOR_SCENARIOS = (
    "broad_colocalization",
    "focal_pattern_colocated",
    "focal_pattern_separated",
    "spatial_gradient_opposed",
    "spatial_gradient_concordant",
)
PAIR_SCENARIOS = (
    "dispersed_infiltration",
    "embedded_domains",
    "nested_foci",
    "contact_interface_responder_activation",
    "contact_interface_anchor_fade",
)


@dataclass
class AuditRow:
    sample_id: str
    dataset: str
    scenario_name: str
    candidate_key: str
    csv_used: str
    dispatched: bool
    dispatch_error: Optional[str]
    emitted_roles: str  # ";"-joined
    positive_role_feasible: Optional[bool]
    comparison_role_feasible: Optional[bool]
    all_roles_feasible: Optional[bool]
    per_role_reasons: str  # JSON dict (infeasibility reasons only)
    # JSON dict of role -> "not_applicable" reason. Kept separate
    # from per_role_reasons so estimand-undefined roles don't
    # conflate with generator failures.
    per_role_not_applicable_reason: str
    # JSON dict of role -> "feasible" | "infeasible" | "not_applicable".
    # Selector reads this as the authoritative feasibility signal;
    # downstream availability counts exclude "not_applicable" from
    # both feasible and infeasible denominators.
    per_role_disposition: str
    # NOTE: the next two are the emitted truth-row counts per role.
    # For a feasible role they equal scaffold.n by design (every
    # sample spot yields one truth row). They are NOT elevated-
    # signal mass or geometric capacity; do not use them for
    # headroom ranking. Scenario-specific realized-signal /
    # geometry fields live in the per-instance manifest's
    # realized_parameters block (surfaced downstream by the
    # selector for Phase D headroom floors).
    positive_role_n_truth_rows: Optional[int]
    comparison_role_n_truth_rows: Optional[int]
    anchor_or_a_n_spots: Optional[int]
    b_n_spots: Optional[int]
    observed_domain_count: Optional[int]
    # Generator-owned realized signal + geometry values for the
    # positive role (JSON-encoded dict). Selector reads scenario-
    # specific headroom floors from this field; does NOT recompute
    # geometry. Empty JSON dict when the positive role is not
    # feasible.
    positive_realized_parameters: str


# --------------------------------------------------------------------------- #
# Substrate enumeration.
# --------------------------------------------------------------------------- #


def _sample_roi_counts(scaffold) -> Dict[str, int]:
    """Return spot counts per roi_id from the loaded scaffold."""
    from collections import Counter
    return dict(Counter(scaffold.source_parent_roi_id))


def _scaffold_from_disk(sample_id: str, dataset: str):
    """Load a sample scaffold from disk.

    All three registered datasets share the same source schema
    (x, y, gene1, gene2, roi_id) and none carry a spot_id. Route
    every dataset through the greenwald adapter, which synthesises
    stable_spot_id and builds the whole-sample SampleScaffold."""
    csv_path, _ = resolve_sample_csv(sample_id, dataset=dataset)
    return load_greenwald_sample_scaffold(csv_path, sample_id=sample_id)


# --------------------------------------------------------------------------- #
# Canonical audit-invocation constructor.
#
# Both the audit driver and the frozen-plan materializer must call the
# same per-scenario builder-argument wiring; the audit constructs a
# plan-shaped invocation dict from AUDIT_DEFAULTS + the candidate_key
# and dispatches via materialize.DISPATCH. This prevents argument
# drift between the two paths.
# --------------------------------------------------------------------------- #


# AUDIT_DEFAULTS keys that never enter the invocation ``args`` block:
# signal_model + audit-only primary_role selectors + mode flags that
# ride on the top-level invocation dict for spatial_gradient.
_NON_ARG_DEFAULT_KEYS = frozenset({
    "signal_model", "primary_positive_role", "comparison_role",
    "gradient_relationship", "mode",
})


def _plan_scenario_name(audit_scenario_name: str) -> str:
    """Collapse audit-side scenario name to the plan/DISPATCH key."""
    if audit_scenario_name.startswith("spatial_gradient"):
        return "spatial_gradient"
    if audit_scenario_name.startswith("contact_interface"):
        return "contact_interface"
    if audit_scenario_name.startswith("focal_pattern"):
        return "focal_pattern"
    return audit_scenario_name


def _extract_args_from_defaults(
    audit_scenario_name: str, candidate_key: str,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    if audit_scenario_name in ANCHOR_SCENARIOS:
        args["anchor_roi_id"] = candidate_key.split("=", 1)[1]
    else:
        parts = dict(kv.split("=") for kv in candidate_key.split(";"))
        args["a_roi_id"] = parts["a"]
        args["b_roi_id"] = parts["b"]
    for k, v in defaults.items():
        if k in _NON_ARG_DEFAULT_KEYS:
            continue
        args[k] = list(v) if isinstance(v, tuple) else v
    return args


def build_audit_invocation(
    audit_scenario_name: str, candidate_key: str,
    counterfactual_set_id: str,
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct a plan-shaped invocation dict from AUDIT_DEFAULTS +
    candidate_key. Used by the audit driver (dispatches via
    materialize.DISPATCH) so audit + materialize share one
    argument-wiring path."""
    d = defaults if defaults is not None else AUDIT_DEFAULTS[audit_scenario_name]
    inv: Dict[str, Any] = {
        "case_id": f"audit__{audit_scenario_name}__{candidate_key}",
        "scenario_name": _plan_scenario_name(audit_scenario_name),
        "source_sample_id": "audit",
        "dataset": "audit",
        "selection_basis": "feasibility_selected",
        "counterfactual_set_id": counterfactual_set_id,
        "feature_a_id": "gene_a",
        "feature_b_id": "gene_b",
        "signal_model": dict(d["signal_model"]),
        "seed": 20260828,
        "nuisance_id": "audit",
        "args": _extract_args_from_defaults(
            audit_scenario_name, candidate_key, d,
        ),
    }
    if audit_scenario_name == "spatial_gradient_opposed":
        inv["gradient_relationship"] = GRADIENT_RELATIONSHIP_OPPOSED
    elif audit_scenario_name == "spatial_gradient_concordant":
        inv["gradient_relationship"] = GRADIENT_RELATIONSHIP_CONCORDANT
    elif audit_scenario_name == "contact_interface_responder_activation":
        inv["mode"] = CONTACT_INTERFACE_MODE_RESPONDER_ACTIVATION
    elif audit_scenario_name == "contact_interface_anchor_fade":
        inv["mode"] = CONTACT_INTERFACE_MODE_ANCHOR_FADE
    elif audit_scenario_name == "focal_pattern_colocated":
        inv["mode"] = FOCAL_PATTERN_MODE_COLOCATED
    elif audit_scenario_name == "focal_pattern_separated":
        inv["mode"] = FOCAL_PATTERN_MODE_SEPARATED
    return inv


# --------------------------------------------------------------------------- #
# Single-audit invocation + row builder.
# --------------------------------------------------------------------------- #


def _audit_one(
    *, sample_id: str, dataset: str,
    scaffold, scenario_name: str, candidate_key: str,
    csv_used: str, roi_counts: Dict[str, int],
    anchor_or_a: str, b: Optional[str],
) -> AuditRow:
    d = AUDIT_DEFAULTS[scenario_name]
    row = AuditRow(
        sample_id=sample_id, dataset=dataset,
        scenario_name=scenario_name, candidate_key=candidate_key,
        csv_used=csv_used, dispatched=False, dispatch_error=None,
        emitted_roles="", positive_role_feasible=None,
        comparison_role_feasible=None, all_roles_feasible=None,
        per_role_reasons="{}",
        per_role_not_applicable_reason="{}",
        per_role_disposition="{}",
        positive_role_n_truth_rows=None, comparison_role_n_truth_rows=None,
        anchor_or_a_n_spots=roi_counts.get(anchor_or_a),
        b_n_spots=(roi_counts.get(b) if b is not None else None),
        observed_domain_count=None,
        positive_realized_parameters="{}",
    )
    cf_id = f"audit__{scenario_name}__{sample_id}__{candidate_key}"
    try:
        inv = build_audit_invocation(scenario_name, candidate_key, cf_id)
        # Late import to avoid cycle: materialize.py imports run_audit
        # + write_availability_summary from feasibility.py.
        from scenario_benchmark.materialize import DISPATCH
        s = DISPATCH[inv["scenario_name"]](scaffold, inv)
    except Exception as e:
        row.dispatch_error = f"{type(e).__name__}: {e}"
        return row

    row.dispatched = True
    row.emitted_roles = ";".join(
        sorted({inst.counterfactual_role for inst in s.instances})
    )
    # Refined: per_role_reasons captures ONLY genuine infeasibilities.
    # not-applicable reasons live in per_role_not_applicable_reason
    # so downstream consumers can distinguish "estimand undefined"
    # from "estimand defined but generator failed" without inspecting
    # generator_infeasible directly.
    reasons: Dict[str, str] = {}
    na_reasons: Dict[str, str] = {}
    dispositions: Dict[str, str] = {}
    for inst in s.instances:
        dispositions[inst.counterfactual_role] = inst.role_disposition
        if inst.role_disposition == "infeasible":
            reasons[inst.counterfactual_role] = (
                inst.fallback_reason or "unknown"
            )
        elif inst.role_disposition == "not_applicable":
            na_reasons[inst.counterfactual_role] = (
                inst.fallback_reason or "unknown"
            )
    row.per_role_reasons = json.dumps(reasons, sort_keys=True)
    row.per_role_not_applicable_reason = json.dumps(
        na_reasons, sort_keys=True,
    )
    row.per_role_disposition = json.dumps(dispositions, sort_keys=True)

    # Feasibility flags (based on three-state disposition).
    positive_role = d["primary_positive_role"]
    comparison_role = d["comparison_role"]
    row.positive_role_feasible = (
        dispositions.get(positive_role) == "feasible"
    )
    row.comparison_role_feasible = (
        None if comparison_role is None
        else dispositions.get(comparison_role) == "feasible"
    )
    # all_roles_feasible excludes not_applicable from denominators:
    # every role must be either feasible or not_applicable, and at
    # least one must be feasible.
    non_na_roles = {
        r: d_ for r, d_ in dispositions.items() if d_ != "not_applicable"
    }
    row.all_roles_feasible = (
        len(non_na_roles) > 0
        and all(v == "feasible" for v in non_na_roles.values())
    )

    # Realized spot counts by instance.
    truth_counts: Dict[str, int] = {}
    for r in s.spot_truth_rows:
        truth_counts[r.counterfactual_role] = (
            truth_counts.get(r.counterfactual_role, 0) + 1
        )
    row.positive_role_n_truth_rows = truth_counts.get(positive_role)
    row.comparison_role_n_truth_rows = (
        None if comparison_role is None
        else truth_counts.get(comparison_role)
    )

    # Surface the observed embedded-domain count.
    # from the positive instance's realized_parameters.
    if scenario_name == "embedded_domains":
        for inst in s.instances:
            if (
                inst.counterfactual_role == "positive"
                and not inst.generator_infeasible
            ):
                oic = inst.realized_parameters.get("observed_component_count")
                if oic is not None:
                    row.observed_domain_count = int(oic)
                break

    # Positive-role realized_parameters snapshot for Phase-D headroom
    # floors. Filtered to JSON-serializable scalars + lists so the
    # audit CSV stays readable; nested dicts / numpy arrays are
    # dropped (rare on positive instances but defensive).
    for inst in s.instances:
        if (
            inst.counterfactual_role == positive_role
            and inst.role_disposition == "feasible"
        ):
            snap: Dict[str, Any] = {}
            for k, v in inst.realized_parameters.items():
                if isinstance(v, (int, float, bool, str)):
                    snap[k] = v
                elif isinstance(v, (list, tuple)):
                    if all(isinstance(x, (int, float, bool, str)) for x in v):
                        snap[k] = list(v)
            row.positive_realized_parameters = json.dumps(
                snap, sort_keys=True,
            )
            break
    return row


# --------------------------------------------------------------------------- #
# Top-level audit runner.
# --------------------------------------------------------------------------- #


def run_audit(
    *, output_path: Path,
    include_datasets: Sequence[str] = tuple(DATASET_REGISTRY),
    log: Callable[[str], None] = print,
) -> List[AuditRow]:
    """Enumerate every candidate across every registered dataset,
    invoke each scenario builder once, write ``output_path`` as CSV,
    return the raw AuditRow list."""
    rows: List[AuditRow] = []
    for dataset in include_datasets:
        samples = enumerate_samples(dataset)
        for sample_id in samples:
            try:
                scaffold = _scaffold_from_disk(sample_id, dataset)
            except Exception as e:
                log(f"  SKIP {dataset}/{sample_id}: scaffold-load failed: {e}")
                continue
            csv_used, _ = resolve_sample_csv(sample_id, dataset=dataset)
            roi_counts = _sample_roi_counts(scaffold)
            rois = sorted(roi_counts)
            log(f"  AUDIT {dataset}/{sample_id} roi_count={len(rois)}")
            for scenario_name in ANCHOR_SCENARIOS:
                for anchor in rois:
                    rows.append(_audit_one(
                        sample_id=sample_id, dataset=dataset,
                        scaffold=scaffold, scenario_name=scenario_name,
                        candidate_key=f"anchor={anchor}",
                        csv_used=str(csv_used), roi_counts=roi_counts,
                        anchor_or_a=anchor, b=None,
                    ))
            for scenario_name in PAIR_SCENARIOS:
                for a in rois:
                    for b in rois:
                        if a == b:
                            continue
                        rows.append(_audit_one(
                            sample_id=sample_id, dataset=dataset,
                            scaffold=scaffold, scenario_name=scenario_name,
                            candidate_key=f"a={a};b={b}",
                            csv_used=str(csv_used), roi_counts=roi_counts,
                            anchor_or_a=a, b=b,
                        ))
    _write_audit_csv(rows, output_path)
    return rows


def _write_audit_csv(rows: Iterable[AuditRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "dataset", "scenario_name",
        "candidate_key", "csv_used", "dispatched", "dispatch_error",
        "emitted_roles", "positive_role_feasible",
        "comparison_role_feasible", "all_roles_feasible",
        "per_role_reasons", "per_role_not_applicable_reason",
        "per_role_disposition",
        "positive_role_n_truth_rows", "comparison_role_n_truth_rows",
        "anchor_or_a_n_spots", "b_n_spots", "observed_domain_count",
        "positive_realized_parameters",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({f: getattr(r, f) for f in fields})


def write_availability_summary(
    rows: Iterable[AuditRow], path: Path,
) -> None:
    """Aggregate per (dataset, scenario_name) totals-vs-feasible.

    Reports scenario availability across the audited denominator
    (dataset × scenario), independent of any plan selection.
    """
    from collections import defaultdict
    agg: Dict[Tuple[str, str], List[int]] = defaultdict(
        lambda: [0, 0, 0]  # total, positive_feasible, all_feasible
    )
    for r in rows:
        key = (r.dataset, r.scenario_name)
        agg[key][0] += 1
        if r.positive_role_feasible:
            agg[key][1] += 1
        if r.all_roles_feasible:
            agg[key][2] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow([
            "dataset", "scenario_name",
            "n_candidates", "n_positive_feasible", "n_all_roles_feasible",
        ])
        for (ds, sc), (n, pf, af) in sorted(agg.items()):
            w.writerow([ds, sc, n, pf, af])
