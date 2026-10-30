"""Rung-0 benchmark corpus materializer (case-oriented, v7).

The external corpus/benchmark_plan.json declares the cases to materialize.
The adjacent corpus_index.json freezes content hashes +
provenance per materialized case. Generated files live outside the repository,
under the configured data root (see ``tilescope_results.paths``).

Layout::

    TileScope-data/scenario_benchmark/corpus/<scenario_name>/<case_id>/
        manifest.json          # case-level; roles[] array
        truth_ledger.csv       # all feasible roles in one file
        <role>.csv             # scenario dataset per feasible role
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from tilescope_results.paths import (
    corpus_root, feasibility_root, generated_corpus_root, input_data_root,
)

from scenario_benchmark import (
    CONTACT_INTERFACE_MODES,
    FOCAL_PATTERN_MODES,
    SignalModel,
    build_broad_colocalization_set,
    build_nested_foci_set,
    build_contact_interface_set,
    build_embedded_domains_set,
    build_focal_pattern_set,
    build_dispersed_infiltration_set,
    build_spatial_gradient_set,
    write_tilescope_input_csv,
    write_truth_ledger_csv,
)
from scenario_benchmark.adapters.greenwald import load_greenwald_sample_scaffold
from scenario_benchmark.background_bank import BackgroundPolicy
from scenario_benchmark.feasibility import (
    run_audit,
    write_availability_summary,
)
from scenario_benchmark.registry import SCENARIO_REGISTRY, registry_fields_for_case
from scenario_benchmark.scenarios.spatial_gradient import (
    GRADIENT_RELATIONSHIP_VALUES,
)
from scenario_benchmark.signal_model import COMPONENT_EXPRESSION_PROFILES


PLAN_SCHEMA_VERSION = 7
CORPUS_INDEX_SCHEMA_VERSION = 2

SELECTION_BASIS_MANUAL = "manually_reviewed_reference"
SELECTION_BASIS_FEASIBILITY = "feasibility_selected"
SELECTION_BASIS_VALUES = (SELECTION_BASIS_MANUAL, SELECTION_BASIS_FEASIBILITY)

# Fixed feature IDs — feature identity is not a per-case scientific
# parameter (it names the synthetic gene columns emitted by every builder).
FEATURE_A_ID = "gene_a"
FEATURE_B_ID = "gene_b"


# ---------------------------------------------------------------------------
# Adapter registry — one per scenario. Each adapter builds ``kwargs`` and
# calls the real builder. No policy is invented here.
# ---------------------------------------------------------------------------


def _signal_model(case: Dict[str, Any]) -> SignalModel:
    """Extract the scenario-level effect amplitude."""
    sm = case["signal_model"]
    return SignalModel(effect_delta=float(sm["effect_delta"]))


def _common_kwargs(
    case: Dict[str, Any],
    background_policy: Optional[BackgroundPolicy] = None,
) -> Dict[str, Any]:
    return {
        "counterfactual_set_id": f"cs__{case['scenario_name']}__{case['case_id']}",
        "feature_a_id": FEATURE_A_ID,
        "feature_b_id": FEATURE_B_ID,
        "signal_model": _signal_model(case),
        "background_policy": background_policy,
        "base_seed": int(case["seed"]),
        "partition": "benchmark_corpus",
        "nuisance_id": case.get("nuisance_id", "default"),
    }


CANONICAL_BACKGROUND_POLICY_KEYS = frozenset({
    "background_level", "participant_amplitude",
    "non_participant_amplitude", "latent_abs_clip",
})


def _validate_background_policy_dict(policy: Dict[str, Any], *, where: str) -> None:
    missing = CANONICAL_BACKGROUND_POLICY_KEYS - set(policy.keys())
    if missing:
        raise ValueError(
            f"{where}: canonical background_policy missing keys {sorted(missing)}"
        )
    extra = set(policy.keys()) - CANONICAL_BACKGROUND_POLICY_KEYS
    if extra:
        raise ValueError(
            f"{where}: canonical background_policy has unexpected keys {sorted(extra)}"
        )


def resolve_case_background_policy(
    canonical: Dict[str, Any], case: Dict[str, Any],
) -> BackgroundPolicy:
    """Merge a per-case background_policy_override (partial dict) onto
    the plan-level canonical policy and return a BackgroundPolicy."""
    merged = dict(canonical)
    override = case.get("background_policy_override")
    if override is not None:
        unknown = set(override.keys()) - CANONICAL_BACKGROUND_POLICY_KEYS
        if unknown:
            raise ValueError(
                f"case {case.get('case_id')!r}: background_policy_override "
                f"has unknown keys {sorted(unknown)}"
            )
        merged.update(override)
    return BackgroundPolicy(**merged)


def _adapt_broad_colocalization(scaffold, case, background_policy=None):
    args = case["args"]
    return build_broad_colocalization_set(
        scaffold=scaffold,
        anchor_roi_id=str(args["anchor_roi_id"]),
        **_common_kwargs(case, background_policy),
    )


def _adapt_focal_pattern(scaffold, case, background_policy=None):
    args = case["args"]
    mode = case["mode"]
    if mode not in FOCAL_PATTERN_MODES:
        raise ValueError(
            f"focal_pattern: unknown mode {mode!r}; "
            f"expected one of {FOCAL_PATTERN_MODES!r}"
        )
    return build_focal_pattern_set(
        scaffold=scaffold,
        anchor_roi_id=str(args["anchor_roi_id"]),
        mode=mode,
        prevalence_fraction=float(args["prevalence_fraction"]),
        n_foci=int(args["n_foci"]),
        min_focus_separation_hops=int(args["min_focus_separation_hops"]),
        **_common_kwargs(case, background_policy),
    )


def _adapt_spatial_gradient(scaffold, case, background_policy=None):
    args = case["args"]
    rel = case["gradient_relationship"]
    if rel not in GRADIENT_RELATIONSHIP_VALUES:
        raise ValueError(
            f"spatial_gradient: unknown gradient_relationship {rel!r}; "
            f"expected one of {GRADIENT_RELATIONSHIP_VALUES!r}"
        )
    return build_spatial_gradient_set(
        scaffold=scaffold,
        anchor_roi_id=str(args["anchor_roi_id"]),
        gradient_relationship=rel,
        truth_tail_fraction=float(args["truth_tail_fraction"]),
        gradient_axis=tuple(float(x) for x in args["gradient_axis"]),
        **_common_kwargs(case, background_policy),
    )


_DECOMPOSITION_KWARGS_COMMON = (
    "a_host_baseline", "b_population_baseline", "b_local_increment",
)


def _forward_decomposition_kwargs(args, keys):
    return {k: float(args[k]) for k in keys if k in args}


def _adapt_dispersed_infiltration(scaffold, case, background_policy=None):
    args = case["args"]
    common = _common_kwargs(case, background_policy)
    if "a_host_effect_scale" in args:
        common["a_host_effect_scale"] = float(args["a_host_effect_scale"])
    common.update(_forward_decomposition_kwargs(args, _DECOMPOSITION_KWARGS_COMMON))
    return build_dispersed_infiltration_set(
        scaffold=scaffold,
        a_roi_id=str(args["a_roi_id"]),
        b_roi_id=str(args["b_roi_id"]),
        **common,
    )


def _adapt_contact_interface(scaffold, case, background_policy=None):
    args = case["args"]
    mode = case["mode"]
    if mode not in CONTACT_INTERFACE_MODES:
        raise ValueError(
            f"contact_interface: unknown mode {mode!r}; "
            f"expected one of {CONTACT_INTERFACE_MODES!r}"
        )
    return build_contact_interface_set(
        scaffold=scaffold,
        a_roi_id=str(args["a_roi_id"]),
        b_roi_id=str(args["b_roi_id"]),
        periphery_hops=int(args["periphery_hops"]),
        mode=mode,
        **_common_kwargs(case, background_policy),
    )


def _adapt_embedded_domains(scaffold, case, background_policy=None):
    args = case["args"]
    common = _common_kwargs(case, background_policy)
    if "a_host_effect_scale" in args:
        common["a_host_effect_scale"] = float(args["a_host_effect_scale"])
    common.update(_forward_decomposition_kwargs(args, _DECOMPOSITION_KWARGS_COMMON))
    return build_embedded_domains_set(
        scaffold=scaffold,
        a_roi_id=str(args["a_roi_id"]),
        b_roi_id=str(args["b_roi_id"]),
        **common,
    )


_NESTED_DECOMPOSITION_KWARGS = (
    "a_host_baseline", "b_population_baseline", "b_focus_amplitude",
)


def _adapt_nested_foci(scaffold, case, background_policy=None):
    args = case["args"]
    common = _common_kwargs(case, background_policy)
    if "a_host_effect_scale" in args:
        common["a_host_effect_scale"] = float(args["a_host_effect_scale"])
    if "b_expression_profile" in args:
        common["b_expression_profile"] = str(args["b_expression_profile"])
    common.update(_forward_decomposition_kwargs(args, _NESTED_DECOMPOSITION_KWARGS))
    return build_nested_foci_set(
        scaffold=scaffold,
        a_roi_id=str(args["a_roi_id"]),
        b_roi_id=str(args["b_roi_id"]),
        focus_sizes=tuple(int(x) for x in args["focus_sizes"]),
        **common,
    )


DISPATCH: Dict[str, Callable] = {
    "broad_colocalization": _adapt_broad_colocalization,
    "focal_pattern": _adapt_focal_pattern,
    "spatial_gradient": _adapt_spatial_gradient,
    "dispersed_infiltration": _adapt_dispersed_infiltration,
    "embedded_domains": _adapt_embedded_domains,
    "nested_foci": _adapt_nested_foci,
    "contact_interface": _adapt_contact_interface,
}
if set(DISPATCH) != set(SCENARIO_REGISTRY):
    raise RuntimeError("DISPATCH and SCENARIO_REGISTRY must name the same scenarios")


# ---------------------------------------------------------------------------
# Typed plan model.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    plan_schema_version: int
    master_seed: int
    background_policy: Dict[str, Any]
    cases: List[Dict[str, Any]]

    def __post_init__(self) -> None:
        if self.plan_schema_version != PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"plan_schema_version {self.plan_schema_version} unsupported; "
                f"driver expects {PLAN_SCHEMA_VERSION}"
            )
        _validate_background_policy_dict(
            self.background_policy, where="plan.background_policy",
        )
        seen: set = set()
        for case in self.cases:
            _validate_case(case)
            # Fail fast on unknown / invalid override keys; forces every
            # case to produce a resolvable BackgroundPolicy at load time.
            resolve_case_background_policy(self.background_policy, case)
            role_contract_for_case(case)
            key = (case["scenario_name"], case["case_id"])
            if key in seen:
                raise ValueError(
                    f"duplicate (scenario_name, case_id) {key!r}"
                )
            seen.add(key)
        _validate_sample_dataset_uniqueness(self.cases)


def live_scenario_role_contract() -> Dict[str, Dict[str, Any]]:
    """Return role contracts from the live scenario modules."""
    import importlib
    out: Dict[str, Dict[str, Any]] = {}
    for scenario_name, entry in SCENARIO_REGISTRY.items():
        mod = importlib.import_module(entry["module_path"])
        by_mode = getattr(mod, "ROLE_CONTRACT_BY_MODE", None)
        if by_mode is not None:
            out[scenario_name] = {
                "by_mode": {
                    mode: {
                        "required_roles": list(decl["required_roles"]),
                        "conditional_roles": list(decl["conditional_roles"]),
                    }
                    for mode, decl in by_mode.items()
                }
            }
        else:
            out[scenario_name] = {
                "required_roles": list(mod.REQUIRED_ROLES),
                "conditional_roles": list(mod.CONDITIONAL_ROLES),
            }
    return out


def role_contract_for_case(
    case: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Resolve one case's role contract from the live scenario module."""
    scenario_name = str(case["scenario_name"])
    decl = live_scenario_role_contract().get(scenario_name)
    if decl is None:
        raise ValueError(f"missing role contract for {scenario_name!r}")
    by_mode = decl.get("by_mode")
    if by_mode is not None:
        mode = case.get("mode")
        if mode not in by_mode:
            raise ValueError(
                f"case {case.get('case_id')!r}: no role contract for "
                f"{scenario_name} mode {mode!r}; known modes: {sorted(by_mode)}"
            )
        sub_decl = by_mode[mode]
    else:
        return {
            "required_roles": list(decl.get("required_roles", [])),
            "conditional_roles": list(decl.get("conditional_roles", [])),
        }
    return {
        "required_roles": list(sub_decl.get("required_roles", [])),
        "conditional_roles": list(sub_decl.get("conditional_roles", [])),
    }


def _validate_sample_dataset_uniqueness(cases: List[Dict[str, Any]]) -> None:
    """Each source_sample_id must map to at most one dataset across
    the plan. Two cases naming the same sample_id under different
    datasets would silently disagree on which raw CSV to load."""
    seen: Dict[str, str] = {}
    for case in cases:
        sid = str(case["source_sample_id"])
        ds = str(case["dataset"])
        prev = seen.get(sid)
        if prev is not None and prev != ds:
            raise ValueError(
                f"plan validation: source_sample_id {sid!r} maps to "
                f"multiple datasets: {prev!r} and {ds!r}."
            )
        seen[sid] = ds


def _validate_case(case: Dict[str, Any]) -> None:
    required = [
        "case_id", "scenario_name", "source_sample_id", "dataset",
        "selection_basis", "seed", "signal_model", "args",
    ]
    missing = [k for k in required if k not in case]
    if missing:
        raise ValueError(
            f"case {case.get('case_id')!r} missing fields: {missing}"
        )
    if case["selection_basis"] not in SELECTION_BASIS_VALUES:
        raise ValueError(
            f"case {case['case_id']!r}: selection_basis must be one of "
            f"{SELECTION_BASIS_VALUES!r}; got {case['selection_basis']!r}"
        )
    if case["scenario_name"] not in DISPATCH:
        raise ValueError(
            f"case {case['case_id']!r}: unknown scenario_name "
            f"{case['scenario_name']!r}; known: {sorted(DISPATCH)}"
        )
    if case["scenario_name"] == "nested_foci":
        if "focus_sizes" not in case.get("args", {}):
            raise ValueError(
                f"case {case['case_id']!r}: nested_foci "
                "requires args.focus_sizes"
            )
    if case["scenario_name"] == "embedded_domains":
        if "mode" in case:
            raise ValueError(
                f"case {case['case_id']!r}: embedded_domains "
                "is observed topology and takes no mode switch"
            )
    if case["scenario_name"] == "nested_foci":
        profile = case.get("args", {}).get("b_expression_profile")
        if profile not in COMPONENT_EXPRESSION_PROFILES:
            raise ValueError(
                f"case {case['case_id']!r}: b_expression_profile must be one of "
                f"{COMPONENT_EXPRESSION_PROFILES!r}; got {profile!r}"
            )
    if case["scenario_name"] == "spatial_gradient":
        if "gradient_relationship" not in case:
            raise ValueError(
                f"case {case['case_id']!r}: spatial_gradient requires "
                "'gradient_relationship'"
            )
    if case["scenario_name"] == "contact_interface":
        if "mode" not in case:
            raise ValueError(
                f"case {case['case_id']!r}: contact_interface requires "
                "'mode'"
            )
    if case["scenario_name"] == "focal_pattern":
        if "mode" not in case:
            raise ValueError(
                f"case {case['case_id']!r}: focal_pattern requires "
                "'mode'"
            )
    sm = case["signal_model"]
    if "effect_delta" not in sm:
        raise ValueError(
            f"case {case['case_id']!r}: signal_model missing 'effect_delta'"
        )
    # background_mean / noise_sd are ignored: BackgroundPolicy owns
    # those values now. The plan payload may still carry them for
    # historical readability and calibration provenance.


def load_plan(path: Path) -> Plan:
    raw = json.loads(path.read_text())
    for key in ("background_policy", "cases"):
        if key not in raw:
            raise ValueError(
                f"plan {path}: v{PLAN_SCHEMA_VERSION} requires top-level {key!r}"
            )
    return Plan(
        plan_schema_version=int(raw["plan_schema_version"]),
        master_seed=int(raw["master_seed"]),
        background_policy=raw["background_policy"],
        cases=raw["cases"],
    )


# ---------------------------------------------------------------------------
# Provenance.
# ---------------------------------------------------------------------------


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _generator_source_state(repo_root: Path) -> str:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if not status:
            return "clean"
        tree = subprocess.check_output(
            ["git", "write-tree"], cwd=repo_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return f"tree:{tree}"
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# Per-case processor.
# ---------------------------------------------------------------------------


@dataclass
class CaseReport:
    case_id: str
    scenario_name: str
    source_sample_id: str
    dispatched: bool
    reason: Optional[str] = None
    role_dispositions: Dict[str, str] = field(default_factory=dict)
    role_inputs_sha: Dict[str, str] = field(default_factory=dict)
    manifest_sha256: Optional[str] = None
    truth_ledger_sha256: Optional[str] = None
    acceptance_gate_failures: List[str] = field(default_factory=list)

    @property
    def case_path(self) -> str:
        return f"{self.scenario_name}/{self.case_id}"

    @property
    def acceptance_gate_passed(self) -> bool:
        return self.dispatched and not self.acceptance_gate_failures


def build_case_manifest_payload(
    *,
    case: Dict[str, Any],
    source_data_sha256: str,
    generator_commit_sha: str,
    generator_source_state: str,
    background_policy: BackgroundPolicy,
    roles_manifest: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "scenario_name": case["scenario_name"],
        "source_sample_id": case["source_sample_id"],
        "dataset": case["dataset"],
        "selection_basis": case["selection_basis"],
        "generator_commit_sha": generator_commit_sha,
        "generator_source_state": generator_source_state,
        "source_data_sha256": source_data_sha256,
        "seed": int(case["seed"]),
        "signal_model": case["signal_model"],
        "args": case["args"],
        "background_policy_resolved": {
            "background_level": background_policy.background_level,
            "participant_amplitude": background_policy.participant_amplitude,
            "non_participant_amplitude": background_policy.non_participant_amplitude,
            "latent_abs_clip": background_policy.latent_abs_clip,
        },
        "role_contract": role_contract_for_case(case),
        "roles": roles_manifest,
    }


def process(
    case: Dict[str, Any],
    *,
    output_root: Path,
    dry_run: bool,
    source_csv: Path,
    generator_commit_sha: str,
    generator_source_state: str,
    background_policy: BackgroundPolicy,
    verify_determinism: bool = False,
    scaffold_cache: Optional[Dict[tuple, Any]] = None,
    source_sha_cache: Optional[Dict[Path, str]] = None,
) -> CaseReport:
    report = CaseReport(
        case_id=case["case_id"],
        scenario_name=case["scenario_name"],
        source_sample_id=case["source_sample_id"],
        dispatched=False,
    )
    csv_path = source_csv
    scaffold_key = (
        str(case["dataset"]), str(case["source_sample_id"]), str(csv_path),
    )
    scaffold = None if scaffold_cache is None else scaffold_cache.get(scaffold_key)
    if scaffold is None:
        try:
            scaffold = load_greenwald_sample_scaffold(
                csv_path, sample_id=case["source_sample_id"],
            )
        except Exception as e:
            report.reason = f"scaffold_load_failed: {type(e).__name__}: {e}"
            return report
        if scaffold_cache is not None:
            scaffold_cache[scaffold_key] = scaffold
    try:
        scenario_set = DISPATCH[case["scenario_name"]](
            scaffold, case, background_policy,
        )
    except Exception as e:
        report.reason = f"builder_failed: {type(e).__name__}: {e}"
        return report

    report.dispatched = True
    for inst in scenario_set.instances:
        report.role_dispositions[inst.counterfactual_role] = inst.role_disposition

    # Acceptance gate: required roles must be feasible + backed by scaffold.n rows.
    contract = role_contract_for_case(case)
    required = list(contract.get("required_roles", []))
    gate_failures: List[str] = []
    by_role = {inst.counterfactual_role: inst for inst in scenario_set.instances}
    truth_counts: Dict[str, int] = {}
    for r in scenario_set.spot_truth_rows:
        truth_counts[r.counterfactual_role] = (
            truth_counts.get(r.counterfactual_role, 0) + 1
        )
    for role in required:
        inst = by_role.get(role)
        if inst is None:
            gate_failures.append(f"{role}: not_emitted")
            continue
        if inst.role_disposition != "feasible":
            gate_failures.append(f"{role}: disposition={inst.role_disposition}")
            continue
        n_rows = truth_counts.get(role, 0)
        if n_rows == 0:
            gate_failures.append(f"{role}: zero_truth_rows")
            continue
        if n_rows != scaffold.n:
            gate_failures.append(
                f"{role}: truth_row_count={n_rows} != scaffold.n={scaffold.n}"
            )
    report.acceptance_gate_failures = gate_failures

    if dry_run:
        return report

    case_dir = output_root / case["scenario_name"] / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    source_data_sha256 = (
        None if source_sha_cache is None else source_sha_cache.get(csv_path)
    )
    if source_data_sha256 is None:
        source_data_sha256 = _sha256_file(csv_path)
        if source_sha_cache is not None:
            source_sha_cache[csv_path] = source_data_sha256

    roles_manifest = _build_roles_manifest(
        scenario_set=scenario_set,
        contract=contract,
        case_dir=case_dir,
    )
    for role_entry in roles_manifest:
        if role_entry["input_file"] is not None:
            report.role_inputs_sha[role_entry["name"]] = role_entry["input_sha256"]

    truth_path = case_dir / "truth_ledger.csv"
    write_truth_ledger_csv(scenario_set.spot_truth_rows, truth_path)
    report.truth_ledger_sha256 = _sha256_file(truth_path)

    manifest_payload = build_case_manifest_payload(
        case=case,
        source_data_sha256=source_data_sha256,
        generator_commit_sha=generator_commit_sha,
        generator_source_state=generator_source_state,
        background_policy=background_policy,
        roles_manifest=roles_manifest,
    )
    if case["scenario_name"] == "spatial_gradient":
        manifest_payload["gradient_relationship"] = case["gradient_relationship"]
    if case["scenario_name"] == "contact_interface":
        manifest_payload["mode"] = case["mode"]
    if case["scenario_name"] == "focal_pattern":
        manifest_payload["mode"] = case["mode"]
    manifest_payload["geometry"] = registry_fields_for_case(case)
    manifest_bytes = (
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
    ).encode()
    (case_dir / "manifest.json").write_bytes(manifest_bytes)
    report.manifest_sha256 = _sha256_bytes(manifest_bytes)

    if verify_determinism:
        try:
            second = DISPATCH[case["scenario_name"]](
                scaffold, case, background_policy,
            )
        except Exception as e:
            raise RuntimeError(
                f"determinism check re-run failed for {case['case_id']!r}: {e}"
            ) from e
        _assert_scenario_sets_equal(scenario_set, second, case["case_id"])
    return report


def _build_roles_manifest(
    *,
    scenario_set,
    contract: Dict[str, List[str]],
    case_dir: Path,
) -> List[Dict[str, Any]]:
    """Return the roles[] block. One entry per declared role
    (required + conditional). Writes each dataset directly in the case folder."""
    from collections import defaultdict

    truth_by_role = defaultdict(list)
    for row in scenario_set.spot_truth_rows:
        truth_by_role[row.counterfactual_role].append(row)

    inst_by_role = {inst.counterfactual_role: inst for inst in scenario_set.instances}
    declared = list(contract.get("required_roles", [])) + list(
        contract.get("conditional_roles", [])
    )

    entries: List[Dict[str, Any]] = []
    for role in declared:
        inst = inst_by_role.get(role)
        rows = truth_by_role.get(role, [])
        if inst is None:
            entries.append({
                "name": role,
                "disposition": "not_emitted",
                "reason": "role_not_emitted_by_builder",
                "input_file": None,
                "input_sha256": None,
                "n_spots": 0,
                "n_truth_rows": 0,
                "realized_parameters": {},
            })
            continue
        entry: Dict[str, Any] = {
            "name": role,
            "disposition": inst.role_disposition,
            "reason": inst.fallback_reason,
            "n_spots": 0,
            "n_truth_rows": len(rows),
            "realized_parameters": dict(inst.realized_parameters or {}),
            "input_file": None,
            "input_sha256": None,
        }
        if inst.role_disposition == "feasible":
            input_path = case_dir / f"{role}.csv"
            write_tilescope_input_csv(rows, input_path)
            entry["input_file"] = input_path.name
            entry["input_sha256"] = _sha256_file(input_path)
            entry["n_spots"] = len(rows)
        entries.append(entry)
    return entries


def _assert_scenario_sets_equal(a, b, case_id: str) -> None:
    if len(a.instances) != len(b.instances):
        raise RuntimeError(
            f"determinism check: instance count differs for {case_id!r} "
            f"({len(a.instances)} vs {len(b.instances)})"
        )
    a_truth = sorted(
        (r.scenario_instance_id, r.counterfactual_role, r.source_spot_id,
         getattr(r, "feature_a_value", None), getattr(r, "feature_b_value", None))
        for r in a.spot_truth_rows
    )
    b_truth = sorted(
        (r.scenario_instance_id, r.counterfactual_role, r.source_spot_id,
         getattr(r, "feature_a_value", None), getattr(r, "feature_b_value", None))
        for r in b.spot_truth_rows
    )
    if a_truth != b_truth:
        raise RuntimeError(
            f"determinism check: truth rows differ for {case_id!r}"
        )


# ---------------------------------------------------------------------------
# Corpus index writer.
# ---------------------------------------------------------------------------


def write_corpus_index(
    *,
    index_path: Path,
    plan: Plan,
    reports: List[CaseReport],
    generator_commit_sha: str,
    generator_source_state: str,
    materialized: bool,
) -> None:
    payload = {
        "corpus_index_schema_version": CORPUS_INDEX_SCHEMA_VERSION,
        "plan_schema_version": plan.plan_schema_version,
        "generator_commit_sha": generator_commit_sha,
        "generator_source_state": generator_source_state,
        "materialized": materialized,
        "cases": [
            {
                "scenario_name": r.scenario_name,
                "case_id": r.case_id,
                "case_path": r.case_path,
                "manifest_sha256": r.manifest_sha256,
                "truth_ledger_sha256": r.truth_ledger_sha256,
                "role_inputs": dict(r.role_inputs_sha),
                "role_dispositions": dict(r.role_dispositions),
                "acceptance_gate_passed": r.acceptance_gate_passed,
                "acceptance_gate_failures": list(r.acceptance_gate_failures),
            }
            for r in reports
        ],
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


# Read the plan and source inventory from Input Data. The generated index
# belongs beside the new case directories, never in the downloaded bundle.
DEFAULT_PLAN = corpus_root() / "benchmark_plan.json"
DEFAULT_SOURCES = input_data_root() / "benchmark" / "source_samples.csv"
DEFAULT_OUTPUT = generated_corpus_root()
DEFAULT_AUDIT_CSV = feasibility_root() / "feasibility_audit.csv"
DEFAULT_AVAILABILITY_CSV = generated_corpus_root() / "feasibility_availability.csv"


def load_source_inventory(path: Path) -> Dict[tuple[str, str], Path]:
    """Resolve source CSVs relative to the inventory, reusing bundled score banks.

    Only x, y and roi_id are used by the scaffold adapter; measured expression
    is not used to generate the synthetic feature values.
    """
    sources = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"dataset", "sample_id", "input_file"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns are {sorted(required)}")
        for row in reader:
            if any(not (row.get(key) or "").strip() for key in required):
                raise ValueError(f"{path}: empty source identifier or input_file")
            key = (row["dataset"].strip(), row["sample_id"].strip())
            if key in sources:
                raise ValueError(f"{path}: duplicate source {key}")
            relative_path = Path(row["input_file"].strip())
            if relative_path.is_absolute():
                raise ValueError(f"{path}: input_file must be relative: {relative_path}")
            source = (path.parent / relative_path).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Source {key} not found: {source}")
            sources[key] = source
    return sources


def _run_audit_mode(args) -> int:
    """--audit: enumerate every candidate across every registered dataset,
    invoke each scenario builder once, write the untracked detailed
    audit CSV + the external scientific-availability summary. Never
    invokes TileScope."""
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    print(f"# scenario_benchmark.materialize - AUDIT")
    print(
        f"# datasets: greenwald + hcc + mouse; "
        f"scenarios: {len(DISPATCH)}; "
        f"writing to {args.audit_output}"
    )
    rows = run_audit(output_path=args.audit_output)
    write_availability_summary(rows, args.availability_output)
    total = len(rows)
    dispatched = sum(1 for r in rows if r.dispatched)
    positive_ok = sum(1 for r in rows if r.positive_role_feasible)
    all_ok = sum(1 for r in rows if r.all_roles_feasible)
    print(
        f"# {total} candidates audited; {dispatched} dispatched; "
        f"{positive_ok} positive-role feasible; "
        f"{all_ok} all-roles feasible"
    )
    print(f"# wrote {args.audit_output}")
    print(f"# wrote {args.availability_output}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index", type=Path, default=None,
                        help="Output index (default: OUTPUT/corpus_index.json).")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES,
                        help="Source CSV inventory; paths are relative to this file.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-determinism", action="store_true",
                        help="Repeat generation and compare outputs (CI/debug).")
    parser.add_argument("--audit", action="store_true",
                        help="Feasibility-audit mode.")
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--availability-output", type=Path,
                        default=DEFAULT_AVAILABILITY_CSV)
    args = parser.parse_args(argv)

    args.index = args.index or args.output / "corpus_index.json"
    outputs = (
        [args.audit_output, args.availability_output] if args.audit
        else [args.output, args.index]
    )
    for output in outputs:
        if output.resolve().is_relative_to(input_data_root().resolve()):
            parser.error(f"Output must be outside the downloaded Input Data: {output}")

    if args.audit:
        return _run_audit_mode(args)

    plan = load_plan(args.plan)
    try:
        sources = load_source_inventory(args.sources)
        missing = {
            (case["dataset"], case["source_sample_id"]) for case in plan.cases
        } - sources.keys()
        if missing:
            raise ValueError(f"{args.sources}: missing sources {sorted(missing)}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.index.resolve() in {
        args.plan.resolve(), args.sources.resolve(), *sources.values(),
    }:
        parser.error("The output index must not overwrite an input file")
    repo_root = Path(__file__).resolve().parent.parent
    generator_commit_sha = _git_sha(repo_root)
    generator_source_state = _generator_source_state(repo_root)

    reports: List[CaseReport] = []
    scaffold_cache: Dict[tuple, Any] = {}
    source_sha_cache: Dict[Path, str] = {}
    for case in plan.cases:
        policy = resolve_case_background_policy(plan.background_policy, case)
        reports.append(process(
            case,
            output_root=args.output,
            dry_run=args.dry_run,
            source_csv=sources[(case["dataset"], case["source_sample_id"])],
            generator_commit_sha=generator_commit_sha,
            generator_source_state=generator_source_state,
            background_policy=policy,
            verify_determinism=args.verify_determinism,
            scaffold_cache=scaffold_cache,
            source_sha_cache=source_sha_cache,
        ))

    if not args.dry_run:
        write_corpus_index(
            index_path=args.index, plan=plan, reports=reports,
            generator_commit_sha=generator_commit_sha,
            generator_source_state=generator_source_state,
            materialized=True,
        )

    n_total = len(reports)
    n_dispatched = sum(1 for r in reports if r.dispatched)
    n_gate_pass = sum(1 for r in reports if r.acceptance_gate_passed)
    mode = "DRY-RUN" if args.dry_run else "MATERIALIZE"
    print(f"# scenario_benchmark.materialize - {mode}")
    print(f"# {n_total} cases declared")
    print(
        f"# {n_dispatched} dispatched, "
        f"{n_total - n_dispatched} dispatch-failed; "
        f"{n_gate_pass} passed acceptance gate"
    )
    for r in reports:
        if not r.dispatched:
            print(f"  DISPATCH-FAIL  {r.case_path}: {r.reason}")
        elif r.acceptance_gate_failures:
            print(
                f"  GATE-FAIL      {r.case_path}: {r.acceptance_gate_failures}"
            )
        else:
            print(
                f"  OK             {r.case_path}: "
                f"roles={sorted(r.role_dispositions)}"
            )
    if not args.dry_run:
        print(f"# wrote corpus to {args.output}/")
        print(f"# wrote index to {args.index}")

    hard_failures: List[str] = []
    for r in reports:
        if not r.dispatched and r.reason:
            hard_failures.append(f"{r.case_path}: dispatch_failed={r.reason}")
        if r.acceptance_gate_failures:
            hard_failures.append(
                f"{r.case_path}: acceptance_gate_failed={r.acceptance_gate_failures}"
            )
    if hard_failures:
        print(f"# HARD FAILURES ({len(hard_failures)}):")
        for msg in hard_failures:
            print(f"    {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
