"""Build the current-core scenario discriminability atlas.

Iterates the frozen corpus_index.json v2 (case-oriented). For each
feasible (scenario, case_id, role) instance, runs TileScope against
its <role>.csv dataset under the frozen tilescope_run_config.json,
extracts declared estimand rows from the emitted CSVs, and writes below the
configured TileScope-data/scenario_benchmark directory:

  CurrentResults/<scenario_name>/<case_id>/<role>/
    (full TileScope output tree; separate from generated inputs)

  scenario_discriminability/<scenario_name>/<case_id>/<role>/
    atlas_rows.csv         # extracted tidy rows for this role
    run_status.json        # provenance + status

  scenario_discriminability/atlas.csv
  scenario_discriminability/run_accounting.csv
  scenario_discriminability/visual_inspection_index.csv

Restartable: a completed role is skipped unless --force. Cache
validation requires: (a) atlas_rows.csv + run_status.json exist,
(b) recorded input / config / core-sha hashes match current,
(c) all standard publication PDFs (expression_summary,
consensus_tessellation_summary, support_provenance, and
tessellation.n density PDFs) exist and are non-empty in the
canonical CurrentResults/<scenario>/<case>/<role>/ directory. Any drift or
missing visual forces a rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tilescope_results.paths import corpus_root, current_results_root, atlas_root, core_root, reference

from analyses.scenario_discriminability.extractors import ATLAS_COLUMNS, extract_all, to_dataframe

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = corpus_root() / "corpus_index.json"
PLAN_PATH = corpus_root() / "benchmark_plan.json"
CONFIG_PATH = current_results_root() / "tilescope_run_config.json"
CORPUS_ROOT = corpus_root()
OUTPUTS_ROOT = atlas_root()

ACCOUNTING_COLUMNS = [
    "scenario_name", "case_id", "role", "role_disposition",
    "selection_basis", "execution_status", "reason",
    "input_hash", "config_hash", "core_sha", "n_atlas_rows",
]

STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_SKIPPED_INFEASIBLE = "skipped_infeasible"
STATUS_SKIPPED_NOT_APPLICABLE = "skipped_not_applicable"

# Standard TileScope publication PDFs that must exist alongside every
# feasible run. The three summary PDFs are per-role fixed; the
# tessellation density PDFs are one per replicate (n = tessellation.n
# from the frozen config).
REQUIRED_SUMMARY_PDFS = (
    "expression_summary.pdf",
    "consensus_tessellation_summary.pdf",
    "support_provenance.pdf",
)


def _tessellation_n_from_config() -> int:
    cfg = json.loads(CONFIG_PATH.read_text())
    return int(cfg["tessellation"]["n"])


@dataclass
class PlannedRole:
    scenario_name: str
    case_id: str
    case_path: str  # relative to corpus/, e.g. "broad_colocalization/242_T__roi3"
    role: str
    role_disposition: str  # feasible / infeasible / not_applicable
    scenario_mode: Optional[str]
    sample_id: str
    dataset: Optional[str]
    selection_basis: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _tilescope_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=core_root(),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _scenario_mode_from_case(case: Dict) -> Optional[str]:
    """Top-level gradient_relationship / realization_mode / mode on the
    case entry supply the atlas's scenario_mode axis. Absent for
    scenarios without a mode axis."""
    for key in ("gradient_relationship", "realization_mode", "mode"):
        if key in case:
            return str(case[key])
    return None


def enumerate_planned_roles(
    plan: Dict, index: Dict,
) -> List[PlannedRole]:
    """Every planned role across the plan × index, keyed by
    (scenario_name, case_id, role). role_disposition + case_path
    come from the index; scenario_mode + sample_id + dataset +
    selection_basis come from the plan case."""
    plan_case_by_key = {
        (c["scenario_name"], c["case_id"]): c for c in plan["cases"]
    }
    out: List[PlannedRole] = []
    for entry in index["cases"]:
        key = (entry["scenario_name"], entry["case_id"])
        case = plan_case_by_key.get(key)
        if case is None:
            raise ValueError(
                f"corpus_index case {key!r} not found in plan"
            )
        dispositions: Dict[str, str] = entry.get("role_dispositions", {})
        for role, disposition in sorted(dispositions.items()):
            out.append(PlannedRole(
                scenario_name=entry["scenario_name"],
                case_id=entry["case_id"],
                case_path=entry["case_path"],
                role=role,
                role_disposition=disposition,
                scenario_mode=_scenario_mode_from_case(case),
                sample_id=case["source_sample_id"],
                dataset=case.get("dataset"),
                selection_basis=case["selection_basis"],
            ))
    return out


def role_output_dir(scenario_name: str, case_id: str, role: str) -> Path:
    """Derived analysis files (atlas_rows, run_status) per role."""
    return OUTPUTS_ROOT / scenario_name / case_id / role


def tilescope_output_dir(
    scenario_name: str, case_id: str, role: str,
) -> Path:
    """Canonical core outputs, separate from the input corpus and atlas."""
    return current_results_root() / scenario_name / case_id / role


def _required_pdfs(tess_n: int) -> List[str]:
    density_pdfs = [
        f"tessellation_densities_{i}.pdf" for i in range(1, tess_n + 1)
    ]
    return list(REQUIRED_SUMMARY_PDFS) + density_pdfs


def _tilescope_pdfs_valid(
    tilescope_dir: Path, tess_n: int,
) -> Tuple[bool, str]:
    """Every required PDF must exist and be non-empty."""
    for name in _required_pdfs(tess_n):
        p = tilescope_dir / name
        if not p.exists():
            return False, f"missing_pdf:{name}"
        if p.stat().st_size == 0:
            return False, f"empty_pdf:{name}"
    return True, ""


def is_role_complete(
    scenario_name: str, case_id: str, role: str, *,
    input_hash: str, config_hash: str, core_sha: str, tess_n: int,
) -> bool:
    """Complete iff run_status.json declares execution_status ==
    STATUS_COMPLETE, atlas_rows.csv exists, provenance hashes match
    current, AND the canonical TileScope PDFs exist in CurrentResults.
    A stale cache (deleted PDFs, config drift, core upgrade) forces
    a rerun."""
    d = role_output_dir(scenario_name, case_id, role)
    status_path = d / "run_status.json"
    rows_path = d / "atlas_rows.csv"
    if not status_path.exists() or not rows_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text())
    except Exception:
        return False
    if status.get("execution_status") != STATUS_COMPLETE:
        return False
    if not (
        status.get("input_hash") == input_hash
        and status.get("config_hash") == config_hash
        and status.get("core_sha") == core_sha
    ):
        return False
    pdfs_ok, _ = _tilescope_pdfs_valid(
        tilescope_output_dir(scenario_name, case_id, role), tess_n,
    )
    return pdfs_ok


def run_one_role(
    planned: PlannedRole, *, force: bool, config_hash: str, core_sha: str,
    tess_n: int,
) -> Dict[str, object]:
    """Execute TileScope + extract atlas rows for one role. Writes
    atlas_rows.csv + run_status.json alongside the TileScope output
    dir on success. Returns a run_accounting record."""
    scenario, case_id, role = (
        planned.scenario_name, planned.case_id, planned.role,
    )
    record = {
        "scenario_name": scenario, "case_id": case_id, "role": role,
        "role_disposition": planned.role_disposition,
        "selection_basis": planned.selection_basis,
        "execution_status": None, "reason": None,
        "input_hash": None, "config_hash": config_hash, "core_sha": core_sha,
        "n_atlas_rows": 0,
    }
    if planned.role_disposition == "infeasible":
        record["execution_status"] = STATUS_SKIPPED_INFEASIBLE
        record["reason"] = "role_disposition=infeasible"
        return record
    if planned.role_disposition == "not_applicable":
        record["execution_status"] = STATUS_SKIPPED_NOT_APPLICABLE
        record["reason"] = "role_disposition=not_applicable"
        return record

    if planned.role_disposition != "feasible":
        record["execution_status"] = STATUS_FAILED
        record["reason"] = (
            f"unknown role_disposition {planned.role_disposition!r}"
        )
        return record

    input_csv = CORPUS_ROOT / planned.case_path / f"{role}.csv"
    if not input_csv.exists():
        record["execution_status"] = STATUS_FAILED
        record["reason"] = f"input_csv_missing: {input_csv}"
        _write_status(scenario, case_id, role, record)
        return record

    input_hash = _sha256_file(input_csv)
    if not force and is_role_complete(
        scenario, case_id, role,
        input_hash=input_hash, config_hash=config_hash, core_sha=core_sha,
        tess_n=tess_n,
    ):
        # Reuse: cached provenance matches current, so reload the
        # prior accounting record verbatim.
        status_path = (
            role_output_dir(scenario, case_id, role) / "run_status.json"
        )
        prior = json.loads(status_path.read_text())
        record.update({
            "execution_status": STATUS_COMPLETE,
            "reason": "reused_prior_run",
            "input_hash": prior.get("input_hash"),
            "config_hash": prior.get("config_hash"),
            "core_sha": prior.get("core_sha"),
            "n_atlas_rows": prior.get("n_atlas_rows", 0),
        })
        return record

    tilescope_run_dir = tilescope_output_dir(scenario, case_id, role)
    tilescope_run_dir.mkdir(parents=True, exist_ok=True)

    try:
        from tilescope.api import run_tilescope_from_csv
        run_tilescope_from_csv(
            str(input_csv),
            run_config=str(CONFIG_PATH),
            output_dir=str(tilescope_run_dir),
            html_report=False,
            dataset_id=0,
            feature_a_id="A",
            feature_b_id="B",
        )
    except Exception as e:
        record["execution_status"] = STATUS_FAILED
        record["reason"] = (
            f"tilescope_failed: {type(e).__name__}: {e}\n"
            + traceback.format_exc(limit=6)
        )
        _write_status(scenario, case_id, role, record)
        return record

    pdfs_ok, pdf_reason = _tilescope_pdfs_valid(tilescope_run_dir, tess_n)
    if not pdfs_ok:
        record["execution_status"] = STATUS_FAILED
        record["reason"] = f"tilescope_pdfs_invalid: {pdf_reason}"
        _write_status(scenario, case_id, role, record)
        return record

    out_dir = role_output_dir(scenario, case_id, role)
    out_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "case_id": case_id,
        "scenario_name": scenario,
        "scenario_mode": planned.scenario_mode,
        "sample_id": planned.sample_id,
        "dataset": planned.dataset,
        "selection_basis": planned.selection_basis,
        "role": role,
        "role_disposition": planned.role_disposition,
    }
    try:
        rows = extract_all(tilescope_run_dir, context)
        atlas_df = to_dataframe(rows)
        atlas_df.to_csv(out_dir / "atlas_rows.csv", index=False)
    except Exception as e:
        record["execution_status"] = STATUS_FAILED
        record["reason"] = (
            f"extraction_failed: {type(e).__name__}: {e}\n"
            + traceback.format_exc(limit=6)
        )
        _write_status(scenario, case_id, role, record)
        return record

    record["execution_status"] = STATUS_COMPLETE
    record["input_hash"] = input_hash
    record["n_atlas_rows"] = len(atlas_df)
    _write_status(scenario, case_id, role, record)
    return record


def _write_status(
    scenario: str, case_id: str, role: str, record: Dict[str, object],
) -> None:
    out_dir = role_output_dir(scenario, case_id, role)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_status.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str)
    )


def _write_visual_inspection_index(
    planned: List[PlannedRole], tess_n: int,
) -> None:
    """Untracked CSV enumerating each feasible role's PDF paths so a
    reader can inspect the standard TileScope publication outputs
    without deriving paths by hand."""
    rows: List[Dict[str, object]] = []
    required_pdfs = _required_pdfs(tess_n)
    for pr in planned:
        if pr.role_disposition != "feasible":
            continue
        tdir = tilescope_output_dir(
            pr.scenario_name, pr.case_id, pr.role,
        )
        row = {
            "scenario_name": pr.scenario_name,
            "scenario_mode": pr.scenario_mode or "",
            "case_id": pr.case_id, "role": pr.role,
            "selection_basis": pr.selection_basis,
            "sample_id": pr.sample_id, "dataset": pr.dataset or "",
        }
        for pdf in required_pdfs:
            p = tdir / pdf
            column_key = pdf.replace(".pdf", "").replace(".", "_")
            row[column_key] = (
                reference(p) if p.exists() else ""
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(
        OUTPUTS_ROOT / "visual_inspection_index.csv",
        index=False, lineterminator="\n",
    )


def aggregate(planned: List[PlannedRole]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate every per-role atlas_rows.csv into atlas.csv +
    build the run_accounting.csv snapshot. Reports missing feasible
    rows so nothing disappears silently."""
    accounting: List[Dict[str, object]] = []
    atlas_frames: List[pd.DataFrame] = []
    missing_feasible: List[str] = []
    for pr in planned:
        status_path = (
            role_output_dir(pr.scenario_name, pr.case_id, pr.role)
            / "run_status.json"
        )
        if not status_path.exists():
            if pr.role_disposition == "feasible":
                missing_feasible.append(
                    f"{pr.scenario_name}/{pr.case_id}/{pr.role}"
                )
                accounting.append({
                    "scenario_name": pr.scenario_name,
                    "case_id": pr.case_id, "role": pr.role,
                    "role_disposition": pr.role_disposition,
                    "selection_basis": pr.selection_basis,
                    "execution_status": "missing",
                    "reason": "no_run_status_on_disk",
                    "input_hash": None, "config_hash": None,
                    "core_sha": None, "n_atlas_rows": 0,
                })
                continue
            accounting.append({
                "scenario_name": pr.scenario_name,
                "case_id": pr.case_id, "role": pr.role,
                "role_disposition": pr.role_disposition,
                "selection_basis": pr.selection_basis,
                "execution_status": (
                    STATUS_SKIPPED_INFEASIBLE
                    if pr.role_disposition == "infeasible"
                    else STATUS_SKIPPED_NOT_APPLICABLE
                ),
                "reason": f"role_disposition={pr.role_disposition}",
                "input_hash": None, "config_hash": None,
                "core_sha": None, "n_atlas_rows": 0,
            })
            continue
        rec = json.loads(status_path.read_text())
        accounting.append({k: rec.get(k) for k in ACCOUNTING_COLUMNS})
        if rec.get("execution_status") == STATUS_COMPLETE:
            rows_path = (
                role_output_dir(pr.scenario_name, pr.case_id, pr.role)
                / "atlas_rows.csv"
            )
            if rows_path.exists():
                # roi_id + responder_roi_id are per-atlas-row identifiers
                # that must round-trip as string literals; without an
                # explicit dtype hint pandas coerces numeric-looking ids
                # to float (e.g. "7" -> 7.0) whenever the column has any
                # NaN, which breaks the downstream slot-match filter.
                atlas_frames.append(pd.read_csv(
                    rows_path,
                    dtype={"roi_id": "string", "responder_roi_id": "string"},
                ))
    accounting_df = pd.DataFrame(accounting, columns=ACCOUNTING_COLUMNS)
    if atlas_frames:
        atlas_df = pd.concat(atlas_frames, ignore_index=True)
        atlas_df = atlas_df[ATLAS_COLUMNS]
    else:
        atlas_df = pd.DataFrame(columns=ATLAS_COLUMNS)
    if missing_feasible:
        print(
            f"ERROR: {len(missing_feasible)} feasible role(s) missing run_status.json:",
            file=sys.stderr,
        )
        for m in missing_feasible:
            print(f"  {m}", file=sys.stderr)
    return atlas_df, accounting_df


def parse_only(value: Optional[str]) -> Optional[Tuple[str, Optional[str]]]:
    """--only CASE or --only CASE:ROLE."""
    if not value:
        return None
    if ":" in value:
        case, role = value.split(":", 1)
        return case, role
    return value, None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", type=str, default=None,
        help="Run only CASE or CASE:ROLE (matched on case_id)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rerun completed roles instead of skipping",
    )
    parser.add_argument(
        "--skip-execution", action="store_true",
        help="Skip per-role execution; only aggregate whatever is on disk",
    )
    args = parser.parse_args(argv)

    plan = json.loads(PLAN_PATH.read_text())
    index = json.loads(INDEX_PATH.read_text())
    planned = enumerate_planned_roles(plan, index)

    only_filter = parse_only(args.only)
    if only_filter:
        case_key, role_key = only_filter
        planned = [
            p for p in planned
            if p.case_id == case_key
            and (role_key is None or p.role == role_key)
        ]
    if not planned:
        print("no planned roles matched filters", file=sys.stderr)
        return 1

    config_hash = _sha256_file(CONFIG_PATH)
    core_sha = _tilescope_sha()
    tess_n = _tessellation_n_from_config()

    print(
        f"# scenario_discriminability: {len(planned)} planned roles; "
        f"config_hash={config_hash[:12]} core_sha={core_sha[:12]} "
        f"tessellation.n={tess_n}"
    )

    if not args.skip_execution:
        for pr in planned:
            rec = run_one_role(
                pr, force=args.force,
                config_hash=config_hash, core_sha=core_sha,
                tess_n=tess_n,
            )
            marker = {
                STATUS_COMPLETE: "OK",
                STATUS_FAILED: "FAIL",
                STATUS_SKIPPED_INFEASIBLE: "SKIP-INF",
                STATUS_SKIPPED_NOT_APPLICABLE: "SKIP-NA",
            }.get(rec["execution_status"], "??")
            print(
                f"  {marker:9s} {pr.scenario_name}/{pr.case_id}/{pr.role} "
                f"n_rows={rec.get('n_atlas_rows', 0)} "
                f"reason={rec.get('reason') or ''}"
            )

    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    atlas_df, accounting_df = aggregate(planned)
    atlas_df.to_csv(OUTPUTS_ROOT / "atlas.csv", index=False, lineterminator="\n")
    accounting_df.to_csv(
        OUTPUTS_ROOT / "run_accounting.csv", index=False, lineterminator="\n",
    )
    _write_visual_inspection_index(planned, tess_n)

    n_complete = int((accounting_df["execution_status"] == STATUS_COMPLETE).sum())
    n_failed = int((accounting_df["execution_status"] == STATUS_FAILED).sum())
    n_missing = int((accounting_df["execution_status"] == "missing").sum())
    print(
        f"# aggregated: {len(accounting_df)} planned rows; "
        f"complete={n_complete} failed={n_failed} missing={n_missing} "
        f"atlas_rows={len(atlas_df)}"
    )
    return 0 if (n_failed == 0 and n_missing == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
