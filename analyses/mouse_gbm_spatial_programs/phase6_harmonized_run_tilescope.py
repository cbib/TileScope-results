"""Run the frozen harmonized mouse-GBM TileScope pair corpus.

Read the distributed score banks and pair inventory from --input-dir.
Write results and run accounting only under the explicit --output-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from tilescope_results.paths import core_root
from tilescope_results.pair_inputs import directory_name, pair_csv, read_scores, run_paths

import pandas as pd
import yaml

CORE_ROOT = (core_root())
CORE_PYTHON = CORE_ROOT / "python"
if str(CORE_PYTHON) not in sys.path:
    sys.path.insert(0, str(CORE_PYTHON))

from tilescope.api.core import run_tilescope_from_csv  # noqa: E402

DATASET_IDS = {"MCT1_CTL": 1, "MCT1_KO": 2, "V4_WT": 3}
IDENTITY_FIELDS = ("input_sha256", "config_sha256", "signature_contract_sha256",
                   "core_git_sha", "core_worktree_diff_sha256")

VALIDATION_KEYS = {
    ("MCT1_CTL", "cell_cycle_proliferation", "interferon_related"),
    ("MCT1_KO", "cell_cycle_proliferation", "interferon_related"),
    ("MCT1_CTL", "antigen_presenting_myeloid", "cytotoxic_t_nk"),
    ("MCT1_KO", "antigen_presenting_myeloid", "cytotoxic_t_nk"),
    ("V4_WT", "myeloid_phagocytic", "v4_reactive_interface"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_text(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=CORE_ROOT, text=True
    ).strip()


def _core_provenance() -> dict:
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=CORE_ROOT)
    status = _git_text("status", "--short")
    return {
        "core_git_sha": _git_text("rev-parse", "HEAD"),
        "core_branch": _git_text("branch", "--show-current"),
        "core_worktree_dirty": bool(status),
        "core_worktree_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "core_worktree_status": status.splitlines() if status else [],
    }


def _safe_len(value) -> int:
    try:
        return 0 if value is None else int(len(value))
    except TypeError:
        return 0


def load_manifest(input_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(input_dir / "pairs.csv", dtype=str)
    keys = ["sample_dir", "signature_a", "signature_b"]
    if not set(keys).issubset(manifest) or manifest.empty:
        raise ValueError("pairs.csv must contain sample_dir, signature_a and signature_b")
    if manifest[keys].isna().any().any() or manifest.duplicated(keys).any():
        raise ValueError("Missing or duplicate sample/pair in pairs.csv")
    contract = yaml.safe_load((input_dir / "signatures_harmonized.yaml").read_text())["signatures"]
    locators = []
    for row in manifest.itertuples(index=False):
        for value in (row.sample_dir, row.signature_a, row.signature_b):
            directory_name(value)
        if row.sample_dir not in DATASET_IDS or row.signature_a == row.signature_b:
            raise ValueError(f"Invalid mouse sample/pair: {row}")
        if row.signature_a not in contract or row.signature_b not in contract:
            raise ValueError(f"Signature absent from signatures_harmonized.yaml: {row}")
        locators.append(any(contract[name]["kind"] == "technical_locator"
                            for name in (row.signature_a, row.signature_b)))
    manifest["contains_technical_locator"] = locators
    return manifest


def run_one(row: dict, input_root: Path, output_root: Path, config_path: Path) -> dict:
    sample = row["sample_dir"]
    signature_a = row["signature_a"]
    signature_b = row["signature_b"]
    output_dir = Path(output_root) / sample / f"{signature_a}__{signature_b}"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    try:
        score_path = Path(input_root) / "scores" / f"{sample}.csv.gz"
        with pair_csv(score_path, f"{signature_a}__score", f"{signature_b}__score") as csv_path:
            result = run_tilescope_from_csv(
                input_csv=str(csv_path),
                run_config=str(config_path),
                feature_a_id=signature_a,
                feature_b_id=signature_b,
                output_dir=str(output_dir),
                export=True,
                return_minimal=True,
                html_report=True,
                dataset_id=DATASET_IDS[sample],
            )
        export_paths = result.get("export_paths", {}) or {}
        return {
            **{key: row[key] for key in IDENTITY_FIELDS},
            "sample_dir": sample,
            "signature_a": signature_a,
            "signature_b": signature_b,
            "contains_technical_locator": bool(row["contains_technical_locator"]),
            "status": "ok",
            "started_utc": started,
            "elapsed_s": round(time.time() - t0, 2),
            "roi_enrichment_n": _safe_len(result.get("roi_enrichment")),
            "roi_correlations_n": _safe_len(result.get("roi_correlations")),
            "roi_cross_grain_n": _safe_len(result.get("roi_cross_grain_sensitivity_df")),
            "component_radial_n": _safe_len(result.get("component_radial_shell_correlogram_df")),
            "export_paths_count": len(export_paths),
            "output_dir": str(output_dir),
            "error_message": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **{key: row[key] for key in IDENTITY_FIELDS},
            "sample_dir": sample,
            "signature_a": signature_a,
            "signature_b": signature_b,
            "contains_technical_locator": bool(row["contains_technical_locator"]),
            "status": "error",
            "started_utc": started,
            "elapsed_s": round(time.time() - t0, 2),
            "roi_enrichment_n": 0,
            "roi_correlations_n": 0,
            "roi_cross_grain_n": 0,
            "component_radial_n": 0,
            "export_paths_count": 0,
            "output_dir": str(output_dir),
            "error_message": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
        }


def _key(frame: pd.DataFrame) -> pd.Series:
    return frame["sample_dir"] + "|" + frame["signature_a"] + "|" + frame["signature_b"]


def _checkpoint(status_path: Path, rows: list[dict]) -> None:
    new = pd.DataFrame(rows)
    if status_path.exists():
        old = pd.read_csv(status_path)
        new_keys = set(_key(new))
        old = old[~_key(old).isin(new_keys)]
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values(["sample_dir", "signature_a", "signature_b"]).to_csv(
        status_path, index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--remainder", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing pairs.csv, scores/ and signature/run settings")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Separate directory for results and run accounting")
    parser.add_argument("--config", type=Path,
                        help="Override INPUT_DIR/tilescope_run_config.json")
    parser.add_argument("--sample", action="append", help="Limit to sample_dir; may repeat")
    parser.add_argument("--pair", action="append", help="Limit to signature_a__signature_b; may repeat")
    parser.add_argument("--dry-run", action="store_true", help="Validate selected inputs without writing")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    try:
        input_root, output_root, config_path = run_paths(args.input_dir, args.output_dir, args.config)
        manifest = load_manifest(input_root)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    from tilescope.settings.loader import load_run_config
    load_run_config(path=config_path)
    pair_names = manifest.signature_a + "__" + manifest.signature_b
    if (set(args.sample or []) - set(manifest.sample_dir)
            or set(args.pair or []) - set(pair_names)):
        parser.error("Unknown sample or physical pair name")

    if args.validate:
        row_keys = list(zip(manifest.sample_dir, manifest.signature_a, manifest.signature_b))
        manifest = manifest[[key in VALIDATION_KEYS for key in row_keys]].copy()
        if len(manifest) != len(VALIDATION_KEYS):
            raise RuntimeError("validation pair set is not closed")
        mode_name = "validation"
    else:
        mode_name = "remainder" if args.remainder else "all"
    if args.sample:
        manifest = manifest[manifest.sample_dir.isin(args.sample)].copy()
    if args.pair:
        manifest = manifest[(manifest.signature_a + "__" + manifest.signature_b).isin(args.pair)].copy()
    if manifest.empty:
        parser.error("No matching inputs")
    input_hashes = {}
    for sample, group in manifest.groupby("sample_dir"):
        score_path = input_root / "scores" / f"{sample}.csv.gz"
        features = set(group.signature_a) | set(group.signature_b)
        read_scores(score_path, [f"{name}__score" for name in features])
        input_hashes[sample] = _sha256(score_path)
    core_provenance = _core_provenance()
    contract_path = input_root / "signatures_harmonized.yaml"
    identity = {key: core_provenance[key] for key in ("core_git_sha", "core_worktree_diff_sha256")}
    identity.update(config_sha256=_sha256(config_path), signature_contract_sha256=_sha256(contract_path))
    for key, value in identity.items():
        manifest[key] = value
    manifest["input_sha256"] = manifest.sample_dir.map(input_hashes)
    if args.dry_run:
        print(f"[harmonized phase6] validated {len(manifest)} pairs; no files written")
        return
    accounting = output_root / "run_accounting"
    status_path = accounting / "run_status.csv"

    if args.remainder and status_path.exists():
        prior = pd.read_csv(status_path)
        complete = set()
        previous = dict(zip(_key(prior), prior.to_dict("records")))
        for key, row in zip(_key(manifest), manifest.to_dict("records")):
            old = previous.get(key)
            if old and old["status"] == "ok":
                if any(old.get(field) != row[field] for field in IDENTITY_FIELDS):
                    parser.error(f"Cannot resume changed inputs/settings for {key}; choose a fresh output directory or --all")
                folder = output_root / row["sample_dir"] / f"{row['signature_a']}__{row['signature_b']}"
                if (folder / "run_metadata.json").is_file():
                    complete.add(key)
        manifest = manifest[~_key(manifest).isin(complete)].copy()

    accounting.mkdir(parents=True, exist_ok=True)
    provenance = {
        "run_config": str(config_path),
        "mode": mode_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "n_manifest_rows_selected": int(len(manifest)),
        "workers": args.workers,
        "signature_contract": str(contract_path),
        "signature_contract_sha256": identity["signature_contract_sha256"],
        "pair_manifest": str(input_root / "pairs.csv"),
        "pair_manifest_sha256": _sha256(input_root / "pairs.csv"),
        "config_sha256": identity["config_sha256"],
        "score_bank_sha256": input_hashes,
        **core_provenance,
    }
    provenance_path = accounting / "run_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    rows = manifest.to_dict("records")
    print(f"[harmonized phase6] {mode_name}: {len(rows)} pairs, workers={args.workers}")
    completed: list[dict] = []
    if args.workers == 1:
        for index, row in enumerate(rows, start=1):
            print(
                f"[{index}/{len(rows)}] {row['sample_dir']} / "
                f"{row['signature_a']} x {row['signature_b']}",
                flush=True,
            )
            result = run_one(row, input_root, output_root, config_path)
            completed.append(result)
            _checkpoint(status_path, completed)
            print(f"  -> {result['status']} ({result['elapsed_s']:.1f}s)", flush=True)
            if result["status"] == "error":
                print(result["error_message"], flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_one, row, input_root, output_root, config_path): row for row in rows
            }
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                completed.append(result)
                _checkpoint(status_path, completed)
                print(
                    f"[{index}/{len(rows)}] {result['sample_dir']} / "
                    f"{result['signature_a']} x {result['signature_b']} "
                    f"-> {result['status']} ({result['elapsed_s']:.1f}s)",
                    flush=True,
                )
                if result["status"] == "error":
                    print(result["error_message"], flush=True)

    errors = [row for row in completed if row["status"] != "ok"]
    print(f"[harmonized phase6] completed={len(completed)} errors={len(errors)}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
