"""Portable Greenwald runner; see README.md for the current workflow.

Reads pairs.csv, scores/<sample>.csv.gz and the shared configuration from
--input-dir; all persistent outputs are written under --output-dir.
Existing runs are reused only when score-bank bytes, resolved configuration, core
revision and the complete quantitative export set match their recorded run.
Historical runs without those identities remain readable, but are not silently
accepted for a new invocation. A fresh output directory permits isolated tests;
--force explicitly authorizes replacement. No result files are patched afterward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

_HERE = Path(__file__).resolve()
_SISTER_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_SISTER_ROOT))
from tilescope_results.paths import core_root
from tilescope_results.pair_inputs import directory_name, pair_csv, read_scores, run_paths
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tilescope_greenwald_mpl"))

# Every current quantitative export used by the analysis, plus semantic metadata.
REQUIRED_OUTPUTS = (
    "roi_correlations.csv",
    "roi_enrichment.csv",
    "roi_tessellation_correlations.csv",
    "roi_tessellation_enrichment.csv",
    "roi_replicate_correlations.csv",
    "roi_replicate_enrichment.csv",
    "roi_consensus_correlations.csv",
    "roi_consensus_enrichment.csv",
    "roi_source_moran.csv",
    "roi_tessellation_moran.csv",
    "roi_consensus_moran.csv",
    "roi_pair_interface_response.csv",
    "roi_support_status.csv",
    "spot_ledger.csv",
    "component_geometry.csv",
    "component_domain_regions.csv",
    "component_region_replicate_status.csv",
    "component_radial_shell_correlogram.csv",
    "roi_cross_grain_sensitivity.csv",
    "consensus_hexes.csv",
    "tessellation_tiles.csv",
    "effective_config.json",
    "run_metadata.json",
)

FAILURE_STATUSES = frozenset({
    "input_missing", "config_missing", "failed", "incomplete_after_run", "reuse_refused",
})

_STATUS_KEY = ("sample_id", "pair")


def _load_existing_status(path: Path) -> pd.DataFrame:
    """Read the on-disk run_status.csv if present. Missing = empty frame
    with the expected column set."""
    if not path.exists():
        return pd.DataFrame(columns=[
            "sample_id", "pair", "checked_at", "status",
            "wall_clock_seconds", "reason", "core_sha",
        ])
    frame = pd.read_csv(path)
    if frame.duplicated(list(_STATUS_KEY)).any():
        raise ValueError(f"Duplicate sample/pair entries: {path}")
    return frame


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolved_config_digest(config: Path) -> str:
    from tilescope.api import validate_run_config
    supplied = json.loads(config.read_text())
    # File-level documentation is not an API configuration option.
    supplied = {k: v for k, v in supplied.items() if not k.startswith("_")}
    resolved = validate_run_config(supplied).model_dump(mode="json")
    resolved.pop("_metadata", None)
    resolved["run_config"].pop("output_dir", None)
    return hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()


def reusable(previous, identity, run_dir):
    """Unknown historical identity is not proof of compatibility."""
    return bool(previous and previous.get("status") == "complete"
                and all(previous.get(k) == v for k, v in identity.items())
                and _is_complete(run_dir))


def _current_core_sha() -> str:
    import tilescope
    expected = (core_root() / "python").resolve()
    if not Path(tilescope.__file__).resolve().is_relative_to(expected):
        raise ValueError(f"Imported TileScope is not the configured checkout: {tilescope.__file__}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "python"], cwd=core_root(), text=True)
    if dirty.strip():
        raise ValueError("Core Python files have uncommitted changes; cannot record a reproducible run")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=core_root(), text=True,
    ).strip()


def _merge_status(existing: pd.DataFrame, new_rows: List[Dict]) -> pd.DataFrame:
    """Upsert new_rows into existing on (sample_id, pair). New rows win
    EXCEPT for status == "already_complete" -- if the prior row already
    records a completion with real wall-clock timing, we preserve it
    rather than overwriting with the no-op re-visit record."""
    if not new_rows:
        return existing
    if existing.empty:
        existing_keys = set()
    else:
        existing_keys = set(zip(existing["sample_id"], existing["pair"]))
    winning_rows = [
        r for r in new_rows
        if r["status"] != "already_complete"
        or (r["sample_id"], r["pair"]) not in existing_keys
    ]
    if not winning_rows:
        return existing
    new_df = pd.DataFrame(winning_rows)
    if existing.empty:
        return new_df.sort_values(list(_STATUS_KEY)).reset_index(drop=True)
    key_pairs = set(zip(new_df["sample_id"], new_df["pair"]))
    keep = existing[
        ~existing.apply(lambda r: (r["sample_id"], r["pair"]) in key_pairs, axis=1)
    ]
    return pd.concat([keep, new_df], ignore_index=True).sort_values(
        list(_STATUS_KEY)
    ).reset_index(drop=True)

def enumerate_tasks(input_dir: Path) -> List[Tuple[str, str, Path]]:
    """Return physical feature order, not an alphabetically renamed pair."""
    manifest = pd.read_csv(input_dir / "pairs.csv", dtype=str)
    if not {"sample", "pair"}.issubset(manifest) or manifest.empty:
        raise ValueError("pairs.csv must contain sample and pair columns and at least one run")
    if manifest[["sample", "pair"]].isna().any().any() or manifest.duplicated(["sample", "pair"]).any():
        raise ValueError("Missing or duplicate sample/pair in pairs.csv")
    tasks: List[Tuple[str, str, Path]] = []
    for row in manifest.itertuples(index=False):
        sample, pair = directory_name(row.sample), directory_name(row.pair)
        features = pair.split("_")
        if len(features) != 2 or not all(features) or features[0] == features[1]:
            raise ValueError(f"Invalid physical Greenwald pair: {pair!r}")
        tasks.append((sample, pair, input_dir / "scores" / f"{sample}.csv.gz"))
    return tasks


def _is_complete(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False
    for name in REQUIRED_OUTPUTS:
        f = run_dir / name
        if not f.exists() or f.stat().st_size == 0:
            return False
    return True


def _run_one(
    sample_id: str, pair: str, score_path: Path, *,
    force: bool, config_path: Path, core_sha: str,
    results_root: Path, previous: dict | None = None,
    config_sha: str | None = None, input_sha: str | None = None,
) -> Dict[str, object]:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row: Dict[str, object] = {
        "sample_id": sample_id, "pair": pair, "checked_at": now,
        "status": "pending", "wall_clock_seconds": None,
        "reason": "", "core_sha": core_sha,
    }
    if not score_path.exists():
        row["status"] = "input_missing"
        row["reason"] = f"missing:{score_path.name}"
        return row
    if not config_path.exists():
        row["status"] = "config_missing"
        row["reason"] = f"missing:{config_path.name}"
        return row
    run_dir = results_root / sample_id / pair
    identity = dict(core_sha=core_sha, input_sha256=input_sha or digest(score_path),
                    config_sha256=config_sha or resolved_config_digest(config_path))
    row.update(identity)
    if not force and run_dir.exists() and any(run_dir.iterdir()):
        if reusable(previous, identity, run_dir):
            row["status"] = "already_complete"
        else:
            row["status"] = "reuse_refused"
            row["reason"] = "Existing outputs have incomplete or different identity; use a fresh --output-dir or explicit --force"
        return row
    run_dir.mkdir(parents=True, exist_ok=True)
    # The manifest's physical order determines directional statistics.
    feature_a, feature_b = pair.split("_", 1)
    t0 = time.monotonic()
    try:
        from tilescope.api import run_tilescope_from_csv
        with pair_csv(score_path, feature_a, feature_b) as csv_path:
            run_tilescope_from_csv(
                input_csv=str(csv_path),
                run_config=str(config_path),
                output_dir=str(run_dir),
                html_report=False,
                dataset_id=0,
                feature_a_id=feature_a,
                feature_b_id=feature_b,
            )
    except Exception as exc:
        row["status"] = "failed"
        row["reason"] = f"{type(exc).__name__}: {exc}"
        row["wall_clock_seconds"] = round(time.monotonic() - t0, 2)
        return row
    row["wall_clock_seconds"] = round(time.monotonic() - t0, 2)
    if _is_complete(run_dir):
        row["status"] = "complete"
    else:
        missing = [
            n for n in REQUIRED_OUTPUTS
            if not (run_dir / n).exists() or (run_dir / n).stat().st_size == 0
        ]
        row["status"] = "incomplete_after_run"
        row["reason"] = f"missing_after_run:{','.join(missing[:5])}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Rerun tasks even if already complete.")
    parser.add_argument("--sample", action="append",
                        help="Limit to specific sample_id(s); may repeat.")
    parser.add_argument("--pair", action="append",
                        help="Limit to specific pair(s); may repeat.")
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel worker processes (default 6).")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing pairs.csv, scores/ and tilescope_run_config.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Separate directory for results and run accounting")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and list tasks without writing")
    parser.add_argument(
        "--config", type=Path,
        help="Override INPUT_DIR/tilescope_run_config.json; used for every selected pair",
    )
    args = parser.parse_args()
    try:
        args.input_dir, args.output_dir, args.config = run_paths(args.input_dir, args.output_dir, args.config)
        tasks = enumerate_tasks(args.input_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.workers < 1:
        parser.error("--workers must be positive")
    status_path = args.output_dir / "run_accounting" / "run_status.csv"
    known_samples = {t[0] for t in tasks}; known_pairs = {t[1] for t in tasks}
    if (set(args.sample or []) - known_samples) or (set(args.pair or []) - known_pairs):
        parser.error("Unknown sample or physical pair name")
    core_sha = _current_core_sha()
    config_sha = resolved_config_digest(args.config)
    if args.sample:
        wanted_samples = set(args.sample)
        tasks = [t for t in tasks if t[0] in wanted_samples]
    if args.pair:
        wanted_pairs = set(args.pair)
        tasks = [t for t in tasks if t[1] in wanted_pairs]
    input_hashes = {}
    for path in sorted({t[2] for t in tasks}):
        features = {f for _, pair, p in tasks if p == path for f in pair.split("_")}
        read_scores(path, features)
        input_hashes[path] = digest(path)

    n_total = len(tasks)
    if not n_total:
        parser.error("No matching inputs")
    print(
        f"# Greenwald full runner: {n_total} (sample, pair) tasks "
        f"across {args.workers} workers"
    )
    print(f"# config: {args.config}")
    print(f"# core SHA: {core_sha}")
    print(f"# results root: {args.output_dir}")
    if args.dry_run:
        print(f"# validated {n_total} tasks; no files written")
        return

    existing = _load_existing_status(status_path)
    prior = {(r['sample_id'], r['pair']): r for r in existing.to_dict('records')}
    # Refuse mismatches before scheduling ANY work or changing the old inventory.
    if not args.force:
        for sample, pair, path in tasks:
            folder = args.output_dir / sample / pair
            identity = dict(core_sha=core_sha, input_sha256=input_hashes[path], config_sha256=config_sha)
            if folder.exists() and any(folder.iterdir()) and not reusable(prior.get((sample, pair)), identity, folder):
                parser.error(f"Cannot safely reuse {sample}/{pair}; use a fresh --output-dir or explicit --force")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not existing.empty:
        print(
            f"# resuming: {len(existing)} existing status rows will be "
            "preserved and updated in place."
        )
    rows: List[Dict] = []
    t_start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                _run_one, sample_id, pair, csv_path,
                force=args.force, config_path=args.config,
                core_sha=core_sha, config_sha=config_sha, results_root=args.output_dir,
                previous=prior.get((sample_id, pair)), input_sha=input_hashes[csv_path],
            ): (i, sample_id, pair)
            for i, (sample_id, pair, csv_path) in enumerate(tasks, 1)
        }
        for fut in as_completed(futures):
            i, sample_id, pair = futures[fut]
            row = fut.result()
            rows.append(row)
            # Incremental write after every task: a kill loses no more
            # than the currently in-flight tasks.
            merged = _merge_status(existing, rows)
            temporary = status_path.with_suffix(".csv.tmp")
            merged.to_csv(temporary, index=False)
            temporary.replace(status_path)
            wt = row["wall_clock_seconds"]
            wt_str = f"{wt:.1f}s" if wt is not None else "-"
            elapsed = time.monotonic() - t_start
            print(
                f"#   [{len(rows):4d}/{n_total}] {sample_id:<18s} "
                f"{pair:<28s} {row['status']:<20s} {wt_str:>7s}  "
                f"(elapsed {elapsed:.1f}s)  {row['reason']}"
            )

    status_df = _merge_status(existing, rows)
    print()
    print(f"# run_status: {len(status_df)} rows -> {status_path}")
    print("# status breakdown:")
    for status, n in status_df["status"].value_counts().items():
        print(f"#   {status:<25s} {int(n)}")
    total_secs = status_df["wall_clock_seconds"].fillna(0).sum()
    print(f"# total worker-seconds: {total_secs:.1f}s")
    print(f"# wall-clock: {time.monotonic() - t_start:.1f}s")
    n_failed = int(status_df["status"].isin(FAILURE_STATUSES).sum())
    if n_failed:
        print(
            f"# ERROR: {n_failed} task(s) did not complete "
            f"(status in {sorted(FAILURE_STATUSES)!r}). See run_status.csv."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
