"""Run the Janesick parent-ROI input using current core without overwriting a run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tilescope-matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tilescope_results.paths import core_root, data_root

DATA = data_root() / "Janesick"
DEFAULT_INPUT = DATA / "sample1_rep1/tilescope_inputs/CSVs/Janesick_S1R1_expression_data_ACTA2_FASN_parent_rois.csv"
DEFAULT_CONFIG = DATA / "CurrentResults/tilescope_run_config.json"
DEFAULT_OUTPUT = DATA / "CurrentResults/sample1_rep1/ACTA2_FASN"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--tessellations", type=int,
                        help="Update the shared configuration before a new run (10 for the manuscript).")
    args = parser.parse_args()
    if not args.validate_only and args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output is not empty: {args.output}; choose a new directory.")
    sys.path.insert(0, str(core_root() / "python"))
    from analyses.janesick_xenium_feasibility.feasibility import load_and_validate_input
    from tilescope.settings import load_run_config
    from tilescope.api import run_tilescope_from_csv
    load_and_validate_input(args.input, expected_roi_ids=("DCIS_1", "DCIS_2", "Invasive"), expected_n_cells=87770)
    recipe = json.loads(args.config.read_text())
    if args.tessellations is not None:
        if args.tessellations < 1:
            raise ValueError("Tessellations must be positive")
        recipe["tessellation"]["n"] = args.tessellations
    config = load_run_config(str(args.config), profile=None, set_overrides=recipe)
    if config.run_config.technology != "xenium":
        raise ValueError("Janesick input requires technology=xenium.")
    if args.validate_only:
        print(f"Input and current-core configuration are valid (K={config.run_config.tessellation.n}); no files changed.")
        return
    if args.tessellations is not None:
        args.config.write_text(json.dumps(recipe, indent=2) + "\n")
    args.output.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "sample": "sample1_rep1", "pair": "ACTA2_FASN",
        "core_git_sha": subprocess.check_output(["git", "-C", str(core_root()), "rev-parse", "HEAD"], text=True).strip(),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "tessellations": config.run_config.tessellation.n,
    }
    status_path = args.output / "runner_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    started = time.monotonic()
    try:
        run_tilescope_from_csv(input_csv=str(args.input), run_config=config,
                              output_dir=str(args.output), export=True, html_report=False,
                              configure_logging=True, return_minimal=True,
                              feature_a_id="ACTA2", feature_b_id="FASN")
        status["status"] = "complete"
    except Exception:
        status["status"] = "failed"
        status["traceback"] = traceback.format_exc()
        raise
    finally:
        status["seconds"] = time.monotonic() - started
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
