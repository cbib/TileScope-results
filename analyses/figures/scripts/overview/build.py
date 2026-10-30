"""Reproducible Figure 1A input, unchanged-core run and three separate panels.

This is an explanatory synthetic illustration, not a benchmark case. No
expression values or component boundaries are edited at rendering time.
"""
from __future__ import annotations
from tilescope_results.paths import data_root

import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from tilescope_results.paths import core_root
import platform
import resource
import shutil
import subprocess
import sys
import time

HERE = data_root() / "figure1_overview"
RESULTS_ROOT = Path(__file__).resolve().parents[4]
CORE_ROOT = core_root()
GENERATION_SEED = 20260916
FEATURE_IDS = ("Synthetic gene A", "Synthetic gene B")
from analyses.figures.common import EXPRESSION_COLOR_STOPS, expression_colormap


# Limit interference with other analyses on this workstation.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

import numpy as np
import pandas as pd


def gene_b_regional_background(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Broad ROI-2-aligned elevation with gradual transitions at its borders.

    This illustrative field shares the analytic tissue geometry used for the
    labels; it is not evidence that the labels were learned from two genes.
    Sigmoid transitions extend across the boundaries rather than switching at
    integer ROI labels. The detached ROI 1 component retains a lower baseline.
    """
    from scipy.special import expit

    left_boundary = -4.0 + 0.6 * np.sin(y / 2.5)
    lower_boundary = -3.4 + 0.35 * np.cos(x / 3.0)
    island_radius = np.sqrt(((x - 7.0) / 3.7) ** 2 + ((y - 3.5) / 3.0) ** 2)
    envelope = (
        expit((x - left_boundary) / 0.85)
        * expit((y - lower_boundary) / 0.90)
        * expit((island_radius - 1.0) / 0.20)
    )
    gentle_variation = 0.92 + 0.08 * np.cos((x - 2.0) / 5.0) * np.cos(y / 4.0)
    return 0.56 * envelope * gentle_variation


def gene_a_boundary_taper(frame: pd.DataFrame) -> pd.DataFrame:
    """Attenuate A only in ROI 2, beyond three spot spacings from ROI 1.

    Both ROI 1 components contribute. This smooth, distance-based synthetic
    field change precedes analysis; core still defines near/far by graph hops.
    Values within three spacings, all other ROIs, and B remain bit-identical.
    """
    from scipy.spatial import cKDTree

    xy = frame[["x", "y"]].to_numpy()
    spacing = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    distance = cKDTree(xy[frame.roi_id.eq(1)]).query(xy)[0] / spacing
    affected = frame.roi_id.eq(2).to_numpy() & (distance > 3.0)
    result = frame.copy()
    result.loc[affected, "gene1"] = (
        0.02 + (frame.loc[affected, "gene1"] - 0.02)
        * np.exp(-((distance[affected] - 3.0) / 1.0) ** 2)
    )
    return result


def make_input(seed: int = GENERATION_SEED, *, include_gene_b_background: bool = True,
               include_gene_a_boundary_taper: bool = True) -> pd.DataFrame:
    """Hexagonal spot lattice; fixed analytic labels and smooth paired fields."""
    row, col = np.mgrid[-12:13, -16:17]
    x = (col + 0.5 * (row % 2)).ravel().astype(float)
    y = (row * np.sqrt(3.0) / 2.0).ravel()
    theta = np.arctan2(y / 9.0, x / 14.0)
    radius = 1.0 + 0.045 * np.sin(3 * theta + 0.4)
    inside = (x / 14.0) ** 2 + (y / 9.0) ** 2 <= radius ** 2
    x, y = x[inside], y[inside]

    # ROI 1: left territory + separate inclusion in the upper right.
    # ROI 2: intervening/upper territory. ROI 3: lower territory.
    main = x < -4.0 + 0.6 * np.sin(y / 2.5)
    island = ((x - 7.0) / 3.7) ** 2 + ((y - 3.5) / 3.0) ** 2 <= 1.0
    lower = (y < -3.4 + 0.35 * np.cos(x / 3.0)) & ~main
    roi = np.where(main | island, 1, np.where(lower, 3, 2))

    def gaussian(cx, cy, sx, sy):
        return np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))

    rng = np.random.default_rng(seed)
    # Smooth baseline fields; the optional A taper is applied after noise.
    # B retains its boundary focus and moderate regional elevation.
    gene_a = 0.045 + 0.88 * gaussian(-6.2, 0.8, 4.8, 5.0)
    gene_a += 0.69 * gaussian(7.0, 3.5, 3.4, 3.3)
    boundary_focus = gaussian(-2.6, 2.0, 3.5, 2.7)
    gene_b = 0.045 + 0.90 * boundary_focus
    if include_gene_b_background:
        # Blend towards the existing peak rather than adding a flat pedestal
        # that would clip many observations to the top of the colour scale.
        gene_b += gene_b_regional_background(x, y) * (1.0 - boundary_focus)
    gene_a += rng.normal(0.0, 0.013, len(x))
    gene_b += rng.normal(0.0, 0.013, len(x))
    frame = pd.DataFrame({
        "x": x, "y": y,
        "gene1": np.clip(gene_a, 0.001, 1.0),
        "gene2": np.clip(gene_b, 0.001, 1.0),
        "roi_id": roi.astype(int),
    })
    return gene_a_boundary_taper(frame) if include_gene_a_boundary_taper else frame


def source_audit(frame: pd.DataFrame) -> dict:
    """Check the intended anatomy using TileScope's own Visium graph."""
    from tilescope.spatial.source_spot_table import build_source_spot_table_from_frame
    from tilescope.spatial.source_spot_adjacency import build_source_spot_adjacency

    graph = build_source_spot_adjacency(
        build_source_spot_table_from_frame(frame), technology="visium",
    )
    components = [{"roi_id": c.roi_id, "n_spots": c.n_spots,
                   "spot_indices": c.positions.tolist()}
                  for c in graph.within_roi_components]
    observed = dict(Counter(c["roi_id"] for c in components))
    if observed != {"1": 2, "2": 1, "3": 1}:
        raise ValueError(f"Unexpected component anatomy: {observed}")
    if min(c["n_spots"] for c in components) < 30:
        raise ValueError("A schematic component has fewer than 30 spots")
    cross = {f"{a}-{b}": len(graph.cross_roi_edges(a, b))
             for a, b in (("1", "2"), ("1", "3"), ("2", "3"))}
    if cross["1-2"] == 0:
        raise ValueError("The intended ROI 1/2 contact is absent")
    return {"n_spots": len(frame), "n_rois": 3, "n_components": len(components),
            "roi_spot_counts": {str(k): int(v) for k, v in
                                frame.groupby("roi_id").size().items()},
            "graph_method": graph.method, "median_nn": graph.sample_median_nn,
            "cross_roi_edge_counts": cross, "components": components}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def core_identity() -> str:
    import tilescope

    loaded = Path(tilescope.__file__).resolve()
    if not loaded.is_relative_to(CORE_ROOT.resolve()):
        raise RuntimeError(f"Unexpected TileScope import: {loaded}")
    dirty = subprocess.check_output(
        ["git", "-C", str(CORE_ROOT), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Tracked core changes are present; use a clean core commit")
    return subprocess.check_output(
        ["git", "-C", str(CORE_ROOT), "rev-parse", "HEAD"], text=True,
    ).strip()


def run_core(input_csv: Path, config: Path, core_sha: str) -> Path:
    """Keep differently configured runs separate and validate cached outputs."""
    from tilescope.api.core import run_tilescope_from_csv

    signature = {"core_sha": core_sha, "input_sha256": sha256(input_csv),
                 "config_sha256": sha256(config), "feature_ids": list(FEATURE_IDS)}
    tag = f"{core_sha[:8]}_{signature['input_sha256'][:8]}_{signature['config_sha256'][:8]}"
    run_dir = HERE / "runs" / tag
    status_file = run_dir / "runner_status.json"
    if run_dir.exists():
        if not status_file.exists():
            raise RuntimeError(f"Incomplete previous run retained at {run_dir}")
        status = json.loads(status_file.read_text())
        valid = status.get("status") == "complete" and status.get("signature") == signature
        hashes = status.get("output_sha256", {})
        valid = valid and bool(hashes) and all(
            (run_dir / name).is_file() and sha256(run_dir / name) == digest
            for name, digest in hashes.items()
        )
        if not valid:
            raise RuntimeError(f"Existing run is incomplete or modified: {run_dir}")
        print(f"Reusing verified core run: {run_dir}", flush=True)
        return run_dir

    run_dir.mkdir(parents=True)
    shutil.copyfile(input_csv, run_dir / "input.csv")
    shutil.copyfile(config, run_dir / "run.config.json")
    status = {"status": "running", "signature": signature,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version()}
    write_json(status_file, status)
    start = time.monotonic()
    print(f"Running unchanged TileScope {core_sha[:8]}: {run_dir}", flush=True)
    try:
        try:
            with (run_dir / "run.log").open("w") as stream:
                logging.basicConfig(stream=stream, level=logging.INFO, force=True)
                with redirect_stdout(stream), redirect_stderr(stream):
                    run_tilescope_from_csv(
                        input_csv=input_csv, run_config=config, profile=None,
                        output_dir=run_dir, export=True, html_report=True,
                        dataset_id=1, feature_a_id=FEATURE_IDS[0], feature_b_id=FEATURE_IDS[1],
                        return_minimal=True,
                    )
        finally:
            # shutdown() alone leaves handlers pointing to the closed run log.
            # Detach them before the independent manuscript renderer logs.
            logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)
        if core_identity() != core_sha:
            raise RuntimeError("Core changed during this run")
        for name in ("expression_summary.pdf", "component_geometry.csv",
                     "spot_ledger.csv", "tessellation_tiles.csv", "consensus_hexes.csv",
                     "roi_pair_interface_response.csv", "effective_config.json"):
            if not (run_dir / name).is_file() or (run_dir / name).stat().st_size == 0:
                raise RuntimeError(f"Missing standard output: {name}")
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(status_file, status)
        raise
    status.update(status="complete", wall_clock_seconds=round(time.monotonic() - start, 3),
                  peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                                     (1024 ** 2 if sys.platform == "darwin" else 1024), 2))
    status["output_sha256"] = {
        str(p.relative_to(run_dir)): sha256(p)
        for p in sorted(run_dir.rglob("*")) if p.is_file() and p != status_file
    }
    write_json(status_file, status)
    print(f"Core run complete in {status['wall_clock_seconds']} s", flush=True)
    return run_dir


def verify_run(frame: pd.DataFrame, preflight: dict, run_dir: Path) -> tuple[pd.DataFrame, dict]:
    ledger = pd.read_csv(run_dir / "spot_ledger.csv", dtype={"roi_id": str, "component_id": str})
    ledger = ledger.sort_values("spot_index").reset_index(drop=True)
    if ledger.spot_index.tolist() != list(range(len(frame))):
        raise ValueError("The source row identity has changed")
    if ledger.roi_id.tolist() != frame.roi_id.astype(str).tolist():
        raise ValueError("The source ROI labels have changed")
    if not ledger.component_status.eq("retained_component").all():
        raise ValueError("The schematic unexpectedly has excluded components")
    if not ledger.tessellation_eligible.eq(True).all():
        raise ValueError("The schematic unexpectedly has non-tessellatable observations")
    observed = {frozenset(g.spot_index) for _, g in ledger.groupby("component_id")}
    expected = {frozenset(c["spot_indices"]) for c in preflight["components"]}
    if observed != expected:
        raise ValueError("Core components differ from the intended source components")
    geometry = pd.read_csv(run_dir / "component_geometry.csv",
                           dtype={"roi_id": str, "component_id": str})
    if len(geometry) != 4 or not geometry.geometry_status.eq("ok").all():
        raise ValueError("Invalid or incomplete exported component geometry")
    from shapely import wkt
    from shapely.geometry import Point
    for component in geometry.itertuples():
        polygon = wkt.loads(component.geometry_wkt)
        members = ledger.loc[ledger.component_id == component.component_id, "spot_index"]
        if not all(polygon.covers(Point(x, y)) for x, y in
                   frame.loc[members, ["x", "y"]].itertuples(index=False, name=None)):
            raise ValueError("An exported component polygon excludes an input spot")
    tiles = pd.read_csv(run_dir / "tessellation_tiles.csv")
    counts = tiles.groupby(["component_id", "replicate_index"]).size()
    if len(counts) != 40 or counts.min() < 3:
        raise ValueError("Every component must have at least three tiles in all ten replicates")
    interface = pd.read_csv(run_dir / "roi_pair_interface_response.csv")
    contact = interface.loc[
        interface.anchor_roi_id.astype(str).eq("1")
        & interface.responder_roi_id.astype(str).eq("2")
        & interface.surface.eq("source_native")
    ]
    if len(contact) != 2 or not contact.availability_status.eq("finite").all():
        raise ValueError("The intended ROI 1→2 interface is not evaluable for both genes")
    audit = {"status": "pass", "all_source_spots_retained": True,
             "all_source_spots_tessellation_eligible": True,
             "n_spots": len(frame), "n_components": len(geometry),
             "n_tessellations": int(tiles.replicate_index.nunique()),
             "minimum_tiles_per_component_replicate": int(counts.min()),
             "n_tiles_all_replicates": len(tiles),
             "n_consensus_hexes": len(pd.read_csv(run_dir / "consensus_hexes.csv")),
             "n_interface_rows": len(interface),
             "roi_1_to_2_source_interface_evaluable_for_both_genes": True,
             "component_membership_matches_preflight": True,
             "all_source_spots_covered_by_exported_component_geometry": True,
             "components": geometry[["roi_id", "component_id", "n_component_spots"]].to_dict("records")}
    return geometry, audit




def render_panels(frame: pd.DataFrame, geometry: pd.DataFrame, output: Path) -> list[Path]:
    """Render separate vector maps using core painters and exported geometry."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
    from shapely import wkt
    from tilescope.viz.painters import SpotSplatPainter, draw_polygon_boundaries

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "tilescope-figure1",
                         "pdf.fonttype": 42, "axes.linewidth": 0.6})
    output.mkdir(parents=True, exist_ok=True)
    polygons = {r.component_id: wkt.loads(r.geometry_wkt) for r in geometry.itertuples()}
    coords = frame[["x", "y"]].to_numpy()
    identity_cmap = ListedColormap(["#62AFA4", "#DDA966", "#9C93C4"])
    identity_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], identity_cmap.N)
    expression_norm = Normalize(0.0, 1.0)  # same, fixed synthetic units on both genes
    expression_cmap = expression_colormap()
    files = []
    specs = [("A1_roi_labels", "ROI labels", "roi_id", identity_cmap, identity_norm),
             ("A2_gene_A_expression", "Gene A", "gene1", expression_cmap, expression_norm),
             ("A3_gene_B_expression", "Gene B", "gene2", expression_cmap, expression_norm)]
    for stem, title, column, cmap, norm in specs:
        # Fixed canvas/axes, not tight bounding boxes: panels align when assembled.
        fig = plt.figure(figsize=(4.2, 3.35), facecolor="white")
        ax = fig.add_axes([0.035, 0.19, 0.93, 0.70])
        ax.set_xlim(-15.5, 15.5)
        ax.set_ylim(-10.5, 10.5)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        fig.canvas.draw()
        collection = SpotSplatPainter(radius_scale=0.35).draw(
            ax, coords, frame[column].to_numpy(), cmap=cmap, norm=norm,
            canvas=None, median_nn_override=1.0,
        )
        if collection is None or len(collection.get_offsets()) != len(frame):
            raise ValueError("A source point was omitted from a figure panel")
        if not np.array_equal(collection.get_array(), frame[column].to_numpy()):
            raise ValueError("Expression values changed during rendering")
        draw_polygon_boundaries(ax, polygons, color="#3B4043", linewidth=0.80)
        fig.text(0.5, 0.94, title, ha="center", va="center", fontsize=12)
        if column == "roi_id":
            for r in geometry.itertuples():
                p = polygons[r.component_id].representative_point()
                ax.text(p.x, p.y, f"ROI {r.roi_id}", ha="center", va="center", fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90,
                              "boxstyle": "round,pad=0.18"})
            # Same palette and explicit duplicate label identify the two ROI 1 pieces.
            for x, label, color in zip((0.27, 0.50, 0.73), ("ROI 1", "ROI 2", "ROI 3"),
                                       identity_cmap.colors):
                fig.text(x - 0.053, 0.092, "●", color=color, fontsize=12, ha="right")
                fig.text(x - 0.042, 0.092, label, fontsize=9, ha="left")
        else:
            cax = fig.add_axes([0.30, 0.098, 0.40, 0.025])
            cbar = fig.colorbar(collection, cax=cax, orientation="horizontal", ticks=[0, 1])
            cbar.ax.set_xticklabels(["Low", "High"])
            cbar.ax.tick_params(length=0, labelsize=8, pad=3)
            cbar.outline.set_visible(False)
        from analyses.figures.common import save_panel
        files.extend(save_panel(fig, output, stem))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Create and audit the CSV only")
    parser.add_argument("--run-analysis", action="store_true", help="Explicitly generate the illustration data and run TileScope; not needed to redraw figures")
    args = parser.parse_args()
    if not args.run_analysis and not args.prepare_only:
        parser.error('Use python -m analyses.figures.build overview to redraw the saved example; --run-analysis is required to generate new results')
    core_sha = core_identity()
    frame = make_input()
    preflight = source_audit(frame)
    inputs = HERE / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    input_csv = inputs / "illustrative_gene_A_gene_B.csv"
    frame.to_csv(input_csv, index=False, float_format="%.15g")
    # Subsequent checks and rendering use the serialized input used by core.
    frame = pd.read_csv(input_csv)
    write_json(inputs / "geometry_audit.json", preflight)
    print(json.dumps({k: v for k, v in preflight.items() if k != "components"}, indent=2), flush=True)
    if args.prepare_only:
        return
    config = HERE / "run.config.json"
    run_dir = run_core(input_csv, config, core_sha)
    geometry, audit = verify_run(frame, preflight, run_dir)
    write_json(HERE / "validation.json", audit)
    write_json(HERE / "provenance.json", {
        "purpose": "Figure 1 schematic, not a biological result or benchmark case",
        "core_sha": core_sha, "core_tracked_tree_clean": True,
        "generation_seed": GENERATION_SEED, "feature_ids": list(FEATURE_IDS),
        "expression_design": "Gene A attenuated in ROI 2 beyond three spot spacings from ROI 1; Gene B retains its smooth ROI-2-aligned background and stronger boundary focus",
        "gene_a_boundary_taper": {"roi_id": 2, "anchor_roi_id": 1,
                                  "onset_nn_spacings": 3.0, "width_nn_spacings": 1.0,
                                  "asymptotic_expression": 0.02},
        "gene_b_background_maximum_amplitude": 0.56,
        "input": str(input_csv.relative_to(HERE)), "input_sha256": sha256(input_csv),
        "config_sha256": sha256(config), "script_sha256": sha256(Path(__file__)),
        "core_run": str(run_dir.relative_to(HERE)),
        "run_status_sha256": sha256(run_dir / "runner_status.json"),
        "source_geometry": str((run_dir / "component_geometry.csv").relative_to(HERE)),
        "source_geometry_sha256": sha256(run_dir / "component_geometry.csv"),
        "rendering": {"core_painters": ["SpotSplatPainter", "draw_polygon_boundaries"],
                      "spot_radius_nn_fraction": 0.35, "expression_limits": [0.0, 1.0],
                      "expression_colormap": "figure1_light_blue_red",
                      "expression_color_stops": EXPRESSION_COLOR_STOPS,
                      "geometry_smoothing": False,
                      "point_relocation_or_value_editing": False,
                      "output": "three independent panels, identical canvas and spatial extent",
                      "standard_pdf_difference": "Light-blue through off-white to red palette, fixed 0–1 colour limits, smaller dots, neutral outlines; no correlation panel"},
    })
    print(json.dumps(audit, indent=2), flush=True)
    print("Saved illustration run prepared. Render it with python -m analyses.figures.build overview", flush=True)


if __name__ == "__main__":
    main()
