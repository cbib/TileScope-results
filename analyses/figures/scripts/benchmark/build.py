"""Figure 2A and supplementary catalogue from checksummed benchmark inputs.

Rendering only: no simulation, TileScope run, spot subsampling, expression
editing or label reassignment. Display crops are explicitly recorded. The
source spot painter and expression colours are shared with Figure 1.
"""
from __future__ import annotations
from tilescope_results.paths import figure2_root

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from tilescope_results.paths import input_data_root, reference, resolve_reference
import sys

CODE = Path(__file__).resolve().parent
HERE = figure2_root()
ROOT = CODE.parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import box

from analyses.figures.common import expression_colormap, spot_label_outlines
from tilescope.viz.painters import SpotSplatPainter, draw_polygon_boundaries


@dataclass(frozen=True)
class Example:
    slug: str
    family: str
    case: str
    title: str
    role: str = "positive"
    framing: str = "roi"
    main: bool = False


# Contrasting modes use the SAME substrate, viewport rule and colour limits.
# The primary examples span expression, label geometry and boundary responses.
EXAMPLES = (
    Example("01_broad_overlap", "broad_colocalization", "CHC20__roi1",
            "Broad overlap", "positive_broad"),
    Example("02_colocated_foci", "focal_pattern", "TS20__roi4__colocated",
            "Colocated foci"),
    Example("03_separated_foci", "focal_pattern", "TS20__roi4__separated",
            "Separated foci"),
    Example("04_concordant_gradients", "spatial_gradient", "242_T__roi3__concordant",
            "Concordant gradients", "positive_spatial_gradient", "main_component"),
    Example("05_opposed_gradients", "spatial_gradient", "242_T__roi3__opposed",
            "Opposed gradients", "positive_spatial_gradient", "main_component", True),
    Example("06_nested_foci", "nested_foci", "262_T__A4_B2", "Nested foci"),
    Example("07_embedded_domain", "embedded_domains", "260_T__A5_B2",
            "Embedded domain", framing="embedded_context", main=True),
    Example("08_dispersed_population", "dispersed_infiltration", "266_T__A7_B2",
            "Dispersed population", framing="host"),
    Example("09_boundary_response", "contact_interface", "CHC20__A4_B3__responder_activation",
            "Boundary response", framing="interface", main=True),
    Example("10_boundary_extension", "contact_interface", "CHC20__A4_B3__anchor_fade",
            "Expression across a boundary", framing="interface"),
)


def corpus_root():
    return input_data_root() / 'benchmark/corpus'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()




def load_example(spec: Example) -> dict:
    """Validate both active-index hashes and input/truth row correspondence."""
    folder = corpus_root() / spec.family / spec.case
    manifest_path = folder / "manifest.json"
    truth_path = folder / "truth_ledger.csv"
    manifest = json.loads(manifest_path.read_text())
    index = json.loads((corpus_root() / "corpus_index.json").read_text())
    entry = next(e for e in index["cases"]
                 if e["scenario_name"] == spec.family and e["case_id"] == spec.case)
    role = next(r for r in manifest["roles"] if r["name"] == spec.role)
    if role["disposition"] != "feasible":
        raise ValueError(f"Selected example is unavailable: {spec.slug}")
    input_path = resolve_reference(folder / role["input_file"])
    if sha256(manifest_path) != entry["manifest_sha256"]:
        raise ValueError(f"Manifest differs from active corpus index: {spec.case}")
    if sha256(input_path) != role["input_sha256"] or sha256(input_path) != entry["role_inputs"][spec.role]:
        raise ValueError(f"Input differs from active corpus index: {spec.case}")
    if sha256(truth_path) != entry["truth_ledger_sha256"]:
        raise ValueError(f"Truth differs from active corpus index: {spec.case}")
    frame = pd.read_csv(input_path, dtype={"roi_id": str}, float_precision="round_trip")
    frame.insert(0, "input_row", np.arange(len(frame)))
    truth = pd.read_csv(truth_path, dtype={
        "source_spot_id": str, "generated_roi_id": str, "source_parent_roi_id": str,
        "source_component_id": str}, float_precision="round_trip")
    truth = truth.loc[truth.counterfactual_role.eq(spec.role)].copy()
    if len(frame) != role["n_spots"] or len(truth) != len(frame):
        raise ValueError("Input and truth row counts differ")
    merged = frame.merge(truth, on=["x", "y"], validate="one_to_one", how="left", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("Input coordinates could not be matched to the truth ledger")
    for left, right in (("gene1", "feature_a_value"), ("gene2", "feature_b_value"),
                        ("roi_id", "generated_roi_id"), ("roi_id", "source_parent_roi_id")):
        if not np.array_equal(merged[left].to_numpy(), merged[right].to_numpy()):
            raise ValueError(f"Input/truth mismatch in {left}: {spec.case}")
    coords = frame[["x", "y"]].to_numpy()
    if not np.isfinite(frame[["x", "y", "gene1", "gene2"]].to_numpy()).all():
        raise ValueError("Nonfinite input")
    spacing = float(np.median(cKDTree(coords).query(coords, k=2)[0][:, 1]))
    if spacing <= 0:
        raise ValueError("Duplicate or degenerate coordinates")
    source_paths = [input_path, manifest_path, truth_path]
    return {"spec": spec, "frame": merged.drop(columns="_merge"), "manifest": manifest,
            "spacing": spacing, "source_paths": source_paths,
            "source_hashes": {reference(p): sha256(p) for p in source_paths}}


def choose_view(data: dict) -> None:
    """Frame observed geometry, never select using a statistic or gene value."""
    spec, f = data["spec"], data["frame"]
    args, spacing = data["manifest"]["args"], data["spacing"]
    roi_a = str(args.get("anchor_roi_id", args.get("a_roi_id")))
    roi_b = str(args["b_roi_id"]) if "b_roi_id" in args else None
    subject_rois = [roi_a]
    framing = f.roi_id.eq(roi_a)
    if spec.framing == "main_component":
        component = f.loc[framing, "source_component_id"].value_counts().index[0]
        framing = f.source_component_id.eq(component)
    elif spec.framing == "embedded_context":
        subject_rois.append(roi_b)
        # Show the whole host AND the inclusion's parent ROI: a detached
        # component should not be mistaken for the entirety of its ROI.
        framing = f.roi_id.isin(subject_rois)
    elif spec.framing == "host":
        subject_rois.append(roi_b)
        framing = f.generated_structure_kind.eq("a_host_support")
        if not framing.any():
            framing = f.roi_id.eq(roi_a)
    elif spec.framing == "interface":
        subject_rois.append(roi_b)
        # All observed A/B contacts, not whichever fragment has strongest signal.
        xy = f[["x", "y"]].to_numpy()
        edges = cKDTree(xy).query_pairs(1.5 * spacing, output_type="ndarray")
        labels = f.roi_id.to_numpy()
        cross = ((labels[edges[:, 0]] == roi_a) & (labels[edges[:, 1]] == roi_b)) | (
            (labels[edges[:, 1]] == roi_a) & (labels[edges[:, 0]] == roi_b))
        endpoints = np.unique(edges[cross])
        framing = f.input_row.isin(endpoints)
        if not len(endpoints):
            raise ValueError("Selected boundary has no observed contacts")
    xy = f.loc[framing, ["x", "y"]].to_numpy()
    if not len(xy):
        raise ValueError(f"Empty framing rule: {spec.slug}")
    pad = {"interface": 4.0}.get(spec.framing, 1.6) * spacing
    lower, upper = xy.min(axis=0) - pad, xy.max(axis=0) + pad
    centre = (lower + upper) / 2.0
    # Square viewports fit paired map cards without shape distortion.
    # Keep spot magnification comparable across the catalogue, including the
    # smaller broad-overlap region. Approved main views already exceed this.
    minimum_width = 50 * spacing
    width = float(max(*(upper - lower), minimum_width))
    bounds = [float(centre[0] - width / 2), float(centre[1] - width / 2),
              float(centre[0] + width / 2), float(centre[1] + width / 2)]
    visible = f.x.between(bounds[0], bounds[2]) & f.y.between(bounds[1], bounds[3])
    f["display_visible"] = visible
    f["display_expression"] = visible & f.roi_id.isin(subject_rois)
    data.update(bounds=bounds, subject_rois=subject_rois, frame=f)




def prepare() -> list[dict]:
    data = [load_example(spec) for spec in EXAMPLES]
    for d in data:
        choose_view(d)
        d["outlines"] = spot_label_outlines(d["frame"], d["spacing"], d["subject_rois"])
    for family in {spec.family for spec in EXAMPLES}:
        group = [d for d in data if d["spec"].family == family]
        values = np.concatenate([d["frame"][["gene1", "gene2"]].to_numpy().ravel() for d in group])
        limits = [float(values.min()), float(values.max())]
        for d in group:
            d["colour_limits"] = limits
    return data


def map_axes(fig, rect, data, feature: str):
    ax = fig.add_axes(rect, facecolor="none")
    xmin, ymin, xmax, ymax = data["bounds"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.canvas.draw()
    f = data["frame"]
    context = f.loc[f.display_visible & ~f.display_expression]
    if len(context) > 1:
        from matplotlib.colors import ListedColormap
        SpotSplatPainter(radius_scale=0.32).draw(
            ax, context[["x", "y"]].to_numpy(), np.zeros(len(context)),
            cmap=ListedColormap(["#E3E6E8"]), norm=Normalize(0, 1), canvas=None,
            median_nn_override=data["spacing"])
    visible = f.loc[f.display_expression]
    painter = SpotSplatPainter(radius_scale=0.38).draw(
        ax, visible[["x", "y"]].to_numpy(), visible[feature].to_numpy(),
        cmap=expression_colormap(), norm=Normalize(*data["colour_limits"]), canvas=None,
        median_nn_override=data["spacing"])
    if painter is None or not np.array_equal(painter.get_array(), visible[feature].to_numpy()):
        raise ValueError("Expression was modified or omitted while rendering")
    painter.set_gid(data["spec"].slug + "_" + feature + "_source_spots")
    draw_polygon_boundaries(ax, data["outlines"], color="#41474B", linewidth=0.52, alpha=0.85)
    crop = box(*data["bounds"])
    for roi, outline in data["outlines"].items():
        clipped = outline.intersection(crop)
        parts = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
        parts = [p for p in parts if p.geom_type == "Polygon" and p.area > 0]
        if not parts:
            continue
        centre = max(parts, key=lambda p: p.area).representative_point()
        style = dict(fontsize=6.8, color="#343A3D", ha="center", va="center",
                     bbox={"facecolor": "white", "alpha": 0.90, "edgecolor": "none", "pad": 1.0})
        if data["spec"].family == "embedded_domains" and roi == data["subject_rois"][1]:
            # Repeat the original ROI ID on its larger territory and inclusion.
            # Keep the small inclusion visible by placing its label outside it.
            ax.text(centre.x, centre.y, f"ROI {roi}", **style)
            selected = f.loc[f.generated_structure_kind.eq("embedded_b_domain")]
            point = selected[["x", "y"]].mean().to_numpy()
            ax.annotate(f"ROI {roi}", point, xytext=(0, -20), textcoords="offset points",
                        arrowprops={"arrowstyle": "-", "color": "#52585B", "lw": 0.65}, **style)
        else:
            ax.text(centre.x, centre.y, f"ROI {roi}", **style)
    return ax
