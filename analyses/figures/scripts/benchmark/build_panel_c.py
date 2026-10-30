"""Comparator statistics on the same 18 cases as Figure 2B0, grouped by method.

Each published spatial method is one group of columns. Most packages publish a
single primary statistic; two of them publish two. SpaceBF reports one
statistic in two directions. The non-spatial within-ROI reference reports
Pearson and Spearman. Row order matches B0 so the two panels can be read
leaf-for-leaf. Values are per-column z-scored across the 18 cases; no
clustering is applied across methods, because methods are not a signature.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tilescope_results.paths import supplementary_data_root, figure2_root

HERE = figure2_root()
CODE = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

NAMES = {
    "broad_colocalization": "Broad overlap", "focal_pattern::colocated": "Colocated foci",
    "focal_pattern::separated": "Separated foci", "spatial_gradient::concordant": "Concordant gradient",
    "spatial_gradient::opposed": "Opposed gradient", "nested_foci": "Nested foci",
}
MODE_COLOURS = dict(zip(NAMES, ["#9B6AA6", "#C88A43", "#B97156", "#438E87", "#577FA5", "#A58C51"]))
CMAP = LinearSegmentedColormap.from_list("comparator_z", ["#5986AC", "#F7F7F4", "#CC7864"])
CLIP = 2.0

# Each method is a group of sub-columns. Within-method stat order is stable but
# carries no scientific ordering; it just has to be deterministic.
METHODS = (
    ("Within-ROI\nreference", (
        ("pearson_within_roi", "", "Pearson r"),
        ("spearman_within_roi", "", "Spearman ρ"),
    )),
    ("MERINGUE", (
        ("MERINGUE spatialCrossCor", "", "SCC"),
    )),
    ("Voyager", (
        ("Voyager lee", "", "Lee's L"),
    )),
    ("SpatialCorr", (
        ("spatialcorr_derived_rho", "", "ρ"),
        ("spatialcorr_native", "", "LLR"),
    )),
    ("STopover", (
        ("STopover J_comp", "", "J"),
    )),
    ("SpaceBF", (
        ("SpaceBF Gaussian global.beta", "a_on_b", "β(A→B)"),
        ("SpaceBF Gaussian global.beta", "b_on_a", "β(B→A)"),
    )),
)

# Flattened stat list in declaration order.
STATS = [(method, sub[0], sub[1], sub[2]) for method, subs in METHODS for sub in subs]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparator_benchmark_root():
    return supplementary_data_root(1) / 'comparator_methods'


def load_values(input_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_path, dtype={"a_roi_id": str})
    required = {"case_id", "class_label", "source_sample_id", "a_roi_id",
                "method", "direction", "value", "status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in comparator_values: {sorted(missing)}")
    if not frame.class_label.isin(NAMES).all():
        unexpected = sorted(set(frame.class_label.unique()) - NAMES.keys())
        raise ValueError(f"Unexpected class labels: {unexpected}")
    return frame


def load_row_order() -> list[str]:
    from .build_panel_b_profiles import prepare
    return prepare()['primary']['order']


def build_matrix(values: pd.DataFrame, row_order: list[str] | None = None) -> dict:
    cases = (values[["case_id", "class_label", "source_sample_id", "a_roi_id"]]
             .drop_duplicates().set_index("case_id"))
    if len(cases) != 18:
        raise ValueError(f"Expected 18 cases, got {len(cases)}")
    if row_order is None:
        row_order = load_row_order()
    missing = set(row_order) - set(cases.index)
    extra = set(cases.index) - set(row_order)
    if missing or extra:
        raise ValueError(f"Row-order mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    cases = cases.loc[row_order]
    raw = pd.DataFrame(index=cases.index, columns=[s[3] for s in STATS], dtype=float)
    direction_or_blank = values.direction.fillna("")
    for _method, source, direction, label in STATS:
        sel = values[(values.method == source) & (direction_or_blank == direction)]
        if sel.empty:
            raise ValueError(f"No rows for stat {source!r} direction={direction!r}")
        sel = sel.set_index("case_id")
        if not raw.index.isin(sel.index).all():
            raise ValueError(f"Missing case rows for stat {source!r}")
        raw[label] = sel.loc[raw.index, "value"].astype(float)
    mu = raw.mean(axis=0, skipna=True)
    sd = raw.std(axis=0, ddof=0, skipna=True)
    if (sd <= 0).any():
        raise ValueError(f"Zero-variance stat(s): {sd[sd <= 0].index.tolist()}")
    standardised = (raw - mu) / sd
    return dict(cases=cases, raw=raw, standardised=standardised,
                scale=pd.DataFrame({"mean": mu, "sd": sd}))


def plot(data: dict):
    standardised = data["standardised"]
    cases = data["cases"]
    n = len(cases)
    # Horizontal budget: cell width 0.78 in, inter-method gap 0.22 in,
    # left margin 0.1 in, right labels 2.6 in. Total ~9.5 in.
    cell = 0.78
    gap = 0.22
    left = 0.10
    right_labels = 2.60
    total_cells = sum(len(subs) for _, subs in METHODS)
    n_methods = len(METHODS)
    matrix_width = total_cells * cell + (n_methods - 1) * gap
    width = left + matrix_width + right_labels
    fig_height = 5.9
    fig = plt.figure(figsize=(width, fig_height))
    # Matrix axes: y measured in rows, x in inches then normalised.
    ax = fig.add_axes((left / width, .18, matrix_width / width, .56), label="comparator_matrix")
    label_ax = fig.add_axes(((left + matrix_width) / width, .18, right_labels / width, .56),
                            label="comparator_labels")
    ax.axis("off")
    label_ax.axis("off")

    # Lay out sub-columns in inches, grouped by method.
    positions, method_bounds, x_in = [], [], 0.0
    for method_name, subs in METHODS:
        start_in = x_in
        for _source, _direction, label in subs:
            positions.append((x_in, label))
            x_in += cell
        method_bounds.append((start_in, x_in, method_name))
        x_in += gap
    x_in -= gap  # last gap should not be included

    ax.set(xlim=(0, matrix_width), ylim=(n, 0))
    label_ax.set(xlim=(0, right_labels), ylim=(n, 0))

    # Column headings (stat labels, horizontal) and method brackets above them.
    for px, label in positions:
        ax.text(px + cell / 2, -.3, label, ha="center", va="bottom", fontsize=7.9,
                clip_on=False)
    for lo, hi, name in method_bounds:
        mid = (lo + hi) / 2
        ax.plot([lo + .04, hi - .04], [-1.55, -1.55], color="#4A5257", lw=1.1, clip_on=False)
        ax.text(mid, -1.95, name, ha="center", va="bottom", fontsize=8.6,
                color="#1F2628", clip_on=False)

    # Cells.
    for px, label in positions:
        for i, case in enumerate(cases.index):
            z = standardised.loc[case, label]
            if np.isnan(z):
                face = "#E6E6E6"
                hatch = "////"
            else:
                face = CMAP(Normalize(-CLIP, CLIP)(np.clip(z, -CLIP, CLIP)))
                hatch = None
            cell_patch = Rectangle((px, i), cell, 1, facecolor=face, edgecolor="white",
                                   linewidth=.55, hatch=hatch)
            cell_patch.set_gid(f"comparator:{case}:{label}")
            ax.add_patch(cell_patch)

    # Thicker separator between methods.
    for lo, hi, _ in method_bounds[:-1]:
        sep_x = hi + gap / 2
        ax.plot([sep_x, sep_x], [-.05, n + .05], color="#C9CFD3", lw=1.3, clip_on=False)

    # Right-side case + scenario labels.
    label_ax.text(.02, -1.95, "Sample · ROI", va="center", fontsize=8.5)
    label_ax.text(1.05, -1.95, "Scenario", va="center", fontsize=8.5)
    modes = cases.class_label.tolist()
    begin = 0
    while begin < n:
        end = begin + 1
        while end < n and modes[end] == modes[begin]:
            end += 1
        mode = modes[begin]
        label_ax.plot([1.00, 1.00], [begin + .17, end - .17],
                      color=MODE_COLOURS[mode], linewidth=2.4)
        label_ax.text(1.05, (begin + end) / 2, NAMES[mode],
                      va="center", fontsize=8.2, color="#263A35")
        begin = end
    for i, row in enumerate(cases.itertuples()):
        label_ax.text(.02, i + .5, f"{row.source_sample_id} · {row.a_roi_id}",
                      va="center", fontsize=7.8)

    # Colourbar; keep enough space below for descenders in the label.
    cax = fig.add_axes((left / width, .090, matrix_width / width, .022))
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(-CLIP, CLIP), cmap=CMAP),
                       cax=cax, orientation="horizontal")
    bar.set_ticks([-CLIP, 0, CLIP])
    bar.set_label("Per-column z-score across the 18 cases (clipped to ±2)", fontsize=8)
    bar.ax.tick_params(labelsize=7.5, length=2)
    bar.outline.set_visible(False)
    return fig


def save(fig, output: Path, stem: str) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ("svg", "pdf", "png"):
        path = output / f"{stem}.{ext}"
        metadata = {"Date": None} if ext == "svg" else (
            {"CreationDate": None, "ModDate": None} if ext == "pdf" else None)
        fig.savefig(path, dpi=240, transparent=True, metadata=metadata)
        files.append(path)
    plt.close(fig)
    return files


def write_source_data(data: dict, source_dir: Path) -> dict:
    source_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    data["standardised"].to_csv(source_dir / "C0_standardised_matrix.csv")
    outputs["C0_standardised_matrix.csv"] = sha256(source_dir / "C0_standardised_matrix.csv")
    data["raw"].to_csv(source_dir / "C0_comparator_values.csv")
    outputs["C0_comparator_values.csv"] = sha256(source_dir / "C0_comparator_values.csv")
    data["scale"].to_csv(source_dir / "C0_scaling.csv")
    outputs["C0_scaling.csv"] = sha256(source_dir / "C0_scaling.csv")
    (pd.DataFrame([{"method": m, "source": src, "direction": d, "label": l}
                   for m, src, d, l in STATS])
     .to_csv(source_dir / "C0_column_definitions.csv", index=False))
    outputs["C0_column_definitions.csv"] = sha256(source_dir / "C0_column_definitions.csv")
    return outputs


def build(root: Path = ROOT) -> dict:
    input_path = comparator_benchmark_root() / "comparator_values.csv"
    values = load_values(input_path)
    data = build_matrix(values)
    panels_dir = HERE / "panels"
    source_dir = HERE / "source_data"
    fig = plot(data)
    panel_paths = save(fig, panels_dir, "C0_statistics_by_method")
    output_hashes = write_source_data(data, source_dir)
    output_hashes.update({p.name: sha256(p) for p in panel_paths})
    return dict(input_path=input_path, input_sha256=sha256(input_path),
                output_hashes=output_hashes, row_order=data["cases"].index.tolist())


if __name__ == "__main__":
    result = build()
    print(json.dumps({k: (str(v) if isinstance(v, Path) else v)
                      for k, v in result.items() if k != "row_order"}, indent=2))
