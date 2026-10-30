"""Render Figure 2B from existing core exports; never run a TileScope analyzer.

One main distance plot and two supplementary boundary heatmaps. Every case of
the selected modes is included. Component selection uses size, not a statistic.
"""
from __future__ import annotations
from tilescope_results.paths import figure2_root

import hashlib
import json
import os
from pathlib import Path
from tilescope_results.paths import input_data_root, reference, resolve_reference

ROOT = Path(__file__).resolve().parents[4]
CODE = Path(__file__).resolve().parent
HERE = figure2_root()
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

CORE_SHA = "ef8ab0df1332bda822a2cf3240fc856c466ed534"
FIELD_MODES = ("broad", "colocated", "concordant")
BOUNDARY_MODES = ("responder_activation", "anchor_fade")
LABELS = {
    "broad": "Broad\noverlap", "colocated": "Colocated\nfoci",
    "concordant": "Concordant\ngradients",
    "responder_activation": "Responder\nexpression increase",
    "anchor_fade": "Expression extending\nacross a boundary",
}
COLOURS = {"broad": "#9B6AA6", "colocated": "#CE873A", "concordant": "#438E87",
           "A": "#467AA2", "B": "#CA7556"}
HEATMAP_LIMITS = {"pearson": (-1.0, 1.0), "response_A": (-0.6, 0.6),
                  "response_B": (-0.6, 0.6)}
LEGACY_STEMS = ("B1_expression_correlation", "B2_cross_correlation",
                "B3_responder_correlation", "B4_boundary_response", "B2_boundary_heatmap")
ID_COLUMNS = {k: str for k in ("sample_id", "roi_id", "component_id", "anchor_roi_id",
                              "responder_roi_id", "responder_feature_id", "target_roi_id",
                              "primary_component_id", "source_sample_id", "core_sample_id")}


def corpus_root():
    return input_data_root() / 'benchmark/corpus'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExportReader:
    def __init__(self, root: Path):
        self.root = root
        self.hashes: dict[str, str] = {}

    def record(self, path: Path, expected: str | None = None) -> Path:
        path = resolve_reference(path, repository=self.root)
        value = sha256(path)
        if expected is not None and value != expected:
            raise ValueError(f"Hash mismatch: {path}")
        self.hashes[reference(path, repository=self.root)] = value
        return path

    def json(self, path: Path) -> dict:
        return json.loads(self.record(path).read_text())

    def csv(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(self.record(path), dtype=ID_COLUMNS, float_precision="round_trip")


def prepare(root: Path = ROOT) -> dict:
    reader = ExportReader(root)
    from tilescope_results.paths import supplementary_data_root
    from tilescope_results.bundles import verify
    bundle = supplementary_data_root(1)
    verify(bundle)
    tables = {name: reader.csv(bundle / 'distance_and_boundary_controls' / f'{name}.csv.gz')
              for name in ('radial', 'correlations', 'interfaces', 'membership', 'inventory')}
    n_runs = len(tables['inventory'])
    counts = tables["inventory"].drop_duplicates("case_id").scenario_mode.value_counts()
    if set(counts.index) != set(FIELD_MODES + BOUNDARY_MODES) or not counts.eq(3).all() or n_runs != 33:
        raise ValueError("Expected every case of five modes: 15 cases, 33 case/control runs")
    return {"tables": tables, "source_sha256": reader.hashes, "n_runs": n_runs}


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.linewidth": 0.65, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "svg.hashsalt": "tilescope-figure2-panel-b"})


def combined_curve_values(tables: dict) -> pd.DataFrame:
    """Append actual same-component Pearson, not a fabricated distance band."""
    f = tables["radial"].loc[lambda x: x.primary_display]
    rows = []
    for _, group in f.groupby(["case_id", "role"], sort=True):
        group = group.sort_values("shell_bin_index")
        first = group.iloc[0]
        meta = {k: first[k] for k in ("case_id", "role", "scenario_mode", "condition",
                                      "source_sample_id", "roi_id", "component_id", "n_component_spots")}
        if group.component_rho_0.nunique(dropna=False) != 1:
            raise ValueError("Inconsistent zero-distance coefficient within a component")
        rho = first.component_rho_0
        rows.append({**meta, "distance": 0.0, "point_type": "same_spot_pearson", "value": rho,
                     "availability_status": "finite" if np.isfinite(rho) else "unavailable"})
        for row in group.itertuples():
            rows.append({**meta, "distance": row.shell_midpoint_normalized,
                         "point_type": "distance_band", "value": row.shell_symmetric,
                         "availability_status": row.component_radial_shell_status})
    return pd.DataFrame(rows)


def plot_distance(tables: dict):
    fig, ax = plt.subplots(figsize=(4.5, 3.25))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.25, top=0.74)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.6, length=3)
    ax.axhline(0, color="#B5BBBF", lw=0.7, zorder=0)
    ax.axvspan(-.27, .3, facecolor="#E7E9EA", alpha=.45, zorder=0)
    f = combined_curve_values(tables)
    for (mode, case, cond), group in f.groupby(["scenario_mode", "case_id", "condition"], sort=True):
        group = group.sort_values("distance")
        positive = cond == "pattern"
        line, = ax.plot(group.distance, group.value,
                        color=COLOURS[mode], marker="o", markersize=3.4 if positive else 2.6,
                        markerfacecolor=COLOURS[mode] if positive else "white",
                        lw=1.15 if positive else 0.65, linestyle="-" if positive else "--",
                        alpha=0.9 if positive else 0.5, zorder=3 if positive else 1)
        line.set_gid(f"curve:{case}:{cond}")
    ax.set(xlim=(-0.3, 6.85), xticks=[0, 1, 2, 3.5, 6.5],
           xticklabels=["0\nPearson r", "1", "2", "3.5", "6.5"],
           ylim=(-0.23, 1.08), yticks=[0, .5, 1],
           ylabel="Cross-correlation", xlabel="Distance / median neighbour spacing")
    ax.xaxis.labelpad = 7
    handles = [Line2D([], [], color=COLOURS[m], lw=1.5, label=LABELS[m].replace("\n", " "))
               for m in FIELD_MODES]
    legend = ax.legend(handles=handles, frameon=False, loc="lower left", bbox_to_anchor=(-0.015, 1.06),
                       fontsize=7.5, handlelength=1.4, borderaxespad=0, labelspacing=0.3)
    ax.add_artist(legend)
    styles = [Line2D([], [], color="#656D72", lw=1, marker="o", markersize=3.5, label="Pattern"),
              Line2D([], [], color="#656D72", lw=.7, linestyle="--", marker="o",
                     markerfacecolor="white", markersize=3.5, label="Matched control")]
    ax.legend(handles=styles, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.06),
              fontsize=7.5, handlelength=1.6, borderaxespad=0, labelspacing=0.4)
    return fig


def boundary_heatmap_values(tables: dict) -> pd.DataFrame:
    """Join actual designated interfaces to their own responder's correlation."""
    f = tables["interfaces"]
    f = f.loc[f.surface.eq("source_native") & f.direction.eq("designated")]
    c = tables["correlations"]
    c = c.loc[c.surface.eq("source_native") & c.scenario_mode.isin(BOUNDARY_MODES)]
    rows = []
    for (case, role), group in f.groupby(["case_id", "role"], sort=True):
        if set(group.responder_feature_id) != {"A", "B"} or len(group) != 2:
            raise ValueError(f"Expected both genes for {case}, {role}")
        first = group.iloc[0]
        meta = {k: first[k] for k in ("case_id", "role", "scenario_mode", "condition",
                                      "source_sample_id", "anchor_roi_id", "responder_roi_id")}
        corr = c.loc[c.case_id.eq(case) & c.role.eq(role) & c.roi_id.eq(first.responder_roi_id)]
        if len(corr) != 1:
            raise ValueError(f"Ambiguous or missing responder correlation: {case}, {role}")
        row = corr.iloc[0]
        rows.append({**meta, "statistic": "pearson", "value": row.correlation_value,
                     "availability_status": row.availability_status})
        for row in group.itertuples():
            rows.append({**meta, "statistic": f"response_{row.responder_feature_id}",
                         "value": row.interface_response, "availability_status": row.availability_status})
    return pd.DataFrame(rows)


def heatmap_cell(value: float, status: str, statistic: str) -> tuple:
    """Absolute scales, centred at zero; unavailable is not a neutral value."""
    if status != "finite" or not np.isfinite(value):
        return "#DFE2E4", "NA", "//"
    low, high = HEATMAP_LIMITS[statistic]
    if not low <= value <= high:
        raise ValueError(f"Heatmap would clip {statistic}={value}; revise the documented scale")
    cmap = LinearSegmentedColormap.from_list("signed_measurement", ["#5584AA", "#FAF9F6", "#CB6D56"])
    text = "0.00" if abs(value) < .005 else f"{value:.2f}".replace("-", "−")
    return cmap(Normalize(low, high)(value)), text, None


def plot_boundary_heatmap(tables: dict, *, relocated: bool = False):
    values = boundary_heatmap_values(tables)
    modes = ("responder_activation",) if relocated else BOUNDARY_MODES
    conditions = ("pattern", "matched_control", "relocated_control") if relocated else (
        "pattern", "matched_control")
    headings = {"pattern": "Pattern", "matched_control": "Boundary term removed",
                "relocated_control": "Added expression relocated"}
    fig = plt.figure(figsize=(8.0, 2.9) if relocated else (6.7, 3.9))
    ax = fig.add_axes([.31, .36, .68, .36] if relocated else [.365, .30, .625, .48])
    statistics = ("pearson", "response_A", "response_B")
    xlocal = (0, 1.2, 2.2)
    nrows = 3 if relocated else 6.7
    ax.set(xlim=(-.5, (len(conditions) - 1) * 3.9 + 2.7), ylim=(nrows - .5, -.5))
    ax.set_axis_off()
    lookup = values.set_index(["case_id", "condition", "statistic"], verify_integrity=True)
    for group_index, mode in enumerate(modes):
        cases = values.loc[values.scenario_mode.eq(mode)].drop_duplicates("case_id").sort_values("case_id")
        y0 = group_index * 3.7
        left = (.02 - ax.get_position().x0) / ax.get_position().width
        bracket = ((.19 if relocated else .215) - ax.get_position().x0) / ax.get_position().width
        ax.text(left, y0 + 1, LABELS[mode], ha="left", va="center", transform=ax.get_yaxis_transform(),
                fontsize=8, clip_on=False)
        ax.plot([bracket, bracket], [y0 - .4, y0 + 2.4], color="#B8BEC2", lw=.7,
                transform=ax.get_yaxis_transform(), clip_on=False)
        for offset, case in enumerate(cases.itertuples()):
            y = y0 + offset
            label = f"{case.source_sample_id}  {case.anchor_roi_id}→{case.responder_roi_id}"
            ax.text(-.7, y, label, ha="right", va="center", fontsize=7.4, clip_on=False)
            for condition_index, cond in enumerate(conditions):
                for j, statistic in enumerate(statistics):
                    row = lookup.loc[(case.case_id, cond, statistic)]
                    colour, text, hatch = heatmap_cell(row.value, row.availability_status, statistic)
                    x = condition_index * 3.9 + xlocal[j]
                    cell = Rectangle((x - .46, y - .44), .92, .88, facecolor=colour,
                                     edgecolor="#E2E3E3", lw=.5, hatch=hatch)
                    cell.set_gid(f"heat:{case.case_id}:{cond}:{statistic}")
                    ax.add_patch(cell)
                    ax.text(x, y, text, ha="center", va="center", fontsize=7.8, color="#273138")
    for i, cond in enumerate(conditions):
        ax.text(i * 3.9 + 1.1, -2.2, headings[cond], ha="center", va="center", fontsize=8.3,
                clip_on=False)
        ax.plot([i * 3.9 - .44, i * 3.9 + 2.64], [-1.93, -1.93], color="#9EA8AD", lw=.6,
                clip_on=False)
        ax.text(i * 3.9, -1.2, "ROI\nPearson r", ha="center", va="center", fontsize=7.4, clip_on=False)
        ax.text(i * 3.9 + 1.7, -1.48, "Near − far", ha="center", va="center", fontsize=7.4, clip_on=False)
        for x, label in zip(xlocal[1:], ("Gene A", "Gene B")):
            ax.text(i * 3.9 + x, -.92, label, ha="center", va="center", fontsize=7.4, clip_on=False)
    cmap = LinearSegmentedColormap.from_list("signed_measurement", ["#5584AA", "#FAF9F6", "#CB6D56"])
    for i, (limits, label) in enumerate(((HEATMAP_LIMITS["pearson"], "Pearson r"),
                                       (HEATMAP_LIMITS["response_A"], "Mean expression: near − far"))):
        cax = fig.add_axes([.38 + .32 * i, .15, .22, .028] if not relocated else
                           [.36 + .32 * i, .17, .22, .035])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(*limits), cmap=cmap), cax=cax,
                          orientation="horizontal", ticks=[limits[0], 0, limits[1]])
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=2, width=.5, pad=2, labelsize=7)
        cb.set_label(label, fontsize=7.5, labelpad=3)
    return fig


def plot_relocated_controls(tables: dict):
    return plot_boundary_heatmap(tables, relocated=True)




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
