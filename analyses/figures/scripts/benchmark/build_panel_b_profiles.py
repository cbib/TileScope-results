"""Cluster saved measurement profiles, not prediction outcomes, for Figure 2B.

Expression modes are a predefined comparison set. Within it, clustering sees
only measurements; scenario labels annotate the resulting leaves. Native
component, ROI and interface measurements retain their different identities.
Writes the retained B0 and its supplementary grain comparison; no core runs.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from analyses.figures.scripts.benchmark import build_panel_b as v1
from analyses.scenario_discriminability.recognition import full_catalogue_clustering as full
from tilescope_results.paths import supplementary_data_root
from tilescope_results.bundles import verify

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, to_tree

RECOGNITION = Path("analyses/scenario_discriminability/recognition")
NAMES = {
    "broad_colocalization": "Broad overlap", "focal_pattern::colocated": "Colocated foci",
    "focal_pattern::separated": "Separated foci", "spatial_gradient::concordant": "Concordant gradient",
    "spatial_gradient::opposed": "Opposed gradient", "nested_foci": "Nested foci",
}
GRAINS = (("source_native", "Spots"), ("tessellation", "Tiles"),
          ("consensus_direct", "Direct consensus"), ("consensus_ensemble", "Ensemble consensus"))
MODE_COLOURS = dict(zip(NAMES, ["#9B6AA6", "#C88A43", "#B97156", "#438E87", "#577FA5", "#A58C51"]))
CMAP = LinearSegmentedColormap.from_list("profile_effect", ["#5986AC", "#F7F7F4", "#CC7864"])
TRACE_COLUMNS = ["raw_value", "value", "status", "transform", "source_file", "source_column",
                 "roi_id", "component_id", "anchor_roi_id", "responder_roi_id"]


def verified_statistics(reader):
    """Verify the frozen numerical inputs, without requiring their original run tree."""
    bundle = supplementary_data_root(1)
    verify(bundle)
    path = bundle / 'measurement_profiles'
    for name in ("cases.csv", "features.csv", "measurements.csv"):
        reader.record(path / name)
    return path


def specifications():
    rows = []
    for channel, label in (("g1", "A"), ("g2", "B"), ("joint_high", "Both\nhigh"), ("joint_low", "Both\nlow")):
        family = "joint_enrichment" if channel.startswith("joint_") else "marginal_enrichment"
        rows.append(dict(feature=f"A_ROI|source_native|enrichment|{channel}|", family=family,
                         label=label, group="Enrichment\nROI", unit="Whole A ROI", grain="source_native"))
    rows += [dict(feature="A_ROI|source_native|pearson||", family="correlation", label="ROI",
                  group="Pearson\nsame spots", unit="Whole A ROI", grain="source_native"),
             dict(feature="A_ROI|largest_component|rho_zero||", family="correlation", label="Comp.",
                  group="Pearson\nsame spots", unit="Largest retained A component", grain="source_native")]
    for direction, label in (("g1_lag_g2", "A→B"), ("g2_lag_g1", "B→A")):
        rows.append(dict(feature=f"A_ROI|source_native|moran||{direction}", family="moran", label=label,
                         group="Moran\nROI", unit="Whole A ROI", grain="source_native"))
    for band, label in enumerate(("1", "2", "3.5", "6.5")):
        rows.append(dict(feature=f"A_ROI|largest_component|shell_{band}||", family="distance", label=label,
                         group="Cross-correlation\ncomponent", unit="Largest retained A component", grain="source_native"))
    return pd.DataFrame(rows)


def grain_specifications():
    rows = []
    for grain, label in GRAINS:
        pearson = "pearson_median" if grain == "tessellation" else "pearson"
        label = label.replace(" consensus", "\nconsensus")
        for stat, direction, short in ((pearson, "", "r"), ("moran", "g1_lag_g2", "I\nA→B"),
                                       ("moran", "g2_lag_g1", "I\nB→A")):
            rows.append(dict(feature=f"A_ROI|{grain}|{stat}||{direction}", family="correlation" if stat.startswith("pearson") else "moran",
                             label=short, group=label, unit="Whole A ROI", grain=grain))
    return pd.DataFrame(rows)


def cluster_profiles(values, families, *, omit=None, only=None, balanced=True):
    """Use the full catalogue's average-linkage implementation on spot statistics.

    The twelve values are complete, so its dissimilarity is weighted Euclidean.
    No alternative Ward renderer or label-dependent selection is retained here.
    """
    columns = [c for c in values if (omit is None or families[c] != omit) and
               (only is None or families[c] == only)]
    matrix = values[columns].sort_index().astype(float)
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Missing or nonfinite clustering measurements; no imputation allowed")
    meta = pd.DataFrame([dict(feature=c, group=families[c], support="source_native") for c in columns])
    result = full.cluster(matrix, meta, method="average",
                          balancing="family_support" if balanced else "column")
    scale = result["scale"].rename(columns={"scale": "sd"})
    counts = pd.Series([families[c] for c in result["z"]]).value_counts()
    weights = pd.Series({c: (1 / np.sqrt(len(counts) * counts[families[c]]) if balanced
                            else 1 / np.sqrt(len(result["z"].columns))) if scale.loc[c, "active"] else 0.
                         for c in columns})
    scale["family_weight"] = weights
    return dict(tree=result["tree"], matrix=result["z"].mul(weights[result["z"].columns], axis=1),
                scale=scale, order=result["order"])


def branch_audit(tree, cases):
    """Post hoc annotation: smallest branch containing all cases of a mode.

    No requested cluster count or cut is fitted to the known mode labels.
    """
    _, nodes = to_tree(tree, rd=True)
    leaves = {n.id: set(n.pre_order(lambda x: x.id)) for n in nodes}
    rows = []
    for mode in sorted(cases.class_label.unique()):
        target = set(np.flatnonzero(cases.class_label.eq(mode)))
        node = min((n for n in nodes if leaves[n.id] >= target), key=lambda n: len(leaves[n.id]))
        indices = sorted(leaves[node.id])
        rows.append(dict(class_label=mode, n_mode_cases=len(target), n_branch_cases=len(indices),
                         n_other_cases=len(indices)-len(target), node_id=node.id, height=node.dist,
                         branch_case_ids=";".join(cases.iloc[indices].index)))
    return pd.DataFrame(rows)


def attach_measurements(reader, path, cases, specs):
    all_values = reader.csv(path / "features.csv").set_index("case_id")
    ledger = reader.csv(path / "measurements.csv").set_index(["case_id", "feature"])
    data = all_values.loc[cases.index, specs.feature]
    traces = []
    display = data.copy()
    for case in cases.index:
        for spec in specs.itertuples():
            trace = ledger.loc[(case, spec.feature)]
            if not np.isclose(data.loc[case, spec.feature], trace.value, rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError("Measurement and feature table disagree")
            if spec.family.endswith("enrichment"):
                if trace.status not in {"finite", "positive_over_zero_context"} or not 0 <= trace.value <= 1:
                    raise ValueError("Unavailable enrichment is not a finite ratio")
                # Existing bounded representation: E/(1+E), with +infinity = 1.
                # Centre E=1 on white. Colour labels remain enrichment ratios.
                effect = 2 * trace.value - 1
            else:
                if trace.status != "finite" or not np.isfinite(trace.value):
                    raise ValueError("Unavailable coefficient in the selected comparison")
                effect = trace.raw_value
            if not np.isfinite(effect) or abs(effect) > 1 + 1e-12:
                raise ValueError("Displayed coefficient would be clipped")
            display.loc[case, spec.feature] = effect
            traces.append(dict(case_id=case, class_label=cases.loc[case, "class_label"],
                               source_sample_id=cases.loc[case, "source_sample_id"], feature=spec.feature,
                               family=spec.family, unit=spec.unit, grain=spec.grain, display_value=effect,
                               **{k: trace[k] for k in TRACE_COLUMNS}))
    return data, display, pd.DataFrame(traces)


def prepare(root=ROOT):
    reader = v1.ExportReader(root)
    path = verified_statistics(reader)
    cases = reader.csv(path / "cases.csv").set_index("case_id")
    cases = cases.loc[cases.class_label.isin(NAMES)].sort_index()
    if len(cases) != 18 or not cases.groupby("class_label").size().eq(3).all():
        raise ValueError("Expected every positive case of all six expression modes")
    specs = specifications()
    numeric, display, traces = attach_measurements(reader, path, cases, specs)
    families = specs.set_index("feature").family.to_dict()
    primary = cluster_profiles(numeric, families)
    audits, variants = [], {}
    settings = [("primary", {})] + [("without_"+f, dict(omit=f)) for f in dict.fromkeys(families.values())]
    settings += [("only_"+f, dict(only=f)) for f in dict.fromkeys(families.values())]
    settings += [("equal_column_weights", dict(balanced=False))]
    for name, kwargs in settings:
        result = primary if name == "primary" else cluster_profiles(numeric, families, **kwargs)
        audit = branch_audit(result["tree"], cases)
        audit["variant"] = name
        audits.append(audit)
        variants[name] = dict(order=result["order"], linkage=result["tree"].tolist())
    grain_specs = grain_specifications()
    _, grain_display, grain_traces = attach_measurements(reader, path, cases, grain_specs)
    return dict(cases=cases, specs=specs, numeric=numeric, display=display, traces=traces,
                primary=primary, audits=pd.concat(audits, ignore_index=True), variants=variants,
                grain_specs=grain_specs, grain_display=grain_display, grain_traces=grain_traces,
                source_sha256=reader.hashes)


def plot_profiles(data, *, grains=False):
    specs = data["grain_specs"] if grains else data["specs"]
    values = data["grain_display"] if grains else data["display"]
    order = data["primary"]["order"]
    cases = data["cases"].loc[order]
    n = len(order)
    width = 7.8
    fig = plt.figure(figsize=(width, 5.6))
    # Keep cell sizes fixed in inches while tightening the label columns.
    # The sample/ROI is next to its row; the grouped scenario is farther right.
    if not grains:
        tree_ax = fig.add_axes((.17/width, .21, .7225/width, .65))
        d = dendrogram(data["primary"]["tree"], orientation="left", no_labels=True, ax=tree_ax,
                       color_threshold=0, above_threshold_color="#76807D")
        if [data["numeric"].index[i] for i in d["leaves"]] != order:
            raise ValueError("Dendrogram and heatmap order differ")
        tree_ax.set_ylim(n*10, 0)
        tree_ax.axis("off")
        fig.text(.17/width, .94, "Grouping", fontsize=8)
    ax = fig.add_axes((1.02/width, .21, 3.9525/width, .65), label="profile_matrix")
    label_ax = fig.add_axes((5.10/width, .21, 2.60/width, .65), label="profile_labels")
    ax.axis("off")
    label_ax.axis("off")
    label_ax.set(xlim=(0, 2.60), ylim=(n, 0))
    x, positions, group_centres = 0., [], []
    for group, sub in specs.groupby("group", sort=False):
        if positions:
            x += .36
        start = x
        for spec in sub.itertuples():
            positions.append((x, spec))
            x += 1
        group_centres.append(((start+x)/2, group))
    ax.set(xlim=(0, x), ylim=(n, 0))
    for centre, group in group_centres:
        ax.text(centre, -2.0, group, ha="center", va="center", fontsize=8.1, clip_on=False)
    for x, spec in positions:
        ax.text(x+.5, -.62, spec.label, ha="center", va="center", fontsize=7.4, clip_on=False)
        for i, case in enumerate(order):
            value = values.loc[case, spec.feature]
            cell = Rectangle((x, i), 1, 1, facecolor=CMAP((value+1)/2), edgecolor="white", linewidth=.55)
            cell.set_gid(f"profile:{case}:{spec.feature}")
            ax.add_patch(cell)
            if spec.family.endswith("enrichment"):
                status = data["traces"].set_index(["case_id", "feature"]).loc[(case, spec.feature), "status"]
                if status == "positive_over_zero_context":
                    ax.text(x+.5, i+.5, "∞", ha="center", va="center", fontsize=7.5, color="#49372D")
    label_ax.text(.02, -2.0, "Sample · ROI", va="center", fontsize=8.1)
    label_ax.text(1.08, -2.0, "Scenario", va="center", fontsize=8.1)
    modes = cases.class_label.tolist()
    begin = 0
    while begin < n:
        end = begin+1
        while end < n and modes[end] == modes[begin]:
            end += 1
        mode = modes[begin]
        label_ax.plot([1.035, 1.035], [begin+.17, end-.17], color=MODE_COLOURS[mode], linewidth=2.2)
        label_ax.text(1.08, (begin+end)/2, NAMES[mode], va="center", fontsize=8, color="#263A35")
        begin = end
    for i, row in enumerate(cases.itertuples()):
        label_ax.text(.02, i+.5, f"{row.source_sample_id} · {row.a_roi_id}", va="center", fontsize=7.5)
    if grains:
        fig.text(1.02/width, .985, "Whole-ROI measurements at each grain", fontsize=9, va="top")
        cax = fig.add_axes((1.70/width, .092, 2.295/width, .024))
    else:
        fig.text(1.02/width, .985, "Expression measurements at spots", fontsize=9, va="top")
        cax = fig.add_axes((3.315/width, .092, 1.6575/width, .024))
        ebar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(-1, 1), cmap=CMAP),
                            cax=fig.add_axes((1.02/width, .092, 1.7425/width, .024)), orientation="horizontal")
        ebar.set_ticks([-1, -.5, 0, .5, 1], labels=["0", "⅓", "1", "3", "∞"])
        ebar.set_label("Enrichment ratio", fontsize=8)
        ebar.ax.tick_params(labelsize=7, length=2)
        ebar.outline.set_visible(False)
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(-1, 1), cmap=CMAP), cax=cax,
                      orientation="horizontal")
    bar.set_ticks([-1, 0, 1])
    bar.set_label("Pearson / Moran / cross-correlation" if not grains else "Pearson r / Moran I", fontsize=7.7)
    bar.ax.tick_params(labelsize=7, length=2)
    bar.outline.set_visible(False)
    return fig
