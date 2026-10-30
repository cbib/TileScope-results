"""Five separate Figure 1C/D graphics from the saved illustrative TileScope run.

No new analysis run. Map masks use core threshold/graph definitions and must
reproduce exported counts and values before any artwork is written.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analyses.figures.scripts.overview.build import expression_colormap
from analyses.figures.scripts.overview.plot_panel_b import load_saved_run, save_artwork

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgba
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge, Rectangle
import numpy as np
import pandas as pd
from shapely import wkt

from tilescope.metrics.enrichment_thresholds import compute_global_thresholds
from tilescope.spatial.source_spot_table import build_source_spot_table_from_frame
from tilescope.spatial.source_spot_adjacency import build_source_spot_adjacency, UNREACHABLE_HOP
from tilescope.analyzers.parent_roi_interface_response import NEAR_HOP_MAX, BUFFER_HOP, FAR_HOP_MIN
from tilescope.viz.painters import draw_polygon_boundaries
from tilescope.viz.geometry_helpers import patch_from_polygon

INK = "#343C43"
FAINT = "#E4E7E7"
GENE_A = "#B18432"
GENE_B = "#34898D"
JOINT = "#85569A"
MAIN = "#B46155"
DETACHED = "#426E96"
NEAR = "#C86C51"
FAR = "#568EB0"
BUFFER = "#C3C9CC"
# Ordered distance only: independent of expression magnitude or effect sign.
DISTANCE_BAND_COLORS = ("#DFEFD8", "#BCDEB9", "#94C9A3", "#66AA8A")
DISTANCE_BAND_ALPHA = 0.80
AGGREGATION_ROI_ID = 3


def one(frame, **conditions):
    selected = frame
    for column, value in conditions.items():
        selected = selected.loc[selected[column].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {conditions}, found {len(selected)}")
    return selected.iloc[0]


def prepare_evidence():
    folder, provenance, frames = load_saved_run()
    def read(name):
        return pd.read_csv(folder / (name + '.csv'), float_precision='round_trip')
    source = frames['source'].copy()
    measurements = read('spot_measurements')
    np.testing.assert_array_equal(measurements.spot_index, np.arange(len(source)))
    source = pd.concat([source, measurements], axis=1)
    thresholds = compute_global_thresholds(source.gene1.to_numpy(), source.gene2.to_numpy(),
                                           q_percentile=.75, q_low_percentile=.25)
    np.testing.assert_array_equal(source.a_high, source.gene1 > thresholds.q_g1_threshold)
    np.testing.assert_array_equal(source.b_high, source.gene2 > thresholds.q_g2_threshold)
    np.testing.assert_array_equal(source.both_high, source.a_high & source.b_high)
    enrichment = read('enrichment')
    for roi in (1, 2, 3):
        row = one(enrichment, roi_id=roi)
        inside = source.roi_id.eq(roi)
        for mask, count, ratio in [('a_high', 'n_g1_high', 'g1_enrichment'),
                                  ('b_high', 'n_g2_high', 'g2_enrichment'),
                                  ('both_high', 'n_both_high', 'joint_high_enrichment_fc')]:
            assert int(source.loc[inside, mask].sum()) == int(row[count])
            np.testing.assert_allclose(source.loc[inside, mask].mean() / source.loc[~inside, mask].mean(),
                                       row[ratio], rtol=1e-12, atol=1e-12)
    components = read('component_correlations')
    for row in components.itertuples():
        subset = source.loc[source.component_id.eq(row.component_id)]
        assert len(subset) == row.n_spots
        np.testing.assert_allclose(np.corrcoef(subset.gene1, subset.gene2)[0, 1], row.pearson, atol=1e-12)
    graph = build_source_spot_adjacency(build_source_spot_table_from_frame(source), technology='visium')
    _, seeds = graph.interface_endpoints('1', '2')
    hops = graph.hop_distance_within_roi('2', seeds)
    np.testing.assert_array_equal(hops, source.hops_from_roi1_contact)
    boundary = read('boundary')
    near, far = source.interface_band.eq('Near'), source.interface_band.eq('Far')
    for row, column in zip(boundary.itertuples(), ['gene1', 'gene2']):
        assert (int(near.sum()), int(far.sum())) == (row.n_near, row.n_far)
        np.testing.assert_allclose(source.loc[near, column].mean() - source.loc[far, column].mean(),
                                   row.response, atol=1e-12)
    reps = read('replicate_correlations')
    np.testing.assert_array_equal(reps.replicate_index, np.arange(10))
    src, ce = read('source_correlation').iloc[0], read('consensus_correlation').iloc[0]
    for frame, columns, expected in [
        (source.loc[source.roi_id.eq(AGGREGATION_ROI_ID)], ['gene1', 'gene2'], src.corr_obs),
        (frames['hexes'].loc[frames['hexes'].roi_id.eq(str(AGGREGATION_ROI_ID))],
         ['gene1_density_ensemble', 'gene2_density_ensemble'], ce.correlation_value),
    ]:
        np.testing.assert_allclose(np.corrcoef(frame[columns].to_numpy().T)[0, 1], expected, atol=1e-12)
    evidence = frames.copy()
    evidence.update(source=source, graph=graph, components=components, boundary=boundary,
                    enrichment=enrichment, display_enrichment=enrichment, moran=read('moran'),
                    distance=read('distance'), aggregation_replicates=reps,
                    aggregation_source=src, aggregation_consensus=ce)
    validation = dict(high_thresholds=thresholds.to_dict(), high_operator='>',
        enrichment_counts_and_ratios_match_exports=True, display_enrichment_surface='source_native',
        component_scatter_correlations_match_exports=True, source_bands_and_mean_differences_match_exports=True,
        near_hops_max=NEAR_HOP_MAX, buffer_hop=BUFFER_HOP, far_hops_min=FAR_HOP_MIN,
        near_count=int(near.sum()), far_count=int(far.sum()),
        buffer_count=int(source.interface_band.eq('Buffer').sum()),
        aggregation_roi_id=AGGREGATION_ROI_ID, aggregation_source_spots=int(src.n_roi),
        aggregation_consensus_hexes=int(ce.sample_size),
        all_aggregation_replicates_finite_and_full_spot_coverage=bool(
            reps.parent_replicate_correlation_status.eq('finite').all()
            and reps.source_roi_spot_coverage_fraction.eq(1).all()),
        original_full_run_available=False)
    return folder, provenance, evidence, validation


def figure(title, *, strip=False):
    fig = plt.figure(figsize=(4.6, 2.65) if strip else (6.4, 3.8), facecolor="none")
    fig.text(0.5, 0.94, title, ha="center", va="center", fontsize=12, color=INK)
    return fig


def clean_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#ABB2B5")
    ax.tick_params(color="#ABB2B5", labelcolor=INK, length=3, labelsize=8)
    ax.set_facecolor("none")


def tissue_axis(fig, rect, evidence, *, labels=False):
    ax = fig.add_axes(rect, facecolor="none")
    ax.set(xlim=(-15.5, 15.5), ylim=(-10.5, 10.5), aspect="equal")
    ax.axis("off")
    polys = {row.component_id: wkt.loads(row.geometry_wkt)
             for row in evidence["geometry"].itertuples()}
    draw_polygon_boundaries(ax, polys, color="#6D777C", linewidth=0.55)
    if labels:
        for row in evidence["geometry"].itertuples():
            p = polys[row.component_id].representative_point()
            ax.text(p.x, p.y, f"ROI {row.roi_id}", ha="center", va="center", fontsize=7.5,
                    zorder=5, bbox=dict(facecolor="white", alpha=0.90, edgecolor="none", pad=1.5))
    return ax


def dots(ax, source, mask, color, *, size=10, zorder=2):
    ax.scatter(source.loc[mask, "x"], source.loc[mask, "y"], s=size,
               c=color, edgecolors="none", zorder=zorder)


def key_handle(color, label, marker="o"):
    return Line2D([], [], marker=marker, linestyle="none", markersize=5,
                  markerfacecolor=color, markeredgecolor="none", label=label)


def enrichment_plot(e):
    fig = figure("Enrichment")
    source = e["source"]
    ax = tissue_axis(fig, [0.015, 0.28, 0.48, 0.56], e, labels=True)
    palette = {"Neither high": FAINT, "A only": GENE_A, "B only": GENE_B, "Both high": JOINT}
    for state, color in palette.items():
        dots(ax, source, source.high_state.eq(state), color, size=9.5)
    fig.legend(handles=[key_handle(palette[s], s) for s in ("A only", "B only", "Both high", "Neither high")],
               loc="lower left", bbox_to_anchor=(0.05, 0.09), ncol=2, fontsize=8,
               frameon=False, columnspacing=1.1, handletextpad=0.3)

    ax = fig.add_axes([0.60, 0.30, 0.35, 0.51])
    values = e["display_enrichment"].set_index("roi_id").loc[[1, 2, 3],
             ["g1_enrichment", "g2_enrichment", "joint_high_enrichment_fc"]].to_numpy()
    log_values = np.full_like(values, -4.)
    np.log2(values, out=log_values, where=values > 0)
    cmap = expression_colormap()
    norm = Normalize(-4, 4, clip=True)
    ax.imshow(log_values, cmap=cmap, norm=norm, aspect="auto")
    for y, x in np.ndindex(values.shape):
        value = values[y, x]
        label = "0" if value == 0 else (f"{value:.2f}" if value < 10 else f"{value:.1f}")
        ax.text(x, y, label, ha="center", va="center", fontsize=10,
                color="white" if abs(log_values[y, x]) >= 3 else INK)
    ax.set_xticks([0, 1, 2], ["A high", "B high", "Both high"], fontsize=8)
    ax.set_yticks([0, 1, 2], ["ROI 1", "ROI 2", "ROI 3"], fontsize=8)
    ax.tick_params(length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Enrichment (fold)", fontsize=9, pad=10)
    cax = fig.add_axes([0.61, 0.12, 0.33, 0.03])
    key = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal", ticks=[-4, 0, 4])
    key.ax.set_xticklabels(["≤1/16×", "1×", "≥16×"])
    key.ax.tick_params(length=0, labelsize=7, pad=4)
    key.outline.set_visible(False)
    return fig


def association_plot(e):
    fig = figure("Expression correlation")
    for x, row, color in zip((0.095, 0.58), e["components"].itertuples(), (MAIN, DETACHED)):
        subset = e["source"].loc[e["source"].component_id.eq(row.component_id)]
        ax = fig.add_axes([x, 0.40, 0.34, 0.39])
        ax.scatter(subset.gene1, subset.gene2, s=12, c=color, alpha=0.70, edgecolors="none")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Gene A expression", ylabel="Gene B expression")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(f"ROI 1 · {row.name.lower()}\nComponent {row.component_id} · {row.n_spots} spots", fontsize=8, pad=8)
        ax.text(0.04, 0.91, f"r = {row.pearson:+.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color=INK)
        clean_axis(ax)

    # A separate, explicitly parent-level inset: not component Moran values.
    ax = fig.add_axes([0.08, 0.055, 0.84, 0.17])
    ax.axis("off")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.text(0.5, 0.99, "Spatial association between neighbouring spots · whole ROI 1",
            ha="center", va="top", fontsize=9)
    for left, direction, centre, neighbor, center_color, neighbor_color in (
        (0.04, "g1_lag_g2", "A", "B", GENE_A, GENE_B),
        (0.56, "g2_lag_g1", "B", "A", GENE_B, GENE_A),
    ):
        row = one(e["moran"], roi_id=1, moran_direction=direction)
        # Explanatory graph glyph, not three selected data edges or observations.
        for y in (0.27, 0.42, 0.57):
            ax.plot([left + 0.02, left + 0.12], [0.42, y], lw=0.7, c="#A0A8AD")
            ax.scatter([left + 0.12], [y], s=15, c=neighbor_color, zorder=3)
        ax.scatter([left + 0.02], [0.42], s=28, c=center_color, zorder=3)
        ax.text(left + 0.02, 0.59, centre, fontsize=7, ha="center", va="bottom")
        ax.text(left + 0.17, 0.42, f"{neighbor} neighbours", fontsize=8, va="center")
        ax.text(left, -0.03, f"Bivariate Moran’s I = {row.moran_value:+.2f}", fontsize=8, va="center")
    return fig


def distance_illustration(e):
    """Actual source anchor/endpoints and actual exported short/long bands.

    Select a geometric central spot, one immediate neighbour towards the
    right, and a distant spot in another direction. Expression is not used
    in selection. This is one geometric illustration, not the estimator's
    complete anchor/pair set.
    """
    subset = e["source"].loc[e["source"].component_id.eq(0)]
    rows = e["distance"].loc[e["distance"].component_id.eq(0)].sort_values("shell_bin_index")
    xy = subset[["x", "y"]].to_numpy()
    origin_idx = int(np.argmin(np.sum((xy - xy.mean(axis=0)) ** 2, axis=1)))
    spacing = float(rows.median_nn_spacing.iloc[0])
    vectors = xy - xy[origin_idx]
    distances = np.linalg.norm(vectors, axis=1) / spacing
    short_candidates = np.flatnonzero(np.isclose(distances, 1.0, rtol=1e-8, atol=1e-8))
    if not len(short_candidates):
        raise ValueError("No immediate neighbour for the distance illustration")
    short_idx = int(short_candidates[np.argmax(vectors[short_candidates, 0])])
    directions = np.divide(vectors, np.linalg.norm(vectors, axis=1)[:, None],
                           out=np.zeros_like(vectors), where=(distances > 0)[:, None])
    long_band = rows.iloc[-1]
    # Require at least 60 degrees of separation from the short spoke.
    long_candidates = np.flatnonzero(
        (distances >= long_band.shell_edge_lower_normalized)
        & (distances < long_band.shell_edge_upper_normalized)
        & (directions @ directions[short_idx] <= 0.5)
    )
    if not len(long_candidates):
        raise ValueError("No long-band endpoint in a distinct direction")
    preference = np.array([-0.5, np.sqrt(3) / 2])
    scores = np.abs(distances[long_candidates] - long_band.shell_midpoint_normalized)
    scores += 0.2 * (1 - directions[long_candidates] @ preference)
    long_idx = int(long_candidates[np.argmin(scores)])
    selections = []
    for label, idx, band in (("short", short_idx, rows.iloc[0]), ("long", long_idx, long_band)):
        lower, upper = float(band.shell_edge_lower_normalized), float(band.shell_edge_upper_normalized)
        if not lower <= distances[idx] < upper:
            raise ValueError("Illustrated endpoint is outside its exported distance band")
        selections.append({"label": label, "spot_index": int(subset.spot_index.iloc[idx]),
                           "point": xy[idx].tolist(), "separation_normalized": float(distances[idx]),
                           "shell_bin_index": int(band.shell_bin_index), "band_edges": [lower, upper]})
    bands = [{"shell_bin_index": int(row.shell_bin_index),
              "band_edges": [float(row.shell_edge_lower_normalized), float(row.shell_edge_upper_normalized)],
              "color": color, "alpha": DISTANCE_BAND_ALPHA, "n_anchors": int(row.n_anchors_per_shell),
              "n_unordered_pairs": int(row.n_unordered_pairs_per_shell)}
             for row, color in zip(rows.itertuples(), DISTANCE_BAND_COLORS)]
    if len(rows) != len(DISTANCE_BAND_COLORS):
        raise ValueError("The illustration requires four exported distance bands")
    return {"component_id": 0, "anchor_spot_index": int(subset.spot_index.iloc[origin_idx]),
            "anchor_point": xy[origin_idx].tolist(), "median_nn_spacing": spacing,
            "selections": selections, "bands": bands}


def distance_plot(e):
    fig = figure("Cross-correlation over distance")
    source = e["source"]
    rows = e["distance"].loc[e["distance"].component_id.eq(0)].sort_values("shell_bin_index")
    assert rows.component_radial_shell_status.eq("finite").all()
    illustration = distance_illustration(e)
    centre = np.asarray(illustration["anchor_point"])
    spacing = illustration["median_nn_spacing"]
    radius = float(rows.shell_edge_upper_normalized.max()) * spacing
    ax = fig.add_axes([0.015, 0.24, 0.45, 0.61], facecolor="none")
    extent = radius + 0.6 * spacing
    ax.set(xlim=(centre[0] - extent, centre[0] + extent),
           ylim=(centre[1] - extent, centre[1] + extent), aspect="equal")
    ax.axis("off")

    polygons = {row.component_id: wkt.loads(row.geometry_wkt) for row in e["geometry"].itertuples()}
    draw_polygon_boundaries(ax, polygons, color="#E0E4E5", linewidth=0.5)
    component = polygons["0"]
    clip = patch_from_polygon(component, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    dots(ax, source, ~source.component_id.eq(0), "#E9ECEC", size=5, zorder=2)
    dots(ax, source, source.component_id.eq(0), "#596660", size=7, zorder=3)
    for band in illustration["bands"]:
        color = band["color"]
        lower, upper = np.asarray(band["band_edges"]) * spacing
        fill = Wedge(centre, upper, 0, 360, width=upper - lower,
                     facecolor=to_rgba(color, DISTANCE_BAND_ALPHA), edgecolor="none", zorder=1)
        fill.set_clip_path(clip)
        fill.set_gid(f"distance_band_{band['shell_bin_index']}")
        ax.add_patch(fill)
        # Four outer circles delimit four bands. The unfilled central radius
        # below 0.5 is excluded from the first band, not an extra fifth band.
        ring = Circle(centre, upper, facecolor="none", edgecolor="#79927F",
                      lw=0.65, ls="--", zorder=2)
        ring.set_gid(f"distance_band_outer_edge_{band['shell_bin_index']}")
        ax.add_patch(ring)
    draw_polygon_boundaries(ax, {"0": component}, color="#758188", linewidth=0.75)
    # Draw the long spoke first, then short; neither endpoint overwrites the anchor.
    for band, color, style in ((illustration["selections"][1], DISTANCE_BAND_COLORS[-1], "--"),
                               (illustration["selections"][0], DISTANCE_BAND_COLORS[0], "-")):
        endpoint = np.asarray(band["point"])
        line, = ax.plot([centre[0], endpoint[0]], [centre[1], endpoint[1]],
                       color=INK, lw=1.2, ls=style, zorder=6)
        line.set_gid(f"{band['label']}_separation")
        ax.scatter([endpoint[0]], [endpoint[1]], s=24, facecolors=[to_rgba(color, DISTANCE_BAND_ALPHA)],
                   edgecolors=INK, linewidths=0.9, zorder=7)
    ax.scatter([centre[0]], [centre[1]], s=16, c=INK, zorder=8)
    ax.annotate("Reference spot", xy=centre, xytext=(centre[0] - 1.0, centre[1] - 3.3),
                ha="center", va="center", fontsize=7, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.5), zorder=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    fig.text(0.24, 0.17, "ROI 1 · main component\nComponent 0 · 149 spots", ha="center", fontsize=8)
    # An ordinal key: exact band limits remain in the separate figure legend.
    centres = np.array([0.14, 0.205, 0.27, 0.335])
    key_y, box_width = 0.078, 0.025
    box_height = box_width * fig.get_figwidth() / fig.get_figheight()
    fig.text(0.065, key_y, "Short", ha="center", va="center", fontsize=8)
    for i, (x, color) in enumerate(zip(centres, DISTANCE_BAND_COLORS)):
        box = Rectangle((x - box_width / 2, key_y - box_height / 2), box_width, box_height,
                        transform=fig.transFigure, facecolor=to_rgba(color, DISTANCE_BAND_ALPHA),
                        edgecolor="#87988D", linewidth=0.5)
        box.set_gid(f"distance_key_{i}")
        fig.add_artist(box)
    for x in (centres[:-1] + centres[1:]) / 2:
        fig.text(x, key_y, "<", ha="center", va="center", fontsize=10, color="#66756C")
    fig.text(0.41, key_y, "Long", ha="center", va="center", fontsize=8)

    ax = fig.add_axes([0.57, 0.23, 0.39, 0.56])
    ax.axhline(0, color="#A5ACB0", lw=0.7)
    ax.plot(rows.shell_midpoint_normalized, rows.shell_symmetric, color="#929B9F", lw=1.3)
    ax.scatter(rows.shell_midpoint_normalized, rows.shell_symmetric,
               c=[to_rgba(c, DISTANCE_BAND_ALPHA) for c in DISTANCE_BAND_COLORS],
               edgecolors="#4B6051", linewidths=0.5, s=31, zorder=4)
    ax.scatter([0], [rows.component_rho_0.iloc[0]], marker="D", s=27,
               color=INK, label="Same spot (Pearson)", zorder=4)
    ax.set(xlim=(-0.35, 7.1), ylim=(-1, 1), ylabel="Cross-correlation",
           xlabel="Separation / typical spot spacing")
    ax.set_xticks([0, 1, 2, 3.5, 6.5])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    clean_axis(ax)
    ax.legend(handles=[Line2D([], [], color="#929B9F", marker="o", lw=1.3,
                              markersize=4, label="Distance bands"),
                       Line2D([], [], color=INK, marker="D", linestyle="none",
                              markersize=4, label="Same spot (Pearson)")],
              frameon=False, fontsize=7, loc="lower left", handlelength=1.5)
    return fig


def boundary_illustration(e):
    """Zoom to all responder spots and every actual contact endpoint."""
    source = e["source"]
    edges = e["graph"].cross_roi_edges("1", "2")
    required = np.union1d(source.loc[source.roi_id.eq(2), "spot_index"], edges.ravel())
    xy = source.iloc[required][["x", "y"]].to_numpy()
    lower, upper = xy.min(axis=0) - 1.0, xy.max(axis=0) + 1.0
    return {"xlim": [float(lower[0]), float(upper[0])],
            "ylim": [float(lower[1]), float(upper[1])],
            "n_responder_spots": int(source.roi_id.eq(2).sum()),
            "n_contact_edges": len(edges), "anchor_components": [0, 1],
            "scope": "All ROI 2 and all ROI 1/2 contacts; exterior tissue cropped for display only"}


def boundary_plot(e):
    fig = figure("Boundary response")
    source = e["source"]
    view = boundary_illustration(e)
    ax = tissue_axis(fig, [0.01, 0.28, 0.50, 0.57], e)
    ax.set(xlim=view["xlim"], ylim=view["ylim"])
    dots(ax, source, np.ones(len(source), dtype=bool), FAINT, size=9)
    for band, color in (("Buffer", BUFFER), ("Far", FAR), ("Near", NEAR)):
        dots(ax, source, source.interface_band.eq(band), color, size=10, zorder=3)
    xy = source[["x", "y"]].to_numpy()
    edges = e["graph"].cross_roi_edges("1", "2")
    ax.add_collection(LineCollection(xy[edges], colors=NEAR, linewidths=0.55, alpha=0.8, zorder=2))
    for roi, component in ((1, 1), (2, 2)):
        row = one(e["geometry"], component_id=str(component))
        point = wkt.loads(row.geometry_wkt).representative_point()
        ax.text(point.x, point.y, f"ROI {roi}", ha="center", va="center", fontsize=7.5,
                zorder=5, bbox=dict(facecolor="white", alpha=.90, edgecolor="none", pad=1.5))
    # The main anchor is intentionally cropped, but its contributing edge remains visible.
    ax.text(view["xlim"][0] + .6, 0, "ROI 1", rotation=90, ha="center", va="center", fontsize=7,
            zorder=5, bbox=dict(facecolor="white", alpha=.90, edgecolor="none", pad=1.0))
    fig.text(0.255, 0.26, "ROI 1 → ROI 2", fontsize=9, ha="center")
    row = e["boundary"].iloc[0]
    fig.legend(handles=[key_handle(NEAR, f"Near ({row.n_near})"),
                        key_handle(BUFFER, f"Buffer ({row.n_buffer})"),
                        key_handle(FAR, f"Far ({row.n_far})")],
               loc="lower left", bbox_to_anchor=(0.025, 0.105), ncol=3,
               fontsize=7.5, frameon=False, handletextpad=0.1, columnspacing=0.65)

    ax = fig.add_axes([0.64, 0.30, 0.31, 0.46])
    for index, row in enumerate(e["boundary"].itertuples()):
        y = 1 - index
        ax.plot([row.mean_far, row.mean_near], [y, y], color="#89949B", lw=1.5)
        ax.scatter([row.mean_far], [y], c=FAR, s=38, zorder=3)
        ax.scatter([row.mean_near], [y], c=NEAR, s=38, zorder=3)
        ax.text((row.mean_far + row.mean_near) / 2, y + 0.20,
                f"Δ = {row.response:+.3f}", ha="center", fontsize=8)
    ax.set_yticks([1, 0], ["Gene A", "Gene B"])
    ax.set(xlim=(0, 0.70), ylim=(-0.45, 1.50), xlabel="Mean expression")
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_title("Measured in ROI 2", fontsize=9, pad=10)
    clean_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    fig.text(0.79, 0.13, "Δ = near − far", ha="center", fontsize=8)
    return fig


def aggregation_plot(e):
    fig = figure("Sensitivity to aggregation", strip=True)
    src, ce, reps = e["aggregation_source"], e["aggregation_consensus"], e["aggregation_replicates"]
    ax = fig.add_axes([0.16, 0.40, 0.80, 0.38])
    clean_axis(ax)
    ax.set(xlim=(-0.5, 2.5), ylim=(0, 1), ylabel=f"Pearson r · ROI {AGGREGATION_ROI_ID}")
    ax.set_yticks([0, 0.5, 1])
    ax.axhline(0, c="#D1D6D8", lw=0.6)
    ax.scatter([0], [src.corr_obs], s=45, c=INK, marker="D", zorder=3)
    # Deterministic offsets identify ten distinct draws; no smoothing/CI.
    # Preserve the physical width of the replicate cluster on the narrower canvas.
    offsets = np.linspace(-0.22, 0.22, len(reps))
    ax.scatter(1 + offsets, reps.correlation_value, c=MAIN, s=25, zorder=3)
    median = float(reps.correlation_value.median())
    ax.plot([0.73, 1.27], [median, median], c=INK, lw=1.0, zorder=2)
    ax.scatter([2], [ce.correlation_value], s=48, c=DETACHED, marker="h", zorder=3)
    for x, y in ((0, src.corr_obs), (1, median), (2, ce.correlation_value)):
        ax.text(x, 0.93, f"{y:+.2f}" + (" median" if x == 1 else ""),
                ha="center", fontsize=8)
    ax.set_xticks([0, 1, 2], ["Source spots", "Tessellations", "Consensus ensemble"], fontsize=8)
    ax.tick_params(axis="x", length=0, pad=7)
    n_spots = int(src.n_roi)
    min_spots, max_spots = reps.n_contributed_spots.min(), reps.n_contributed_spots.max()
    assert min_spots == max_spots == n_spots
    min_tiles, max_tiles = reps.n_tiles_contributed.min(), reps.n_tiles_contributed.max()
    counts = [f"{n_spots}/{n_spots} spots",
              f"{len(reps)}/10 partitions\n{min_tiles}–{max_tiles} tiles each\n{n_spots}/{n_spots} spots each",
              f"{int(ce.n_hexes_with_finite_density_pooled)}/{int(ce.n_hexes_pooled)} hexagons\n"
              f"{int(ce.n_contributing_components_consensus)}/{int(ce.n_eligible_consensus_components)} component"]
    for x, text in zip((0, 1, 2), counts):
        ax.text(x, -0.47, text, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="#596970", linespacing=1.5)
    return fig


PLOTS = (
    ("C1_enrichment", enrichment_plot),
    ("C2_internal_association", association_plot),
    ("C3_association_over_distance", distance_plot),
    ("C4_boundary_response", boundary_plot),
    ("C5_sensitivity_to_aggregation", aggregation_plot),
)


def render(evidence, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "text.color": INK, "axes.labelcolor": INK,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "svg.hashsalt": "tilescope-figure1-C"})
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for stem, painter in PLOTS:
        files.extend(save_artwork(painter(evidence), output, stem))
    return files
