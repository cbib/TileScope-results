"""Verify Figure 1C against saved core values, identities and plotting units."""
from pathlib import Path
import importlib.util
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
from PIL import Image
import pytest

SPEC = importlib.util.spec_from_file_location("figure1_panel_c", Path(__file__).with_name("plot_panel_c.py"))
panel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(panel)


@pytest.fixture(scope="module")
def prepared():
    return panel.prepare_evidence()


def test_high_expression_uses_global_full_precision_thresholds(prepared):
    _, _, e, validation = prepared
    source = e["source"]
    q = validation["high_thresholds"]
    np.testing.assert_array_equal(source.a_high, source.gene1 > q["q_g1_threshold"])
    np.testing.assert_array_equal(source.b_high, source.gene2 > q["q_g2_threshold"])
    np.testing.assert_array_equal(source.both_high, source.a_high & source.b_high)
    assert source.both_high.sum() == 29
    assert source.loc[source.roi_id.eq(3), "b_high"].sum() == 0
    assert validation["enrichment_counts_and_ratios_match_exports"]


def test_component_correlations_are_not_parent_roi_correlations(prepared):
    _, _, e, _ = prepared
    assert e["components"].component_id.tolist() == [0, 1]
    assert e["components"].n_spots.tolist() == [149, 43]
    assert e["components"].pearson.iloc[0] > 0
    assert e["components"].pearson.iloc[1] < 0
    pooled = e['source'].loc[e['source'].roi_id.eq(1)]
    assert len(pooled) == 192
    assert pooled.gene1.corr(pooled.gene2) != e['components'].pearson.iloc[0]


def test_enrichment_map_and_matrix_use_source_values(prepared):
    from matplotlib.colors import to_rgba
    _, _, e, validation = prepared
    assert validation["display_enrichment_surface"] == "source_native"
    assert validation["enrichment_counts_and_ratios_match_exports"]
    assert e["display_enrichment"].equals(e["enrichment"])
    fig = panel.enrichment_plot(e)
    left, matrix, _ = fig.axes
    assert len(left.collections) == 4
    palette = [("Neither high", panel.FAINT), ("A only", panel.GENE_A),
               ("B only", panel.GENE_B), ("Both high", panel.JOINT)]
    for collection, (state, color) in zip(left.collections, palette):
        selected = e["source"].loc[e["source"].high_state.eq(state)]
        np.testing.assert_array_equal(collection.get_offsets(), selected[["x", "y"]].to_numpy())
        np.testing.assert_allclose(collection.get_facecolors()[0], to_rgba(color))
    assert sum(len(c.get_offsets()) for c in left.collections) == 457
    assert not any((p.get_gid() or "").startswith("joint_enrichment_") for p in left.patches)
    values = e["display_enrichment"].set_index("roi_id").loc[[1, 2, 3],
        ["g1_enrichment", "g2_enrichment", "joint_high_enrichment_fc"]].to_numpy()
    expected = np.full_like(values, -4.)
    np.log2(values, out=expected, where=values > 0)
    np.testing.assert_array_equal(matrix.images[0].get_array(), expected)
    plt.close(fig)


def test_expression_and_spatial_association_headings_are_explicit(prepared):
    _, _, e, _ = prepared
    fig = panel.association_plot(e)
    assert fig.texts[0].get_text() == "Expression correlation"
    text = [t.get_text() for t in fig.axes[2].texts]
    assert "Spatial association between neighbouring spots · whole ROI 1" in text
    assert sum(t.startswith("Bivariate Moran’s I") for t in text) == 2
    plt.close(fig)


def test_boundary_includes_contacts_with_both_anchor_components(prepared):
    _, _, e, validation = prepared
    source, graph = e["source"], e["graph"]
    anchor, responder = graph.interface_endpoints("1", "2")
    assert set(source.iloc[anchor].component_id) == {0, 1}
    assert source.iloc[responder].roi_id.eq(2).all()
    assert source.loc[source.interface_band.isin(["Near", "Far", "Buffer"]), "roi_id"].eq(2).all()
    assert [validation[k] for k in ("near_count", "buffer_count", "far_count")] == [81, 40, 56]
    for row in e["boundary"].itertuples():
        assert row.n_near + row.n_buffer + row.n_far == 177
        np.testing.assert_allclose(row.mean_near - row.mean_far, row.response, atol=1e-12)
    assert e["boundary"].set_index("gene").loc["Gene A", "response"] > .31
    np.testing.assert_allclose(e["boundary"].set_index("gene").loc["Gene B", "response"], .0229959222784326)


def test_boundary_zoom_preserves_every_measured_spot_and_contact(prepared):
    _, _, e, _ = prepared
    fig = panel.boundary_plot(e)
    ax = fig.axes[0]
    source = e["source"]
    edges = e["graph"].cross_roi_edges("1", "2")
    required = np.union1d(source.loc[source.roi_id.eq(2), "spot_index"], edges.ravel())
    subset = source.iloc[required]
    assert subset.x.between(*ax.get_xlim()).all()
    assert subset.y.between(*ax.get_ylim()).all()
    assert ax.get_xlim()[1] - ax.get_xlim()[0] < 31  # genuinely zoomed
    assert source.iloc[edges.ravel()].component_id.nunique() == 3
    plt.close(fig)


def test_distance_curve_uses_all_exported_shells_of_one_component(prepared):
    _, _, e, _ = prepared
    rows = e["distance"].loc[e["distance"].component_id.eq(0)].sort_values("shell_bin_index")
    assert rows.roi_id.eq(1).all()
    assert rows.n_anchors_per_shell.eq(149).all()
    fig = panel.distance_plot(e)
    assert fig.texts[0].get_text() == "Cross-correlation over distance"
    assert fig.axes[1].get_ylabel() == "Cross-correlation"
    curve = fig.axes[1].lines[1]
    np.testing.assert_array_equal(curve.get_xdata(), rows.shell_midpoint_normalized)
    np.testing.assert_array_equal(curve.get_ydata(), rows.shell_symmetric)
    plt.close(fig)


def test_distance_bands_use_real_points_in_distinct_directions(prepared):
    _, _, e, _ = prepared
    illustrated = panel.distance_illustration(e)
    source = e["source"].set_index("spot_index")
    centre = np.asarray(illustrated["anchor_point"])
    anchor = source.loc[illustrated["anchor_spot_index"]]
    np.testing.assert_array_equal(centre, anchor[["x", "y"]].to_numpy())
    assert anchor.component_id == 0
    directions = []
    indices = [illustrated["anchor_spot_index"]]
    for selected in illustrated["selections"]:
        spot = source.loc[selected["spot_index"]]
        assert spot.component_id == 0
        np.testing.assert_array_equal(selected["point"], spot[["x", "y"]].to_numpy())
        delta = np.asarray(selected["point"]) - centre
        d = np.linalg.norm(delta) / illustrated["median_nn_spacing"]
        np.testing.assert_allclose(d, selected["separation_normalized"])
        lower, upper = selected["band_edges"]
        assert lower <= d < upper
        row = panel.one(e["distance"], component_id=0, shell_bin_index=selected["shell_bin_index"])
        assert [lower, upper] == [row.shell_edge_lower_normalized, row.shell_edge_upper_normalized]
        directions.append(delta / np.linalg.norm(delta))
        indices.append(selected["spot_index"])
    assert len(set(indices)) == 3
    np.testing.assert_allclose(illustrated["selections"][0]["separation_normalized"], 1)
    assert np.dot(*directions) <= 0.5


def test_distance_ring_clipping_and_curve_colours_match(prepared):
    from matplotlib.colors import to_rgba
    _, _, e, _ = prepared
    fig = panel.distance_plot(e)
    fig.canvas.draw()
    left, right = fig.axes
    expected_ids = {f"distance_band_{i}" for i in range(4)}
    bands = {p.get_gid(): p for p in left.patches if p.get_gid() in expected_ids}
    assert set(bands) == expected_ids
    assert all(p.get_clip_path() is not None for p in bands.values())
    outlines = [p for p in left.patches if (p.get_gid() or "").startswith("distance_band_outer_edge_")]
    assert len(outlines) == 4
    rows = e["distance"].loc[e["distance"].component_id.eq(0)].sort_values("shell_bin_index")
    np.testing.assert_allclose([p.radius for p in outlines], rows.shell_edge_upper_normalized * rows.median_nn_spacing)
    short = next(line for line in left.lines if line.get_gid() == "short_separation")
    endpoints = left.transData.transform(np.column_stack((short.get_xdata(), short.get_ydata())))
    assert np.linalg.norm(endpoints[1] - endpoints[0]) * 72 / fig.dpi > 8
    colours = right.collections[0].get_facecolors()
    assert len(colours) == 4
    assert len(set(panel.DISTANCE_BAND_COLORS)) == 4
    for i, colour in enumerate(panel.DISTANCE_BAND_COLORS):
        np.testing.assert_allclose(colours[i], to_rgba(colour, panel.DISTANCE_BAND_ALPHA))
        np.testing.assert_allclose(bands[f"distance_band_{i}"].get_facecolor(), colours[i])
        swatch = next(p for p in fig.artists if p.get_gid() == f"distance_key_{i}")
        np.testing.assert_allclose(swatch.get_facecolor(), colours[i])
        np.testing.assert_allclose(bands[f"distance_band_{i}"].width,
                                   (rows.iloc[i].shell_edge_upper_normalized - rows.iloc[i].shell_edge_lower_normalized)
                                   * rows.iloc[i].median_nn_spacing)
    assert rows.n_anchors_per_shell.eq(149).all()
    assert rows.n_unordered_pairs_per_shell.gt(0).all()
    assert [text.get_text() for text in fig.texts].count("<") == 3
    labels = [text.get_text() for text in fig.texts]
    assert "Short" in labels and "Long" in labels
    assert not any("0.5–1.5" in label or "4.5–8.5" in label for label in labels)
    plt.close(fig)


def test_aggregation_shows_ten_actual_estimates_not_a_confidence_interval(prepared):
    _, _, e, _ = prepared
    fig = panel.aggregation_plot(e)
    source, replicates, consensus = fig.axes[0].collections
    assert len(replicates.get_offsets()) == 10
    np.testing.assert_array_equal(replicates.get_offsets()[:, 1], e["aggregation_replicates"].correlation_value)
    assert source.get_offsets()[0, 1] == e["aggregation_source"].corr_obs
    assert consensus.get_offsets()[0, 1] == e["aggregation_consensus"].correlation_value
    assert fig.axes[0].get_ylabel() == "Pearson r · ROI 3"
    assert e["aggregation_source"].n_roi == 88
    assert e["aggregation_consensus"].sample_size == 24
    assert e["aggregation_replicates"].roi_id.eq(3).all()
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "10/10 partitions\n11–17 tiles each\n88/88 spots each" in texts
    assert "24/24 hexagons\n1/1 component" in texts
    plt.close(fig)


def test_five_separate_graphics_with_no_retired_classes(prepared, tmp_path):
    _, _, e, _ = prepared
    files = panel.render(e, tmp_path)
    assert len(files) == 15
    for stem, _ in panel.PLOTS:
        svg = ET.parse(tmp_path / f"{stem}.svg").getroot()
        labels = ["".join(item.itertext()) for item in svg.iter("{http://www.w3.org/2000/svg}text")]
        assert len(labels) > 1
        for retired in ("MAUP-sensitive", "zoning-sensitive", "scale-sensitive"):
            assert retired not in " ".join(labels)
        with Image.open(tmp_path / f"{stem}.png") as image:
            assert image.mode == "RGBA"
            assert image.convert('RGB').getpixel((0, 0)) == (255, 255, 255)
    assert (tmp_path / "C5_sensitivity_to_aggregation.pdf").exists()


def test_compact_aggregation_keeps_category_labels_and_counts_separate(prepared):
    _, _, e, _ = prepared
    fig = panel.aggregation_plot(e)
    np.testing.assert_allclose(fig.get_size_inches(), [4.6, 2.65])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    tick_boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
    count_boxes = [t.get_window_extent(renderer) for t in ax.texts if "spots" in t.get_text()
                   or "hexagons" in t.get_text()]
    for boxes in (tick_boxes, count_boxes):
        assert len(boxes) == 3
        for left, right in zip(boxes, boxes[1:]):
            assert right.x0 - left.x1 >= 8 * fig.dpi / 72
    assert all(t.get_fontsize() == 8 for t in ax.get_xticklabels())
    plt.close(fig)


@pytest.mark.parametrize("stem,painter", panel.PLOTS, ids=[p[0] for p in panel.PLOTS])
def test_no_visible_text_is_clipped(prepared, stem, painter):
    _, _, e, _ = prepared
    fig = painter(e)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        # Tick labels outside the selected limits are not drawn by their axes.
        if text.axes is not None and not text.axes.axison:
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1, (stem, text.get_text(), box)
        assert box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, (stem, text.get_text(), box)
    plt.close(fig)
