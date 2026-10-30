"""Input fidelity, coverage and display semantics for the scenario catalogue."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analyses.figures.scripts.benchmark import build

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point


@pytest.fixture(scope="module")
def catalogue():
    if not (build.corpus_root() / "corpus_index.json").exists():
        pytest.skip("Materialized benchmark corpus is required")
    return build.prepare()


def test_seven_families_ten_modes_three_main_examples():
    assert len(build.EXAMPLES) == 10
    assert len({e.family for e in build.EXAMPLES}) == 7
    assert len({e.slug for e in build.EXAMPLES}) == 10
    assert [e.family for e in build.EXAMPLES if e.main] == [
        "spatial_gradient", "embedded_domains", "contact_interface"]


def test_current_corpus_and_truth_are_preserved(catalogue):
    for d in catalogue:
        f = d["frame"]
        assert len(f) == len(f.input_row.unique()) == len(f.source_spot_id.unique())
        assert np.array_equal(f.roi_id, f.generated_roi_id)
        assert np.array_equal(f.roi_id, f.source_parent_roi_id)
        assert np.array_equal(f.gene1, f.feature_a_value)
        assert np.array_equal(f.gene2, f.feature_b_value)
        assert d["source_hashes"] == {
            build.reference(p): build.sha256(p) for p in d["source_paths"]}


def test_crops_are_viewports_not_subsampling(catalogue):
    for d in catalogue:
        f = d["frame"]
        xmin, ymin, xmax, ymax = d["bounds"]
        expected = f.x.between(xmin, xmax) & f.y.between(ymin, ymax)
        assert np.array_equal(f.display_visible, expected)
        assert np.array_equal(f.display_expression, expected & f.roi_id.isin(d["subject_rois"]))
        assert f.display_expression.sum() > 5
        assert np.isclose(xmax - xmin, ymax - ymin)


@pytest.mark.parametrize("family", ["focal_pattern", "spatial_gradient", "contact_interface"])
def test_contrasting_modes_share_geometry_viewport_and_colour_scale(catalogue, family):
    left, right = [d for d in catalogue if d["spec"].family == family]
    assert left["bounds"] == right["bounds"]
    assert left["colour_limits"] == right["colour_limits"]
    np.testing.assert_array_equal(left["frame"][["x", "y", "roi_id"]],
                                  right["frame"][["x", "y", "roi_id"]])


def test_colour_limits_use_full_inputs_and_include_negative_background(catalogue):
    for d in catalogue:
        values = np.concatenate([item["frame"][["gene1", "gene2"]].to_numpy().ravel()
                                 for item in catalogue if item["spec"].family == d["spec"].family])
        assert d["colour_limits"] == [values.min(), values.max()]
        assert d["colour_limits"][0] < 0  # Do not silently clip signed synthetic fields.


def test_embedded_component_is_real_and_completely_visible(catalogue):
    d = next(d for d in catalogue if d["spec"].family == "embedded_domains")
    f = d["frame"]
    inclusion = f.generated_structure_kind.eq("embedded_b_domain")
    assert inclusion.sum() == 15
    assert f.loc[inclusion, "display_expression"].all()
    assert f.loc[inclusion, "source_component_id"].nunique() == 1
    assert f.loc[inclusion, "roi_id"].eq("2").all()
    assert len(f) == 2971  # Cropping does not alter the benchmark input.


def test_embedded_view_includes_parent_territory_and_host(catalogue):
    d = next(d for d in catalogue if d["spec"].family == "embedded_domains")
    f = d["frame"]
    assert d["spec"].framing == "embedded_context"
    assert f.loc[f.roi_id.isin(["2", "5"]), "display_expression"].all()
    assert f.loc[f.roi_id.eq("2"), "display_expression"].sum() == 782
    main_component = f.source_component_id.eq("2::c0000")
    assert main_component.sum() == 412
    assert f.loc[main_component, "display_expression"].all()
    fig = plt.figure(figsize=(2.4, 2.65))
    ax = build.map_axes(fig, [0.045, 0.08, 0.91, 0.79], d, "gene1")
    assert sum(t.get_text() == "ROI 2" for t in ax.texts) == 2
    plt.close(fig)


def test_main_maps_have_comparable_spot_magnification(catalogue):
    widths = [(d["bounds"][2] - d["bounds"][0]) / d["spacing"]
              for d in catalogue if d["spec"].main]
    assert max(widths) / min(widths) < 1.2


def test_full_catalogue_has_comparable_spot_magnification(catalogue):
    widths = [(d["bounds"][2] - d["bounds"][0]) / d["spacing"] for d in catalogue]
    assert min(widths) >= 50 - 1e-9
    assert max(widths) / min(widths) < 1.3


def test_label_outlines_preserve_small_components(catalogue):
    d = next(d for d in catalogue if d["spec"].family == "embedded_domains")
    for roi, geometry in d["outlines"].items():
        assert geometry.is_valid
        xy = d["frame"].loc[d["frame"].roi_id.eq(roi), ["x", "y"]].to_numpy()
        assert all(geometry.covers(Point(p)) for p in xy)
    first, second = [d["outlines"][roi] for roi in d["subject_rois"]]
    assert first.intersection(second).area < d["spacing"] ** 2 * 1e-8


def test_painter_draws_the_unmodified_expression_values(catalogue):
    d = next(d for d in catalogue if d["spec"].family == "embedded_domains")
    for feature in ("gene1", "gene2"):
        fig = plt.figure(figsize=(2.5, 2.5))
        ax = build.map_axes(fig, [0.05, 0.05, 0.9, 0.9], d, feature)
        layer = next(c for c in ax.collections if c.get_gid() and c.get_gid().endswith("_source_spots"))
        visible = d["frame"].loc[d["frame"].display_expression]
        np.testing.assert_array_equal(layer.get_array(), visible[feature].to_numpy())
        np.testing.assert_array_equal(layer.get_offsets(), visible[["x", "y"]].to_numpy())
        plt.close(fig)
