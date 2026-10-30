"""Measurement semantics, unsupervised ordering and artwork checks."""
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analyses.figures.scripts.benchmark import build_panel_b_profiles as p


@pytest.fixture(scope="module")
def data():
    return p.prepare()


def test_complete_positive_expression_inventory(data):
    assert data["numeric"].shape == (18, 12)
    assert data["cases"].groupby("class_label").size().eq(3).all()
    assert len(data["traces"]) == 216
    assert data["specs"].grain.eq("source_native").all()
    assert set(data["specs"].family) == {"marginal_enrichment", "joint_enrichment", "correlation", "moran", "distance"}


def test_colour_is_effect_not_zscore_and_infinity_not_missing(data):
    traces = data["traces"]
    enr = traces.loc[traces.family.str.endswith("enrichment")]
    np.testing.assert_allclose(enr.display_value, 2*enr.value-1)
    infinite = enr.loc[enr.status.eq("positive_over_zero_context")]
    assert len(infinite) == 10
    assert infinite.raw_value.isna().all()
    assert infinite.display_value.eq(1).all()
    finite = enr.loc[enr.status.eq("finite")]
    np.testing.assert_allclose(finite.display_value, (finite.raw_value-1)/(finite.raw_value+1))
    coeff = traces.loc[~traces.family.str.endswith("enrichment")]
    np.testing.assert_array_equal(coeff.display_value, coeff.raw_value)


def test_source_component_identity_and_grain_comparison(data):
    rows = data["traces"]
    component = rows.loc[rows.unit.eq("Largest retained A component")]
    assert component.component_id.notna().all()
    assert component.groupby("case_id").component_id.nunique().eq(1).all()
    assert component.groupby("case_id").size().eq(5).all()
    assert rows.loc[rows.unit.eq("Whole A ROI"), "component_id"].isna().all()
    assert data["grain_display"].shape == (18, 12)
    grain = data["grain_traces"]
    assert grain.groupby(["case_id", "grain"]).size().eq(3).all()
    assert grain.component_id.isna().all()
    assert set(grain.grain) == dict(p.GRAINS).keys()


def test_family_weights_give_equal_total_variance(data):
    matrix = data["primary"]["matrix"]
    groups = data["specs"].set_index("feature").family
    for family in groups.unique():
        columns = groups.index[groups.eq(family)]
        assert matrix[columns].var(ddof=0).sum() == pytest.approx(1 / groups.nunique())
    assert not np.allclose(data["display"], matrix)


def test_order_does_not_depend_on_scenario_labels_or_input_row_order(data):
    # The clustering function has no label input. Its numeric rows are keyed
    # independently; shuffling their input order cannot change the hierarchy.
    families = data["specs"].set_index("feature").family.to_dict()
    result = p.cluster_profiles(data["numeric"].sample(frac=1, random_state=82), families)
    np.testing.assert_array_equal(result["tree"], data["primary"]["tree"])
    assert result["order"] == data["primary"]["order"]
    assert set(result["order"]) == set(data["cases"].index)


def test_uses_full_catalogue_grouping_and_average_linkage(data):
    from scipy.spatial.distance import pdist, squareform
    from tilescope_results.paths import supplementary_data_root
    catalogue = pd.read_csv(supplementary_data_root(1) / 'statistic_catalogue.csv')
    meta = catalogue.set_index("feature").loc[data["numeric"].columns].rename_axis("feature").reset_index()
    assert meta.group.tolist() == data["specs"].family.tolist()
    expected = p.full.cluster(data["numeric"], meta, method="average")
    np.testing.assert_array_equal(data["primary"]["tree"], expected["tree"])
    np.testing.assert_allclose(pdist(data["primary"]["matrix"]), squareform(expected["distances"]), atol=1e-12)


def test_constant_columns_and_missing_measurements():
    values = pd.DataFrame({"a": [1., 2., 5.], "b": [3., 3., 3.]}, index=["c", "b", "a"])
    result = p.cluster_profiles(values, {"a": "first", "b": "constant"})
    assert list(result["matrix"]) == ["a"]
    assert not result["scale"].loc["b", "active"]
    assert result["scale"].loc["b", "family_weight"] == 0
    values.loc["a", "a"] = np.nan
    with pytest.raises(ValueError, match="no imputation"):
        p.cluster_profiles(values, {"a": "first", "b": "constant"})


def test_audit_does_not_select_a_cut_or_cluster_count(data):
    audits = data["audits"]
    assert len(audits) == 72
    primary = audits.loc[audits.variant.eq("primary")]
    assert primary.n_branch_cases.eq(3).all() and primary.n_other_cases.eq(0).all()
    omissions = audits.loc[audits.variant.str.startswith("without_")]
    failed = omissions.loc[omissions.n_other_cases.gt(0), ["variant", "class_label", "n_other_cases"]]
    assert failed.to_dict("records") == [dict(variant="without_joint_enrichment",
        class_label="spatial_gradient::concordant", n_other_cases=3)]
    assert audits.loc[audits.variant.eq("only_distance") & audits.class_label.eq("nested_foci"), "n_other_cases"].item() == 3


@pytest.mark.parametrize("grains", [False, True])
def test_heatmap_cell_identity_and_visible_text(data, grains):
    p.v1.style()
    fig = p.plot_profiles(data, grains=grains)
    fig.canvas.draw()
    cells = [r for ax in fig.axes for r in ax.patches if (r.get_gid() or "").startswith("profile:")]
    assert len(cells) == 216
    specs = data["grain_specs"] if grains else data["specs"]
    values = data["grain_display"] if grains else data["display"]
    by_gid = {c.get_gid(): c for c in cells}
    for i, case in enumerate(data["primary"]["order"]):
        for feature in specs.feature:
            cell = by_gid[f"profile:{case}:{feature}"]
            assert cell.get_y() == i
            np.testing.assert_allclose(cell.get_facecolor(), p.CMAP((values.loc[case, feature]+1)/2))
    texts = list(fig.texts) + [t for ax in fig.axes for t in ax.texts]
    for ax in fig.axes:
        if ax.axison:
            texts += [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]
    renderer = fig.canvas.get_renderer()
    label_axis = next(ax for ax in fig.axes if ax.get_label() == "profile_labels")
    matrix_axis = next(ax for ax in fig.axes if ax.get_label() == "profile_matrix")
    sample_header = next(t for t in label_axis.texts if t.get_text() == "Sample · ROI")
    scenario_header = next(t for t in label_axis.texts if t.get_text() == "Scenario")
    sample_x = sample_header.get_window_extent(renderer).x0
    assert sample_x < scenario_header.get_window_extent(renderer).x0
    assert 0 < sample_x - matrix_axis.get_window_extent(renderer).x1 < .2 * fig.dpi
    assert fig.get_size_inches()[0] == 7.8
    for t in texts:
        if not t.get_visible() or not t.get_text():
            continue
        b = t.get_window_extent(renderer)
        assert b.x0 >= -1 and b.y0 >= -1, t.get_text()
        assert b.x1 <= fig.bbox.width+1 and b.y1 <= fig.bbox.height+1, t.get_text()
    for ax in fig.axes:
        for i, a in enumerate(ax.texts):
            ba = a.get_window_extent(renderer)
            for b in ax.texts[i+1:]:
                bb = b.get_window_extent(renderer)
                assert not (min(ba.x1, bb.x1)-max(ba.x0, bb.x0)>1 and
                            min(ba.y1, bb.y1)-max(ba.y0, bb.y0)>1), (a.get_text(), b.get_text())
    plt.close(fig)







