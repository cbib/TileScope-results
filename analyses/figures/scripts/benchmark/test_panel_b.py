"""Unit identity, completeness and rendering checks for Figure 2B."""
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analyses.figures.scripts.benchmark import build_panel_b as panel


@pytest.fixture(scope="module")
def data():
    return panel.prepare()


def test_all_cases_and_controls_of_selected_modes_are_included(data):
    f = data["tables"]["inventory"]
    assert len(f) == 33
    assert f.case_id.nunique() == 15
    assert f.drop_duplicates("case_id").groupby("scenario_mode").size().eq(3).all()
    assert f.tessellation_n.eq(3).all()
    assert set(f.core_sha) == {panel.CORE_SHA}
    assert f.loc[f.scenario_mode.eq("responder_activation")].groupby("case_id").size().eq(3).all()


def test_selected_component_is_largest_in_the_saved_membership(data):
    membership = data['tables']['membership']
    for _, rows in membership.groupby(['case_id', 'role']):
        counts = rows.groupby('component_id').size()
        selected = rows.loc[rows.primary_display, 'component_id'].unique()
        assert len(selected) == 1
        assert counts.loc[selected[0]] == counts.max()


def test_per_run_metadata_is_not_required(monkeypatch):
    record = panel.ExportReader.record

    def without_metadata(self, path, expected=None):
        assert Path(path).name not in {"effective_config.json", "run_metadata.json"}
        return record(self, path, expected)

    monkeypatch.setattr(panel.ExportReader, "record", without_metadata)
    assert panel.prepare()["n_runs"] == 33


def test_hash_mismatch_is_rejected(tmp_path):
    f = tmp_path / "example.csv"
    f.write_text("x,y\n1,2\n")
    with pytest.raises(ValueError, match="Hash mismatch"):
        panel.ExportReader(tmp_path).record(f, "wrong")


def test_plotted_zero_lag_uses_exactly_the_radial_component_membership(data):
    tables = data["tables"]
    primary = tables["radial"].loc[lambda f: f.primary_display]
    assert primary.groupby(["case_id", "role"]).size().eq(4).all()
    assert primary.component_radial_shell_status.eq("finite").all()
    for (case, role), shell in primary.groupby(["case_id", "role"]):
        meta = tables["inventory"].set_index(["case_id", "role"]).loc[(case, role)]
        expressions = pd.read_csv(panel.corpus_root() / meta.scenario_name / case / f"{role}.csv", float_precision="round_trip")
        ledger = tables["membership"]
        rows = ledger.loc[ledger.case_id.eq(case) & ledger.role.eq(role) & ledger.primary_display, "spot_index"]
        expected = np.corrcoef(expressions.iloc[rows].gene1, expressions.iloc[rows].gene2)[0, 1]
        assert shell.component_rho_0.nunique() == 1
        assert shell.component_rho_0.iloc[0] == pytest.approx(expected, abs=1e-12)
        assert len(rows) == shell.n_component_spots.iloc[0]


def test_no_control_changes_component_membership(data):
    f = data["tables"]["membership"]
    for _, group in f.groupby("case_id"):
        subframes = [g.sort_values("spot_index")[["spot_index", "component_id", "primary_display"]]
                     .reset_index(drop=True) for _, g in group.groupby("role")]
        for other in subframes[1:]:
            pd.testing.assert_frame_equal(subframes[0], other)


def test_boundary_endpoints_and_both_features_are_retained(data):
    f = data["tables"]["interfaces"]
    assert set(f.direction) == {"designated", "reverse"}
    assert set(f.surface) == {"source_native", "tessellation", "consensus_direct", "consensus_ensemble"}
    assert f.groupby(["case_id", "role", "direction", "surface"]).responder_feature_id.nunique().eq(2).all()
    designated = f.loc[f.direction.eq("designated") & f.surface.eq("source_native")]
    assert designated.responder_roi_id.eq(designated.target_roi_id).all()
    assert designated.availability_status.eq("finite").all()
    for _, group in designated.groupby(["case_id", "responder_feature_id"]):
        assert group.source_near_mass.nunique() == group.source_far_mass.nunique() == 1
    activation = designated.loc[designated.scenario_mode.eq("responder_activation") &
                                designated.responder_feature_id.eq("B")]
    paired = activation.pivot(index="case_id", columns="condition", values="interface_response")
    np.testing.assert_allclose(paired.pattern - paired.matched_control, .5, atol=1e-12)


def test_complete_export_hashes_still_match(data):
    assert all(panel.sha256(panel.resolve_reference(p)) == checksum for p, checksum in data["source_sha256"].items())


def test_combined_curves_use_exported_pearson_then_four_unchanged_bands(data):
    f = panel.combined_curve_values(data["tables"])
    assert len(f) == 90  # Nine cases, two roles, five points.
    radial = data["tables"]["radial"].loc[lambda x: x.primary_display]
    for (case, role), rows in f.groupby(["case_id", "role"]):
        expected = radial.loc[radial.case_id.eq(case) & radial.role.eq(role)].sort_values("shell_bin_index")
        rows = rows.sort_values("distance")
        assert list(rows.distance) == [0, 1, 2, 3.5, 6.5]
        assert rows.iloc[0].point_type == "same_spot_pearson"
        assert rows.iloc[0].value == expected.iloc[0].component_rho_0
        np.testing.assert_array_equal(rows.value.iloc[1:], expected.shell_symmetric)


def test_heatmap_has_raw_values_from_the_correct_responder(data):
    f = panel.boundary_heatmap_values(data["tables"])
    assert len(f) == 45  # Fifteen case/roles, three measurements.
    assert f.groupby(["case_id", "condition"]).size().eq(3).all()
    corr = data["tables"]["correlations"]
    for row in f.loc[f.statistic.eq("pearson")].itertuples():
        expected = corr.loc[corr.case_id.eq(row.case_id) & corr.role.eq(row.role) &
                            corr.surface.eq("source_native") & corr.roi_id.eq(row.responder_roi_id)]
        assert len(expected) == 1
        assert row.value == expected.iloc[0].correlation_value
    assert f.condition.eq("relocated_control").sum() == 9


def test_heatmap_separate_fixed_scales_and_missingness():
    assert panel.HEATMAP_LIMITS["pearson"] == (-1, 1)
    assert panel.HEATMAP_LIMITS["response_A"] == panel.HEATMAP_LIMITS["response_B"] == (-.6, .6)
    colour, text, hatch = panel.heatmap_cell(-.001, "finite", "pearson")
    assert text == "0.00" and hatch is None
    assert panel.heatmap_cell(np.nan, "unavailable", "pearson")[1:] == ("NA", "//")
    assert panel.heatmap_cell(.05, "insufficient_support", "pearson")[1:] == ("NA", "//")
    with pytest.raises(ValueError, match="would clip"):
        panel.heatmap_cell(.7, "finite", "response_A")




@pytest.mark.parametrize("renderer", [panel.plot_distance, panel.plot_boundary_heatmap,
                                      panel.plot_relocated_controls])
def test_artwork_has_no_narrative_title_and_visible_labels(data, renderer):
    panel.style()
    fig = renderer(data["tables"])
    fig.canvas.draw()
    assert len(fig.axes) == (1 if renderer is panel.plot_distance else 3)
    for ax in fig.axes:
        assert ax.get_title() == ""
        text = list(ax.texts)
        if ax.axison:
            text += [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]
        for item in text:
            if not item.get_text() or not item.get_visible():
                continue
            extent = item.get_window_extent(fig.canvas.get_renderer())
            assert extent.x0 >= -1 and extent.y0 >= -1, item.get_text()
            assert extent.x1 <= fig.bbox.width + 1 and extent.y1 <= fig.bbox.height + 1, item.get_text()
    if renderer is panel.plot_distance:
        lines = [line for line in fig.axes[0].lines if (line.get_gid() or "").startswith("curve:")]
        assert len(lines) == 18
        assert all(len(line.get_xdata()) == 5 and line.get_xdata()[0] == 0 for line in lines)
    else:
        cells = [p for p in fig.axes[0].patches if (p.get_gid() or "").startswith("heat:")]
        assert len(cells) == (36 if renderer is panel.plot_boundary_heatmap else 27)
        texts = fig.axes[0].texts
        for i, left in enumerate(texts):
            a = left.get_window_extent(fig.canvas.get_renderer())
            for right in texts[i + 1:]:
                b = right.get_window_extent(fig.canvas.get_renderer())
                assert not (min(a.x1, b.x1) - max(a.x0, b.x0) > 1 and
                            min(a.y1, b.y1) - max(a.y0, b.y0) > 1), (left.get_text(), right.get_text())
    plt.close(fig)
