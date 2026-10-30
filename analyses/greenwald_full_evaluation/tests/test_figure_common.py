"""Small semantic checks for the manuscript-only rendering layer."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_common import component_mask, one
import figure_common as panels
from local_effect_atlas import (
    availability_table, pair_pages, plot_source_atlas, prepare_source_rows,
)


def source_fixture():
    corr = pd.DataFrame(dict(
        sample_id=["0"] * 3, roi_id=["1", "2", "3"], g1_id=["A"] * 3, g2_id=["B"] * 3,
        n_roi=[100, 450, 20], corr_obs=[-0.71, 0.34, np.nan],
        parent_correlation_status=["finite", "finite", "insufficient_roi_spots"],
    ))
    support = pd.DataFrame(dict(roi_id=["1", "2", "3"], source_spots=[100, 450, 20],
                                spatial_support_class=["scattered", "mixed", "scattered"]))
    return corr, support


def prepared_fixture():
    return prepare_source_rows(*source_fixture(), sample="sample", pair="A_B", features=("A", "B"))


def test_both_signs_and_scattered_source_support_are_retained():
    rows = prepared_fixture()
    assert rows.corr_obs.iloc[:2].tolist() == [-0.71, 0.34]
    assert rows.display_status.tolist() == ["finite", "finite", "unavailable"]
    assert rows.sample_id.eq("sample").all()  # Not the raw core sample_id=0.
    assert rows.display_support_bin.tolist() == ["100–399 spots", "≥400 spots", "<100 spots"]
    assert len(rows) == 3


def test_display_offsets_depend_on_roi_identity_not_input_order():
    corr, support = source_fixture()
    actual = prepare_source_rows(corr.iloc[::-1], support.iloc[::-1], sample="sample", pair="A_B", features=("A", "B"))
    pd.testing.assert_frame_equal(actual, prepared_fixture(), check_exact=True)


def test_renderer_keeps_raw_x_values_and_unavailable_separate():
    fig = plot_source_atlas(prepared_fixture(), ["A_B"], ["sample"])
    try:
        assert len(fig.axes) == 1
        assert fig.axes[0].get_title() == "A\n× B"
        assert [text.get_text() for text in fig.texts] == ["Within-ROI source Pearson r"]
        offsets = np.vstack([collection.get_offsets() for collection in fig.axes[0].collections])
        np.testing.assert_array_equal(offsets[:, 0], [-0.71, 0.34, 1.25])
        assert [tick.get_text() for tick in fig.axes[0].get_yticklabels()] == ["sample"]
        fig.canvas.draw()
    finally:
        panels.plt.close(fig)


def test_accounting_preserves_unavailable_mass_without_detection_labels():
    rows = availability_table(prepared_fixture())
    r = rows.iloc[0]
    assert (r.n_rois_total, r.n_rois_finite, r.n_rois_unavailable) == (3, 2, 1)
    assert (r.n_source_spots_total, r.n_source_spots_finite_rois, r.n_source_spots_unavailable_rois) == (570, 550, 20)
    assert r.n_finite_scattered_rois == 1
    assert r.unavailable_statuses == "insufficient_roi_spots"
    assert not any("positive" in c or "negative" in c or "detected" in c for c in rows.columns)


@pytest.mark.parametrize("mutation, message", [
    ("duplicate", "duplicate"), ("missing", "universes"),
    ("count", "count mismatch"), ("feature", "Feature"),
    ("finite_nan", "Status/value"), ("unavailable_value", "Status/value"),
    ("bounds", "outside"),
])
def test_source_integrity_errors_fail_loudly(mutation, message):
    corr, support = source_fixture()
    if mutation == "duplicate":
        corr = pd.concat([corr, corr.iloc[:1]])
    elif mutation == "missing":
        corr = corr.iloc[:2]
    elif mutation == "count":
        corr.loc[0, "n_roi"] = 99
    elif mutation == "feature":
        corr.loc[0, "g1_id"] = "wrong"
    elif mutation == "finite_nan":
        corr.loc[0, "corr_obs"] = np.nan
    elif mutation == "unavailable_value":
        corr.loc[2, "corr_obs"] = 0.0
    elif mutation == "bounds":
        corr.loc[0, "corr_obs"] = 1.1
    with pytest.raises(ValueError, match=message):
        prepare_source_rows(corr, support, sample="sample", pair="A_B", features=("A", "B"))


def test_missing_sample_pair_is_not_treated_as_unavailability():
    with pytest.raises(ValueError, match="inventory"):
        plot_source_atlas(prepared_fixture(), ["A_B"], ["sample", "missing"])


def test_all_pairs_once_and_main_is_supplement_first_page():
    pairs = [f"pair{i:02d}" for i in range(91)]
    main = pairs[30:38]
    pages = pair_pages(pairs, main)
    assert pages[0] == main
    assert len(pages) == 12 and len(pages[-1]) == 3
    flattened = [p for page in pages for p in page]
    assert len(flattened) == 91 and set(flattened) == set(pairs)
    with pytest.raises(ValueError):
        pair_pages(pairs, ["unknown"])


def test_atlas_selection_ignores_unrequested_pairs():
    rows = prepared_fixture()
    extra = rows.assign(pair="C_D", program_g1="C", program_g2="D", display_pair="C × D")
    main = plot_source_atlas(rows, ["A_B"], ["sample"])
    supplement = plot_source_atlas(pd.concat([rows, extra]), ["A_B"], ["sample"])
    try:
        main.canvas.draw(); supplement.canvas.draw()
        np.testing.assert_array_equal(np.asarray(main.canvas.buffer_rgba()), np.asarray(supplement.canvas.buffer_rgba()))
    finally:
        panels.plt.close(main); panels.plt.close(supplement)


def test_component_selection_uses_ledger_identity_not_geometry():
    frame = pd.DataFrame({"roi_id": ["1", "1", "2"]})
    ledger = pd.DataFrame({"spot_index": [2, 0, 1], "roi_id": ["2", "1", "1"],
                           "component_id": ["8", "4", "5"]})
    np.testing.assert_array_equal(component_mask(frame, ledger, "4"), [True, False, False])
    ledger.loc[0, "roi_id"] = "1"
    with pytest.raises(ValueError, match="ROI order"):
        component_mask(frame, ledger, "4")


def test_ledger_requires_exact_input_row_coverage():
    frame = pd.DataFrame({"roi_id": ["1", "1"]})
    ledger = pd.DataFrame({"spot_index": [0, 0], "roi_id": ["1", "1"],
                           "component_id": ["4", "4"]})
    with pytest.raises(ValueError, match="exactly once"):
        component_mask(frame, ledger, "4")


def test_exact_one_row_required_before_plotting():
    frame = pd.DataFrame({"roi_id": ["1", "1"]})
    with pytest.raises(ValueError):
        one(frame, roi_id="1")
