from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context_matching import (CONTEXTS, official_to_current,
                              roi_context_inventory, validate_annotation_join,
                              validate_prepared_source)


def test_explicit_author_sample_alias():
    assert official_to_current("ZH881inf") == "GBM_ZH881_inf"
    assert official_to_current("UKF243") == "243_T"
    with pytest.raises(ValueError):
        official_to_current("unknown")


def annotation_fixture():
    source = pd.DataFrame(dict(x=[0., 1., 2.], y=[0., 0., 0.], roi_id=["1", "1", "2"]))
    ledger = pd.DataFrame(dict(spot_index=[0, 1, 2], roi_id=["1", "1", "2"]))
    annotation = pd.DataFrame(dict(spot_index=[0, 1, 2], source_roi_id=["1", "1", "2"],
                                   spot_barcode=["A", "B", "C"], x=[0., 1., 2.], y=[0., 0., 0.]))
    return source, ledger, annotation


def test_annotation_join_requires_exact_identity_not_coordinate_nearest_match():
    source, ledger, annotation = annotation_fixture()
    assert validate_annotation_join(source, ledger, annotation) == 0
    for broken in (annotation.iloc[::-1], annotation.assign(source_roi_id="9"),
                   annotation.assign(spot_barcode="A"), annotation.assign(x=[0, 1, 2.1])):
        with pytest.raises(ValueError):
            validate_annotation_join(source, ledger, broken)


def test_prepared_mask_checks_coordinates_even_when_roi_order_is_unchanged():
    source, _, _ = annotation_fixture()
    membership = source.assign(spot_index=np.arange(len(source)))
    validate_prepared_source(source, membership)
    with pytest.raises(ValueError, match="coordinate identity"):
        validate_prepared_source(source.assign(x=[1., 0., 2.]), membership)


def test_intermediate_and_unknown_spots_are_not_promoted_to_context():
    spots = pd.DataFrame(dict(sample_id=["S"]*5, roi_id=["1"]*5, spot_index=range(5),
                              structure_class=["structured", "structured", "disorganized", "intermediate", "unannotated"],
                              annotation_available=True))
    cohorts = pd.DataFrame([dict(context=c, sample_id="S", in_author_context_cohort=(c=="structured")) for c in CONTEXTS])
    result = roi_context_inventory(spots, cohorts).set_index("context")
    row = result.loc["structured"]
    assert row.n_context_spots == 2
    assert row.fraction_original_roi_in_context == .4
    assert row.mask_status == "roi_context_intersection"
    assert row.n_unannotated_spots == 1 and row.n_intermediate_spots == 1
    assert result.loc["disorganized"].constructed_status == "not_in_author_context_cohort"


def test_annotation_absence_is_not_zero_effect():
    spots = pd.DataFrame(dict(sample_id=["S"]*3, roi_id=["1"]*3, spot_index=range(3),
                              structure_class="annotation_unavailable", annotation_available=False))
    cohorts = pd.DataFrame([dict(context=c, sample_id="S", in_author_context_cohort=False) for c in CONTEXTS])
    result = roi_context_inventory(spots, cohorts)
    assert result.mask_status.eq("annotation_unavailable").all()
    assert result.n_annotation_unavailable_spots.eq(3).all()
