"""Contract tests for the sister-side saturating-enrichment recognition transform.

Real atlas rows encode saturation as
``availability_status="undefined"`` + ``undefined_reason="positive_over_zero_context"``;
Recognition consumers previously collapsed both
undefined states into missing_feature, discarding informative
saturating-high cases. These tests pin the corrected contract and
drive the real `_build_case_features` + `_panel_status` code path on
a production-shaped atlas fixture so the fix does not depend on any
regenerated summary CSV.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enrichment_bounded import (  # noqa: E402
    saturating_enrichment,
    STATUS_FINITE,
    STATUS_UNDEFINED,
    REASON_INDETERMINATE,
    REASON_SATURATING_HIGH,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recognition"))


# --- saturating_enrichment: helper contract on real (status, reason) pairs ---


def test_finite_maps_r_over_one_plus_r():
    assert saturating_enrichment(2.0, STATUS_FINITE, None) == pytest.approx(2 / 3)
    assert saturating_enrichment(9.0, STATUS_FINITE, None) == pytest.approx(0.9)
    assert saturating_enrichment(0.0, STATUS_FINITE, None) == 0.0


def test_undefined_positive_over_zero_context_saturates_high():
    v = saturating_enrichment(None, STATUS_UNDEFINED, REASON_SATURATING_HIGH)
    assert v == 1.0


def test_undefined_zero_over_zero_context_is_missing():
    v = saturating_enrichment(None, STATUS_UNDEFINED, REASON_INDETERMINATE)
    assert v is None


def test_undefined_with_no_or_unknown_reason_is_missing():
    assert saturating_enrichment(None, STATUS_UNDEFINED, None) is None
    assert saturating_enrichment(None, STATUS_UNDEFINED, "some_other_reason") is None


def test_finite_negative_or_nonnumeric_is_missing():
    assert saturating_enrichment(-0.5, STATUS_FINITE, None) is None
    assert saturating_enrichment("nan", STATUS_FINITE, None) is None


# --- Regression: dispersed-shape case with saturated A-ROI enrichment ---
