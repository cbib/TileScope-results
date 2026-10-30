"""Scientific extraction semantics; classifier tests retired with the classifier."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import all_statistics as audit


def test_invalid_status_does_not_promote_numeric_placeholder(monkeypatch):
    monkeypatch.setattr(audit, "ROWS", [])
    monkeypatch.setattr(audit, "COLUMN_USE", {("f.csv", "x"): set()})
    case = SimpleNamespace(case_id="c", class_label="A", source_sample_id="s", a_roi_id="1", b_roi_id="2")
    audit.register(case, "A_ROI", "source_native", "pearson", "", "correlation", 0.75,
                   "insufficient_support", "f.csv", "x")
    assert np.isnan(audit.ROWS[0]["value"])
    assert audit.ROWS[0]["raw_value"] == 0.75


def test_ratio_saturation_and_zero_over_zero_differ(monkeypatch):
    monkeypatch.setattr(audit, "ROWS", [])
    monkeypatch.setattr(audit, "COLUMN_USE", {("f.csv", "x"): set()})
    case = SimpleNamespace(case_id="c", class_label="A", source_sample_id="s", a_roi_id="1", b_roi_id="2")
    for status in ("positive_over_zero_context", "zero_over_zero_context"):
        audit.register(case, "A_ROI", "source_native", "enrichment", "g1", "marginal_enrichment",
                       np.nan, status, "f.csv", "x")
    assert audit.ROWS[0]["value"] == 1.
    assert np.isnan(audit.ROWS[1]["value"])


def test_zero_distance_is_correlation_not_distance():
    catalog = pd.read_csv(audit.OUT / "feature_catalogue.csv")
    assert catalog.loc[catalog.statistic.eq("rho_zero"), "group"].eq("correlation").all()
    assert set(catalog.loc[catalog.group.eq("distance"), "statistic"]) == {"shell_0", "shell_1", "shell_2", "shell_3"}


def test_effect_features_do_not_include_inference_or_support():
    catalog = pd.read_csv(audit.OUT / "feature_catalogue.csv")
    effects = catalog.loc[catalog.use.eq("effect")]
    assert set(effects.group) == set(audit.GROUPS)
    assert not effects.group.str.contains("diagnostic").any()
    assert not effects.feature.str.contains("replicate=").any()
