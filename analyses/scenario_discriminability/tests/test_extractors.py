"""Extractor registry contract tests.

Kept intentionally small: one row-schema test per representative
registry class, plus the two contracts corrections must not
regress (partial per-channel tessellation availability, missing-
export raises).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractors import (  # noqa: E402
    ATLAS_COLUMNS, REGISTRY, MissingExportError,
    extract_all, extract_one_csv, to_dataframe,
)


CONTEXT = {
    "case_id": "242_T__roi3",
    "scenario_name": "broad_colocalization",
    "scenario_mode": None,
    "sample_id": "242_T",
    "dataset": "greenwald",
    "selection_basis": "manually_reviewed_reference",
    "role": "positive_broad",
    "role_disposition": "feasible",
}


def _make_csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_roi_correlations_shape(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "roi_id": [1, 2],
        "corr_obs": [0.42, -0.10],
        "parent_correlation_status": ["finite", "insufficient_data"],
    })
    csv = _make_csv(tmp_path, "roi_correlations.csv", df)
    spec = next(s for s in REGISTRY if s.filename == "roi_correlations.csv")
    rows = extract_one_csv(csv, spec, CONTEXT)
    assert len(rows) == 2
    finite = next(r for r in rows if r["roi_id"] == 1)
    undef = next(r for r in rows if r["roi_id"] == 2)
    assert finite["statistic"] == "pearson"
    assert finite["value"] == pytest.approx(0.42)
    assert finite["availability_status"] == "finite"
    assert finite["undefined_reason"] is None
    assert finite["surface"] == "source_native"
    assert finite["grain"] == "roi"
    assert undef["value"] is None
    assert undef["undefined_reason"] == "insufficient_data"


def test_roi_replicate_correlations_passthrough(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "roi_id": [1, 1, 2],
        "replicate_index": [0, 1, 0],
        "correlation_value": [0.3, 0.5, 0.7],
        "parent_replicate_correlation_status": ["finite", "finite", "finite"],
    })
    csv = _make_csv(tmp_path, "roi_replicate_correlations.csv", df)
    spec = next(
        s for s in REGISTRY if s.filename == "roi_replicate_correlations.csv"
    )
    rows = extract_one_csv(csv, spec, CONTEXT)
    assert len(rows) == 3
    assert [r["replicate_index"] for r in rows] == [0, 1, 0]
    assert rows[0]["grain"] == "roi_replicate"
    assert rows[0]["surface"] == "tessellation"


def test_tessellation_enrichment_preserves_partial_channels(
    tmp_path: Path,
) -> None:
    """A ROI with aggregate partial_ratio_availability where only
    joint_high has zero valid replicates MUST still emit finite
    rows for the other three channels + their effect-size
    reductions."""
    # Construct one ROI that has partial availability: g1 / g2 /
    # joint_low have 3 valid replicates; joint_high has 0.
    row = {
        "roi_id": 1,
        "parent_tessellation_enrichment_status": "partial_ratio_availability",
        "parent_tessellation_effect_size_status": "finite",
        # Per-channel counts
        "n_valid_g1_enrichment_replicates": 3,
        "n_valid_g2_enrichment_replicates": 3,
        "n_valid_joint_high_enrichment_replicates": 0,
        "n_valid_joint_low_enrichment_replicates": 3,
        "n_valid_effect_size_replicates": 3,
        # Enrichment reductions
        "g1_enrichment_median": 1.5, "g1_enrichment_q25": 1.2, "g1_enrichment_q75": 1.8,
        "g2_enrichment_median": 1.1, "g2_enrichment_q25": 1.0, "g2_enrichment_q75": 1.2,
        "joint_high_enrichment_fc_median": None,
        "joint_high_enrichment_fc_q25": None,
        "joint_high_enrichment_fc_q75": None,
        "joint_low_enrichment_fc_median": 0.9,
        "joint_low_enrichment_fc_q25": 0.8, "joint_low_enrichment_fc_q75": 1.0,
        # Effect-size reductions
        "log2fc_g1_median": 0.4, "log2fc_g1_q25": 0.3, "log2fc_g1_q75": 0.5,
        "log2fc_g2_median": 0.1, "log2fc_g2_q25": 0.0, "log2fc_g2_q75": 0.2,
        "CLES_g1_median": 0.6, "CLES_g1_q25": 0.5, "CLES_g1_q75": 0.7,
        "CLES_g2_median": 0.55, "CLES_g2_q25": 0.5, "CLES_g2_q75": 0.6,
    }
    df = pd.DataFrame([row])
    csv = _make_csv(tmp_path, "roi_tessellation_enrichment.csv", df)
    spec = next(
        s for s in REGISTRY if s.filename == "roi_tessellation_enrichment.csv"
    )
    rows = extract_one_csv(csv, spec, CONTEXT)
    # 4 enrichment channels × 3 reductions + 2 log2fc channels × 3
    # + 2 CLES channels × 3 = 12 + 6 + 6 = 24 rows per ROI.
    assert len(rows) == 24
    # g1 median finite; joint_high median undefined with the aggregate
    # parent status as the reason (not "count_missing").
    g1_med = next(
        r for r in rows
        if r["statistic"] == "enrichment_median" and r["channel"] == "g1"
    )
    jhi_med = next(
        r for r in rows
        if r["statistic"] == "enrichment_median" and r["channel"] == "joint_high"
    )
    assert g1_med["availability_status"] == "finite"
    assert g1_med["value"] == pytest.approx(1.5)
    assert jhi_med["availability_status"] == "undefined"
    assert jhi_med["undefined_reason"] == "partial_ratio_availability"
    assert jhi_med["value"] is None
    # Effect-size reductions gated on n_valid_effect_size_replicates.
    log2_g1_med = next(
        r for r in rows
        if r["statistic"] == "log2fc_median" and r["channel"] == "g1"
    )
    assert log2_g1_med["availability_status"] == "finite"
    assert log2_g1_med["value"] == pytest.approx(0.4)


def test_moran_direction_passthrough(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "roi_id": [1, 1, 2, 2],
        "moran_direction": ["g1_lag_g2", "g2_lag_g1", "g1_lag_g2", "g2_lag_g1"],
        "moran_value": [0.11, 0.22, None, None],
        "parent_moran_status": [
            "finite", "finite",
            "no_tessellation_eligible_component",
            "no_tessellation_eligible_component",
        ],
        "moran_reason": [None, None, None, None],
    })
    csv = _make_csv(tmp_path, "roi_source_moran.csv", df)
    spec = next(s for s in REGISTRY if s.filename == "roi_source_moran.csv")
    rows = extract_one_csv(csv, spec, CONTEXT)
    assert len(rows) == 4
    assert [r["moran_direction"] for r in rows] == [
        "g1_lag_g2", "g2_lag_g1", "g1_lag_g2", "g2_lag_g1",
    ]
    assert rows[0]["statistic"] == "moran"
    assert rows[0]["surface"] == "source"
    assert rows[0]["value"] == pytest.approx(0.11)
    assert rows[2]["value"] is None
    assert rows[2]["undefined_reason"] == "no_tessellation_eligible_component"


def test_missing_source_file_raises(tmp_path: Path) -> None:
    spec = next(s for s in REGISTRY if s.filename == "roi_correlations.csv")
    with pytest.raises(MissingExportError, match="declared source CSV missing"):
        extract_one_csv(tmp_path / "roi_correlations.csv", spec, CONTEXT)


def test_missing_required_column_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"roi_id": [1], "corr_obs": [0.5]})  # no status col
    csv = _make_csv(tmp_path, "roi_correlations.csv", df)
    spec = next(s for s in REGISTRY if s.filename == "roi_correlations.csv")
    with pytest.raises(MissingExportError, match="missing required columns"):
        extract_one_csv(csv, spec, CONTEXT)


def test_to_dataframe_pads_missing_columns() -> None:
    df = to_dataframe([{"case_id": "x", "statistic": "pearson"}])
    assert list(df.columns) == ATLAS_COLUMNS
    assert df.iloc[0]["case_id"] == "x"
    assert pd.isna(df.iloc[0]["scenario_name"])
