"""Sister-side support-class classifier == TileScope's canonical.

CONTRACT.md § 1.1 forbids TileScope imports from
``src/scenario_benchmark/v2/`` but not from tests. This module
imports TileScope solely to prove that our sister-side classifier
reproduces ``derive_roi_support_status`` byte-for-byte on real
substrates.

Any drift here means the sister-side gate has silently diverged
from what TileScope will actually do downstream.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from tilescope_results.paths import data_root

from scenario_benchmark import load_greenwald_sample_scaffold
from scenario_benchmark.support_class import (
    SAMPLE_SUPPORT_FACTOR,
    SUPPORT_FLOOR_STANDARD_V2,
    classify_roi_support,
    compute_component_sizes_by_roi,
)


DATA_ROOT = data_root()
CASES = [
    (
        "gw_259T_ACNPC",
        DATA_ROOT / "heiland_tyrosh" / "CSVs"
        / "259_T_expression_data_AC_NPC.csv",
        "259_T",
    ),
    (
        "hcc_TS20_FibrosisFAP",
        DATA_ROOT / "HCC_human" / "TS20" / "dchalopin_signatures" / "CSVs"
        / "TS20_expression_data_Fibrosis_FAP_CAF.csv",
        "TS20",
    ),
]


def _tilescope_reference(csv_path: Path) -> pd.DataFrame:
    """Run TileScope's canonical classifier on the same substrate.
    Returns the DataFrame emitted by
    ``derive_roi_support_status`` under standard-v2 policy."""
    from tilescope.analyzers.support_eligibility import (
        DEFAULT_SAMPLE_SUPPORT_FACTOR,
        classify_sample_support,
        derive_roi_support_status,
    )
    assert DEFAULT_SAMPLE_SUPPORT_FACTOR == SAMPLE_SUPPORT_FACTOR, (
        "Sister SAMPLE_SUPPORT_FACTOR drifted from TileScope default"
    )
    tsdf = pd.read_csv(csv_path)
    tsdf["roi_id"] = tsdf["roi_id"].astype(str)
    comp_sizes, _median_nn = classify_sample_support(
        tsdf, factor=SAMPLE_SUPPORT_FACTOR,
    )
    return derive_roi_support_status(
        comp_sizes, min_component_spots=SUPPORT_FLOOR_STANDARD_V2,
    )


@pytest.mark.parametrize("label,csv_path,sample_id", CASES)
def test_sister_matches_tilescope_component_sizes(label, csv_path, sample_id):
    """Per-ROI component-size vectors are byte-identical."""
    if not csv_path.exists():
        pytest.skip(f"substrate not present: {csv_path}")
    scaf = load_greenwald_sample_scaffold(
        csv_path, sample_id=sample_id,
        radius_factor=SAMPLE_SUPPORT_FACTOR,
    )
    sister_sizes = compute_component_sizes_by_roi(scaf)
    ref_df = _tilescope_reference(csv_path)

    from tilescope.analyzers.support_eligibility import classify_sample_support
    tsdf = pd.read_csv(csv_path)
    tsdf["roi_id"] = tsdf["roi_id"].astype(str)
    ref_sizes, _ = classify_sample_support(tsdf, factor=SAMPLE_SUPPORT_FACTOR)

    assert set(sister_sizes) == set(ref_sizes), (
        f"ROI sets differ: sister={set(sister_sizes)} "
        f"ref={set(ref_sizes)}"
    )
    for roi, ref in ref_sizes.items():
        assert sister_sizes[roi] == ref, (
            f"{label} ROI={roi}: sister={sister_sizes[roi]} != "
            f"ref={ref}"
        )


@pytest.mark.parametrize("label,csv_path,sample_id", CASES)
def test_sister_matches_tilescope_spatial_support_class(
    label, csv_path, sample_id,
):
    """Per-ROI 4-class classification matches TileScope's."""
    if not csv_path.exists():
        pytest.skip(f"substrate not present: {csv_path}")
    scaf = load_greenwald_sample_scaffold(
        csv_path, sample_id=sample_id,
        radius_factor=SAMPLE_SUPPORT_FACTOR,
    )
    ref_df = _tilescope_reference(csv_path).set_index("roi_id")
    for roi, ref_row in ref_df.iterrows():
        status = classify_roi_support(scaf, roi)
        assert status.spatial_support_class == ref_row["spatial_support_class"], (
            f"{label} ROI={roi}: sister={status.spatial_support_class!r} "
            f"!= ref={ref_row['spatial_support_class']!r}"
        )
        assert status.source_spots == int(ref_row["source_spots"])
        assert status.support_component_count == int(
            ref_row["support_component_count"]
        )
        assert status.support_qualified_component_count == int(
            ref_row["support_qualified_component_count"]
        )
        assert status.support_qualified_spots == int(
            ref_row["support_qualified_spots"]
        )
        assert status.largest_support_component_spots == int(
            ref_row["largest_support_component_spots"]
        )


def test_policy_constants_frozen_at_standard_v2():
    """Sister policy constants match TileScope's defaults."""
    from tilescope.defaults import DEFAULT_MIN_COMPONENT_SPOTS
    from tilescope.analyzers.support_eligibility import (
        DEFAULT_SAMPLE_SUPPORT_FACTOR,
    )
    # The production component floor is user-configurable in the new
    # technology-aware schema; its packaged Visium default remains 15.
    assert DEFAULT_MIN_COMPONENT_SPOTS == SUPPORT_FLOOR_STANDARD_V2
    assert DEFAULT_SAMPLE_SUPPORT_FACTOR == SAMPLE_SUPPORT_FACTOR
