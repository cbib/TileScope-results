"""Greenwald 242_T ROI-surface E2E acceptance test.

Opt-in expensive test: exercises `tilescope.api.run_tilescope_from_csv`
on the vendored 242_T two-pair fixture and verifies TileScope's
canonical ROI-grain export surface end-to-end.

Scope split.

- **Structural / identity contract (hard-fail)** — every canonical
  ROI output written, absence of `component_id` from ROI-grain
  scientific outputs, ledger completeness, geometry WKT round-trip,
  multi-component structural complexity, total-table row counts,
  Moran row accounting + bidirectional finite / null coupling, exact
  Cartesian key set on ROI x replicate frames.

- **Independent numeric recomputation on canonical outputs
  (hard-fail)** — native ROI Pearson from fixture spots grouped by
  `roi_id`; consensus Pearson pooled from `consensus_hexes.csv` on
  BOTH `consensus_ensemble` and `consensus_direct` surfaces; native
  enrichment sufficient counts recomputed from fixture spots against
  the sample-wide `q_high` / `q_low` thresholds; consensus ROI
  ensemble enrichment sufficient stats pooled from
  `consensus_hexes.csv`. Tessellation-summary reduction from
  `roi_replicate_enrichment.csv` medians / Q25 / Q75. Two-status
  finite / null coupling on every enrichment surface (ratio vs
  effect-size axes independently).

- **Not covered here — commit-#3 prerequisite** — exact independent
  verification that the correct tiles were pooled into
  `roi_replicate_correlations.csv` / `roi_replicate_enrichment.csv`.
  That requires a canonical per-tile observation surface (per-tile
  densities + gene sums + high / low sufficient counts +
  replicate / component / region provenance) which TileScope does
  not currently export. This test does not fall back on the
  transitional `tessellation_tiles_by_roi` return-dict key, which is
  scheduled for retirement. ROI x replicate frames get their
  structural contract verified here; numeric independent recompute
  waits on the per-tile canonical export landing in commit #3.

- **Also not covered — per-hex ensemble-weighted high / low counts
  from `consensus_hexes.csv`** — the exported hex row carries
  ensemble-weighted `n_spots`, `sum_g1`, `sum_g2` primitives but no
  `n_g1_high_ensemble_weighted` / `n_both_high_ensemble_weighted`
  columns. This test recomputes consensus enrichment target
  sufficient stats to the extent the exported hex columns allow
  (n_spots / sum_g1 / sum_g2 pooling), and verifies target
  algebra + two-status coupling on the ROI-consensus enrichment
  surface; independently reconstructing pooled g1_high / joint_high
  ratios also waits on those primitive columns being added to the
  canonical hex export.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from shapely import wkt

pytestmark = [pytest.mark.real_data, pytest.mark.slow]


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "greenwald_242T"
SAMPLE = "242_T"
PAIRS: Tuple[Tuple[str, str], ...] = (
    ("AC_Chromatin-Reg", "AC_vs_Chromatin-Reg"),
    ("MES_Reactive-Ast", "MES_vs_Reactive-Ast"),
)

# Canonical ROI-grain outputs.
CANONICAL_ROI_OUTPUTS = (
    "roi_correlations.csv",
    "roi_enrichment.csv",
    "roi_support_status.csv",
    "roi_replicate_correlations.csv",
    "roi_replicate_enrichment.csv",
    "roi_tessellation_correlations.csv",
    "roi_tessellation_enrichment.csv",
    "roi_consensus_correlations.csv",
    "roi_consensus_enrichment.csv",
    "roi_source_moran.csv",
    "roi_tessellation_moran.csv",
    "roi_consensus_moran.csv",
)


# Retired shards + return-dict keys that must not resurface.
RETIRED_CSV_SHARDS = (
    "tessellation_0_correlation.csv",
    "tessellation_1_correlation.csv",
    "tessellation_0_enrichment.csv",
    "tessellation_1_enrichment.csv",
    "component_replicate_correlations.csv",
    "consensus_enrichment.csv",
    "spot_ledger_summary.csv",
    # Wrong-grain component-consensus outcomes retired; ROI-grain
    # replacements are roi_consensus_*.csv + the three per-grain
    # roi_*_moran.csv files.
    "consensus_component_outcomes.csv",
    # Consolidated single-grain Moran retired; replaced by the three
    # per-grain files above.
    "roi_moran.csv",
)


# Retired per-hex enrichment columns on consensus_hexes.csv — replaced
# by hex_* columns keyed by hex_enrichment_status. Absence is asserted.
RETIRED_CONSENSUS_HEXES_COLUMNS = (
    "unit_enrichment_status",
    "g1_enrichment", "g2_enrichment",
    "joint_high_enrichment_fc", "joint_low_enrichment_fc",
    "joint_confidence", "joint_confidence_adj",
    "enrichment_direction",
)


# --------------------------------------------------------------------------- #
# Invocation
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pair_outputs(tmp_path_factory) -> Dict[str, Path]:
    """Run TileScope on both fixture pairs once per module. Returns
    ``{dir_pair: out_dir}``.

    The vendored ``run.config.json`` is passed directly — it has
    already been sanitized once (retired ``hull`` / ``hull_adaptive``
    sections removed) at fixture creation and committed as the
    schema-valid current-core snapshot. No runtime compatibility
    shims are applied here.
    """
    from tilescope.api import run_tilescope_from_csv

    root = tmp_path_factory.mktemp("greenwald_242T_e2e")
    outputs: Dict[str, Path] = {}
    for csv_pair, dir_pair in PAIRS:
        out = root / dir_pair
        out.mkdir(parents=True, exist_ok=True)
        input_csv = FIXTURE_DIR / f"{SAMPLE}_expression_data_{csv_pair}.csv"
        run_tilescope_from_csv(
            str(input_csv),
            run_config=str(FIXTURE_DIR / "run.config.json"),
            output_dir=str(out),
            html_report=False,
            dataset_id=0,
            feature_a_id=csv_pair.split("_", 1)[0],
            feature_b_id=csv_pair.split("_", 1)[1],
        )
        outputs[dir_pair] = out
    return outputs


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _pearson_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _component_to_roi(geom: pd.DataFrame) -> Dict[str, int]:
    return dict(
        zip(geom["component_id"].astype(str), geom["roi_id"].astype(int))
    )


# --------------------------------------------------------------------------- #
# Hard-fail structural / identity gates — parametrised across pairs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_every_canonical_roi_output_written(csv_pair, dir_pair, pair_outputs):
    out = pair_outputs[dir_pair]
    for name in CANONICAL_ROI_OUTPUTS:
        assert (out / name).exists(), (
            f"{dir_pair}: canonical ROI-grain output {name!r} not written"
        )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_roi_outputs_do_not_leak_component_id(csv_pair, dir_pair, pair_outputs):
    """Every ROI-grain scientific output must NOT carry a
    ``component_id`` column — component grain is construction, not
    reporting."""
    out = pair_outputs[dir_pair]
    forbidden = "component_id"
    for name in CANONICAL_ROI_OUTPUTS:
        cols = pd.read_csv(out / name, nrows=0).columns
        assert forbidden not in cols, (
            f"{dir_pair}/{name}: leaks component_id — grain contract broken"
        )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_spot_ledger_covers_every_input_spot(csv_pair, dir_pair, pair_outputs):
    input_csv = FIXTURE_DIR / f"{SAMPLE}_expression_data_{csv_pair}.csv"
    spots = pd.read_csv(input_csv)
    ledger = pd.read_csv(pair_outputs[dir_pair] / "spot_ledger.csv")
    assert len(ledger) == len(spots), (
        f"{dir_pair}: ledger row count {len(ledger)} != input CSV rows "
        f"{len(spots)}"
    )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_component_geometry_identity_and_wkt(csv_pair, dir_pair, pair_outputs):
    geom = pd.read_csv(pair_outputs[dir_pair] / "component_geometry.csv")
    assert geom.duplicated(subset=["roi_id", "component_id"]).sum() == 0, (
        f"{dir_pair}: duplicate (roi_id, component_id) rows in "
        "component_geometry.csv"
    )
    for _, row in geom.iterrows():
        poly = wkt.loads(row["geometry_wkt"])
        assert poly.is_valid, (
            f"{dir_pair}/({row['roi_id']},{row['component_id']}): invalid "
            "polygon"
        )
        recorded = float(row["component_domain_area"])
        tol = max(1e-6 * recorded, 1e-9)
        assert abs(poly.area - recorded) < tol, (
            f"{dir_pair}/({row['roi_id']},{row['component_id']}): area drift "
            f"wkt={poly.area:.6f} recorded={recorded:.6f}"
        )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_multi_component_parent_present(csv_pair, dir_pair, pair_outputs):
    """242_T is expected to have at least one parent ROI that splits
    into more than one analysis component under the standard config
    — this exercises sibling-component pooling."""
    ledger = pd.read_csv(pair_outputs[dir_pair] / "spot_ledger.csv")
    counts = (
        ledger.dropna(subset=["component_id"])
        .groupby("roi_id")["component_id"].nunique()
    )
    assert int((counts > 1).sum()) > 0, (
        f"{dir_pair}: no multi-component parent ROI — fixture selection "
        "premise (structural complexity) violated"
    )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_total_table_row_counts(csv_pair, dir_pair, pair_outputs):
    """Every ROI-grain surface is a total table over source parents,
    with surface / direction multipliers where applicable."""
    out = pair_outputs[dir_pair]
    n_rois = len(pd.read_csv(out / "roi_support_status.csv"))
    checks = {
        "roi_correlations.csv": n_rois,
        "roi_enrichment.csv": n_rois,
        "roi_tessellation_correlations.csv": n_rois,
        "roi_tessellation_enrichment.csv": n_rois,
        # consensus correlation: both ensemble + direct surfaces
        "roi_consensus_correlations.csv": 2 * n_rois,
        # consensus enrichment: ensemble surface only
        "roi_consensus_enrichment.csv": n_rois,
        # Moran: three per-grain files.
        # Source: both directions per ROI.
        "roi_source_moran.csv": 2 * n_rois,
        # Tessellation: both directions per ROI.
        "roi_tessellation_moran.csv": 2 * n_rois,
        # Consensus: both directions per ROI × both surfaces (ensemble + direct).
        "roi_consensus_moran.csv": 4 * n_rois,
    }
    for name, expected in checks.items():
        n = len(pd.read_csv(out / name))
        assert n == expected, (
            f"{dir_pair}/{name}: row count {n} != expected {expected} "
            f"(total table over {n_rois} source parents)"
        )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_status_numeric_coupling(csv_pair, dir_pair, pair_outputs):
    """Numeric estimand fields are finite iff the typed status is
    ``finite``. Only surfaces with a single availability status are
    checked here; two-status surfaces (enrichment, effect-size) have
    their own dedicated coupling tests below."""
    out = pair_outputs[dir_pair]
    for name, status_col, numeric_cols in (
        (
            "roi_correlations.csv",
            "parent_correlation_status",
            # Native ROI Pearson uses the legacy `corr_obs` column name;
            # ROI-consensus Pearson uses `correlation_value`. Both are
            # verified here as the same surface-specific contract.
            ("corr_obs",),
        ),
        (
            "roi_tessellation_correlations.csv",
            "parent_tessellation_correlation_status",
            ("corr_median", "corr_q25", "corr_q75"),
        ),
        (
            "roi_consensus_correlations.csv",
            "parent_consensus_correlation_status",
            ("correlation_value",),
        ),
    ):
        df = pd.read_csv(out / name)
        finite = df[status_col] == "finite"
        for col in numeric_cols:
            n_null_finite = int(df.loc[finite, col].isna().sum())
            n_val_nonfinite = int(df.loc[~finite, col].notna().sum())
            assert n_null_finite == 0, (
                f"{dir_pair}/{name}: {n_null_finite} rows with "
                f"status=finite but null {col}"
            )
            assert n_val_nonfinite == 0, (
                f"{dir_pair}/{name}: {n_val_nonfinite} rows with "
                f"non-finite status but populated {col}"
            )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_moran_bidirectional_status_value_coupling(
    csv_pair, dir_pair, pair_outputs,
):
    """For each per-grain Moran surface: no duplicate composite key and
    bidirectional finite / null coupling (``parent_moran_status ==
    'finite'`` iff ``moran_value`` is non-null)."""
    out = pair_outputs[dir_pair]
    for filename, key_cols in (
        ("roi_source_moran.csv",
         ["sample_id", "roi_id", "moran_direction"]),
        ("roi_tessellation_moran.csv",
         ["sample_id", "roi_id", "moran_direction"]),
        ("roi_consensus_moran.csv",
         ["sample_id", "roi_id", "value_surface", "moran_direction"]),
    ):
        df = pd.read_csv(out / filename)
        assert df.duplicated(subset=key_cols).sum() == 0, (
            f"{dir_pair}/{filename}: duplicate {key_cols}"
        )
        finite = df["parent_moran_status"] == "finite"
        n_finite_null = int(df.loc[finite, "moran_value"].isna().sum())
        n_nonfinite_val = int(df.loc[~finite, "moran_value"].notna().sum())
        assert n_finite_null == 0, (
            f"{dir_pair}/{filename}: {n_finite_null} rows with "
            "finite status but null moran_value"
        )
        assert n_nonfinite_val == 0, (
            f"{dir_pair}/{filename}: {n_nonfinite_val} rows with "
            "non-finite status but populated moran_value"
        )


def _n_planned_replicates(out: Path) -> int:
    """Planned tessellation replicate count for this run — read from
    the effective config TileScope emits alongside its outputs."""
    cfg = json.loads((out / "effective_config.json").read_text())
    return int(cfg["run_config"]["tessellation"]["n"])


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_roi_x_replicate_frames_are_exact_cartesian_totals(
    csv_pair, dir_pair, pair_outputs,
):
    """ROI x replicate frames are total tables — assert the exact
    Cartesian key set (``source ROI set`` x ``planned replicate
    set``), not just uniqueness. Structural: reveals both silent
    row drops and silent row duplication."""
    out = pair_outputs[dir_pair]
    n_planned = _n_planned_replicates(out)
    source_rois = set(
        pd.read_csv(out / "roi_support_status.csv")["roi_id"].astype(int)
    )
    expected_keys = {
        (r, i) for r in source_rois for i in range(n_planned)
    }
    for name in ("roi_replicate_correlations.csv",
                 "roi_replicate_enrichment.csv"):
        df = pd.read_csv(out / name)
        emitted_keys = set(
            zip(df["roi_id"].astype(int), df["replicate_index"].astype(int))
        )
        assert emitted_keys == expected_keys, (
            f"{dir_pair}/{name}: emitted key set "
            f"{len(emitted_keys)} rows over "
            f"{len(source_rois)} ROIs x {n_planned} replicates does "
            f"not match the expected Cartesian total table\n"
            f"  missing: {sorted(expected_keys - emitted_keys)}\n"
            f"  extra:   {sorted(emitted_keys - expected_keys)}"
        )


# --------------------------------------------------------------------------- #
# Independent numeric recomputes — canonical outputs only
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_native_roi_pearson_matches_recompute(csv_pair, dir_pair, pair_outputs):
    """Native-spot ROI Pearson recomputed from the fixture spots
    (grouped by ``roi_id``) must match ``roi_correlations.csv``
    (finite rows only) exactly, up to floating-point tolerance.

    Estimand contract. The parent-ROI headline correlation is pooled
    over **all source spots with that ``roi_id``** — not just those
    the ledger later assigns to an analysis component. Confirmed by
    the emitted ``n_roi == sample_size == len(input spots with
    roi_id==r)``. Spots that fail component ownership (e.g. isolated
    spots outside any component's owned support) are included in the
    native ROI Pearson; they only get excluded from the component-
    grain surfaces.
    """
    out = pair_outputs[dir_pair]
    input_csv = FIXTURE_DIR / f"{SAMPLE}_expression_data_{csv_pair}.csv"
    spots = pd.read_csv(input_csv)
    emitted = pd.read_csv(out / "roi_correlations.csv")

    finite = emitted[emitted["parent_correlation_status"] == "finite"]
    assert len(finite) > 0, (
        f"{dir_pair}: no FINITE rows in roi_correlations.csv — cannot "
        "verify against independent recompute"
    )
    for _, row in finite.iterrows():
        roi_id = int(row["roi_id"])
        sub = spots[spots["roi_id"] == roi_id]
        recomputed = _pearson_or_nan(
            sub["gene1"].values.astype(float),
            sub["gene2"].values.astype(float),
        )
        # Sanity check on the estimand population contract itself:
        # the emitted `n_roi` MUST equal the count of source spots
        # for that ROI. If this diverges the estimand contract has
        # silently shifted.
        assert int(row["n_roi"]) == len(sub), (
            f"{dir_pair}/roi={roi_id}: emitted n_roi={row['n_roi']} "
            f"!= source spot count {len(sub)} — parent-ROI native "
            "correlation is no longer pooled over all source spots"
        )
        # Native ROI Pearson is emitted under `corr_obs` (legacy column
        # name on this pre-migration surface); the ROI-consensus
        # surface uses `correlation_value`.
        emitted_val = float(row["corr_obs"])
        assert math.isfinite(recomputed), (
            f"{dir_pair}/roi={roi_id}: emitted FINITE but independent "
            "recompute produced NaN — source-population disagreement"
        )
        assert abs(recomputed - emitted_val) < 1e-9, (
            f"{dir_pair}/roi={roi_id}: native Pearson emitted="
            f"{emitted_val:.12f} vs recomputed={recomputed:.12f}"
        )


_CONSENSUS_SURFACES = (
    ("consensus_ensemble", "gene1_density_ensemble", "gene2_density_ensemble"),
    ("consensus_direct",   "gene1_density_direct",   "gene2_density_direct"),
)


@pytest.mark.parametrize("surface,g1_col,g2_col", _CONSENSUS_SURFACES)
@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_consensus_roi_pearson_matches_recompute(
    csv_pair, dir_pair, surface, g1_col, g2_col, pair_outputs,
):
    """Pooled per-hex Pearson on the specified consensus surface
    (ensemble or direct), computed independently from
    ``consensus_hexes.csv``, must match
    ``roi_consensus_correlations.csv`` (that surface, FINITE rows)
    exactly up to floating-point tolerance.

    The consensus builder documents finite-density hexes as
    equally-weighted (no area / exposure / tessellation-count
    weighting), so a plain ``np.corrcoef`` on the pooled per-hex
    densities is the correct independent estimator.
    """
    out = pair_outputs[dir_pair]
    hexes = pd.read_csv(out / "consensus_hexes.csv")
    geom = pd.read_csv(out / "component_geometry.csv")
    emitted = pd.read_csv(out / "roi_consensus_correlations.csv")

    density_cols = (g1_col, g2_col)
    if not all(c in hexes.columns for c in density_cols):
        pytest.skip(
            f"consensus_hexes.csv missing density columns {density_cols!r} "
            f"for surface {surface!r}"
        )

    df = hexes.copy()
    df["roi_id"] = df["component_id"].astype(str).map(
        _component_to_roi(geom)
    )
    df = df[df[list(density_cols)].notna().all(axis=1)]

    surf_rows = emitted[emitted["value_surface"] == surface]
    finite = surf_rows[
        surf_rows["parent_consensus_correlation_status"] == "finite"
    ]
    assert len(finite) > 0, (
        f"{dir_pair}: no FINITE rows in "
        f"roi_consensus_correlations.csv/{surface} — cannot verify "
        "against independent recompute"
    )
    for _, row in finite.iterrows():
        roi_id = int(row["roi_id"])
        pooled = df[df["roi_id"] == roi_id]
        recomputed = _pearson_or_nan(
            pooled[g1_col].values.astype(float),
            pooled[g2_col].values.astype(float),
        )
        emitted_val = float(row["correlation_value"])
        assert math.isfinite(recomputed), (
            f"{dir_pair}/{surface}/roi={roi_id}: emitted FINITE but "
            "independent recompute produced NaN — pooled-hex population "
            "disagreement"
        )
        assert abs(recomputed - emitted_val) < 1e-9, (
            f"{dir_pair}/{surface}/roi={roi_id}: consensus Pearson "
            f"emitted={emitted_val:.12f} vs recomputed={recomputed:.12f}"
        )


# --------------------------------------------------------------------------- #
# Enrichment recomputes + two-status coupling + tessellation reduction
# --------------------------------------------------------------------------- #


_RATIO_COLS = (
    "g1_enrichment", "g2_enrichment",
    "joint_high_enrichment_fc", "joint_low_enrichment_fc",
)
_RATIO_STATUS_BY_COL = {
    "g1_enrichment": "g1_enrichment_status",
    "g2_enrichment": "g2_enrichment_status",
    "joint_high_enrichment_fc": "joint_high_enrichment_status",
    "joint_low_enrichment_fc": "joint_low_enrichment_status",
}
_RATIO_COUNT_BY_COL = {
    "g1_enrichment": "n_valid_g1_enrichment_replicates",
    "g2_enrichment": "n_valid_g2_enrichment_replicates",
    "joint_high_enrichment_fc": "n_valid_joint_high_enrichment_replicates",
    "joint_low_enrichment_fc": "n_valid_joint_low_enrichment_replicates",
}
_EFFECT_COLS = ("log2fc_g1", "log2fc_g2", "CLES_g1", "CLES_g2")


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_native_roi_enrichment_sufficient_counts_recompute(
    csv_pair, dir_pair, pair_outputs,
):
    """Recompute the native-ROI sufficient counts (`n_g1_high`,
    `n_g2_high`, `n_both_high`, `n_g1_low`, `n_g2_low`, `n_both_low`,
    plus `n_roi`, `n_ctx`) from the fixture input CSV alone, using
    the sample-wide ``q_high`` / ``q_low`` thresholds recorded in the
    effective config. Exact match required for every emitted ROI row.

    The estimand population is all source spots (grouped by
    ``roi_id``) — same contract as the native ROI Pearson, cross-
    checked here on the enrichment side.
    """
    out = pair_outputs[dir_pair]
    cfg = json.loads((out / "effective_config.json").read_text())
    q_high = float(cfg["resolved_policy"]["thresholds"]["q_high"])
    q_low = float(cfg["resolved_policy"]["thresholds"]["q_low"])

    spots = pd.read_csv(
        FIXTURE_DIR / f"{SAMPLE}_expression_data_{csv_pair}.csv"
    )
    g1_hi = float(np.quantile(spots["gene1"], q_high))
    g2_hi = float(np.quantile(spots["gene2"], q_high))
    g1_lo = float(np.quantile(spots["gene1"], q_low))
    g2_lo = float(np.quantile(spots["gene2"], q_low))

    emitted = pd.read_csv(out / "roi_enrichment.csv")
    n_total = len(spots)
    for _, row in emitted.iterrows():
        if row["parent_enrichment_status"] != "finite":
            continue
        roi_id = int(row["roi_id"])
        target = spots[spots["roi_id"] == roi_id]
        n_roi = len(target)
        n_ctx = n_total - n_roi
        expected = {
            "n_roi": n_roi,
            "n_ctx": n_ctx,
            "n_g1_high": int((target["gene1"] >= g1_hi).sum()),
            "n_g2_high": int((target["gene2"] >= g2_hi).sum()),
            "n_both_high": int(
                ((target["gene1"] >= g1_hi) & (target["gene2"] >= g2_hi)).sum()
            ),
            "n_g1_low": int((target["gene1"] <= g1_lo).sum()),
            "n_g2_low": int((target["gene2"] <= g2_lo).sum()),
            "n_both_low": int(
                ((target["gene1"] <= g1_lo) & (target["gene2"] <= g2_lo)).sum()
            ),
        }
        for col, want in expected.items():
            assert int(row[col]) == want, (
                f"{dir_pair}/roi={roi_id}: native enrichment "
                f"{col} emitted={row[col]} vs recomputed={want}"
            )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_consensus_ensemble_enrichment_sufficient_stats_recompute(
    csv_pair, dir_pair, pair_outputs,
):
    """Recompute the ROI-consensus ensemble sufficient masses
    (`obs_equiv_spot_mass`, `obs_equiv_g*_sum_mass`) from
    ``consensus_hexes.csv`` alone. ``consensus_hexes.csv`` publishes
    the normalized masses directly (no raw accumulator columns), so
    the recompute is a simple groupby-sum across the ROI's
    finite-density sibling hexes.

    All three masses are asserted to floating-point tolerance —
    every field is Float64 end-to-end (no rounding on export).

    Also asserts every emitted enrichment ratio reconstructs from
    target vs context mass pairs (``obs_equiv_g1_high_mass /
    obs_equiv_spot_mass ÷ context_obs_equiv_g1_high_mass /
    context_obs_equiv_spot_mass``) to floating-point tolerance —
    the "self-contained row" contract on FINITE rows.
    """
    out = pair_outputs[dir_pair]
    hexes = pd.read_csv(out / "consensus_hexes.csv")
    geom = pd.read_csv(out / "component_geometry.csv")
    emitted = pd.read_csv(out / "roi_consensus_enrichment.csv")

    df = hexes.copy()
    df["roi_id"] = df["component_id"].astype(str).map(
        _component_to_roi(geom)
    )
    # Builder's FINITE gate on the ensemble surface: hex has finite
    # ensemble densities on both genes AND positive spot mass.
    df = df[
        df["gene1_density_ensemble"].notna()
        & df["gene2_density_ensemble"].notna()
        & (df["obs_equiv_spot_mass"] > 0)
    ]
    pooled = df.groupby("roi_id")[
        [
            "obs_equiv_spot_mass",
            "obs_equiv_g1_sum_mass", "obs_equiv_g2_sum_mass",
        ]
    ].sum()

    ens = emitted[emitted["value_surface"] == "consensus_ensemble"]
    finite = ens[ens["parent_consensus_enrichment_status"] == "finite"]
    assert len(finite) > 0, (
        f"{dir_pair}: no FINITE rows in "
        "roi_consensus_enrichment.csv/consensus_ensemble — cannot "
        "verify against independent recompute"
    )
    for _, row in finite.iterrows():
        roi_id = int(row["roi_id"])
        assert roi_id in pooled.index, (
            f"{dir_pair}/roi={roi_id}: FINITE enrichment row but no "
            "matching finite-density hexes in consensus_hexes.csv"
        )
        p = pooled.loc[roi_id]
        for col in (
            "obs_equiv_spot_mass",
            "obs_equiv_g1_sum_mass", "obs_equiv_g2_sum_mass",
        ):
            emitted_val = float(row[col])
            recomputed = float(p[col])
            assert abs(emitted_val - recomputed) < 1e-9, (
                f"{dir_pair}/roi={roi_id}: {col} emitted={emitted_val} "
                f"vs pooled-from-consensus_hexes={recomputed}"
            )

        # Self-contained ratio-reconstruction from target + context
        # mass pairs on the same row.
        for emitted_col, tgt_num_col, ctx_num_col in (
            ("g1_enrichment",
             "obs_equiv_g1_high_mass", "context_obs_equiv_g1_high_mass"),
            ("g2_enrichment",
             "obs_equiv_g2_high_mass", "context_obs_equiv_g2_high_mass"),
            ("joint_high_enrichment_fc",
             "obs_equiv_both_high_mass", "context_obs_equiv_both_high_mass"),
            ("joint_low_enrichment_fc",
             "obs_equiv_both_low_mass", "context_obs_equiv_both_low_mass"),
        ):
            tgt_rate = float(row[tgt_num_col]) / float(row["obs_equiv_spot_mass"])
            ctx_rate = (
                float(row[ctx_num_col])
                / float(row["context_obs_equiv_spot_mass"])
            )
            if ctx_rate == 0.0:
                # A required context proportion of zero would push the
                # enrichment status off FINITE; skip defensively.
                continue
            recomputed = tgt_rate / ctx_rate
            assert abs(recomputed - float(row[emitted_col])) < 1e-9, (
                f"{dir_pair}/roi={roi_id}: {emitted_col} "
                f"emitted={row[emitted_col]} vs reconstructed from "
                f"target/context mass pair={recomputed}"
            )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_replicate_enrichment_target_algebra(csv_pair, dir_pair, pair_outputs):
    """Structural algebra on the emitted replicate sufficient stats:
    for every FINITE-enrichment replicate row, target `n_both_high`
    is bounded by both `n_g1_high` and `n_g2_high`, `n_both_low` by
    the two `n_g*_low`, and every count is bounded above by
    `n_spots`. Guards against silent count corruption in the
    replicate-level extractor even without per-tile independent
    reconstruction.
    """
    df = pd.read_csv(pair_outputs[dir_pair] / "roi_replicate_enrichment.csv")
    finite = df[df["parent_replicate_enrichment_status"] == "finite"]
    assert len(finite) > 0, (
        f"{dir_pair}: no FINITE rows in roi_replicate_enrichment.csv"
    )
    for _, row in finite.iterrows():
        n = int(row["n_spots"])
        for lo, hi in (
            ("n_both_high", "n_g1_high"),
            ("n_both_high", "n_g2_high"),
            ("n_both_low", "n_g1_low"),
            ("n_both_low", "n_g2_low"),
        ):
            assert int(row[lo]) <= int(row[hi]), (
                f"{dir_pair}/roi={row['roi_id']}/rep={row['replicate_index']}: "
                f"{lo}={row[lo]} > {hi}={row[hi]}"
            )
        for c in ("n_g1_high", "n_g2_high", "n_both_high",
                  "n_g1_low", "n_g2_low", "n_both_low"):
            assert 0 <= int(row[c]) <= n, (
                f"{dir_pair}/roi={row['roi_id']}/rep={row['replicate_index']}: "
                f"{c}={row[c]} outside [0, n_spots={n}]"
            )


def _quantiles(series: pd.Series) -> Tuple[float, float, float]:
    return (
        float(series.median()),
        float(series.quantile(0.25)),
        float(series.quantile(0.75)),
    )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_tessellation_enrichment_matches_replicate_reduction(
    csv_pair, dir_pair, pair_outputs,
):
    """`roi_tessellation_enrichment.csv` is a pure reduction of
    `roi_replicate_enrichment.csv`. Verify:

    - Each ratio median / Q25 / Q75 reduces over rows where that ratio's
      own typed status is ``finite``. This includes available ratios on a
      row whose aggregate status is ``context_proportion_zero`` because a
      different denominator was zero.
    - Effect-size medians / Q25 / Q75 (log2fc_g1/g2, CLES_g1/g2)
      reduce INDEPENDENTLY over rows where
      ``parent_replicate_effect_size_status == 'finite'``.

    Exact equality (float tolerance).
    """
    out = pair_outputs[dir_pair]
    rep = pd.read_csv(out / "roi_replicate_enrichment.csv")
    summary = pd.read_csv(out / "roi_tessellation_enrichment.csv")

    for _, row in summary.iterrows():
        roi_id = int(row["roi_id"])
        rep_r = rep[rep["roi_id"] == roi_id]

        # Ratio reduction.
        for col in _RATIO_COLS:
            status_col = _RATIO_STATUS_BY_COL[col]
            ratio_finite = rep_r[rep_r[status_col] == "finite"]
            count_col = _RATIO_COUNT_BY_COL[col]
            assert int(row[count_col]) == len(ratio_finite)
            if len(ratio_finite) > 0:
                med, q25, q75 = _quantiles(ratio_finite[col])
                for suffix, want in (
                    ("median", med), ("q25", q25), ("q75", q75),
                ):
                    got = float(row[f"{col}_{suffix}"])
                    assert abs(got - want) < 1e-9, (
                        f"{dir_pair}/roi={roi_id}: {col}_{suffix} "
                        f"emitted={got} vs recomputed={want}"
                    )
            else:
                assert all(
                    pd.isna(row[f"{col}_{suffix}"])
                    for suffix in ("median", "q25", "q75")
                )

        # Effect-size reduction.
        eff_finite = rep_r[
            rep_r["parent_replicate_effect_size_status"] == "finite"
        ]
        if row["parent_tessellation_effect_size_status"] == "finite":
            assert len(eff_finite) > 0, (
                f"{dir_pair}/roi={roi_id}: summary effect-size FINITE "
                "but no FINITE replicate rows to reduce over"
            )
            for col in _EFFECT_COLS:
                med, q25, q75 = _quantiles(eff_finite[col])
                for suffix, want in (
                    ("median", med), ("q25", q25), ("q75", q75),
                ):
                    got = float(row[f"{col}_{suffix}"])
                    assert abs(got - want) < 1e-9, (
                        f"{dir_pair}/roi={roi_id}: {col}_{suffix} "
                        f"emitted={got} vs recomputed={want}"
                    )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_enrichment_two_status_coupling(csv_pair, dir_pair, pair_outputs):
    """Per-ratio and effect-size finite/null coupling on every surface.

    Each ratio value follows its own typed status; the aggregate enrichment
    status may be ``context_proportion_zero`` when only one ratio is
    unavailable. Effect-size columns independently follow their aggregate
    effect-size status. The tessellation-summary reductions are checked above.
    """
    out = pair_outputs[dir_pair]
    surfaces = (
        (
            "roi_replicate_enrichment.csv",
            "parent_replicate_enrichment_status",
            "parent_replicate_effect_size_status",
            _RATIO_COLS,
            _EFFECT_COLS,
        ),
        (
            "roi_consensus_enrichment.csv",
            "parent_consensus_enrichment_status",
            "parent_consensus_effect_size_status",
            _RATIO_COLS,
            _EFFECT_COLS,
        ),
    )
    for name, ratio_status, effect_status, ratio_cols, effect_cols in surfaces:
        df = pd.read_csv(out / name)
        effect_finite = df[effect_status] == "finite"
        for col in ratio_cols:
            per_ratio_status = _RATIO_STATUS_BY_COL[col]
            ratio_finite = df[per_ratio_status] == "finite"
            n_null_finite = int(df.loc[ratio_finite, col].isna().sum())
            n_val_nonfinite = int(df.loc[~ratio_finite, col].notna().sum())
            assert n_null_finite == 0 and n_val_nonfinite == 0, (
                f"{dir_pair}/{name}: two-status coupling broken on "
                f"ratio column {col!r} (status={per_ratio_status!r}; "
                f"aggregate={ratio_status!r}): "
                f"finite/null={n_null_finite}, "
                f"non_finite/value={n_val_nonfinite}"
            )
        for col in effect_cols:
            n_null_finite = int(df.loc[effect_finite, col].isna().sum())
            n_val_nonfinite = int(df.loc[~effect_finite, col].notna().sum())
            assert n_null_finite == 0 and n_val_nonfinite == 0, (
                f"{dir_pair}/{name}: two-status coupling broken on "
                f"effect-size column {col!r} (status={effect_status!r}): "
                f"finite/null={n_null_finite}, "
                f"non_finite/value={n_val_nonfinite}"
            )


# --------------------------------------------------------------------------- #
# Canonical per-tile surface.
#
# Since `tessellation_tiles.csv` is a first-class publication frame,
# the sister smoke drives the ROI x replicate Pearson +
# enrichment recompute from per-tile densities, cross-checked against the
# corresponding aggregate ROI x replicate frames.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_tessellation_tiles_csv_present_and_key_shape(
    csv_pair, dir_pair, pair_outputs,
):
    """`tessellation_tiles.csv` is written, composite key
    `(sample_id, component_id, domain_region_id, replicate_index,
    tile_id)` is unique, and `tile_id` is a plain integer (no
    `"{tessellation_id}_tile_{n}"` string encoding).
    """
    out = pair_outputs[dir_pair]
    tiles_path = out / "tessellation_tiles.csv"
    assert tiles_path.exists(), (
        f"{dir_pair}: tessellation_tiles.csv not written"
    )
    tiles = pd.read_csv(tiles_path)
    assert len(tiles) > 0, f"{dir_pair}: tessellation_tiles.csv is empty"

    key_cols = [
        "sample_id", "component_id", "domain_region_id",
        "replicate_index", "tile_id",
    ]
    assert not tiles.duplicated(subset=key_cols).any(), (
        f"{dir_pair}/tessellation_tiles.csv: duplicate composite key rows"
    )

    # Plain-integer tile_id — no legacy string encoding.
    assert pd.api.types.is_integer_dtype(tiles["tile_id"]), (
        f"{dir_pair}/tessellation_tiles.csv: tile_id is not an integer "
        f"dtype (got {tiles['tile_id'].dtype!r}); "
        "legacy `\"{tessellation_id}_tile_{n}\"` encoding must not "
        "resurface"
    )
    for offender in ("_tile_",):
        assert not tiles["tile_id"].astype(str).str.contains(offender).any(), (
            f"{dir_pair}/tessellation_tiles.csv: tile_id values contain "
            f"legacy substring {offender!r}"
        )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_tessellation_tiles_roi_id_functional_dependency(
    csv_pair, dir_pair, pair_outputs,
):
    """`(sample_id, component_id) -> exactly one roi_id`, cross-checked
    against `component_geometry.csv`'s authoritative mapping.
    """
    out = pair_outputs[dir_pair]
    tiles = pd.read_csv(out / "tessellation_tiles.csv")
    geom = pd.read_csv(out / "component_geometry.csv")

    per_component_roi = (
        tiles.groupby(["sample_id", "component_id"])["roi_id"].nunique()
    )
    offenders = per_component_roi[per_component_roi > 1]
    assert offenders.empty, (
        f"{dir_pair}/tessellation_tiles.csv: (sample_id, component_id) "
        f"with multiple roi_id values: {offenders.to_dict()!r}"
    )

    tile_map = (
        tiles.drop_duplicates(subset=["component_id"])
        .set_index("component_id")["roi_id"]
        .astype(int)
        .to_dict()
    )
    geom_map = _component_to_roi(geom)
    for cid, tile_roi in tile_map.items():
        auth_roi = geom_map.get(str(cid))
        assert auth_roi is not None, (
            f"{dir_pair}: tessellation_tiles.csv component_id={cid!r} "
            "missing from component_geometry.csv"
        )
        assert int(auth_roi) == int(tile_roi), (
            f"{dir_pair}/cid={cid}: tessellation_tiles.csv roi_id="
            f"{tile_roi} disagrees with component_geometry.csv roi_id="
            f"{auth_roi}"
        )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_tessellation_tiles_sufficient_count_coherence(
    csv_pair, dir_pair, pair_outputs,
):
    """Per-tile sufficient counts are internally coherent:

    - every `n_g*_high` / `n_g*_low` / `n_both_*` in `[0, n_spots]`
    - `n_both_high <= min(n_g1_high, n_g2_high)`
    - `n_both_low  <= min(n_g1_low, n_g2_low)`
    - `gene*_density` null iff `n_spots == 0`
    """
    tiles = pd.read_csv(
        pair_outputs[dir_pair] / "tessellation_tiles.csv"
    )
    n_spots = tiles["n_spots"].astype("int64")

    for col in (
        "n_g1_high", "n_g2_high", "n_both_high",
        "n_g1_low", "n_g2_low", "n_both_low",
    ):
        c = tiles[col].astype("int64")
        assert (c >= 0).all(), (
            f"{dir_pair}: negative sufficient count in {col}"
        )
        assert (c <= n_spots).all(), (
            f"{dir_pair}: {col} exceeds n_spots on some tile"
        )

    assert (
        tiles["n_both_high"].astype("int64")
        <= tiles[["n_g1_high", "n_g2_high"]].astype("int64").min(axis=1)
    ).all(), f"{dir_pair}: n_both_high > min(n_g1_high, n_g2_high)"
    assert (
        tiles["n_both_low"].astype("int64")
        <= tiles[["n_g1_low", "n_g2_low"]].astype("int64").min(axis=1)
    ).all(), f"{dir_pair}: n_both_low > min(n_g1_low, n_g2_low)"

    zero = n_spots == 0
    if zero.any():
        assert tiles.loc[zero, "g1_density"].isna().all(), (
            f"{dir_pair}: g1_density non-null on zero-spot tile"
        )
        assert tiles.loc[zero, "g2_density"].isna().all(), (
            f"{dir_pair}: g2_density non-null on zero-spot tile"
        )
    assert tiles.loc[~zero, "g1_density"].notna().all(), (
        f"{dir_pair}: g1_density null on nonzero-spot tile"
    )
    assert tiles.loc[~zero, "g2_density"].notna().all(), (
        f"{dir_pair}: g2_density null on nonzero-spot tile"
    )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_retired_return_keys_and_shards_absent(
    csv_pair, dir_pair, pair_outputs,
):
    """Retired CSV shards must not resurface in the output directory,
    and the retired return-dict keys must not appear on the API
    surface either.
    """
    out = pair_outputs[dir_pair]
    for name in RETIRED_CSV_SHARDS:
        assert not (out / name).exists(), (
            f"{dir_pair}: retired shard {name!r} present in output dir"
        )

    from tilescope.api import run_tilescope_from_csv
    input_csv = FIXTURE_DIR / f"{SAMPLE}_expression_data_{csv_pair}.csv"
    results = run_tilescope_from_csv(
        str(input_csv),
        run_config=str(FIXTURE_DIR / "run.config.json"),
        output_dir=str(out / "__retirement_probe"),
        html_report=False,
        dataset_id=0,
        feature_a_id=csv_pair.split("_", 1)[0],
        feature_b_id=csv_pair.split("_", 1)[1],
    )
    for retired_key in (
        "tessellation_tiles_by_roi",
        "tessellation_tiles_objects_by_roi",
    ):
        assert retired_key not in results, (
            f"{dir_pair}: retired return-dict key {retired_key!r} "
            "resurfaced on the run_tilescope_from_csv API"
        )
    assert "tessellation_tiles_df" in results, (
        f"{dir_pair}: canonical tessellation_tiles_df absent from "
        "return dict"
    )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_roi_replicate_pearson_matches_pooled_tile_recompute(
    csv_pair, dir_pair, pair_outputs,
):
    """For every FINITE row in `roi_replicate_correlations.csv`,
    pool all sibling-component tiles of that source ROI at that
    replicate from `tessellation_tiles.csv` and recompute the Pearson
    of `(g1_density, g2_density)`. Must match the emitted
    `correlation_value` to floating-point tolerance.

    This is the exact numeric independent recompute the pre-#3b
    docstring flagged as blocked pending the canonical per-tile
    export.
    """
    out = pair_outputs[dir_pair]
    tiles = pd.read_csv(out / "tessellation_tiles.csv")
    emitted = pd.read_csv(out / "roi_replicate_correlations.csv")

    # Tile-level density filter: skip zero-spot tiles (density undefined
    # by contract) — the ROI x replicate estimator equally weights the
    # finite-density tiles.
    tile_pool = tiles[
        tiles["g1_density"].notna()
        & tiles["g2_density"].notna()
    ]

    finite = emitted[emitted["parent_replicate_correlation_status"] == "finite"]
    assert len(finite) > 0, (
        f"{dir_pair}: no FINITE rows in roi_replicate_correlations.csv"
    )
    for _, row in finite.iterrows():
        roi_id = int(row["roi_id"])
        rep = int(row["replicate_index"])
        pooled = tile_pool[
            (tile_pool["roi_id"].astype(int) == roi_id)
            & (tile_pool["replicate_index"].astype(int) == rep)
        ]
        recomputed = _pearson_or_nan(
            pooled["g1_density"].values.astype(float),
            pooled["g2_density"].values.astype(float),
        )
        emitted_val = float(row["correlation_value"])
        assert math.isfinite(recomputed), (
            f"{dir_pair}/roi={roi_id}/rep={rep}: FINITE emitted but "
            "pooled-tile recompute NaN — pool population disagreement"
        )
        assert abs(recomputed - emitted_val) < 1e-9, (
            f"{dir_pair}/roi={roi_id}/rep={rep}: pooled-tile Pearson "
            f"emitted={emitted_val:.12f} vs "
            f"recomputed={recomputed:.12f}"
        )
        # Population coherence: emitted `sample_size` == number of
        # finite-density tiles pooled from the canonical surface.
        assert int(row["sample_size"]) == int(len(pooled)), (
            f"{dir_pair}/roi={roi_id}/rep={rep}: sample_size="
            f"{row['sample_size']} != pooled tile count {len(pooled)}"
        )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_roi_replicate_enrichment_sufficient_counts_from_tiles(
    csv_pair, dir_pair, pair_outputs,
):
    """For every FINITE-enrichment row in
    `roi_replicate_enrichment.csv`, recompute the target sufficient
    stats from `tessellation_tiles.csv` by classifying each finite-
    density tile against the sample-wide density thresholds
    (`q_g*_threshold` derived from source spots at
    ``thresholds.q_high`` / ``q_low``) and pooling siblings of the
    source ROI at that replicate. The emitted `n_spots` on this
    surface is the count of finite-density tiles (not source-spot
    count); the emitted `sum_g*` is the sum of tile densities
    (not the sum of tile ``sum_g*`` accumulators).

    This is the exact independent recompute the pre-#3b docstring
    flagged as blocked pending the per-tile canonical export.
    """
    out = pair_outputs[dir_pair]
    tiles = pd.read_csv(out / "tessellation_tiles.csv")
    emitted = pd.read_csv(out / "roi_replicate_enrichment.csv")
    cfg = json.loads((out / "effective_config.json").read_text())
    spots = pd.read_csv(
        FIXTURE_DIR / f"{SAMPLE}_expression_data_{csv_pair}.csv"
    )
    q_high = float(cfg["resolved_policy"]["thresholds"]["q_high"])
    q_low = float(cfg["resolved_policy"]["thresholds"]["q_low"])
    q_g1_hi = float(np.quantile(spots["gene1"], q_high))
    q_g2_hi = float(np.quantile(spots["gene2"], q_high))
    q_g1_lo = float(np.quantile(spots["gene1"], q_low))
    q_g2_lo = float(np.quantile(spots["gene2"], q_low))

    finite_tiles = tiles[
        tiles["g1_density"].notna() & tiles["g2_density"].notna()
    ].copy()
    finite_tiles["is_g1_high"] = finite_tiles["g1_density"] >= q_g1_hi
    finite_tiles["is_g2_high"] = finite_tiles["g2_density"] >= q_g2_hi
    finite_tiles["is_g1_low"] = finite_tiles["g1_density"] <= q_g1_lo
    finite_tiles["is_g2_low"] = finite_tiles["g2_density"] <= q_g2_lo

    finite = emitted[emitted["parent_replicate_enrichment_status"] == "finite"]
    assert len(finite) > 0, (
        f"{dir_pair}: no FINITE rows in roi_replicate_enrichment.csv"
    )
    for _, row in finite.iterrows():
        roi_id = int(row["roi_id"])
        rep = int(row["replicate_index"])
        pool = finite_tiles[
            (finite_tiles["roi_id"].astype(int) == roi_id)
            & (finite_tiles["replicate_index"].astype(int) == rep)
        ]

        # Sufficient counts — tile-classification counts, not source-spot
        # rollups.
        want_counts = {
            "n_spots": len(pool),
            "n_g1_high": int(pool["is_g1_high"].sum()),
            "n_g2_high": int(pool["is_g2_high"].sum()),
            "n_both_high": int(
                (pool["is_g1_high"] & pool["is_g2_high"]).sum()
            ),
            "n_g1_low": int(pool["is_g1_low"].sum()),
            "n_g2_low": int(pool["is_g2_low"].sum()),
            "n_both_low": int(
                (pool["is_g1_low"] & pool["is_g2_low"]).sum()
            ),
        }
        for col, want in want_counts.items():
            got = int(row[col])
            assert got == want, (
                f"{dir_pair}/roi={roi_id}/rep={rep}: {col}={got} "
                f"!= tile-classification recompute {want}"
            )
        # Density sums (sum of per-tile densities).
        for col, series in (
            ("sum_g1", pool["g1_density"]),
            ("sum_g2", pool["g2_density"]),
        ):
            want = float(series.sum())
            got = float(row[col])
            assert abs(got - want) < 1e-9, (
                f"{dir_pair}/roi={roi_id}/rep={rep}: {col}={got:.9f} "
                f"!= sum-of-tile-density {want:.9f}"
            )


# --------------------------------------------------------------------------- #
# Consensus_hexes.csv retirement + per-hex enrichment recompute
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_retired_consensus_shards_and_columns_absent(
    csv_pair, dir_pair, pair_outputs,
):
    """consensus_component_outcomes.csv must NOT be written (wrong grain);
    consensus_hexes.csv must NOT carry the legacy per-hex enrichment
    columns (replaced by hex_* + hex_enrichment_status)."""
    out = pair_outputs[dir_pair]
    assert not (out / "consensus_component_outcomes.csv").exists(), (
        f"{dir_pair}: consensus_component_outcomes.csv survived — retired "
        "with the component-grain consensus statistics."
    )
    hexes_cols = pd.read_csv(out / "consensus_hexes.csv", nrows=0).columns
    leaked = set(RETIRED_CONSENSUS_HEXES_COLUMNS).intersection(hexes_cols)
    assert not leaked, (
        f"{dir_pair}: consensus_hexes.csv carries retired per-hex enrichment "
        f"column(s) {sorted(leaked)!r}; use hex_* + hex_enrichment_status."
    )
    for required in (
        "hex_enrichment_status",
        "parent_consensus_enrichment_status",
        "hex_g1_enrichment", "hex_g2_enrichment",
        "hex_joint_high_enrichment_fc", "hex_joint_low_enrichment_fc",
    ):
        assert required in hexes_cols, (
            f"{dir_pair}: consensus_hexes.csv missing required column {required!r}."
        )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_hex_enrichment_ratios_reconstruct_from_masses(
    csv_pair, dir_pair, pair_outputs,
):
    """For every finite hex row, hex_* ratios must equal the target rate
    (obs_equiv_*_mass / obs_equiv_spot_mass) divided by the ROI-level
    context rate on roi_consensus_enrichment.csv (ensemble surface)."""
    out = pair_outputs[dir_pair]
    hexes = pd.read_csv(out / "consensus_hexes.csv")
    geom = pd.read_csv(out / "component_geometry.csv")
    roi_enr = pd.read_csv(out / "roi_consensus_enrichment.csv")

    hexes = hexes.copy()
    hexes["roi_id"] = hexes["component_id"].astype(str).map(
        _component_to_roi(geom)
    )
    ens = roi_enr[roi_enr["value_surface"] == "consensus_ensemble"]
    roi_by_id = {
        int(row["roi_id"]): row for _, row in ens.iterrows()
    }
    finite = hexes[hexes["hex_enrichment_status"] == "finite"]
    if finite.empty:
        pytest.skip("no finite hexes to verify")
    for _, hex_row in finite.iterrows():
        roi_id = int(hex_row["roi_id"])
        roi_row = roi_by_id[roi_id]
        spot = float(hex_row["obs_equiv_spot_mass"])
        ctx_spot = float(roi_row["context_obs_equiv_spot_mass"])
        for hex_col, tgt_col, ctx_col in (
            ("hex_g1_enrichment",
             "obs_equiv_g1_high_mass", "context_obs_equiv_g1_high_mass"),
            ("hex_g2_enrichment",
             "obs_equiv_g2_high_mass", "context_obs_equiv_g2_high_mass"),
            ("hex_joint_high_enrichment_fc",
             "obs_equiv_both_high_mass", "context_obs_equiv_both_high_mass"),
            ("hex_joint_low_enrichment_fc",
             "obs_equiv_both_low_mass", "context_obs_equiv_both_low_mass"),
        ):
            tgt_rate = float(hex_row[tgt_col]) / spot
            ctx_rate = float(roi_row[ctx_col]) / ctx_spot
            recomputed = tgt_rate / ctx_rate
            emitted = float(hex_row[hex_col])
            assert abs(recomputed - emitted) < 1e-9, (
                f"{dir_pair}/roi={roi_id}/hex_id={hex_row['hex_id']}: "
                f"{hex_col} emitted={emitted} vs reconstructed={recomputed}"
            )


@pytest.mark.parametrize("csv_pair,dir_pair", PAIRS)
def test_weighted_hex_ratio_matches_roi_fc(csv_pair, dir_pair, pair_outputs):
    """Weighted coherence identity: for each ROI with any finite hex,
    Σ_h (m_h_spot · hex_FC_h) / Σ_h m_h_spot equals the ROI's FC on
    roi_consensus_enrichment.csv to floating-point tolerance."""
    out = pair_outputs[dir_pair]
    hexes = pd.read_csv(out / "consensus_hexes.csv")
    geom = pd.read_csv(out / "component_geometry.csv")
    roi_enr = pd.read_csv(out / "roi_consensus_enrichment.csv")

    hexes = hexes.copy()
    hexes["roi_id"] = hexes["component_id"].astype(str).map(
        _component_to_roi(geom)
    )
    ens = roi_enr[roi_enr["value_surface"] == "consensus_ensemble"]
    finite = hexes[hexes["hex_enrichment_status"] == "finite"]
    if finite.empty:
        pytest.skip("no finite hexes to verify")
    for roi_id, group in finite.groupby("roi_id"):
        roi_row = ens[ens["roi_id"] == int(roi_id)]
        if roi_row.empty:
            continue
        roi_row = roi_row.iloc[0]
        weights = group["obs_equiv_spot_mass"].astype(float).to_numpy()
        w_sum = float(weights.sum())
        if w_sum <= 0:
            continue
        for hex_col, roi_col in (
            ("hex_g1_enrichment", "g1_enrichment"),
            ("hex_g2_enrichment", "g2_enrichment"),
            ("hex_joint_high_enrichment_fc", "joint_high_enrichment_fc"),
            ("hex_joint_low_enrichment_fc", "joint_low_enrichment_fc"),
        ):
            hex_vals = group[hex_col].astype(float).to_numpy()
            weighted_mean = float((weights * hex_vals).sum() / w_sum)
            roi_fc = float(roi_row[roi_col])
            assert abs(weighted_mean - roi_fc) < 1e-9, (
                f"{dir_pair}/roi={roi_id}: weighted-hex mean of "
                f"{hex_col}={weighted_mean} vs ROI FC {roi_col}={roi_fc}"
            )
