"""scenario_discriminability atlas extractors.

Each extractor consumes one TileScope output CSV and yields tidy atlas
rows. Two grain families:

- Per-ROI grains (``roi``, ``roi_replicate``): keyed by
  ``(source_file, statistic, channel, grain, surface, value_surface,
  moran_direction, replicate_index, roi_id)``. ``responder_roi_id`` is
  NaN.
- Pair-scoped grain (``roi_pair``): additionally keyed by
  ``responder_roi_id``. ``roi_id`` carries the anchor ROI id and
  ``responder_roi_id`` carries the responder ROI id, so directional
  pair statistics survive the atlas + response-matrix merge without
  collapsing.

The value + availability_status per row come from a declared column
pair inside that source CSV; provenance / QC / sample-size columns
are NOT promoted to atlas rows.

Availability contract per ValueSpec:
  - If ``min_valid_count_column`` is set: the row is finite iff that
    column's integer value is > 0. Otherwise availability is
    undefined and the reason is drawn from ``fallback_status_column``.
    Used for tessellation reductions where an aggregate status may
    read "partial_ratio_availability" but individual channels still
    carry valid median / Q25 / Q75 values.
  - Else ``status_column`` is authoritative: finite iff it equals
    FINITE_STATUS. When the per-row status is missing (NaN), the
    optional ``fallback_status_column`` supplies the reason so the
    atlas records e.g. "no_tessellation_eligible_component" instead
    of a generic "status_missing".

Missing declared files or required columns raise MissingExportError
— a broken TileScope export must never look like scientific
unavailability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# Atlas row schema (kept in one place to prevent drift between extractors).
# --------------------------------------------------------------------------- #

ATLAS_COLUMNS: List[str] = [
    "case_id",
    "scenario_name",
    "scenario_mode",
    "sample_id",
    "dataset",
    "selection_basis",
    "role",
    "role_disposition",
    "roi_id",
    "responder_roi_id",
    "surface",
    "grain",
    "replicate_index",
    "value_surface",
    "moran_direction",
    "statistic",
    "channel",
    "value",
    "availability_status",
    "undefined_reason",
    "source_file",
]

# Availability convention: a run emitted a numeric value iff its
# per-row status column carries this literal.
FINITE_STATUS = "finite"


class MissingExportError(RuntimeError):
    """Declared source CSV or required column absent from a role's output."""


@dataclass(frozen=True)
class ValueSpec:
    """One declared estimand extracted from a source CSV.

    Exactly one of ``status_column`` or ``min_valid_count_column``
    must be set. ``fallback_status_column`` is optional in both
    modes.
    """
    statistic: str
    channel: Optional[str]
    value_column: str
    status_column: Optional[str] = None
    min_valid_count_column: Optional[str] = None
    fallback_status_column: Optional[str] = None

    def __post_init__(self) -> None:
        has_status = self.status_column is not None
        has_count = self.min_valid_count_column is not None
        if has_status == has_count:
            raise ValueError(
                f"ValueSpec {self.statistic!r}/{self.channel!r}: exactly "
                f"one of status_column or min_valid_count_column must be set"
            )


@dataclass(frozen=True)
class SourceSpec:
    """Extraction recipe for one source CSV. All declared columns are
    required; missing files or columns raise MissingExportError."""
    filename: str
    grain: str
    surface: str
    values: List[ValueSpec]
    # Optional per-row columns lifted verbatim into the tidy row.
    passthrough: Dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Registry — declared estimands only.
#
# Deliberately NOT included in the first atlas:
#   - obs_equiv_* mass columns (provenance sums, not estimands).
#   - Per-row sample sizes / eligibility counts / coverage fractions.
#   - Analytic / permutation / FDR p-values (QC signals, not the
#     scientific statistic under evaluation in this atlas).
#   - Categorical enrichment_direction strings.
# --------------------------------------------------------------------------- #

_ENRICHMENT_CHANNELS = [
    ValueSpec(
        "enrichment", "g1", "g1_enrichment",
        status_column="g1_enrichment_status",
        fallback_status_column="parent_enrichment_status",
    ),
    ValueSpec(
        "enrichment", "g2", "g2_enrichment",
        status_column="g2_enrichment_status",
        fallback_status_column="parent_enrichment_status",
    ),
    ValueSpec(
        "enrichment", "joint_high", "joint_high_enrichment_fc",
        status_column="joint_high_enrichment_status",
        fallback_status_column="parent_enrichment_status",
    ),
    ValueSpec(
        "enrichment", "joint_low", "joint_low_enrichment_fc",
        status_column="joint_low_enrichment_status",
        fallback_status_column="parent_enrichment_status",
    ),
]


def _tessellation_reduction_values() -> List[ValueSpec]:
    """Median + Q25 + Q75 per channel for enrichment + effect-size
    families. Each channel is gated on its own per-channel valid-
    replicate count so that aggregate ``partial_ratio_availability``
    does not discard channels with real reductions."""
    out: List[ValueSpec] = []
    enrichment_channel_map = {
        "g1": ("g1_enrichment", "n_valid_g1_enrichment_replicates"),
        "g2": ("g2_enrichment", "n_valid_g2_enrichment_replicates"),
        "joint_high": (
            "joint_high_enrichment_fc",
            "n_valid_joint_high_enrichment_replicates",
        ),
        "joint_low": (
            "joint_low_enrichment_fc",
            "n_valid_joint_low_enrichment_replicates",
        ),
    }
    for channel, (base, count_col) in enrichment_channel_map.items():
        for stat_suffix, col_suffix in (
            ("median", "median"), ("q25", "q25"), ("q75", "q75"),
        ):
            out.append(ValueSpec(
                f"enrichment_{stat_suffix}", channel,
                f"{base}_{col_suffix}",
                min_valid_count_column=count_col,
                fallback_status_column="parent_tessellation_enrichment_status",
            ))
    # Effect-size reductions share n_valid_effect_size_replicates.
    for family, col_prefix in (("log2fc", "log2fc"), ("cles", "CLES")):
        for channel, suffix in (("g1", "g1"), ("g2", "g2")):
            for stat_suffix, col_suffix in (
                ("median", "median"), ("q25", "q25"), ("q75", "q75"),
            ):
                out.append(ValueSpec(
                    f"{family}_{stat_suffix}", channel,
                    f"{col_prefix}_{suffix}_{col_suffix}",
                    min_valid_count_column="n_valid_effect_size_replicates",
                    fallback_status_column=(
                        "parent_tessellation_effect_size_status"
                    ),
                ))
    return out


REGISTRY: List[SourceSpec] = [
    SourceSpec(
        filename="roi_correlations.csv",
        grain="roi",
        surface="source_native",
        values=[
            ValueSpec(
                "pearson", None, "corr_obs",
                status_column="parent_correlation_status",
            ),
        ],
    ),
    SourceSpec(
        filename="roi_enrichment.csv",
        grain="roi",
        surface="source_native",
        values=_ENRICHMENT_CHANNELS + [
            ValueSpec(
                "log2fc", "g1", "log2fc_g1",
                status_column="parent_effect_size_status",
            ),
            ValueSpec(
                "log2fc", "g2", "log2fc_g2",
                status_column="parent_effect_size_status",
            ),
            ValueSpec(
                "cles", "g1", "CLES_g1",
                status_column="parent_effect_size_status",
            ),
            ValueSpec(
                "cles", "g2", "CLES_g2",
                status_column="parent_effect_size_status",
            ),
        ],
    ),
    SourceSpec(
        filename="roi_replicate_correlations.csv",
        grain="roi_replicate",
        surface="tessellation",
        values=[
            ValueSpec(
                "pearson", None, "correlation_value",
                status_column="parent_replicate_correlation_status",
            ),
        ],
        passthrough={"replicate_index": "replicate_index"},
    ),
    SourceSpec(
        filename="roi_tessellation_correlations.csv",
        grain="roi",
        surface="tessellation",
        values=[
            ValueSpec(
                "pearson_median", None, "corr_median",
                status_column="parent_tessellation_correlation_status",
            ),
            ValueSpec(
                "pearson_q25", None, "corr_q25",
                status_column="parent_tessellation_correlation_status",
            ),
            ValueSpec(
                "pearson_q75", None, "corr_q75",
                status_column="parent_tessellation_correlation_status",
            ),
        ],
    ),
    SourceSpec(
        filename="roi_replicate_enrichment.csv",
        grain="roi_replicate",
        surface="tessellation",
        values=[
            ValueSpec(
                "enrichment", "g1", "g1_enrichment",
                status_column="g1_enrichment_status",
                fallback_status_column="parent_replicate_enrichment_status",
            ),
            ValueSpec(
                "enrichment", "g2", "g2_enrichment",
                status_column="g2_enrichment_status",
                fallback_status_column="parent_replicate_enrichment_status",
            ),
            ValueSpec(
                "enrichment", "joint_high", "joint_high_enrichment_fc",
                status_column="joint_high_enrichment_status",
                fallback_status_column="parent_replicate_enrichment_status",
            ),
            ValueSpec(
                "enrichment", "joint_low", "joint_low_enrichment_fc",
                status_column="joint_low_enrichment_status",
                fallback_status_column="parent_replicate_enrichment_status",
            ),
            ValueSpec(
                "log2fc", "g1", "log2fc_g1",
                status_column="parent_replicate_effect_size_status",
            ),
            ValueSpec(
                "log2fc", "g2", "log2fc_g2",
                status_column="parent_replicate_effect_size_status",
            ),
            ValueSpec(
                "cles", "g1", "CLES_g1",
                status_column="parent_replicate_effect_size_status",
            ),
            ValueSpec(
                "cles", "g2", "CLES_g2",
                status_column="parent_replicate_effect_size_status",
            ),
        ],
        passthrough={"replicate_index": "replicate_index"},
    ),
    SourceSpec(
        filename="roi_tessellation_enrichment.csv",
        grain="roi",
        surface="tessellation",
        values=_tessellation_reduction_values(),
    ),
    SourceSpec(
        filename="roi_consensus_correlations.csv",
        grain="roi",
        surface="consensus",
        values=[
            ValueSpec(
                "pearson", None, "correlation_value",
                status_column="parent_consensus_correlation_status",
            ),
        ],
        passthrough={"value_surface": "value_surface"},
    ),
    SourceSpec(
        filename="roi_consensus_enrichment.csv",
        grain="roi",
        surface="consensus",
        values=[
            ValueSpec(
                "enrichment", "g1", "g1_enrichment",
                status_column="g1_enrichment_status",
                fallback_status_column="parent_consensus_enrichment_status",
            ),
            ValueSpec(
                "enrichment", "g2", "g2_enrichment",
                status_column="g2_enrichment_status",
                fallback_status_column="parent_consensus_enrichment_status",
            ),
            ValueSpec(
                "enrichment", "joint_high", "joint_high_enrichment_fc",
                status_column="joint_high_enrichment_status",
                fallback_status_column="parent_consensus_enrichment_status",
            ),
            ValueSpec(
                "enrichment", "joint_low", "joint_low_enrichment_fc",
                status_column="joint_low_enrichment_status",
                fallback_status_column="parent_consensus_enrichment_status",
            ),
            ValueSpec(
                "log2fc", "g1", "log2fc_g1",
                status_column="parent_consensus_effect_size_status",
            ),
            ValueSpec(
                "log2fc", "g2", "log2fc_g2",
                status_column="parent_consensus_effect_size_status",
            ),
            ValueSpec(
                "cles", "g1", "CLES_g1",
                status_column="parent_consensus_effect_size_status",
            ),
            ValueSpec(
                "cles", "g2", "CLES_g2",
                status_column="parent_consensus_effect_size_status",
            ),
        ],
        passthrough={"value_surface": "value_surface"},
    ),
    SourceSpec(
        filename="roi_source_moran.csv",
        grain="roi",
        surface="source",
        values=[
            ValueSpec(
                "moran", None, "moran_value",
                status_column="parent_moran_status",
                fallback_status_column="moran_reason",
            ),
        ],
        passthrough={"moran_direction": "moran_direction"},
    ),
    SourceSpec(
        filename="roi_tessellation_moran.csv",
        grain="roi",
        surface="tessellation",
        values=[
            ValueSpec(
                "moran", None, "moran_value",
                status_column="parent_moran_status",
                fallback_status_column="moran_reason",
            ),
        ],
        passthrough={"moran_direction": "moran_direction"},
    ),
    SourceSpec(
        filename="roi_consensus_moran.csv",
        grain="roi",
        surface="consensus",
        values=[
            ValueSpec(
                "moran", None, "moran_value",
                status_column="parent_moran_status",
                fallback_status_column="moran_reason",
            ),
        ],
        passthrough={
            "moran_direction": "moran_direction",
            "value_surface": "value_surface",
        },
    ),
]


def _resolve_availability(
    csv_row: pd.Series, value_spec: ValueSpec,
) -> tuple[str, Optional[str]]:
    """Compute (availability_status, undefined_reason) per the row +
    the value spec's finite-gate mode."""
    fallback_raw = (
        csv_row[value_spec.fallback_status_column]
        if value_spec.fallback_status_column is not None else None
    )
    fallback_reason = (
        str(fallback_raw) if fallback_raw is not None and not pd.isna(fallback_raw)
        else None
    )
    if value_spec.min_valid_count_column is not None:
        n_raw = csv_row[value_spec.min_valid_count_column]
        if pd.isna(n_raw):
            return "undefined", fallback_reason or "count_missing"
        if int(n_raw) > 0:
            return FINITE_STATUS, None
        return "undefined", fallback_reason or "no_valid_replicates"
    # status_column mode
    status_raw = csv_row[value_spec.status_column]
    if pd.isna(status_raw):
        return "undefined", fallback_reason or "status_missing"
    text = str(status_raw)
    if text == FINITE_STATUS:
        return FINITE_STATUS, None
    return "undefined", text


def _required_columns(spec: SourceSpec) -> List[str]:
    cols: List[str] = []
    for v in spec.values:
        cols.append(v.value_column)
        if v.status_column is not None:
            cols.append(v.status_column)
        if v.min_valid_count_column is not None:
            cols.append(v.min_valid_count_column)
        if v.fallback_status_column is not None:
            cols.append(v.fallback_status_column)
    cols.extend(spec.passthrough.values())
    # roi_id is universal across the 9 source CSVs.
    cols.append("roi_id")
    return sorted(set(cols))


def extract_one_csv(
    csv_path: Path, spec: SourceSpec, context: Dict[str, object],
) -> List[Dict[str, object]]:
    """Extract tidy atlas rows from one source CSV per its registry spec.

    Raises MissingExportError if the file is missing or any declared
    column is absent. Missing values within a row are handled by the
    availability contract (never a raise).
    """
    if not csv_path.exists():
        raise MissingExportError(
            f"declared source CSV missing: {csv_path}"
        )
    df = pd.read_csv(csv_path)
    required = _required_columns(spec)
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise MissingExportError(
            f"{csv_path.name} missing required columns: {missing_cols}"
        )
    rows: List[Dict[str, object]] = []
    for _, csv_row in df.iterrows():
        base = dict(context)
        base["grain"] = spec.grain
        base["surface"] = spec.surface
        base["source_file"] = spec.filename
        base["roi_id"] = csv_row["roi_id"]
        for atlas_col, source_col in spec.passthrough.items():
            base[atlas_col] = csv_row[source_col]
        for value_spec in spec.values:
            row = dict(base)
            row["statistic"] = value_spec.statistic
            row["channel"] = value_spec.channel
            availability, undefined_reason = _resolve_availability(
                csv_row, value_spec,
            )
            row["availability_status"] = availability
            row["undefined_reason"] = undefined_reason
            raw_value = csv_row[value_spec.value_column]
            row["value"] = (
                float(raw_value)
                if availability == FINITE_STATUS and not pd.isna(raw_value)
                else None
            )
            rows.append(row)
    return rows


INTERFACE_RESPONSE_STATISTIC = "interface_response"
INTERFACE_RESPONSE_GRAIN = "roi_pair"
INTERFACE_RESPONSE_SOURCE = "roi_pair_interface_response.csv"
INTERFACE_RESPONSE_SURFACES = ("source_native",)


def _extract_interface_response(
    run_dir: Path, context: Dict[str, object],
) -> List[Dict[str, object]]:
    """Emit atlas rows for every ordered adjacent (anchor, responder)
    ROI pair at the declared surfaces.

    Scenario-blind: no participant filter, no plan / manifest lookup.
    Selection of the case-defined A/B pair belongs in the recognition
    layer, which resolves the pair via ``roi_id`` (anchor) +
    ``responder_roi_id``. Background adjacent pairs remain available
    for generic real-data analyses that reason across all adjacent
    pairs.

    Channel mapping is generic: ``responder_feature_id`` is matched
    per-row against the CSV's ``g1_id`` / ``g2_id`` columns and
    translated to ``g1`` / ``g2``. Rows whose responder feature does
    not equal either sample-declared identity are rejected — the
    identity mapping must be unambiguous per pair.
    """
    csv_path = run_dir / INTERFACE_RESPONSE_SOURCE
    if not csv_path.exists():
        raise MissingExportError(f"declared source CSV missing: {csv_path}")
    df = pd.read_csv(
        csv_path,
        dtype={"anchor_roi_id": "string", "responder_roi_id": "string",
               "responder_feature_id": "string", "surface": "string",
               "availability_status": "string",
               "g1_id": "string", "g2_id": "string"},
    )
    required = [
        "anchor_roi_id", "responder_roi_id", "responder_feature_id",
        "g1_id", "g2_id",
        "surface", "availability_status", "interface_response",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise MissingExportError(
            f"{csv_path.name} missing required columns: {missing_cols}"
        )

    rows: List[Dict[str, object]] = []
    for _, r in df.iterrows():
        surface = r["surface"]
        if surface not in INTERFACE_RESPONSE_SURFACES:
            continue
        responder_feat = str(r["responder_feature_id"])
        g1_id = str(r["g1_id"])
        g2_id = str(r["g2_id"])
        if responder_feat == g1_id:
            channel = "g1"
        elif responder_feat == g2_id:
            channel = "g2"
        else:
            raise MissingExportError(
                f"{csv_path.name} row has responder_feature_id "
                f"{responder_feat!r} not matching g1_id {g1_id!r} or "
                f"g2_id {g2_id!r}; core-side identity mapping is ambiguous"
            )
        status_raw = r["availability_status"]
        status_str = str(status_raw) if not pd.isna(status_raw) else None
        raw_value = r["interface_response"]
        if status_str == FINITE_STATUS and not pd.isna(raw_value):
            availability, undefined_reason = FINITE_STATUS, None
            value: Optional[float] = float(raw_value)
        else:
            availability = "undefined"
            undefined_reason = status_str or "status_missing"
            value = None
        row = dict(context)
        row["grain"] = INTERFACE_RESPONSE_GRAIN
        row["surface"] = surface
        row["source_file"] = INTERFACE_RESPONSE_SOURCE
        row["roi_id"] = str(r["anchor_roi_id"])
        row["responder_roi_id"] = str(r["responder_roi_id"])
        row["statistic"] = INTERFACE_RESPONSE_STATISTIC
        row["channel"] = channel
        row["availability_status"] = availability
        row["undefined_reason"] = undefined_reason
        row["value"] = value
        rows.append(row)
    return rows


def extract_all(
    run_dir: Path, context: Dict[str, object],
) -> List[Dict[str, object]]:
    """Run every registry spec + interface_response extractor against
    `run_dir`, concatenating rows. Propagates MissingExportError."""
    out: List[Dict[str, object]] = []
    for spec in REGISTRY:
        out.extend(extract_one_csv(run_dir / spec.filename, spec, context))
    out.extend(_extract_interface_response(run_dir, context))
    return out


def to_dataframe(rows: Iterable[Dict[str, object]]) -> pd.DataFrame:
    """Assemble atlas rows with the frozen column order + nulls filled."""
    df = pd.DataFrame(list(rows))
    for col in ATLAS_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[ATLAS_COLUMNS]
