"""Serialization surfaces for v2 scenario instances.

Two projections:

- **TileScope input** — exactly the 5-column ``(x, y, gene1, gene2,
  roi_id)`` format required by :func:`tilescope.api.core.run_tilescope_from_csv`.
  Field mapping: ``feature_a_value -> gene1``, ``feature_b_value ->
  gene2``, ``generated_roi_id -> roi_id``. Emitted by
  :func:`write_tilescope_input_csv`.
- **Truth ledger** — every :class:`SpotTruthRow` field (elevation
  flags, structure metadata, lineage, source-interface hops). Emitted
  by :func:`write_truth_ledger_csv`.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Iterable

import pandas as pd

from scenario_benchmark.truth_schemas import SpotTruthRow


# TileScope input — exactly the columns TileScope core requires.
TILESCOPE_INPUT_COLUMNS = ("x", "y", "gene1", "gene2", "roi_id")

# Full enriched truth ledger — every SpotTruthRow field.
TRUTH_LEDGER_COLUMNS = tuple(f.name for f in fields(SpotTruthRow))


def rows_to_tilescope_dataframe(rows: Iterable[SpotTruthRow]) -> pd.DataFrame:
    """Emit exactly ``TILESCOPE_INPUT_COLUMNS`` from truth rows.

    Field mapping: feature_a_value -> gene1, feature_b_value -> gene2,
    generated_roi_id -> roi_id.
    """
    records = [
        {
            "x": r.x, "y": r.y,
            "gene1": r.feature_a_value,
            "gene2": r.feature_b_value,
            "roi_id": r.generated_roi_id,
        }
        for r in rows
    ]
    return pd.DataFrame.from_records(
        records, columns=list(TILESCOPE_INPUT_COLUMNS),
    )


def rows_to_truth_dataframe(rows: Iterable[SpotTruthRow]) -> pd.DataFrame:
    """Enriched-schema DataFrame with all :class:`SpotTruthRow` fields."""
    records = [
        {c: getattr(r, c) for c in TRUTH_LEDGER_COLUMNS}
        for r in rows
    ]
    df = pd.DataFrame.from_records(
        records, columns=list(TRUTH_LEDGER_COLUMNS),
    )
    return df.sort_values(
        by=["scenario_instance_id", "counterfactual_role", "source_spot_id"],
    ).reset_index(drop=True)


def write_tilescope_input_csv(
    rows: Iterable[SpotTruthRow], path: Path,
) -> Path:
    """Write the 5-column TileScope-input CSV for one instance."""
    df = rows_to_tilescope_dataframe(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_truth_ledger_csv(
    rows: Iterable[SpotTruthRow], path: Path,
) -> Path:
    """Write the enriched-schema truth ledger CSV."""
    df = rows_to_truth_dataframe(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
