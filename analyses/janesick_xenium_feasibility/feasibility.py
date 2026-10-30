"""Validate the published breast input; graph construction belongs to TileScope."""
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = ("x", "y", "gene1", "gene2", "roi_id")
EXPECTED_ROI_IDS = ("DCIS_1", "DCIS_2", "Invasive")
EXPECTED_N_CELLS = 87_770


def load_and_validate_input(
    path: Path,
    *,
    expected_roi_ids: Iterable[str] = EXPECTED_ROI_IDS,
    expected_n_cells: int | None = EXPECTED_N_CELLS,
) -> pd.DataFrame:
    """Load the predeclared ACTA2--FASN input and enforce its contract."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Janesick input CSV not found: {path}")
    frame = pd.read_csv(path)
    if tuple(frame.columns) != CANONICAL_COLUMNS:
        raise ValueError(
            "Input must contain exactly the ordered TileScope columns "
            f"{CANONICAL_COLUMNS!r}; got {tuple(frame.columns)!r}."
        )
    if expected_n_cells is not None and len(frame) != expected_n_cells:
        raise ValueError(
            f"Expected {expected_n_cells:,} cells; found {len(frame):,}."
        )
    if frame.isna().any().any():
        columns = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"Input contains missing values in columns {columns!r}.")

    numeric = frame.loc[:, ["x", "y", "gene1", "gene2"]]
    if not all(pd.api.types.is_numeric_dtype(numeric[c]) for c in numeric):
        raise ValueError("x, y, gene1, and gene2 must all be numeric.")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Input coordinates and expression values must be finite.")
    if (frame[["gene1", "gene2"]].to_numpy(dtype=float) < 0).any():
        raise ValueError("Raw Xenium transcript counts cannot be negative.")

    observed = set(frame["roi_id"].astype(str).unique())
    expected = set(str(v) for v in expected_roi_ids)
    if observed != expected:
        raise ValueError(
            "ROI identity mismatch: "
            f"missing={sorted(expected - observed)!r}, "
            f"unexpected={sorted(observed - expected)!r}."
        )
    return frame
