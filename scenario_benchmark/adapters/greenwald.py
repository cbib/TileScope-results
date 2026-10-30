"""Greenwald real-data adapter.

``load_greenwald_sample_scaffold`` is the single geometry loader. It
returns a whole-sample :class:`SampleScaffold` with one acquisition
graph and per-ROI connected components.

Both fabricate a stable ``source_spot_id`` via :func:`stable_spot_id`
(SHA-256 over ``"{sample_id}::{roi_id}::{x:.6f},{y:.6f}"``).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from scenario_benchmark.sample_scaffold import (
    DEFAULT_RADIUS_FACTOR, SampleScaffold, load_sample_scaffold,
)


def stable_spot_id(
    *, sample_id: str, roi_id: str, x: float, y: float,
) -> str:
    """Deterministic per-(sample, roi, coord) spot id — SHA-256 over
    ``"{sample_id}::{roi_id}::{x:.6f},{y:.6f}"``, first 12 hex chars."""
    key = f"{sample_id}::{roi_id}::{x:.6f},{y:.6f}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def load_greenwald_sample_scaffold(
    csv_path: Path,
    *, sample_id: str,
    radius_factor: float = DEFAULT_RADIUS_FACTOR,
) -> SampleScaffold:
    """Load a Greenwald expression CSV as a whole-sample scaffold.

    All rows read; ``source_parent_roi_id`` comes from ``roi_id``;
    sample-wide acquisition graph built once; components per ROI
    derived by CC of the induced subgraph. Coordinates never move.
    """
    df = pd.read_csv(csv_path)
    for col in ("x", "y", "roi_id"):
        if col not in df.columns:
            raise ValueError(
                f"load_greenwald_sample_scaffold: CSV {csv_path!s} missing "
                f"required column {col!r}; got {list(df.columns)!r}."
            )
    df["roi_id"] = df["roi_id"].astype(str)
    df["spot_id"] = [
        stable_spot_id(sample_id=sample_id, roi_id=str(roi),
                       x=float(x), y=float(y))
        for roi, x, y in zip(df["roi_id"].to_numpy(),
                             df["x"].to_numpy(), df["y"].to_numpy())
    ]
    if df["spot_id"].duplicated().any():
        dupes = df.loc[df["spot_id"].duplicated(keep=False),
                       ["roi_id", "x", "y"]]
        raise ValueError(
            f"load_greenwald_sample_scaffold: duplicate (roi, coord) "
            f"in sample {sample_id!r}; hash collisions imply true "
            f"duplicates. Offending rows:\n{dupes.head(6)!s}"
        )
    return load_sample_scaffold(
        df, sample_id=sample_id, radius_factor=radius_factor,
    )
