"""Lightweight pair identities from the existing completed-run inventory.

No canonical shards, cache paths or aggregated statistics are read or written.
Physical program order is retained for directional core outputs.
"""
from tilescope_results.paths import data_root
from pathlib import Path
import pandas as pd

RUN_STATUS = data_root() / "heiland_tyrosh/CurrentResults/run_accounting/run_status.csv"


def pair_identity(pair):
    parts = pair.split("_", 1)
    if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
        raise ValueError(f"invalid Greenwald pair directory name: {pair!r}")
    g1, g2 = parts
    return g1, g2, "||".join(sorted((g1, g2)))


def load_pair_catalog(path=RUN_STATUS):
    status = pd.read_csv(path, dtype={"sample_id": str, "pair": str})
    required = {"sample_id", "pair", "status", "core_sha"}
    if not required.issubset(status) or status.empty:
        raise ValueError("Missing/empty completed-run inventory")
    if status[list(required)].isna().any().any():
        raise ValueError("Missing run identity or provenance")
    if status.duplicated(["sample_id", "pair"]).any():
        raise ValueError("Duplicate sample/pair run")
    if not status.status.eq("complete").all() or status.core_sha.nunique() != 1:
        raise ValueError("Run inventory must be complete and have one core revision")
    samples = set(status.sample_id)
    rows = []
    for pair, group in status.groupby("pair", sort=True):
        if set(group.sample_id) != samples:
            raise ValueError("Pairs do not cover the same sample universe")
        g1, g2, key = pair_identity(pair)
        rows.append(dict(pair=pair, program_g1=g1, program_g2=g2,
                         unordered_pair_key=key, n_samples=len(samples)))
    result = pd.DataFrame(rows)
    if result.unordered_pair_key.duplicated().any():
        raise ValueError("Ambiguous physical order: both orientations of one pair")
    return result
