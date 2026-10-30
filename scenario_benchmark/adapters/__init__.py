"""Adapters for externally supplied sample CSV schemas."""

from scenario_benchmark.adapters.greenwald import (
    load_greenwald_sample_scaffold,
    stable_spot_id,
)


__all__ = [
    "load_greenwald_sample_scaffold",
    "stable_spot_id",
]
