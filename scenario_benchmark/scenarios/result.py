"""Common return shape for scenario orchestrators."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from scenario_benchmark.truth_schemas import (
    InstanceManifestRow, InterfaceRow, SpotTruthRow,
)


@dataclass(frozen=True)
class ScenarioSet:
    """One counterfactual set.

    ``interface_rows`` is populated by challenge scenarios that
    perform generated-ROI relabelling and derive contact geometry
    from the acquisition graph under generated labels. Easy
    calibration scenarios leave it empty (no A-B interface truth to
    emit for within-anchor-ROI signals).
    """
    counterfactual_set_id: str
    instances: List[InstanceManifestRow] = field(default_factory=list)
    spot_truth_rows: List[SpotTruthRow] = field(default_factory=list)
    interface_rows: List[InterfaceRow] = field(default_factory=list)
