"""Recognition-facing saturating transformation for enrichment ratios.

TileScope core reports enrichment across two atlas columns:

  ``availability_status``   ``finite`` or ``undefined``
  ``undefined_reason``      populated only when availability_status is
                            ``undefined``; drawn from the enrichment
                            per-ratio enum (``positive_over_zero_context``
                            when numerator > 0 and context proportion
                            == 0; ``zero_over_zero_context`` when both
                            are zero).

``positive_over_zero_context`` marks an informative saturating high
(the ratio is +infinity), which recognition + response-matrix
consumers previously collapsed into ``missing_feature`` alongside
genuine 0/0 indeterminacy. This helper maps the (value,
availability_status, undefined_reason) triple into a bounded [0, 1]
recognition feature so saturated cases contribute rather than get
dropped:

  finite r >= 0                       -> r / (1 + r)   in [0, 1)
  undefined + positive_over_zero_context -> 1.0        saturating high
  undefined + zero_over_zero_context, any other state -> None

The r/(1+r) transform is monotone on r >= 0, so the sign of any paired
difference (used by gap-map's directional classification) is
preserved. Canonical TileScope CSVs keep the raw numeric + typed
status unchanged; this transformation lives strictly inside the
benchmark analysis and applies only to raw
enrichment (not tessellation median rollups, whose per-ratio
saturation is already collapsed at the reduction step).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


STATUS_FINITE = "finite"
STATUS_UNDEFINED = "undefined"
REASON_SATURATING_HIGH = "positive_over_zero_context"
REASON_INDETERMINATE = "zero_over_zero_context"

ENRICHMENT_STATISTICS = frozenset({"enrichment"})


def is_enrichment_statistic(statistic: str) -> bool:
    """True when the statistic uses the saturating enrichment convention."""
    return statistic in ENRICHMENT_STATISTICS


def saturating_enrichment(value, availability_status, undefined_reason) -> Optional[float]:
    """Map an enrichment cell into a bounded [0, 1] recognition feature.

    Returns None for indeterminate or unrecognised states so the
    caller marks the feature missing.
    """
    if (
        availability_status == STATUS_UNDEFINED
        and undefined_reason == REASON_SATURATING_HIGH
    ):
        return 1.0
    if availability_status != STATUS_FINITE:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v) or v < 0.0:
        return None
    return v / (1.0 + v)
