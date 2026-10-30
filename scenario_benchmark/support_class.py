"""Independent implementation of TileScope's parent-ROI spatial support
classification.

Reproduces TileScope's canonical four-class partition of parent ROIs
(``scattered`` / ``mixed`` / ``fully_connected`` /
``fully_supported_multicomponent``) under the frozen ``standard-v2``
policy. Used to gate observed-mode eligibility on
``embedded_domains`` (``B not scattered``) and
``dispersed_infiltration`` (``B is scattered``).

Policy sources (mirrored, not imported — the generator obeys the
``No TileScope dependency`` invariant from CONTRACT.md § 1.1;
conformance verified by
``tests/tilescope_conformance/test_support_class_conformance.py``):

- ``TILESCOPE_STANDARD_V2_MIN_COMPONENT_SPOTS = 15`` mirrors
  ``tilescope.settings.models.RoiFiltersModel.min_component_spots``.
- ``SAMPLE_SUPPORT_FACTOR = 1.5`` mirrors
  ``tilescope.analyzers.support_eligibility.DEFAULT_SAMPLE_SUPPORT_FACTOR``.
- Class definitions + column set match
  ``tilescope.analyzers.support_eligibility.derive_roi_support_status``.

Conformance tests check agreement between this implementation and TileScope.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from scenario_benchmark.sample_scaffold import SampleScaffold


TILESCOPE_STANDARD_V2_MIN_COMPONENT_SPOTS: int = 15
TILESCOPE_STANDARD_V2_SAMPLE_SUPPORT_FACTOR: float = 1.5

# Backward-compatible aliases (retained for downstream code that has
# not migrated to the tilescope-prefixed constants).
SUPPORT_FLOOR_STANDARD_V2 = TILESCOPE_STANDARD_V2_MIN_COMPONENT_SPOTS
SAMPLE_SUPPORT_FACTOR = TILESCOPE_STANDARD_V2_SAMPLE_SUPPORT_FACTOR

SPATIAL_SUPPORT_CLASSES: Tuple[str, ...] = (
    "scattered",
    "mixed",
    "fully_connected",
    "fully_supported_multicomponent",
)

SUPPORT_POLICY_ID: str = "tilescope-standard-v2"


class NonStandardScaffoldError(ValueError):
    """Raised when a caller supplies a scaffold whose adjacency method
    or radius factor does not match ``tilescope-standard-v2``. The
    manifest's support-policy hash and the scaffold's actual graph must
    agree; drift between them silently misrepresents the policy."""


def assert_scaffold_matches_standard_v2(scaffold: SampleScaffold) -> None:
    """Reject scaffolds whose adjacency graph was not built at the
    frozen ``tilescope-standard-v2`` parameters. Call at every public
    scenario-builder entry point that emits a support-class manifest."""
    method = getattr(scaffold, "adjacency_method", None)
    params = getattr(scaffold, "adjacency_parameters", None) or {}
    if method != "radius_multiplier":
        raise NonStandardScaffoldError(
            f"scaffold.adjacency_method must be 'radius_multiplier'; "
            f"got {method!r}."
        )
    rf = params.get("radius_factor")
    if rf is None or float(rf) != float(
        TILESCOPE_STANDARD_V2_SAMPLE_SUPPORT_FACTOR
    ):
        raise NonStandardScaffoldError(
            f"scaffold radius_factor must be "
            f"{TILESCOPE_STANDARD_V2_SAMPLE_SUPPORT_FACTOR}; got {rf!r}."
        )


@dataclass(frozen=True)
class RoiSupportStatus:
    """Per-ROI spatial-support classification. Field set matches
    ``derive_roi_support_status`` output columns byte-for-byte."""
    roi_id: str
    source_spots: int
    support_component_count: int
    support_qualified_component_count: int
    support_qualified_spots: int
    support_qualified_fraction: float
    largest_support_component_spots: int
    largest_support_component_fraction: float
    spatial_support_class: str


def compute_component_sizes_by_roi(
    scaffold: SampleScaffold,
) -> Dict[str, List[int]]:
    """Per-ROI same-label component-size list (sorted descending).

    Uses the scaffold's precomputed ``source_component_id`` (built
    via same-label CC on the ``factor * median_nn`` radius graph)
    — no separate graph traversal.
    """
    parents = scaffold.source_parent_roi_id.astype(str)
    comps = scaffold.source_component_id.astype(str)
    by_roi: Dict[str, Counter] = {}
    for p, c in zip(parents, comps):
        by_roi.setdefault(p, Counter())[c] += 1
    return {roi: sorted(counter.values(), reverse=True)
            for roi, counter in by_roi.items()}


def classify_roi_support(
    scaffold: SampleScaffold, roi_id: str,
    *, min_component_spots: int = TILESCOPE_STANDARD_V2_MIN_COMPONENT_SPOTS,
) -> RoiSupportStatus:
    """Four-class support status for one parent ROI."""
    if min_component_spots < 1:
        raise ValueError(
            f"min_component_spots must be >= 1; got {min_component_spots!r}"
        )
    sizes = compute_component_sizes_by_roi(scaffold).get(str(roi_id))
    if sizes is None or not sizes:
        return RoiSupportStatus(
            roi_id=str(roi_id),
            source_spots=0,
            support_component_count=0,
            support_qualified_component_count=0,
            support_qualified_spots=0,
            support_qualified_fraction=0.0,
            largest_support_component_spots=0,
            largest_support_component_fraction=0.0,
            spatial_support_class="scattered",
        )
    source_spots = int(sum(sizes))
    qualifying = [s for s in sizes if s >= min_component_spots]
    q_count = len(qualifying)
    q_spots = int(sum(qualifying))
    q_frac = q_spots / source_spots if source_spots else 0.0
    largest = int(max(sizes))
    largest_frac = largest / source_spots if source_spots else 0.0
    if q_count == 0:
        cls = "scattered"
    elif q_spots == source_spots:
        cls = "fully_connected" if q_count == 1 else "fully_supported_multicomponent"
    else:
        cls = "mixed"
    return RoiSupportStatus(
        roi_id=str(roi_id),
        source_spots=source_spots,
        support_component_count=len(sizes),
        support_qualified_component_count=q_count,
        support_qualified_spots=q_spots,
        support_qualified_fraction=q_frac,
        largest_support_component_spots=largest,
        largest_support_component_fraction=largest_frac,
        spatial_support_class=cls,
    )


def support_policy_hash(
    *, min_component_spots: int = TILESCOPE_STANDARD_V2_MIN_COMPONENT_SPOTS,
    sample_support_factor: float = SAMPLE_SUPPORT_FACTOR,
) -> str:
    """Stable hash of the policy tuple recorded on manifests."""
    payload = (
        f"{SUPPORT_POLICY_ID}|"
        f"min_component_spots={int(min_component_spots)}|"
        f"sample_support_factor={float(sample_support_factor):.6f}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def support_class_fields(
    scaffold: SampleScaffold, roi_id: str,
    *, prefix: str,
    min_component_spots: int = TILESCOPE_STANDARD_V2_MIN_COMPONENT_SPOTS,
) -> Dict[str, object]:
    """ROI-specific support-class fields with the given prefix
    ("a" or "b"). Does NOT include the invariant policy fields —
    combine with :func:`support_policy_fields` at the caller."""
    status = classify_roi_support(
        scaffold, roi_id, min_component_spots=min_component_spots,
    )
    sizes = compute_component_sizes_by_roi(scaffold).get(str(roi_id), [])
    return {
        f"{prefix}_spatial_support_class": status.spatial_support_class,
        f"{prefix}_support_component_sizes": list(sizes),
        f"{prefix}_support_component_count": status.support_component_count,
        f"{prefix}_support_qualified_component_count":
            status.support_qualified_component_count,
        f"{prefix}_support_qualified_fraction":
            status.support_qualified_fraction,
        f"{prefix}_largest_support_component_spots":
            status.largest_support_component_spots,
    }


def support_policy_fields(
    *, min_component_spots: int = TILESCOPE_STANDARD_V2_MIN_COMPONENT_SPOTS,
    sample_support_factor: float = SAMPLE_SUPPORT_FACTOR,
) -> Dict[str, object]:
    """Policy-invariant fields shared across all ROIs on the manifest."""
    return {
        "min_component_spots": int(min_component_spots),
        "sample_support_factor": float(sample_support_factor),
        "support_policy_id": SUPPORT_POLICY_ID,
        "support_policy_hash": support_policy_hash(
            min_component_spots=min_component_spots,
            sample_support_factor=sample_support_factor,
        ),
    }


def support_manifest_fields(
    scaffold: SampleScaffold, roi_id: str,
    *, min_component_spots: int = TILESCOPE_STANDARD_V2_MIN_COMPONENT_SPOTS,
    sample_support_factor: float = SAMPLE_SUPPORT_FACTOR,
) -> Dict[str, object]:
    """Backward-compat helper: returns b-prefixed ROI fields + policy."""
    return {
        **support_class_fields(
            scaffold, roi_id, prefix="b",
            min_component_spots=min_component_spots,
        ),
        **support_policy_fields(
            min_component_spots=min_component_spots,
            sample_support_factor=sample_support_factor,
        ),
    }


def qualified_component_ids(
    scaffold: SampleScaffold, roi_id: str,
    *, min_component_spots: int = TILESCOPE_STANDARD_V2_MIN_COMPONENT_SPOTS,
) -> Dict[str, int]:
    """Per-component ids in ``roi_id`` that are geometry-qualified
    (size >= min_component_spots). Maps id -> component size.
    Empty dict when the ROI is scattered."""
    parents = scaffold.source_parent_roi_id.astype(str)
    comps = scaffold.source_component_id.astype(str)
    target = str(roi_id)
    sizes: Dict[str, int] = {}
    for p, c in zip(parents, comps):
        if p != target:
            continue
        sizes[c] = sizes.get(c, 0) + 1
    return {cid: sz for cid, sz in sizes.items() if sz >= min_component_spots}
