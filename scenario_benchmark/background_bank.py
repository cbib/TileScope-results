"""Sample-wide feature-independent background field authority.

Sample-scoped latent draws + case-scoped amplitude profile.

Latent noise per (sample, ROI, feature, spot) is a deterministic IID
Gaussian draw with unit SD. The amplitude applied per spot depends on
whether that spot's ROI is a *participant* in the invoking scenario:

    background = background_level + amplitude(ROI) * latent_noise
    role_field = background + scenario_overlay

Amplitudes:

- participant ROI     → participant_amplitude   (small so overlay reads clean)
- non-participant ROI → non_participant_amplitude (subordinate to overlays)

Principle for choosing amplitudes: background variation should usually
remain subordinate to the weakest overlay a scenario applies, so that
non-participant ROIs do not visually or numerically rival the signal
they are meant to contrast with. Draws are Gaussian; unbounded by
default (occasional exceedances expected across large spot counts) but
optionally clipped to a finite absolute bound via
``BackgroundPolicy.latent_abs_clip``, which caps the field to
``background_level ± amplitude * latent_abs_clip``.

Deliberately not encoded here: the identity of "the weakest overlay in
the corpus". Coupling the background primitive to today's scenario
inventory would bake current cases into a general-purpose authority.
Amplitude calibration is a scenario-benchmark design decision;
sensitivity to it is part of the benchmark, not a fixed property of
this module.

Changing which ROIs are participants only rescales those ROIs' latent
draws; it does not resample. Cross-scenario comparability of the
underlying random variates is preserved.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from scenario_benchmark.sample_scaffold import SampleScaffold


@dataclass(frozen=True)
class BackgroundPolicy:
    """Frozen amplitude contract for the background bank."""
    background_level: float = 0.0
    participant_amplitude: float = 0.03
    non_participant_amplitude: float = 0.15
    latent_abs_clip: Optional[float] = None

    def __post_init__(self) -> None:
        if self.latent_abs_clip is not None and self.latent_abs_clip <= 0.0:
            raise ValueError(
                f"latent_abs_clip must be positive when set; got "
                f"{self.latent_abs_clip!r}."
            )


DEFAULT_BACKGROUND_POLICY = BackgroundPolicy()


class SampleBackgroundBank:
    """Sample-wide latent noise bank with case-scoped amplitude profile.

    Latent draws are seeded from (sample_id, roi_id, feature_id) only —
    identical across scenarios sharing the same source sample. The
    ``participant_roi_ids`` selects an amplitude per ROI at field
    extraction time; it does not affect the latent random draws.
    """

    def __init__(
        self, *, scaffold: SampleScaffold,
        participant_roi_ids: Iterable[str],
        policy: Optional[BackgroundPolicy] = None,
    ) -> None:
        self._scaffold = scaffold
        self._participant_roi_ids = frozenset(
            str(r) for r in participant_roi_ids
        )
        self._policy = policy if policy is not None else DEFAULT_BACKGROUND_POLICY
        self._latent_cache: Dict[str, np.ndarray] = {}
        self._amplitude_vector = self._build_amplitude_vector()

    @property
    def participant_roi_ids(self) -> frozenset:
        return self._participant_roi_ids

    @property
    def degenerate_rois(self) -> list:
        """ROIs with fewer than two source spots (no within-ROI structure)."""
        rois = self._scaffold.source_parent_roi_id.astype(str)
        counts: Dict[str, int] = {}
        for r in rois:
            counts[str(r)] = counts.get(str(r), 0) + 1
        return sorted(r for r, n in counts.items() if n < 2)

    def field(self, feature_id: str) -> np.ndarray:
        """Sample-wide (N,) background field for ``feature_id`` (cached)."""
        key = str(feature_id)
        latent = self._latent_cache.get(key)
        if latent is None:
            latent = self._materialize_latent(key)
            self._latent_cache[key] = latent
        clip = self._policy.latent_abs_clip
        effective_latent = np.clip(latent, -clip, +clip) if clip is not None else latent
        return (
            self._policy.background_level
            + self._amplitude_vector * effective_latent
        ).copy()

    def latent(self, feature_id: str) -> np.ndarray:
        """Underlying zero-mean unit-SD latent draw for ``feature_id``.

        Exposed so callers that need the amplitude-independent random
        vector (tests, diagnostics) can obtain it without recomputing.
        """
        key = str(feature_id)
        latent = self._latent_cache.get(key)
        if latent is None:
            latent = self._materialize_latent(key)
            self._latent_cache[key] = latent
        return latent.copy()

    def _build_amplitude_vector(self) -> np.ndarray:
        rois = self._scaffold.source_parent_roi_id.astype(str)
        is_participant = np.isin(
            rois, np.array(sorted(self._participant_roi_ids)),
        ).copy()
        amp = np.where(
            is_participant,
            self._policy.participant_amplitude,
            self._policy.non_participant_amplitude,
        )
        return amp.astype(float)

    def _materialize_latent(self, feature_id: str) -> np.ndarray:
        scaffold = self._scaffold
        n = scaffold.n
        latent = np.zeros(n, dtype=float)
        rois = sorted(set(scaffold.source_parent_roi_id.tolist()))
        for roi_id in rois:
            r_indices = np.where(scaffold.source_parent_roi_id == roi_id)[0]
            rng = _rng_for(
                sample_id=scaffold.sample_id, roi_id=str(roi_id),
                feature_id=feature_id, tag="latent",
            )
            latent[r_indices] = rng.standard_normal(size=r_indices.size)
        return latent


def _rng_for(
    *, sample_id: str, roi_id: str, feature_id: str, tag: str,
) -> np.random.Generator:
    """Deterministic Generator from sha256 of the stream key."""
    key = f"{sample_id}::bg::{roi_id}::{feature_id}::{tag}".encode()
    digest = hashlib.sha256(key).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return np.random.default_rng(seed)
