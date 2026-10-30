"""Feature-signal composition primitives.

Every role's per-feature field is composed as
``role_field = bg_bank.field(feature) + role_overlay``. Overlays are
zero outside the role's elevated positions and may carry either a
constant increment or a distribution-matched component profile.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import hashlib

import numpy as np


COMPONENT_EXPRESSION_PROFILES = (
    "core_high", "rim_high", "axial_gradient",
)
DEFAULT_COMPONENT_PROFILE_CONTRAST = 0.30


# ---------------------------------------------------------------------------
# RNG substreams
# ---------------------------------------------------------------------------

def substream_rng(*, base_seed: int, tag: str) -> np.random.Generator:
    """Deterministic role-independent RNG substream keyed by ``tag``."""
    h = hashlib.sha256(f"{base_seed}::{tag}".encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big", signed=False)
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# SignalModel: single scenario-level knob (increment magnitude)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalModel:
    """Per-scenario increment magnitude on top of the sample background bank."""
    effect_delta: float


# ---------------------------------------------------------------------------
# Compositional primitives
# ---------------------------------------------------------------------------

def compose_role_field(*, bg: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Return ``bg + overlay``. Shape-checked; both must be 1-D of equal size."""
    bg_arr = np.asarray(bg, dtype=float)
    ov_arr = np.asarray(overlay, dtype=float)
    if bg_arr.shape != ov_arr.shape:
        raise ValueError(
            f"compose_role_field: bg shape {bg_arr.shape} != "
            f"overlay shape {ov_arr.shape}"
        )
    return bg_arr + ov_arr


def constant_overlay(
    *, n: int, positions: np.ndarray, magnitude: float,
) -> np.ndarray:
    """Zero everywhere; ``magnitude`` at each index in ``positions``."""
    overlay = np.zeros(int(n), dtype=float)
    if positions is not None and np.asarray(positions).size:
        overlay[np.asarray(positions, dtype=int)] = float(magnitude)
    return overlay


def profiled_component_overlay(
    *, n: int, components: Sequence[Sequence[int]],
    coords: np.ndarray, spot_ids: np.ndarray,
    magnitude: float, profile: str,
    contrast: float = DEFAULT_COMPONENT_PROFILE_CONTRAST,
) -> np.ndarray:
    """Component-shaped overlay with a frozen value multiset.

    ``profile`` controls only the rank assignment of amplitudes to
    component positions. For a fixed total number of positions,
    every geometry receives exactly the same linearly spaced value
    multiset with mean ``magnitude``. This lets matched scenario roles
    vary topology without changing the B marginal distribution.
    """
    if profile not in COMPONENT_EXPRESSION_PROFILES:
        raise ValueError(
            f"profile must be one of {COMPONENT_EXPRESSION_PROFILES!r}; "
            f"got {profile!r}."
        )
    if not (0.0 <= float(contrast) < 1.0):
        raise ValueError(f"contrast must be in [0, 1); got {contrast!r}.")
    if float(magnitude) <= 0.0:
        raise ValueError(f"magnitude must be positive; got {magnitude!r}.")
    coords_arr = np.asarray(coords, dtype=float)
    ids = np.asarray(spot_ids).astype(str)
    if coords_arr.shape != (int(n), 2) or ids.shape != (int(n),):
        raise ValueError("coords / spot_ids must match n.")

    component_arrays = [
        np.asarray(tuple(int(p) for p in component), dtype=int)
        for component in components
    ]
    positions = (
        np.concatenate(component_arrays)
        if component_arrays else np.array([], dtype=int)
    )
    if positions.size == 0:
        return np.zeros(int(n), dtype=float)
    if np.unique(positions).size != positions.size:
        raise ValueError("component positions must be unique.")
    if positions.min() < 0 or positions.max() >= int(n):
        raise ValueError("component position outside [0, n).")

    axis = _sample_major_axis(coords_arr)
    ranked: list[tuple[float, str, int]] = []
    for component in component_arrays:
        xy = coords_arr[component]
        if profile in {"core_high", "rim_high"}:
            radius = np.linalg.norm(xy - xy.mean(axis=0), axis=1)
            score = _unit_interval(radius)
            if profile == "core_high":
                score = -score
        else:
            score = _unit_interval(xy @ axis)
        ranked.extend(
            (float(s), str(ids[int(p)]), int(p))
            for s, p in zip(score, component)
        )

    ranked.sort(key=lambda item: (item[0], item[1]))
    if positions.size == 1:
        amplitudes = np.array([float(magnitude)], dtype=float)
    else:
        amplitudes = float(magnitude) * (
            1.0 + float(contrast) * np.linspace(-1.0, 1.0, positions.size)
        )
    overlay = np.zeros(int(n), dtype=float)
    for amplitude, (_, _, position) in zip(amplitudes, ranked):
        overlay[position] = float(amplitude)
    return overlay


def _unit_interval(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = float(values.max() - values.min())
    if span == 0.0:
        return np.zeros(values.size, dtype=float)
    return (values - float(values.min())) / span


def _sample_major_axis(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0)
    covariance = centered.T @ centered
    _, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1].astype(float)
    # Resolve the arbitrary eigenvector sign deterministically.
    if axis[0] < 0.0 or (axis[0] == 0.0 and axis[1] < 0.0):
        axis = -axis
    return axis


def assert_composition(
    *, field: np.ndarray, bg: np.ndarray, overlay: np.ndarray,
    scenario_name: str, role: str, feature: str,
) -> None:
    """Compositional contract: ``field == bg + overlay`` byte-exact."""
    if not np.array_equal(field, bg + overlay):
        raise RuntimeError(
            f"{scenario_name}: composition invariant broken on role "
            f"{role!r} feature {feature!r} -- role_field != bg + overlay."
        )
