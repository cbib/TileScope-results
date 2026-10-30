"""Scenario ``spatial_gradient`` (concordant + opposed modes).

Estimand (H1): on the anchor ROI, feature A and feature B receive a
midpoint/range-centered physical-projection ramp overlay along the
caller-supplied ``gradient_axis``:

    proj_i = coords_i . axis_unit                    (anchor spots only)
    u_i    = (proj_i - min_proj) / (max_proj - min_proj)   in [0, 1]
    z_i    = u_i - 0.5                               in [-0.5, +0.5]
    overlay_a[i] = effect_delta * z_i                (anchor only)
    overlay_b[i] = s * effect_delta * z_i            (anchor only)

with ``s = +1`` (concordant) or ``s = -1`` (opposed). Outside the
anchor, both overlays are zero and the role field is exactly the
sample background bank.

Roles (per mode; each mode is a distinct counterfactual set):

- ``positive_spatial_gradient`` -- full continuous A + B overlays as above.
- ``paired_randomized`` -- same A overlay; B overlay values permuted
  across the anchor positions (feature-B ramp shape destroyed while
  the anchor-restricted overlay multiset is preserved). The bank
  itself is byte-identical to the positive role's B bank.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from scenario_benchmark.sample_scaffold import SampleScaffold
from scenario_benchmark.scenario_assembly import ScenarioSetBuilder
from scenario_benchmark.scenarios.result import ScenarioSet
from scenario_benchmark.background_bank import SampleBackgroundBank
from scenario_benchmark.scenarios.support import (
    build_all_infeasible_set,
)
from scenario_benchmark.signal_model import (
    SignalModel, assert_composition, compose_role_field, substream_rng,
)


SCENARIO_NAME = "spatial_gradient"
ROLES: Tuple[str, ...] = ("positive_spatial_gradient", "paired_randomized")
REQUIRED_ROLES: Tuple[str, ...] = ROLES
CONDITIONAL_ROLES: Tuple[str, ...] = ()

GRADIENT_RELATIONSHIP_OPPOSED = "opposed"
GRADIENT_RELATIONSHIP_CONCORDANT = "concordant"
GRADIENT_RELATIONSHIP_VALUES = (
    GRADIENT_RELATIONSHIP_OPPOSED,
    GRADIENT_RELATIONSHIP_CONCORDANT,
)


def build_spatial_gradient_set(
    *, scaffold: SampleScaffold, anchor_roi_id: str,
    gradient_relationship: str,
    counterfactual_set_id: str,
    feature_a_id: str, feature_b_id: str,
    truth_tail_fraction: float,
    gradient_axis: Tuple[float, float],
    signal_model: SignalModel,
    background_policy=None,
    nuisance_id: str = "default",
    nuisance_parameters: Optional[dict] = None,
    base_seed: int = 0,
    partition: str = "dev",
    related_sample_group_id: str = "",
) -> ScenarioSet:
    """Emit the two-role spatial_gradient counterfactual set for the
    given ``gradient_relationship`` mode."""
    if gradient_relationship not in GRADIENT_RELATIONSHIP_VALUES:
        raise ValueError(
            f"gradient_relationship must be one of "
            f"{GRADIENT_RELATIONSHIP_VALUES!r}; got "
            f"{gradient_relationship!r}."
        )
    b_slope_sign = (
        +1.0 if gradient_relationship == GRADIENT_RELATIONSHIP_CONCORDANT
        else -1.0
    )
    if not (0.0 < float(truth_tail_fraction) < 1.0):
        raise ValueError(
            f"truth_tail_fraction must be in (0, 1); got {truth_tail_fraction!r}."
        )
    axis = np.asarray(gradient_axis, dtype=float)
    if axis.shape != (2,):
        raise ValueError("gradient_axis must be a 2-vector.")
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm <= 0.0:
        raise ValueError("gradient_axis must be finite and nonzero.")
    axis_unit = axis / axis_norm

    source_roi = scaffold.source_parent_roi_id.astype(str)
    anchor_indices = np.where(source_roi == str(anchor_roi_id))[0]
    if anchor_indices.size == 0:
        raise ValueError(
            f"spatial_gradient: no spots with source_parent_roi_id="
            f"{anchor_roi_id!r}."
        )
    n_anchor = int(anchor_indices.size)
    k_tail = int(round(truth_tail_fraction * n_anchor))

    requested = {
        "gradient_relationship": gradient_relationship,
        "truth_tail_fraction": float(truth_tail_fraction),
        "gradient_axis_requested": [float(a) for a in gradient_axis],
        "gradient_axis_normalized": [float(a) for a in axis_unit],
        "anchor_roi_id": str(anchor_roi_id),
        "effect_delta": float(signal_model.effect_delta),
    }
    nuisance_parameters = dict(nuisance_parameters or {})
    nuisance_parameters.update({
        "gradient_relationship": gradient_relationship,
        "truth_tail_fraction": float(truth_tail_fraction),
        "gradient_axis": [float(a) for a in axis_unit],
        "effect_delta": float(signal_model.effect_delta),
    })
    infeas_kwargs = dict(
        scaffold=scaffold, counterfactual_set_id=counterfactual_set_id,
        scenario_name=SCENARIO_NAME, roles=ROLES,
        feature_a_id=feature_a_id, feature_b_id=feature_b_id,
        nuisance_id=nuisance_id, nuisance_parameters=nuisance_parameters,
        requested_parameters=requested,
        rng_seed=base_seed, partition=partition,
        related_sample_group_id=related_sample_group_id,
    )
    if k_tail < 1 or k_tail > n_anchor:
        return build_all_infeasible_set(
            reason=f"truth_tail_fraction_out_of_range: k_tail={k_tail} "
                   f"anchor_size={n_anchor}",
            **infeas_kwargs,
        )

    coords = scaffold.coords[anchor_indices]
    projections = coords @ axis_unit
    proj_min = float(projections.min())
    proj_max = float(projections.max())
    proj_range = proj_max - proj_min
    if not np.isfinite(proj_range) or proj_range <= 0.0:
        return build_all_infeasible_set(
            reason="degenerate_projection_range_on_anchor",
            **infeas_kwargs,
        )
    u = (projections - proj_min) / proj_range
    if not np.all(np.isfinite(u)):
        return build_all_infeasible_set(
            reason="non_finite_projection_ratio",
            **infeas_kwargs,
        )
    z = u - 0.5
    effect_delta = float(signal_model.effect_delta)

    # Positive overlays: anchor-only ramp on top of the bank; zero
    # everywhere else. No independent anchor background draw.
    overlay_a = np.zeros(scaffold.n, dtype=float)
    overlay_b_pos = np.zeros(scaffold.n, dtype=float)
    overlay_a[anchor_indices] = effect_delta * z
    overlay_b_pos[anchor_indices] = b_slope_sign * effect_delta * z

    bg_bank = SampleBackgroundBank(
        scaffold=scaffold, participant_roi_ids=(str(anchor_roi_id),),
        policy=background_policy,
    )
    bg_a = bg_bank.field(feature_a_id)
    bg_b = bg_bank.field(feature_b_id)
    a_field = compose_role_field(bg=bg_a, overlay=overlay_a)
    b_field_pos = compose_role_field(bg=bg_b, overlay=overlay_b_pos)

    spot_ids = scaffold.spot_ids[anchor_indices]
    a_top_local = _top_tail_indices(
        anchor_vals=a_field[anchor_indices], spot_ids=spot_ids, k=k_tail,
    )
    b_top_local_pos = _top_tail_indices(
        anchor_vals=b_field_pos[anchor_indices], spot_ids=spot_ids, k=k_tail,
    )
    elevated_a_positions = np.array(
        sorted(int(anchor_indices[i]) for i in a_top_local), dtype=int,
    )
    elevated_b_positions_pos = np.array(
        sorted(int(anchor_indices[i]) for i in b_top_local_pos), dtype=int,
    )

    # Null: permute the B OVERLAY values across anchor positions.
    # Bank stays byte-identical between positive and null.
    rng_null_b = substream_rng(base_seed=base_seed, tag="null_permute_b")
    anchor_overlay_b = overlay_b_pos[anchor_indices].copy()
    perm = rng_null_b.permutation(anchor_overlay_b.size)
    overlay_b_null = np.zeros(scaffold.n, dtype=float)
    overlay_b_null[anchor_indices] = anchor_overlay_b[perm]
    b_field_null = compose_role_field(bg=bg_b, overlay=overlay_b_null)
    b_top_local_null = _top_tail_indices(
        anchor_vals=b_field_null[anchor_indices], spot_ids=spot_ids, k=k_tail,
    )
    null_b_elevated_positions = np.array(
        sorted(int(anchor_indices[i]) for i in b_top_local_null), dtype=int,
    )

    assert_composition(field=a_field, bg=bg_a, overlay=overlay_a,
                       scenario_name=SCENARIO_NAME, role="positive_spatial_gradient", feature="a")
    assert_composition(field=b_field_pos, bg=bg_b, overlay=overlay_b_pos,
                       scenario_name=SCENARIO_NAME, role="positive_spatial_gradient", feature="b")
    assert_composition(field=b_field_null, bg=bg_b, overlay=overlay_b_null,
                       scenario_name=SCENARIO_NAME, role="paired_randomized", feature="b")

    builder = ScenarioSetBuilder(
        scaffold=scaffold, counterfactual_set_id=counterfactual_set_id,
        scenario_name=SCENARIO_NAME,
        feature_a_id=feature_a_id, feature_b_id=feature_b_id,
        nuisance_id=nuisance_id, nuisance_parameters=nuisance_parameters,
        requested_parameters=requested, rng_seed=base_seed,
        partition=partition, related_sample_group_id=related_sample_group_id,
    )
    common_realized = {
        "gradient_relationship": gradient_relationship,
        "truth_tail_fraction": float(truth_tail_fraction),
        "anchor_roi_size": n_anchor,
        "k_tail": k_tail,
        "gradient_axis_normalized": [float(a) for a in axis_unit],
        "projection_range": proj_range,
        "background_degenerate_rois": list(bg_bank.degenerate_rois),
    }
    builder.add_instance(
        role="positive_spatial_gradient",
        target_selected_a=elevated_a_positions,
        target_selected_b=elevated_b_positions_pos,
        target_a_field=a_field,
        target_b_field=b_field_pos,
        realized_parameters=common_realized,
    )
    builder.add_instance(
        role="paired_randomized",
        target_selected_a=elevated_a_positions,
        target_selected_b=null_b_elevated_positions,
        target_a_field=a_field,
        target_b_field=b_field_null,
        realized_parameters=common_realized,
    )

    return ScenarioSet(
        counterfactual_set_id=counterfactual_set_id,
        instances=builder.instances,
        spot_truth_rows=builder.spot_rows,
        interface_rows=builder.interface_rows,
    )


def _top_tail_indices(
    *, anchor_vals: np.ndarray, spot_ids: np.ndarray, k: int,
) -> np.ndarray:
    """Return the k local indices with the largest ``anchor_vals``,
    ties broken by ascending ``spot_ids``. Deterministic."""
    order = sorted(
        range(anchor_vals.size),
        key=lambda i: (-float(anchor_vals[i]), str(spot_ids[i])),
    )
    return np.array(sorted(int(i) for i in order[:k]), dtype=int)
