"""Scenario family ``focal_pattern`` (colocated + separated modes).

Both modes share the same substrate + anchor + A foci + bg bank + B focus
sizes; only B focus PLACEMENT differs. Colocated: A and B on the same
connected foci. Separated: B on connected foci at min-hop from every A
focus (same count and size multiset as A). Two roles per mode: ``positive``
+ ``b_independent_background``.
"""
from __future__ import annotations

import itertools
from typing import List, Optional, Tuple

import numpy as np

from scenario_benchmark.background_bank import SampleBackgroundBank
from scenario_benchmark.sample_scaffold import SampleScaffold
from scenario_benchmark.scenario_assembly import (
    InvariantPolicy, ScenarioSetBuilder, StructureAssignment,
)
from scenario_benchmark.scenarios.result import ScenarioSet
from scenario_benchmark.scenarios.support import build_all_infeasible_set
from scenario_benchmark.signal_model import (
    SignalModel, assert_composition, compose_role_field, constant_overlay,
    substream_rng,
)


SCENARIO_NAME = "focal_pattern"

MODE_COLOCATED = "colocated"
MODE_SEPARATED = "separated"
MODES: Tuple[str, ...] = (MODE_COLOCATED, MODE_SEPARATED)

ROLES: Tuple[str, ...] = ("positive", "b_independent_background")
REQUIRED_ROLES: Tuple[str, ...] = ROLES
CONDITIONAL_ROLES: Tuple[str, ...] = ()
COMPARISON_KIND = {
    "positive": "primary_positive",
    "b_independent_background": "diagnostic_baseline",
}

STRUCTURE_KIND_FOCUS = "focus"


# Same role contract across both modes; a single flat policy suffices.
FOCAL_PATTERN_INVARIANTS = InvariantPolicy(
    expected_roles=ROLES,
    feature_a_byte_identical_roles=ROLES,
    permitted_role_level_infeasibilities=(),
    require_non_pair_background_identity=True,
)


def build_focal_pattern_set(
    *, scaffold: SampleScaffold, anchor_roi_id: str,
    mode: str,
    counterfactual_set_id: str,
    feature_a_id: str, feature_b_id: str,
    prevalence_fraction: float,
    n_foci: int,
    min_focus_separation_hops: int,
    signal_model: SignalModel,
    background_policy=None,
    nuisance_id: str = "default",
    nuisance_parameters: Optional[dict] = None,
    base_seed: int = 0,
    partition: str = "dev",
    related_sample_group_id: str = "",
    seed_max_attempts: int = 200,
) -> ScenarioSet:
    """Emit one mode of the focal_pattern family."""
    if mode not in MODES:
        raise ValueError(
            f"focal_pattern: mode must be one of {MODES!r}; got {mode!r}."
        )
    if not (0.0 < float(prevalence_fraction) < 1.0):
        raise ValueError("prevalence_fraction must be in (0, 1).")
    if not isinstance(n_foci, int) or n_foci < 1:
        raise ValueError("n_foci must be a positive int.")
    if not isinstance(min_focus_separation_hops, int) or min_focus_separation_hops < 1:
        raise ValueError("min_focus_separation_hops must be a positive int.")

    effect_delta = float(signal_model.effect_delta)
    broad_a_increment_focal = effect_delta
    broad_b_increment_focal = effect_delta
    min_a_b_focus_separation_hops = int(min_focus_separation_hops)

    source_roi = scaffold.source_parent_roi_id.astype(str)
    anchor_indices = np.where(source_roi == str(anchor_roi_id))[0]
    if anchor_indices.size == 0:
        raise ValueError(
            f"focal_pattern: no spots with source_parent_roi_id="
            f"{anchor_roi_id!r}."
        )
    n_anchor = int(anchor_indices.size)
    k_total = int(round(prevalence_fraction * n_anchor))
    per_focus = _balanced_focus_sizes(k_total, n_foci)

    requested = {
        "prevalence_fraction": float(prevalence_fraction),
        "n_foci": int(n_foci),
        "min_focus_separation_hops": int(min_focus_separation_hops),
        "anchor_roi_id": str(anchor_roi_id),
        "mode": mode,
    }
    nuisance_parameters = dict(nuisance_parameters or {})
    nuisance_parameters.update({
        "prevalence_fraction": float(prevalence_fraction),
        "n_foci": int(n_foci),
        "min_focus_separation_hops": int(min_focus_separation_hops),
        "mode": mode,
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
    if k_total < n_foci or k_total > n_anchor:
        return build_all_infeasible_set(
            reason=(
                f"prevalence_out_of_range: k_total={k_total} "
                f"n_foci={n_foci} anchor_size={n_anchor}"
            ),
            **infeas_kwargs,
        )

    a_adj = scaffold.restricted_adj(anchor_indices)
    seeds = _pick_focus_seeds(
        a_adj, n_foci=n_foci,
        min_hop=min_focus_separation_hops,
        rng=substream_rng(base_seed=base_seed, tag="focal_seeds"),
        max_attempts=seed_max_attempts,
    )
    if seeds is None:
        return build_all_infeasible_set(
            reason="cannot_select_focus_seeds_with_min_separation",
            **infeas_kwargs,
        )
    patches = _grow_disjoint_patches_with_buffer(
        a_adj, seeds=seeds, per_seed_target=per_focus,
        min_hop=min_focus_separation_hops,
    )
    total = sum(len(p) for p in patches)
    if total != k_total or any(len(p) != per_focus[i] for i, p in enumerate(patches)):
        return build_all_infeasible_set(
            reason="focus_growth_did_not_reach_requested_sizes_under_min_hop_buffer",
            **infeas_kwargs,
        )
    if not _verify_pairwise_min_hop(a_adj, patches, min_focus_separation_hops):
        return build_all_infeasible_set(
            reason="focus_pairwise_min_hop_violated_post_growth",
            **infeas_kwargs,
        )
    a_focus_positions_global: List[np.ndarray] = [
        np.array(sorted(int(anchor_indices[i]) for i in patch), dtype=int)
        for patch in patches
    ]
    a_all_positions_global = np.array(
        sorted(int(x) for patch in a_focus_positions_global for x in patch),
        dtype=int,
    )

    bg_bank = SampleBackgroundBank(
        scaffold=scaffold, participant_roi_ids=(str(anchor_roi_id),),
        policy=background_policy,
    )
    bg_a = bg_bank.field(feature_a_id)
    bg_b = bg_bank.field(feature_b_id)

    # Mode differs only in B focus positions + B focus structures.
    if mode == MODE_COLOCATED:
        b_positions_global = a_all_positions_global
        b_focus_positions_global = a_focus_positions_global
        b_structure_prefix = "focus"
    else:
        separated_b_patches_local = _grow_separated_b_foci(
            local_adj=a_adj,
            a_focus_local_patches=patches,
            per_focus_sizes=per_focus,
            min_hop=min_a_b_focus_separation_hops,
            rng=substream_rng(base_seed=base_seed, tag="separated_b_focus_seeds"),
            max_attempts=seed_max_attempts,
        )
        if separated_b_patches_local is None:
            return build_all_infeasible_set(
                reason="insufficient_separated_b_foci_pool", **infeas_kwargs,
            )
        b_focus_positions_global = [
            np.array(sorted(int(anchor_indices[i]) for i in patch), dtype=int)
            for patch in separated_b_patches_local
        ]
        b_positions_global = np.array(
            sorted(int(x) for patch in b_focus_positions_global for x in patch),
            dtype=int,
        )
        b_structure_prefix = "b_focus"

    return _assemble_focal_set(
        scaffold=scaffold,
        counterfactual_set_id=counterfactual_set_id,
        feature_a_id=feature_a_id, feature_b_id=feature_b_id,
        anchor_roi_id=str(anchor_roi_id),
        bg_a=bg_a, bg_b=bg_b, bg_bank=bg_bank,
        a_focus_positions_global=a_focus_positions_global,
        a_all_positions_global=a_all_positions_global,
        b_focus_positions_global=b_focus_positions_global,
        b_positions_global=b_positions_global,
        b_structure_prefix=b_structure_prefix,
        a_structure_prefix="focus" if mode == MODE_COLOCATED else "a_focus",
        mode=mode, per_focus=per_focus,
        n_anchor=n_anchor, k_total=k_total, n_foci=int(n_foci),
        broad_a_increment_focal=broad_a_increment_focal,
        broad_b_increment_focal=broad_b_increment_focal,
        prevalence_fraction=float(prevalence_fraction),
        requested=requested, nuisance_id=nuisance_id,
        nuisance_parameters=nuisance_parameters,
        base_seed=base_seed, partition=partition,
        related_sample_group_id=related_sample_group_id,
    )


# ---------------------------------------------------------------------------
# Shared assembly: compose fields, emit positive + b_independent_background.
# Both modes converge here after choosing B focus positions + structure ids.
# ---------------------------------------------------------------------------

def _assemble_focal_set(
    *, scaffold: SampleScaffold,
    counterfactual_set_id: str,
    feature_a_id: str, feature_b_id: str,
    anchor_roi_id: str,
    bg_a: np.ndarray, bg_b: np.ndarray, bg_bank: SampleBackgroundBank,
    a_focus_positions_global: List[np.ndarray],
    a_all_positions_global: np.ndarray,
    b_focus_positions_global: List[np.ndarray],
    b_positions_global: np.ndarray,
    a_structure_prefix: str, b_structure_prefix: str,
    mode: str, per_focus: List[int],
    n_anchor: int, k_total: int, n_foci: int,
    broad_a_increment_focal: float,
    broad_b_increment_focal: float,
    prevalence_fraction: float,
    requested: dict, nuisance_id: str,
    nuisance_parameters: dict,
    base_seed: int, partition: str,
    related_sample_group_id: str,
) -> ScenarioSet:
    a_overlay = constant_overlay(
        n=scaffold.n, positions=a_all_positions_global,
        magnitude=broad_a_increment_focal,
    )
    b_overlay_positive = constant_overlay(
        n=scaffold.n, positions=b_positions_global,
        magnitude=broad_b_increment_focal,
    )
    b_overlay_background = np.zeros(scaffold.n, dtype=float)
    a_field = compose_role_field(bg=bg_a, overlay=a_overlay)
    b_field_positive = compose_role_field(bg=bg_b, overlay=b_overlay_positive)
    b_field_background = compose_role_field(bg=bg_b, overlay=b_overlay_background)

    assert_composition(field=a_field, bg=bg_a, overlay=a_overlay,
                       scenario_name=SCENARIO_NAME, role="positive", feature="a")
    assert_composition(field=b_field_positive, bg=bg_b, overlay=b_overlay_positive,
                       scenario_name=SCENARIO_NAME, role="positive", feature="b")
    assert_composition(field=b_field_background, bg=bg_b, overlay=b_overlay_background,
                       scenario_name=SCENARIO_NAME,
                       role="b_independent_background", feature="b")

    common_realized = {
        "mode": mode,
        "prevalence_fraction_requested": prevalence_fraction,
        "prevalence_denominator_anchor_roi_size": n_anchor,
        "n_foci": n_foci,
        "per_focus_sizes": [int(x) for x in per_focus],
        "n_elevated_a": int(k_total),
        "n_elevated_b_positive": int(k_total),
        "n_elevated_b_independent": 0,
        "background_degenerate_rois": list(bg_bank.degenerate_rois),
    }

    builder = ScenarioSetBuilder(
        scaffold=scaffold, counterfactual_set_id=counterfactual_set_id,
        scenario_name=SCENARIO_NAME,
        feature_a_id=feature_a_id, feature_b_id=feature_b_id,
        nuisance_id=nuisance_id, nuisance_parameters=nuisance_parameters,
        requested_parameters=requested, rng_seed=base_seed,
        partition=partition, related_sample_group_id=related_sample_group_id,
        pair_a_roi_id=anchor_roi_id, pair_b_roi_id=anchor_roi_id,
    )

    a_focus_structures = [
        StructureAssignment(
            positions=positions, structure_kind=STRUCTURE_KIND_FOCUS,
            structure_id=f"{a_structure_prefix}_{i:02d}",
        )
        for i, positions in enumerate(a_focus_positions_global)
    ]
    b_focus_structures = [
        StructureAssignment(
            positions=positions, structure_kind=STRUCTURE_KIND_FOCUS,
            structure_id=f"{b_structure_prefix}_{i:02d}",
        )
        for i, positions in enumerate(b_focus_positions_global)
    ]

    builder.add_instance(
            role="positive",
            target_selected_a=a_all_positions_global,
            target_selected_b=b_positions_global,
            target_a_field=a_field,
            target_b_field=b_field_positive,
            structures_a=a_focus_structures,
            structures_b=b_focus_structures,
            realized_parameters=common_realized,
            role_specific_metadata={
                "comparison_kind": COMPARISON_KIND["positive"],
            },
    )

    builder.add_instance(
            role="b_independent_background",
            target_selected_a=a_all_positions_global,
            target_selected_b=np.array([], dtype=int),
            target_a_field=a_field,
            target_b_field=b_field_background,
            structures_a=a_focus_structures,
            structures_b=None,
            realized_parameters=common_realized,
            role_specific_metadata={
                "comparison_kind": COMPARISON_KIND[
                    "b_independent_background"
                ],
            },
    )

    return builder.finalize(invariant_policy=FOCAL_PATTERN_INVARIANTS)



# ---------------------------------------------------------------------------
# Focus construction
# ---------------------------------------------------------------------------

def _balanced_focus_sizes(k_total: int, n_foci: int) -> List[int]:
    """Balanced sizes: sizes differ by at most 1, summing to k_total."""
    base = k_total // n_foci
    rem = k_total % n_foci
    return [base + (1 if i < rem else 0) for i in range(n_foci)]


def _pick_focus_seeds(
    local_adj: List[List[int]], *, n_foci: int, min_hop: int,
    rng: np.random.Generator, max_attempts: int,
    excluded: Optional[set] = None,
) -> Optional[List[int]]:
    """Pick n_foci seeds with pairwise hop distance >= min_hop.

    ``excluded`` (local indices) is treated as a forbidden pool.
    """
    n = len(local_adj)
    allowed = [i for i in range(n) if not excluded or i not in excluded]
    if n_foci > len(allowed):
        return None
    allowed_arr = np.array(allowed, dtype=int)
    for _ in range(max_attempts):
        picks = rng.choice(allowed_arr, size=n_foci, replace=False)
        pairs = list(itertools.combinations(picks.tolist(), 2))
        dists = [_hop_distance(local_adj, a, b, cutoff=min_hop) for a, b in pairs]
        if all(d is None or d >= min_hop for d in dists):
            return sorted(int(i) for i in picks)
    return None


def _hop_distance(
    local_adj: List[List[int]], src: int, dst: int, *, cutoff: int,
) -> Optional[int]:
    """BFS hop distance capped at ``cutoff`` for early return."""
    if src == dst:
        return 0
    visited = {int(src)}
    frontier = [int(src)]
    d = 0
    while frontier and d < cutoff:
        d += 1
        nxt: List[int] = []
        for u in frontier:
            for v in local_adj[u]:
                if int(v) == dst:
                    return d
                if int(v) not in visited:
                    visited.add(int(v))
                    nxt.append(int(v))
        frontier = nxt
    return d if any(int(dst) in local_adj[u] for u in frontier) else None


def _grow_disjoint_patches_with_buffer(
    local_adj: List[List[int]], *, seeds: List[int],
    per_seed_target: List[int], min_hop: int,
    initially_excluded: Optional[set] = None,
) -> List[List[int]]:
    """Grow patches sequentially. Each patch's growth excludes every position within ``min_hop - 1`` hops of any spot already claimed by another patch (or in ``initially_excluded``)."""
    global_excluded: set = set(int(x) for x in (initially_excluded or set()))
    patches: List[List[int]] = []
    for pi, seed in enumerate(seeds):
        target = per_seed_target[pi]
        if int(seed) in global_excluded:
            patches.append([])
            continue
        patch = _bfs_patch_local(
            local_adj, seed=int(seed), target=target, excluded=global_excluded,
        )
        patches.append(patch)
        buffer = _kth_hop_neighborhood(local_adj, patch, hops=min_hop - 1)
        global_excluded.update(buffer)
        global_excluded.update(int(x) for x in patch)
    return patches


def _bfs_patch_local(
    local_adj: List[List[int]], *, seed: int, target: int, excluded: set,
) -> List[int]:
    n = len(local_adj)
    if n == 0 or int(seed) in excluded:
        return []
    visited = {int(seed)}
    order = [int(seed)]
    frontier = [int(seed)]
    while frontier and len(order) < target:
        nxt: List[int] = []
        cands = set()
        for u in frontier:
            for v in local_adj[u]:
                if int(v) in visited or int(v) in excluded:
                    continue
                cands.add(int(v))
        for v in sorted(cands):
            if len(order) >= target:
                break
            visited.add(int(v)); order.append(int(v)); nxt.append(int(v))
        frontier = nxt
    return order


def _kth_hop_neighborhood(
    local_adj: List[List[int]], seeds: List[int], hops: int,
) -> set:
    """All positions within ``hops`` graph hops of any seed."""
    if hops <= 0:
        return set()
    frontier = set(int(s) for s in seeds)
    visited = set(frontier)
    for _ in range(hops):
        nxt: set = set()
        for u in frontier:
            for v in local_adj[u]:
                if int(v) not in visited:
                    nxt.add(int(v))
        visited.update(nxt)
        frontier = nxt
    return visited


def _verify_pairwise_min_hop(
    local_adj: List[List[int]], patches: List[List[int]], min_hop: int,
) -> bool:
    """Every pair of spots from distinct patches has hop distance >= min_hop."""
    for i, patch_i in enumerate(patches):
        for j, patch_j in enumerate(patches):
            if j <= i or not patch_j:
                continue
            for a in patch_i:
                d = _bounded_distance(local_adj, int(a), set(int(x) for x in patch_j),
                                       cutoff=min_hop)
                if d < min_hop:
                    return False
    return True


def _bounded_distance(
    local_adj: List[List[int]], src: int, targets: set, *, cutoff: int,
) -> int:
    """Hop distance from ``src`` to the nearest node in ``targets``, capped at ``cutoff``. Returns ``cutoff`` if not reached."""
    if int(src) in targets:
        return 0
    visited = {int(src)}
    frontier = [int(src)]
    d = 0
    while frontier and d < cutoff:
        d += 1
        nxt: List[int] = []
        for u in frontier:
            for v in local_adj[u]:
                if int(v) in targets:
                    return d
                if int(v) not in visited:
                    visited.add(int(v))
                    nxt.append(int(v))
        frontier = nxt
    return cutoff


# ---------------------------------------------------------------------------
# Separated B focus growth
# ---------------------------------------------------------------------------

def _grow_separated_b_foci(
    *, local_adj: List[List[int]],
    a_focus_local_patches: List[List[int]],
    per_focus_sizes: List[int],
    min_hop: int, rng: np.random.Generator, max_attempts: int,
) -> Optional[List[List[int]]]:
    """Relocate n_foci connected B patches on the anchor's restricted adjacency,
    keeping every separated-B spot >= ``min_hop`` hops from every A focus spot.
    Retries the full seed->grow->verify cycle up to ``max_attempts`` times."""
    n_foci = len(a_focus_local_patches)
    if n_foci == 0:
        return []
    a_spots = [int(x) for patch in a_focus_local_patches for x in patch]
    excluded = set(a_spots)
    if min_hop > 1:
        excluded.update(_kth_hop_neighborhood(local_adj, a_spots, hops=min_hop - 1))
    expected_total = sum(per_focus_sizes)
    for _ in range(max_attempts):
        seeds = _pick_focus_seeds(
            local_adj, n_foci=n_foci, min_hop=min_hop,
            rng=rng, max_attempts=1, excluded=excluded,
        )
        if seeds is None:
            continue
        patches = _grow_disjoint_patches_with_buffer(
            local_adj, seeds=seeds, per_seed_target=per_focus_sizes,
            min_hop=min_hop, initially_excluded=excluded,
        )
        if sum(len(p) for p in patches) != expected_total:
            continue
        if any(len(p) != per_focus_sizes[i] for i, p in enumerate(patches)):
            continue
        if not _verify_pairwise_min_hop(local_adj, patches, min_hop):
            continue
        return patches
    return None
