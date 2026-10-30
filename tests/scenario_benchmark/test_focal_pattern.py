"""Focused invariant tests for focal_pattern (colocated + separated modes)."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from scenario_benchmark import (
    FOCAL_PATTERN_MODE_COLOCATED, FOCAL_PATTERN_MODE_SEPARATED,
    SampleScaffold, SignalModel,
    build_focal_pattern_set,
    load_sample_scaffold,
)
from scenario_benchmark.background_bank import SampleBackgroundBank


ANCHOR = "ANCHOR"


def _multi_roi_scaffold() -> SampleScaffold:
    """20x6 grid; 12-col anchor + two 4-col non-anchor ROIs."""
    xs = np.repeat(np.arange(20), 6).astype(float)
    ys = np.tile(np.arange(6), 20).astype(float)
    roi = np.where(xs < 12, ANCHOR, np.where(xs < 16, "R1", "R2"))
    spot_id = [f"s{i:03d}" for i in range(len(xs))]
    df = pd.DataFrame({"x": xs, "y": ys, "spot_id": spot_id, "roi_id": roi})
    return load_sample_scaffold(df, sample_id="hand", radius_factor=1.5)


def _tight_anchor_scaffold() -> SampleScaffold:
    """1-D chain: 10 anchor spots + 4 non-anchor. Big enough for 3 A foci
    at hop-3 separation; too tight for 3 additional separated B foci at the
    same separation from A."""
    xs = np.arange(14).astype(float)
    ys = np.zeros_like(xs)
    roi = np.array([ANCHOR] * 10 + ["OTHER"] * 4, dtype=object)
    spot_id = [f"t{i:03d}" for i in range(len(xs))]
    df = pd.DataFrame({"x": xs, "y": ys, "spot_id": spot_id, "roi_id": roi})
    return load_sample_scaffold(df, sample_id="tight", radius_factor=1.5)


def _model(delta: float = 1.0) -> SignalModel:
    return SignalModel(effect_delta=delta)


def _build(scaf: SampleScaffold, mode: str, *, delta: float = 1.0,
           prevalence_fraction: float = 0.10, n_foci: int = 3,
           min_hop: int = 2, base_seed: int = 1):
    return build_focal_pattern_set(
        scaffold=scaf, anchor_roi_id=ANCHOR, mode=mode,
        counterfactual_set_id=f"cf__{mode}",
        feature_a_id="gene_a", feature_b_id="gene_b",
        prevalence_fraction=prevalence_fraction, n_foci=n_foci,
        min_focus_separation_hops=min_hop,
        signal_model=_model(delta), base_seed=base_seed,
    )


def _rows_by_role(result) -> Dict[str, list]:
    by_iid = {i.scenario_instance_id: (i.counterfactual_role, i.generator_infeasible)
              for i in result.instances}
    out: Dict[str, list] = {}
    for r in result.spot_truth_rows:
        role, infeas = by_iid[r.scenario_instance_id]
        if infeas:
            continue
        out.setdefault(role, []).append(r)
    return out


def _field_from_rows(rows, field: str, scaf: SampleScaffold) -> np.ndarray:
    idx = {str(scaf.spot_ids[i]): i for i in range(scaf.n)}
    out = np.zeros(scaf.n, dtype=float)
    for r in rows:
        out[idx[r.source_spot_id]] = float(getattr(r, field))
    return out


def _elevated_positions(rows, field: str, scaf: SampleScaffold) -> set:
    idx = {str(scaf.spot_ids[i]): i for i in range(scaf.n)}
    return {idx[r.source_spot_id] for r in rows if getattr(r, field)}


def _connected_components(positions: set, scaf: SampleScaffold) -> List[set]:
    seen: set = set()
    out: List[set] = []
    for start in sorted(positions):
        if start in seen:
            continue
        comp: set = set()
        stack = [start]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u); comp.add(u)
            for v in scaf.adj[u]:
                if int(v) in positions and int(v) not in seen:
                    stack.append(int(v))
        out.append(comp)
    return out


def _hop_distance(scaf: SampleScaffold, src: int, targets: set, cutoff: int) -> int:
    if src in targets:
        return 0
    visited = {src}
    frontier = [src]
    d = 0
    while frontier and d < cutoff:
        d += 1
        nxt: List[int] = []
        for u in frontier:
            for v in scaf.adj[u]:
                if int(v) in targets:
                    return d
                if int(v) not in visited:
                    visited.add(int(v)); nxt.append(int(v))
        frontier = nxt
    return cutoff


def test_colocated_positive_A_and_B_share_focus_positions_byte_exact():
    """Colocated positive: A and B fields = bg + delta * 1[union_of_a_foci] byte-exact."""
    scaf = _multi_roi_scaffold()
    delta = 1.5
    res = _build(scaf, FOCAL_PATTERN_MODE_COLOCATED, delta=delta)
    per = _rows_by_role(res)
    pos = per["positive"]
    a_foci = _elevated_positions(pos, "feature_a_is_elevated", scaf)
    b_foci = _elevated_positions(pos, "feature_b_is_elevated", scaf)
    assert a_foci == b_foci
    bg = SampleBackgroundBank(scaffold=scaf, participant_roi_ids=(ANCHOR,))
    bg_a = bg.field("gene_a"); bg_b = bg.field("gene_b")
    overlay = np.zeros(scaf.n)
    overlay[np.array(sorted(a_foci), dtype=int)] = delta
    got_a = _field_from_rows(pos, "feature_a_value", scaf)
    got_b = _field_from_rows(pos, "feature_b_value", scaf)
    assert np.array_equal(got_a, bg_a + overlay)
    assert np.array_equal(got_b, bg_b + overlay)


def test_separated_positive_B_disjoint_from_A_and_min_hop_apart():
    """Separated positive: A byte-identical to colocated positive A; B on
    separated foci disjoint from A_foci at hop distance >= min_hop; B field
    = bg_b + delta * 1[separated_b_foci] byte-exact."""
    scaf = _multi_roi_scaffold()
    delta = 1.25
    min_hop = 3
    coloc = _build(scaf, FOCAL_PATTERN_MODE_COLOCATED, delta=delta, min_hop=min_hop)
    sep = _build(scaf, FOCAL_PATTERN_MODE_SEPARATED, delta=delta, min_hop=min_hop)
    coloc_pos = _rows_by_role(coloc)["positive"]
    sep_pos = _rows_by_role(sep)["positive"]
    a_coloc = _field_from_rows(coloc_pos, "feature_a_value", scaf)
    a_sep = _field_from_rows(sep_pos, "feature_a_value", scaf)
    assert np.array_equal(a_coloc, a_sep), "A drifted between colocated and separated positive"
    a_foci = _elevated_positions(sep_pos, "feature_a_is_elevated", scaf)
    b_foci = _elevated_positions(sep_pos, "feature_b_is_elevated", scaf)
    assert a_foci.isdisjoint(b_foci)
    for spot in b_foci:
        d = _hop_distance(scaf, int(spot), a_foci, cutoff=min_hop)
        assert d >= min_hop, f"separated-B spot {spot} at hop {d} < min_hop {min_hop} from A"
    bg_b = SampleBackgroundBank(scaffold=scaf, participant_roi_ids=(ANCHOR,)).field("gene_b")
    overlay = np.zeros(scaf.n)
    overlay[np.array(sorted(b_foci), dtype=int)] = delta
    got_b = _field_from_rows(sep_pos, "feature_b_value", scaf)
    assert np.array_equal(got_b, bg_b + overlay)


def test_b_independent_background_both_modes_A_identical_and_B_is_bank_only():
    """Both modes' b_independent_background: A byte-identical to that mode's
    positive; B = bg_b bit-exact (no overlay)."""
    scaf = _multi_roi_scaffold()
    bg_b = SampleBackgroundBank(scaffold=scaf, participant_roi_ids=(ANCHOR,)).field("gene_b")
    for mode in (FOCAL_PATTERN_MODE_COLOCATED, FOCAL_PATTERN_MODE_SEPARATED):
        res = _build(scaf, mode)
        per = _rows_by_role(res)
        pos = per["positive"]; bg = per["b_independent_background"]
        a_pos = _field_from_rows(pos, "feature_a_value", scaf)
        a_bg = _field_from_rows(bg, "feature_a_value", scaf)
        assert np.array_equal(a_pos, a_bg), f"{mode}: A drifted positive vs background"
        b_bg = _field_from_rows(bg, "feature_b_value", scaf)
        assert np.array_equal(b_bg, bg_b), f"{mode}: B on background != bg_b"


def test_pair_lock_across_modes_bg_a_foci_and_b_focus_sizes_match():
    """Colocated and separated (built as separate sets on the same substrate
    with the same seed) share byte-identical bg_a, bg_b, A_foci positions,
    and B focus SIZE MULTISET."""
    scaf = _multi_roi_scaffold()
    min_hop = 3
    coloc = _build(scaf, FOCAL_PATTERN_MODE_COLOCATED, min_hop=min_hop)
    sep = _build(scaf, FOCAL_PATTERN_MODE_SEPARATED, min_hop=min_hop)
    coloc_pos = _rows_by_role(coloc)["positive"]
    sep_pos = _rows_by_role(sep)["positive"]
    a_coloc_foci = _elevated_positions(coloc_pos, "feature_a_is_elevated", scaf)
    a_sep_foci = _elevated_positions(sep_pos, "feature_a_is_elevated", scaf)
    assert a_coloc_foci == a_sep_foci, "A foci positions drifted across modes"
    coloc_bg_field = _field_from_rows(
        _rows_by_role(coloc)["b_independent_background"], "feature_b_value", scaf,
    )
    sep_bg_field = _field_from_rows(
        _rows_by_role(sep)["b_independent_background"], "feature_b_value", scaf,
    )
    assert np.array_equal(coloc_bg_field, sep_bg_field), (
        "bg_b drifted across modes (background role should be bank-only in both)"
    )
    coloc_b_foci = _elevated_positions(coloc_pos, "feature_b_is_elevated", scaf)
    sep_b_foci = _elevated_positions(sep_pos, "feature_b_is_elevated", scaf)
    coloc_sizes = Counter(len(c) for c in _connected_components(coloc_b_foci, scaf))
    sep_sizes = Counter(len(c) for c in _connected_components(sep_b_foci, scaf))
    assert coloc_sizes == sep_sizes, (
        f"B focus size multisets differ across modes: coloc={coloc_sizes} sep={sep_sizes}"
    )


def test_separated_infeasibility_on_tight_anchor_colocated_stays_feasible():
    """On the tight-chain anchor, `separated` mode cannot place 3 additional
    B foci at hop-3 from every A focus; it emits scenario-set-level
    infeasibility. `colocated` mode is unaffected and stays feasible."""
    scaf = _tight_anchor_scaffold()
    sep = _build(scaf, FOCAL_PATTERN_MODE_SEPARATED, prevalence_fraction=0.30,
                 n_foci=3, min_hop=3)
    for inst in sep.instances:
        assert inst.role_disposition == "infeasible"
        assert inst.fallback_reason == "insufficient_separated_b_foci_pool"
    coloc = _build(scaf, FOCAL_PATTERN_MODE_COLOCATED, prevalence_fraction=0.30,
                   n_foci=3, min_hop=3)
    by_role = {i.counterfactual_role: i for i in coloc.instances}
    assert by_role["positive"].role_disposition == "feasible"
    assert by_role["b_independent_background"].role_disposition == "feasible"


def test_within_anchor_pearson_contract():
    """Semantic sanity: colocated positive Pearson > 0.5, separated positive
    Pearson < 0 (structural anti-overlap), background near zero in both
    modes, and colocated - separated > 0.5 (morphology specificity)."""
    scaf = _multi_roi_scaffold()
    anchor_mask = scaf.source_parent_roi_id.astype(str) == ANCHOR
    delta = 2.0

    def _r(rows, mask):
        a = _field_from_rows(rows, "feature_a_value", scaf)[mask]
        b = _field_from_rows(rows, "feature_b_value", scaf)[mask]
        return float(np.corrcoef(a, b)[0, 1])

    coloc = _build(scaf, FOCAL_PATTERN_MODE_COLOCATED, delta=delta)
    sep = _build(scaf, FOCAL_PATTERN_MODE_SEPARATED, delta=delta)
    r_coloc_pos = _r(_rows_by_role(coloc)["positive"], anchor_mask)
    r_sep_pos = _r(_rows_by_role(sep)["positive"], anchor_mask)
    r_coloc_bg = _r(_rows_by_role(coloc)["b_independent_background"], anchor_mask)
    r_sep_bg = _r(_rows_by_role(sep)["b_independent_background"], anchor_mask)

    assert r_coloc_pos > 0.5, f"colocated positive r={r_coloc_pos} not > 0.5"
    assert r_sep_pos < 0.0, f"separated positive r={r_sep_pos} not < 0"
    assert abs(r_coloc_bg) < 0.2, f"colocated background r={r_coloc_bg} not near 0"
    assert abs(r_sep_bg) < 0.2, f"separated background r={r_sep_bg} not near 0"
    assert r_coloc_pos - r_sep_pos > 0.5, (
        f"morphology specificity: coloc={r_coloc_pos} sep={r_sep_pos}"
    )


def test_non_anchor_background_byte_identical_across_roles_both_modes():
    """Non-anchor spots byte-identical across roles in each mode
    (delivered by bg_bank + overlay with overlay zero off-anchor)."""
    scaf = _multi_roi_scaffold()
    for mode in (FOCAL_PATTERN_MODE_COLOCATED, FOCAL_PATTERN_MODE_SEPARATED):
        res = _build(scaf, mode)
        per = _rows_by_role(res)
        pos = sorted(per["positive"], key=lambda r: r.source_spot_id)
        bg = sorted(per["b_independent_background"], key=lambda r: r.source_spot_id)
        for p, q in zip(pos, bg):
            if p.source_parent_roi_id != ANCHOR:
                assert p.feature_a_value == q.feature_a_value
                assert p.feature_b_value == q.feature_b_value
