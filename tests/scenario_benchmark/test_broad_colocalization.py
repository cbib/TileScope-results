"""Focused invariant tests for broad_colocalization (whole-ROI co-enrichment)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from scenario_benchmark import (
    SampleScaffold, SignalModel,
    build_broad_colocalization_set,
    load_sample_scaffold,
)
from scenario_benchmark.background_bank import SampleBackgroundBank
from scenario_benchmark.scenarios.broad_colocalization import (
    BROAD_COLOCALIZATION_SHARED_LATENT_RATIO, MIN_TARGET_ROI_SPOTS,
)
from scenario_benchmark.signal_model import substream_rng


ANCHOR = "ANCHOR"


def _multi_roi_scaffold() -> SampleScaffold:
    """20x6 grid; anchor (12 cols) + two 4-col non-anchor ROIs."""
    xs = np.repeat(np.arange(20), 6).astype(float)
    ys = np.tile(np.arange(6), 20).astype(float)
    roi = np.where(xs < 12, ANCHOR, np.where(xs < 16, "R1", "R2"))
    spot_id = [f"s{i:03d}" for i in range(len(xs))]
    df = pd.DataFrame({"x": xs, "y": ys, "spot_id": spot_id, "roi_id": roi})
    return load_sample_scaffold(df, sample_id="hand", radius_factor=1.5)


def _tiny_anchor_scaffold(anchor_size: int) -> SampleScaffold:
    xs = np.arange(anchor_size + 8).astype(float)
    ys = np.zeros_like(xs)
    roi = np.array([ANCHOR] * anchor_size + ["OTHER"] * 8, dtype=object)
    spot_id = [f"t{i:03d}" for i in range(len(xs))]
    df = pd.DataFrame({"x": xs, "y": ys, "spot_id": spot_id, "roi_id": roi})
    return load_sample_scaffold(df, sample_id="tiny", radius_factor=1.5)


def _model() -> SignalModel:
    return SignalModel(effect_delta=1.0)


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


def test_positive_field_equals_bg_plus_symmetric_shared_latent():
    scaf = _multi_roi_scaffold()
    effect_delta = 1.25
    res = build_broad_colocalization_set(
        scaffold=scaf, anchor_roi_id=ANCHOR,
        counterfactual_set_id="cf",
        feature_a_id="gene_a", feature_b_id="gene_b",
        signal_model=SignalModel(effect_delta=effect_delta), base_seed=1,
    )
    pos = _rows_by_role(res)["positive_broad"]
    target = np.where(scaf.source_parent_roi_id.astype(str) == ANCHOR)[0]
    bg = SampleBackgroundBank(scaffold=scaf, participant_roi_ids=(ANCHOR,))
    bg_a = bg.field("gene_a")
    bg_b = bg.field("gene_b")
    latent = substream_rng(
        base_seed=1, tag="broad_colocalization::shared_latent",
    ).standard_normal(target.size)
    latent = (latent - latent.mean()) / latent.std()
    scale = np.sqrt(0.5 * (np.var(bg_a[target]) + np.var(bg_b[target])))
    modulation = BROAD_COLOCALIZATION_SHARED_LATENT_RATIO * scale * latent
    a_overlay = np.zeros(scaf.n)
    a_overlay[target] = effect_delta + modulation
    b_overlay = np.zeros(scaf.n)
    b_overlay[target] = effect_delta + modulation
    got_a = _field_from_rows(pos, "feature_a_value", scaf)
    got_b = _field_from_rows(pos, "feature_b_value", scaf)
    assert np.array_equal(got_a, bg_a + a_overlay)
    assert np.array_equal(got_b, bg_b + b_overlay)


def test_null_A_byte_identical_and_B_equals_bank_and_means_align():
    scaf = _multi_roi_scaffold()
    res = build_broad_colocalization_set(
        scaffold=scaf, anchor_roi_id=ANCHOR,
        counterfactual_set_id="cf",
        feature_a_id="gene_a", feature_b_id="gene_b",
        signal_model=_model(), base_seed=1,
    )
    per = _rows_by_role(res)
    a_pos = _field_from_rows(per["positive_broad"], "feature_a_value", scaf)
    a_null = _field_from_rows(
        per["b_independent_background"], "feature_a_value", scaf,
    )
    assert np.array_equal(a_pos, a_null)
    bg = SampleBackgroundBank(scaffold=scaf, participant_roi_ids=(ANCHOR,)).field("gene_b")
    b_null = _field_from_rows(
        per["b_independent_background"], "feature_b_value", scaf,
    )
    assert np.array_equal(b_null, bg)
    null_inst = next(
        i for i in res.instances
        if i.counterfactual_role == "b_independent_background"
    )
    target = np.where(scaf.source_parent_roi_id.astype(str) == ANCHOR)[0]
    non_target = np.where(scaf.source_parent_roi_id.astype(str) != ANCHOR)[0]
    assert float(null_inst.realized_parameters["target_roi_mean_b"]) == float(
        np.mean(bg[target])
    )
    assert float(null_inst.realized_parameters["non_target_mean_b"]) == float(
        np.mean(bg[non_target])
    )


def test_positive_target_roi_pearson_exceeds_null_by_meaningful_margin():
    """Semantic test: A-B target-ROI Pearson under positive must
    exceed null by a large margin. Null Pearson equals Corr(bg_a[target],
    bg_b[target]) which is fixed by the bank seed; positive adds the
    shared latent modulation, which raises Pearson symmetrically."""
    scaf = _multi_roi_scaffold()
    res = build_broad_colocalization_set(
        scaffold=scaf, anchor_roi_id=ANCHOR,
        counterfactual_set_id="cf",
        feature_a_id="gene_a", feature_b_id="gene_b",
        signal_model=_model(), base_seed=1,
    )
    per = _rows_by_role(res)
    target = np.where(scaf.source_parent_roi_id.astype(str) == ANCHOR)[0]
    a_pos = _field_from_rows(per["positive_broad"], "feature_a_value", scaf)[target]
    b_pos = _field_from_rows(per["positive_broad"], "feature_b_value", scaf)[target]
    a_null = _field_from_rows(
        per["b_independent_background"], "feature_a_value", scaf,
    )[target]
    b_null = _field_from_rows(
        per["b_independent_background"], "feature_b_value", scaf,
    )[target]
    r_pos = float(np.corrcoef(a_pos, b_pos)[0, 1])
    r_null = float(np.corrcoef(a_null, b_null)[0, 1])
    # The shared latent produces correlation without making either
    # feature's nuisance field the causal source of the other.
    assert r_pos > 0.5, (r_pos, r_null)
    assert r_pos - r_null > 0.4, (r_pos, r_null)


@pytest.mark.parametrize(
    ("anchor_id", "expected_reason"),
    [(ANCHOR, "target_roi_too_small"), ("NOT_A_ROI", "target_roi_missing")],
)
def test_feasibility_gates(anchor_id, expected_reason):
    if expected_reason == "target_roi_too_small":
        scaf = _tiny_anchor_scaffold(anchor_size=MIN_TARGET_ROI_SPOTS - 1)
    else:
        scaf = _multi_roi_scaffold()
    res = build_broad_colocalization_set(
        scaffold=scaf, anchor_roi_id=anchor_id,
        counterfactual_set_id="cf",
        feature_a_id="gene_a", feature_b_id="gene_b",
        signal_model=_model(), base_seed=1,
    )
    reasons = {i.counterfactual_role: i.fallback_reason for i in res.instances}
    assert reasons == {
        "positive_broad": expected_reason,
        "b_independent_background": expected_reason,
    }
    assert res.spot_truth_rows == []
