"""SampleBackgroundBank contract tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scenario_benchmark import (
    SignalModel, build_broad_colocalization_set, load_sample_scaffold,
)
from scenario_benchmark.background_bank import (
    DEFAULT_BACKGROUND_POLICY, SampleBackgroundBank,
)


def _multi_roi_scaffold():
    """20x6 grid; 3 ROIs, all with >= 2 spots so no degenerates."""
    xs = np.repeat(np.arange(20), 6).astype(float)
    ys = np.tile(np.arange(6), 20).astype(float)
    roi = np.where(xs < 12, "ANCHOR", np.where(xs < 16, "R1", "R2"))
    spot_id = [f"s{i:03d}" for i in range(len(xs))]
    df = pd.DataFrame({"x": xs, "y": ys, "spot_id": spot_id, "roi_id": roi})
    return load_sample_scaffold(df, sample_id="hand", radius_factor=1.5)


def _scaffold_with_singleton():
    """Include one ROI holding exactly one spot to trigger degenerate path."""
    xs = np.repeat(np.arange(20), 6).astype(float)
    ys = np.tile(np.arange(6), 20).astype(float)
    roi = np.where(xs < 12, "ANCHOR", np.where(xs < 16, "R1", "R2"))
    spot_id = [f"s{i:03d}" for i in range(len(xs))]
    roi = roi.astype(object)
    roi[0] = "LONE"
    xs = xs.copy(); ys = ys.copy()
    xs[0] = 999.0; ys[0] = 999.0
    df = pd.DataFrame({"x": xs, "y": ys, "spot_id": spot_id, "roi_id": roi})
    return load_sample_scaffold(df, sample_id="hand_lone", radius_factor=1.5)


def _tiny_two_roi_scaffold():
    """Two 5-spot rows so the compositional-invariant integration test
    has a real scaffold to run against."""
    xs = np.repeat(np.arange(5), 5).astype(float)
    ys = np.tile(np.arange(5), 5).astype(float)
    roi = np.where(xs < 4, "ANCHOR", "R1")
    spot_id = [f"t{i:03d}" for i in range(len(xs))]
    df = pd.DataFrame({"x": xs, "y": ys, "spot_id": spot_id, "roi_id": roi})
    return load_sample_scaffold(df, sample_id="tiny", radius_factor=1.5)


def test_two_instances_same_scaffold_produce_identical_fields():
    scaf = _multi_roi_scaffold()
    b1 = SampleBackgroundBank(scaffold=scaf, participant_roi_ids=("ANCHOR",))
    b2 = SampleBackgroundBank(scaffold=scaf, participant_roi_ids=("ANCHOR",))
    f1 = b1.field("gene_a")
    f2 = b2.field("gene_a")
    assert np.array_equal(f1, f2)


def test_feature_independence_element_wise():
    scaf = _multi_roi_scaffold()
    b = SampleBackgroundBank(scaffold=scaf, participant_roi_ids=("ANCHOR",))
    fa = b.field("gene_a")
    fb = b.field("gene_b")
    non_singleton = np.isin(
        scaf.source_parent_roi_id, np.array(["ANCHOR", "R1", "R2"]),
    )
    diff = fa[non_singleton] != fb[non_singleton]
    assert diff.mean() > 0.9


def test_singleton_roi_field_finite_and_bounded():
    scaf = _scaffold_with_singleton()
    b = SampleBackgroundBank(scaffold=scaf, participant_roi_ids=("ANCHOR",))
    field = b.field("gene_a")
    lone_idx = np.where(scaf.source_parent_roi_id == "LONE")[0]
    assert lone_idx.size == 1
    v = float(field[int(lone_idx[0])])
    assert np.isfinite(v)
    p = DEFAULT_BACKGROUND_POLICY
    # LONE is non-participant → amplitude = non_participant_amplitude on unit-SD latent
    assert abs(v - p.background_level) < 6.0 * p.non_participant_amplitude


def test_cache_returns_equal_but_separate_arrays():
    scaf = _multi_roi_scaffold()
    b = SampleBackgroundBank(scaffold=scaf, participant_roi_ids=("ANCHOR",))
    a1 = b.field("gene_a")
    a2 = b.field("gene_a")
    assert np.array_equal(a1, a2)
    assert a1 is not a2  # defensive copy


def test_participant_status_changes_amplitude_not_latent():
    """Changing participant_roi_ids scales the same latent draws — it
    does not resample. Bank_A treats ANCHOR as participant; Bank_B
    treats R1 as participant. Latent per feature is identical; only
    the amplitude vector differs."""
    scaf = _multi_roi_scaffold()
    b_anchor = SampleBackgroundBank(scaffold=scaf, participant_roi_ids=("ANCHOR",))
    b_r1 = SampleBackgroundBank(scaffold=scaf, participant_roi_ids=("R1",))
    assert np.array_equal(b_anchor.latent("gene_a"), b_r1.latent("gene_a"))
    # Fields differ on ANCHOR and R1 rows (different amplitude), match on R2.
    fa_anchor = b_anchor.field("gene_a")
    fa_r1 = b_r1.field("gene_a")
    r2_mask = scaf.source_parent_roi_id == "R2"
    assert np.array_equal(fa_anchor[r2_mask], fa_r1[r2_mask])
    anchor_mask = scaf.source_parent_roi_id == "ANCHOR"
    assert not np.array_equal(fa_anchor[anchor_mask], fa_r1[anchor_mask])


def test_participant_amplitude_smaller_than_non_participant():
    """Empirical SD on participant ROIs < empirical SD on non-participant."""
    scaf = _multi_roi_scaffold()
    b = SampleBackgroundBank(scaffold=scaf, participant_roi_ids=("ANCHOR",))
    fa = b.field("gene_a")
    anchor_sd = float(fa[scaf.source_parent_roi_id == "ANCHOR"].std(ddof=1))
    r1_sd = float(fa[scaf.source_parent_roi_id == "R1"].std(ddof=1))
    p = DEFAULT_BACKGROUND_POLICY
    assert anchor_sd < r1_sd
    # SDs should be close to their target amplitudes (loose CLT band).
    assert 0.3 * p.participant_amplitude < anchor_sd < 3.0 * p.participant_amplitude
    assert 0.3 * p.non_participant_amplitude < r1_sd < 3.0 * p.non_participant_amplitude


def test_broad_colocalization_overlay_is_target_only_and_mean_preserving():
    """End-to-end proof of the compositional invariant.

    Run the real broad_colocalization builder, extract the positive
    role's feature-a field, and verify byte-for-byte that
    the role differs from its frozen bank only on the target ROI and
    the zero-mean shared modulation preserves the requested broad mean.
    """
    scaf = _tiny_two_roi_scaffold()
    effect_delta = 1.25
    result = build_broad_colocalization_set(
        scaffold=scaf, anchor_roi_id="ANCHOR",
        counterfactual_set_id="cf",
        feature_a_id="gene_a", feature_b_id="gene_b",
        signal_model=SignalModel(effect_delta=effect_delta), base_seed=1,
    )
    pos_rows = [
        r for r in result.spot_truth_rows
        if r.counterfactual_role == "positive_broad"
    ]
    id_to_pos = {str(scaf.spot_ids[i]): int(i) for i in range(scaf.n)}
    positive_field = np.zeros(scaf.n, dtype=float)
    elevated = np.zeros(scaf.n, dtype=bool)
    for r in pos_rows:
        i = id_to_pos[r.source_spot_id]
        positive_field[i] = float(r.feature_a_value)
        elevated[i] = bool(r.feature_a_is_elevated)

    bg = SampleBackgroundBank(
        scaffold=scaf, participant_roi_ids=("ANCHOR",),
    ).field("gene_a")
    # Byte-exact bg-only outside elevated positions.
    non_elev = ~elevated
    assert np.array_equal(positive_field[non_elev], bg[non_elev])
    observed_overlay = positive_field - bg
    assert np.array_equal(observed_overlay[non_elev], np.zeros(non_elev.sum()))
    assert np.isclose(np.mean(observed_overlay[elevated]), effect_delta)
