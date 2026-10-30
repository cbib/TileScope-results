"""Semantic invariants for spatial_gradient."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scenario_benchmark import (
    SignalModel, build_spatial_gradient_set, load_sample_scaffold,
)
from scenario_benchmark.background_bank import (
    BackgroundPolicy, SampleBackgroundBank,
)


ANCHOR_ID, OTHER_ID = "A", "B"


def _scaffold():
    rows, cols = 12, 20
    xs = np.repeat(np.arange(cols), rows).astype(float)
    ys = np.tile(np.arange(rows), cols).astype(float)
    roi = np.array(
        [ANCHOR_ID] * (10 * rows) + [OTHER_ID] * (10 * rows),
        dtype=object,
    )
    spot_ids = [f"s{i:04d}" for i in range(xs.size)]
    df = pd.DataFrame({
        "x": xs, "y": ys, "spot_id": spot_ids, "roi_id": roi.tolist(),
    })
    return load_sample_scaffold(df, sample_id="grad", radius_factor=1.5)


def test_background_policy_passthrough_clips_nonparticipant_bg():
    """The bounded-background policy must reach the gradient generator:
    nonparticipant-ROI spots on both features are exactly bg_policy-
    scaled clipped latent draws (no gradient overlay applies there)."""
    scaf = _scaffold()
    policy = BackgroundPolicy(
        participant_amplitude=0.05,
        non_participant_amplitude=0.12,
        latent_abs_clip=1.75,
    )
    s = build_spatial_gradient_set(
        scaffold=scaf, anchor_roi_id=ANCHOR_ID,
        gradient_relationship="opposed",
        counterfactual_set_id="cs",
        feature_a_id="fA", feature_b_id="fB",
        truth_tail_fraction=0.3,
        gradient_axis=(1.0, 0.0),
        signal_model=SignalModel(effect_delta=1.0),
        background_policy=policy,
        base_seed=42,
    )
    rows = sorted(
        [r for r in s.spot_truth_rows if r.counterfactual_role
         == "positive_spatial_gradient"],
        key=lambda r: r.source_spot_id,
    )
    id_to_idx = {sid: i for i, sid in enumerate(scaf.spot_ids.astype(str))}
    order = np.argsort([id_to_idx[r.source_spot_id] for r in rows])
    fa = np.array([r.feature_a_value for r in rows])[order]
    fb = np.array([r.feature_b_value for r in rows])[order]
    nonpart_mask = scaf.source_parent_roi_id.astype(str) == OTHER_ID

    latent_bank = SampleBackgroundBank(
        scaffold=scaf, participant_roi_ids=(), policy=policy,
    )
    clip = policy.latent_abs_clip
    latent_a_clipped = np.clip(latent_bank.latent("fA"), -clip, clip)
    latent_b_clipped = np.clip(latent_bank.latent("fB"), -clip, clip)
    expected_a = policy.background_level + policy.non_participant_amplitude * latent_a_clipped
    expected_b = policy.background_level + policy.non_participant_amplitude * latent_b_clipped
    assert np.allclose(fa[nonpart_mask], expected_a[nonpart_mask], atol=1e-12)
    assert np.allclose(fb[nonpart_mask], expected_b[nonpart_mask], atol=1e-12)
