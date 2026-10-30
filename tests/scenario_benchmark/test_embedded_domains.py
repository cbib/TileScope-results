"""Focused biological invariants for observed embedded B domains."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scenario_benchmark import (
    SignalModel, build_embedded_domains_set, load_sample_scaffold,
)
from scenario_benchmark.scenarios.embedded_domains import (
    MIN_EMBEDDED_DOMAIN_SPOTS, POSITIVE_ROLE, ROLES,
)


A_ID, B_ID, C_ID = "A", "B", "C"


def _scaffold():
    """A territory with two enclosed five-spot B domains and one
    non-enclosed B component beside C."""
    rows, cols = 12, 20
    xs = np.repeat(np.arange(cols), rows).astype(float)
    ys = np.tile(np.arange(rows), cols).astype(float)
    roi = np.array(
        [A_ID] * (10 * rows) + [B_ID] * (7 * rows) + [C_ID] * (3 * rows),
        dtype=object,
    )

    def set_roi(col, row, value):
        roi[int(col * rows + row)] = value

    for col, row in [(4, 4), (3, 4), (5, 4), (4, 3), (4, 5)]:
        set_roi(col, row, B_ID)
    for col, row in [(4, 8), (3, 8), (5, 8), (4, 7), (4, 9)]:
        set_roi(col, row, B_ID)
    for col in (7, 8):
        for row in range(1, 11):
            set_roi(col, row, B_ID)
    set_roi(9, 5, C_ID)
    return load_sample_scaffold(pd.DataFrame({
        "x": xs,
        "y": ys,
        "spot_id": [f"s{i:04d}" for i in range(xs.size)],
        "roi_id": roi,
    }), sample_id="embedded", radius_factor=1.5)


def _build(**kwargs):
    scaffold = _scaffold()
    result = build_embedded_domains_set(
        scaffold=scaffold,
        a_roi_id=A_ID,
        b_roi_id=B_ID,
        counterfactual_set_id="cs",
        feature_a_id="fA",
        feature_b_id="fB",
        signal_model=SignalModel(effect_delta=1.0),
        base_seed=42,
        **kwargs,
    )
    return scaffold, result


def _positions(scaffold, result, *, feature):
    by_id = {
        str(spot_id): i for i, spot_id in enumerate(scaffold.spot_ids)
    }
    flag = f"feature_{feature}_is_elevated"
    return {
        by_id[row.source_spot_id]
        for row in result.spot_truth_rows
        if row.counterfactual_role == POSITIVE_ROLE and getattr(row, flag)
    }


def test_observed_domains_are_the_only_role_and_keep_source_labels():
    scaffold, result = _build()
    assert ROLES == (POSITIVE_ROLE,)
    assert [instance.counterfactual_role for instance in result.instances] == [
        POSITIVE_ROLE,
    ]
    instance = result.instances[0]
    assert instance.role_disposition == "feasible"
    assert instance.realized_parameters["observed_component_sizes"] == [5, 5]
    assert MIN_EMBEDDED_DOMAIN_SPOTS == 5
    for row in result.spot_truth_rows:
        assert str(row.generated_roi_id) == str(row.source_parent_roi_id)


def test_a_surrounds_b_without_forced_carry_over():
    scaffold, result = _build()
    a_positions = _positions(scaffold, result, feature="a")
    b_labelled = {
        i for i, roi in enumerate(scaffold.source_parent_roi_id.astype(str))
        if roi == B_ID
    }
    assert a_positions
    assert not (a_positions & b_labelled)


def test_b_population_baseline_covers_all_b_labelled_spots():
    scaffold, result = _build(
        a_host_baseline=0.25,
        b_population_baseline=0.25,
        b_local_increment=0.0,
    )
    b_positions = _positions(scaffold, result, feature="b")
    expected = {
        i for i, roi in enumerate(scaffold.source_parent_roi_id.astype(str))
        if roi == B_ID
    }
    assert b_positions == expected
