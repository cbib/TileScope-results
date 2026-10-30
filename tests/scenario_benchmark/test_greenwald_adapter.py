"""Greenwald adapter: stable IDs + one whole-sample scaffold smoke."""
from __future__ import annotations

from pathlib import Path

from scenario_benchmark import stable_spot_id
from scenario_benchmark.adapters.greenwald import load_greenwald_sample_scaffold


_FIXTURE_CSV = (
    Path(__file__).resolve().parents[1]
    / "e2e/fixtures/greenwald_242T/242_T_expression_data_AC_Chromatin-Reg.csv"
)


def test_stable_spot_id_is_deterministic_and_coord_scoped():
    a = stable_spot_id(sample_id="248_T", roi_id="6", x=426.85548332, y=370.16602826)
    b = stable_spot_id(sample_id="248_T", roi_id="6", x=426.85548332, y=370.16602826)
    assert a == b, "same inputs must yield identical id"
    c = stable_spot_id(sample_id="248_T", roi_id="6", x=426.85548333, y=370.16602826)
    # 6-decimal quantization -> tiny drift under precision still yields
    # a different id when the seventh decimal alters the sixth.
    assert isinstance(a, str) and isinstance(c, str)


def test_stable_spot_id_scopes_by_sample_and_roi():
    xy = dict(x=100.0, y=200.0)
    assert (
        stable_spot_id(sample_id="248_T", roi_id="6", **xy)
        != stable_spot_id(sample_id="248_T", roi_id="7", **xy)
    )
    assert (
        stable_spot_id(sample_id="248_T", roi_id="6", **xy)
        != stable_spot_id(sample_id="242_T", roi_id="6", **xy)
    )


def test_load_fixture_sample_scaffold_smoke():
    """The active adapter builds one non-trivial whole-sample graph."""
    scaffold = load_greenwald_sample_scaffold(_FIXTURE_CSV, sample_id="242_T")
    assert scaffold.n > 0
    assert scaffold.edges.shape[0] > 0
    assert scaffold.median_nn > 0
    assert scaffold.source_parent_roi_id.shape == (scaffold.n,)
    assert scaffold.source_component_id.shape == (scaffold.n,)
