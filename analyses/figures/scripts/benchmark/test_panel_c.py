"""Comparator panel: shape, scope, scenario-matched row order and NaN handling."""
from pathlib import Path
import sys
import shutil

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analyses.figures.scripts.benchmark import build_panel_c as p
from tilescope_results.bundles import verify


@pytest.fixture(scope="module")
def values():
    return p.load_values(p.comparator_benchmark_root() / "comparator_values.csv")


@pytest.fixture(scope="module")
def data(values):
    return p.build_matrix(values)


def test_eighteen_cases_six_modes(data):
    assert data["cases"].shape[0] == 18
    assert data["cases"].groupby("class_label").size().eq(3).all()
    assert set(data["cases"].class_label.unique()) == set(p.NAMES)


def test_columns_match_declared_methods(data):
    expected_labels = [s[3] for s in p.STATS]
    assert list(data["raw"].columns) == expected_labels
    assert data["raw"].shape == (18, len(expected_labels))


def test_row_order_matches_b0(data):
    from analyses.figures.scripts.benchmark.build_panel_b_profiles import prepare
    b0_order = prepare()['primary']['order']
    assert data["cases"].index.tolist() == b0_order


def test_each_method_group_is_non_empty(data):
    for method_name, subs in p.METHODS:
        assert len(subs) >= 1, f"Method {method_name} has no sub-columns"


def test_only_stopover_has_missing_values(data):
    missing_per_column = data["raw"].isna().sum()
    stopover = next(s[3] for s in p.STATS if s[1] == "STopover J_comp")
    assert missing_per_column[stopover] > 0
    assert missing_per_column.drop(stopover).eq(0).all()


def test_zscore_is_column_wise_over_finite_rows(data):
    for col in data["raw"].columns:
        finite = data["raw"][col].dropna()
        standardised_finite = data["standardised"].loc[finite.index, col]
        np.testing.assert_allclose(standardised_finite.mean(), 0, atol=1e-12)
        np.testing.assert_allclose(standardised_finite.std(ddof=0), 1, atol=1e-12)


def test_plot_renders(data):
    fig = p.plot(data)
    assert fig.axes, "no axes rendered"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_comparator_schema_and_download_integrity(tmp_path):
    source = p.comparator_benchmark_root() / "comparator_values.csv"
    bundle = source.parent.parent
    assert source in verify(bundle)
    schema = pd.read_csv(source.with_name("comparator_methods_schema.csv"), keep_default_na=False)
    assert set(zip(schema.method, schema.direction)) == {
        (source, direction) for _, source, direction, _ in p.STATS
    }
    output = tmp_path / 'Supplementary_Data_1'
    relative = "comparator_methods/comparator_values.csv"
    (output / relative).parent.mkdir(parents=True)
    shutil.copy2(source, output / relative)
    index = pd.read_csv(bundle / "file_index.csv")
    index.loc[index.file.eq(relative)].to_csv(output / "file_index.csv", index=False)
    verify(output)
    (output / relative).write_text('changed')
    with pytest.raises(ValueError, match='Bundle file changed'):
        verify(output)
