"""All-case scope, explicit missingness and label-free clustering contracts."""
import numpy as np
import pandas as pd
import pytest

from analyses.scenario_discriminability.recognition import full_catalogue_clustering as f


def metadata(columns, groups, supports):
    return pd.DataFrame(dict(feature=columns, group=groups, support=supports))


def test_pairwise_missing_is_neither_zero_nor_availability_distance():
    z = pd.DataFrame([[0., np.nan, 1.], [2., 9., np.nan], [0., 3., 1.]],
                     columns=["s", "boundary", "t"], index=["one", "two", "three"])
    m = metadata(z.columns, ["corr", "boundary", "corr"], ["source", "source", "tile"])
    d, coverage = f.distances(z, m)
    assert d[0, 1] == 2.  # NOT sqrt(mean([4,81,1])). Only s is shared.
    assert d[0, 2] == 0.  # Missing boundary is not a phenotype.
    assert coverage.iloc[0].n_common_values == 1
    assert coverage.iloc[0].n_common_families == 1


def test_family_and_representation_balancing():
    # Family A: source differences 2,2; tile difference 4. Family B: difference 6.
    z = pd.DataFrame([[0., 0., 0., 0.], [2., 2., 4., 6.]], columns=list("abcd"))
    m = metadata(z.columns, ["A", "A", "A", "B"], ["source", "source", "tile", "source"])
    d, _ = f.distances(z, m)
    assert d[0, 1] == pytest.approx(np.sqrt(((4+16)/2+36)/2))
    duplicated = z.assign(a_copy=z.a)
    mm = pd.concat([m, metadata(["a_copy"], ["A"], ["source"])])
    dd, _ = f.distances(duplicated, mm)
    np.testing.assert_allclose(d, dd)


def test_no_overlap_fails_instead_of_inventing_a_distance():
    z = pd.DataFrame([[1., np.nan], [np.nan, 2.]], columns=["a", "b"])
    m = metadata(z.columns, ["A", "B"], ["S", "S"])
    with pytest.raises(ValueError, match="No common"):
        f.distances(z, m)
    d, _ = f.distances(z, m, allow_missing=True)
    assert np.isnan(d[0, 1])


def test_ward_rejects_pairwise_available_data():
    x = pd.DataFrame([[0., 1.], [1., np.nan], [2., 3.]], columns=["a", "b"])
    m = metadata(x.columns, ["A", "B"], ["S", "S"])
    with pytest.raises(ValueError, match="Ward requires"):
        f.cluster(x, m, method="ward")


def test_scaling_preserves_missing_and_excludes_constant():
    x = pd.DataFrame({"a": [1., np.nan, 3.], "b": [7., 7., 7.], "c": [np.nan, 1., np.nan]})
    z, audit = f.standardize(x)
    assert list(z.columns) == ["a"]
    assert np.isnan(z.loc[1, "a"])
    np.testing.assert_allclose(z.a.dropna(), [-1., 1.])
    assert not audit.loc[["b", "c"], "active"].any()


def test_enrichment_recovery_all_entries_and_undefined(tmp_path, monkeypatch):
    monkeypatch.setenv("TILESCOPE_DATA_DIR", str(tmp_path))
    cases = pd.DataFrame([dict(case_id="c", scenario_name="mode", positive_role="positive", a_roi_id=1, b_roi_id=1)]).set_index("case_id")
    path = tmp_path/"mode/c/positive"; path.mkdir(parents=True)
    channels = ["g1", "g2", "joint_high", "joint_low"]
    pd.DataFrame([dict(roi_id=1, statistic="enrichment", channel=c, target_surface="tessellation_median",
        target_effect_value=v, target_raw_status=s, n_comparable_replicates=n)
        for c, v, s, n in zip(channels, [.5, -.1, np.nan, -.5],
            ["positive_over_zero_context", "finite", "zero_over_zero_context", "finite"], [3, 2, 0, 3])]).to_csv(path/"roi_cross_grain_sensitivity.csv", index=False)
    x = pd.DataFrame(np.nan, index=["c"], columns=[f"{slot}|tessellation|enrichment_median|{c}|" for slot in ["A_ROI", "B_ROI"] for c in channels])
    x.iloc[0, 1] = .8  # Also repair finite-only summaries; not just missing entries.
    y, corrections = f.recover_tessellation_enrichment(cases, x, runs_root=tmp_path)
    np.testing.assert_allclose(y.iloc[0, :4], [1., .4, np.nan, 0.], equal_nan=True)
    assert len(corrections) == 8 and corrections.recovered_from_missing.sum() == 5


@pytest.fixture(scope="module")
def data():
    return f.prepare()


def test_full_scope_and_all_export_accounting(data):
    assert len(data["cases"]) == 30
    assert data["cases"].groupby("class_label").size().eq(3).all()
    assert data["numeric"].shape == (30, 160)
    assert set(data["meta"].group) == set(f.FAMILIES)
    assert set(data["meta"].slot) == {"A_ROI", "B_ROI", "A_to_B", "B_to_A"}
    assert len(data["catalogue"]) == 352
    assert data["export_inventory"].disposition.notna().all()
    assert len(data["corrections"]) == 240
    assert data["corrections"].recovered_from_missing.any()


def test_all_variants_keep_all_cases_and_do_not_assert_success(data):
    for name, result in data["variants"].items():
        assert result["tree"].shape == (29, 4), name
        assert set(result["order"]) == set(data["cases"].index)
    assert data["audits"].groupby("variant").size().eq(10).all()
    coverage = data["variants"]["primary"]["coverage"]
    assert len(coverage) == 435
    assert coverage.distance.notna().all()
    assert coverage.n_common_values.min() > 0
    assert len(data["family_contrasts"]) == 100
    assert data["family_contrasts"].query('family == "boundary"').all_nine_other_modes_comparable.eq(False).all()


def test_identity_and_labels_not_predictors(data):
    x = data["numeric"].copy()
    x.index = [f"anonymous_{i}" for i in range(30)]
    fresh = f.cluster(x, data["meta"])
    np.testing.assert_array_equal(fresh["tree"], data["variants"]["primary"]["tree"])
    assert not any("class_label" in c or "sample_id" in c or "case_id" in c for c in x)


def test_complete_case_ward_is_fixed_euclidean(data):
    result = data["variants"]["complete_values_ward"]
    assert result["z"].notna().all().all()
    assert result["z"].shape[0] == 30
    assert not any("interface_response" in c for c in result["z"].columns)


def test_statistic_redundancy_has_actual_pair_counts(data):
    r = data["redundancy"]
    assert len(r) == 160*159//2
    row = r.iloc[0]
    assert row.n_common_cases == data["numeric"][[row.feature_A, row.feature_B]].notna().all(axis=1).sum()
