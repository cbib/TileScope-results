import json
from pathlib import Path

import pandas as pd
import pytest

from analyses.greenwald_full import run_all_pairs as runner


def complete(folder):
    folder.mkdir()
    for name in runner.REQUIRED_OUTPUTS:
        (folder / name).write_text("fixture\n")


def test_resume_requires_known_matching_identity(tmp_path):
    folder = tmp_path / "run"
    complete(folder)
    identity = dict(core_sha="abc", input_sha256="input", config_sha256="config")
    previous = dict(status="complete", **identity)
    assert runner.reusable(previous, identity, folder)
    assert not runner.reusable(None, identity, folder)
    assert not runner.reusable(dict(status="complete", core_sha="abc"), identity, folder)
    for field in identity:
        assert not runner.reusable(previous, dict(identity, **{field: "changed"}), folder)
    (folder / "roi_cross_grain_sensitivity.csv").unlink()
    assert not runner.reusable(previous, identity, folder)


def test_config_identity_ignores_output_path_but_not_settings(tmp_path):
    doc = {"technology": "visium", "tessellation": {"n": 10}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(doc))
    initial = runner.resolved_config_digest(path)
    doc["output_dir"] = "elsewhere"
    path.write_text(json.dumps(doc))
    assert initial == runner.resolved_config_digest(path)
    doc["tessellation"]["n"] = 3
    path.write_text(json.dumps(doc))
    assert initial != runner.resolved_config_digest(path)


def test_resume_preserves_original_identity():
    old = pd.DataFrame([dict(sample_id="S", pair="A_B", status="complete", core_sha="old", input_sha256="x")])
    result = runner._merge_status(old, [dict(sample_id="S", pair="A_B", status="already_complete", core_sha="new")])
    pd.testing.assert_frame_equal(old, result)


def test_physical_pair_orientation_and_preparation_recipe():
    from analyses.greenwald_full.pair_catalog import load_pair_catalog
    from itertools import combinations
    recipe = json.loads((runner._HERE.parent / "preparation.json").read_text())
    expected = {"_".join(p) for p in combinations(recipe["programs"], 2)}
    assert expected == set(load_pair_catalog().pair)
    assert len(expected) == 91
    assert recipe["nbin"] == 24 and recipe["ctrl"] == 50 and recipe["seed"] == 1


def test_duplicate_status_rejected(tmp_path):
    path = tmp_path / "status.csv"
    pd.DataFrame([dict(sample_id="S", pair="A_B")] * 2).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        runner._load_existing_status(path)


def test_preflight_refuses_unknown_existing_run_without_writes(tmp_path, monkeypatch):
    import sys
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    (inputs / 'scores').mkdir()
    (inputs / 'pairs.csv').write_text('sample,pair_id,pair\nS,A__B,A_B\n')
    pd.DataFrame(dict(x=[0], y=[0], roi_id=[1], A=[1], B=[2])).to_csv(inputs / 'scores/S.csv.gz', index=False)
    (inputs / 'tilescope_run_config.json').write_text('{}')
    outputs = tmp_path / 'outputs'; (outputs / 'S').mkdir(parents=True)
    complete(outputs / 'S' / 'A_B')
    monkeypatch.setattr(runner, '_current_core_sha', lambda: 'test')
    monkeypatch.setattr(runner, 'resolved_config_digest', lambda p: 'config')
    monkeypatch.setattr(sys, 'argv', ['run', '--input-dir', str(inputs), '--output-dir', str(outputs)])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not (outputs / 'run_accounting').exists()
    assert (outputs / 'S/A_B/spot_ledger.csv').read_text() == 'fixture\n'


def test_score_bank_pair_order_and_worker_output(monkeypatch, tmp_path):
    import tilescope.api
    inputs = tmp_path / 'download'; (inputs / 'scores').mkdir(parents=True)
    (inputs / 'pairs.csv').write_text('sample,pair_id,pair\nS,A__B,B_A\n')
    bank = pd.DataFrame(dict(x=[3, 2], y=[5, 7], roi_id=[4, 2], A=[-.1, .7], B=[.5, -.4]))
    path = inputs / 'scores/S.csv.gz'; bank.to_csv(path, index=False)
    before = path.read_bytes()
    config = inputs / 'tilescope_run_config.json'; config.write_text('{}')
    captured = {}
    def run(**kwargs):
        captured.update(kwargs)
        captured['frame'] = pd.read_csv(kwargs['input_csv'])
        folder = Path(kwargs['output_dir'])
        for name in runner.REQUIRED_OUTPUTS:
            (folder / name).write_text('fixture\n')
    monkeypatch.setattr(tilescope.api, 'run_tilescope_from_csv', run)
    sample, pair, score_path = runner.enumerate_tasks(inputs)[0]
    result = runner._run_one(sample, pair, score_path, force=False, config_path=config,
                             core_sha='core', config_sha='config', results_root=tmp_path / 'chosen-output')
    assert result['status'] == 'complete'
    assert (captured['feature_a_id'], captured['feature_b_id']) == ('B', 'A')
    assert captured['run_config'] == str(config)
    expected = bank[['x', 'y', 'B', 'A', 'roi_id']].set_axis(['x', 'y', 'gene1', 'gene2', 'roi_id'], axis=1)
    pd.testing.assert_frame_equal(captured['frame'], expected)
    assert captured['output_dir'] == str(tmp_path / 'chosen-output/S/B_A')
    assert not Path(captured['input_csv']).exists()
    assert path.read_bytes() == before


def test_score_inputs_reject_missing_values_and_nested_outputs(tmp_path):
    from tilescope_results.pair_inputs import read_scores, run_paths
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    (inputs / 'tilescope_run_config.json').write_text('{}')
    with pytest.raises(ValueError, match='separate'):
        run_paths(inputs, inputs / 'outputs')
    bank = pd.DataFrame(dict(x=[1], y=[2], roi_id=[1], A=[float('inf')], B=[3]))
    path = inputs / 'scores.csv'; bank.to_csv(path, index=False)
    with pytest.raises(ValueError, match='nonfinite'):
        read_scores(path, ['A', 'B'])
    with pytest.raises(ValueError, match='columns'):
        read_scores(path, ['missing'])


def test_moved_evidence_uses_configured_root(tmp_path, monkeypatch):
    from tilescope_results.paths import greenwald_evidence, resolve_reference
    monkeypatch.setenv('TILESCOPE_DATA_DIR', str(tmp_path))
    expected = tmp_path / 'heiland_tyrosh/analysis/scenario_cohort/a.csv'
    assert greenwald_evidence('scenario_cohort') / 'a.csv' == expected
    assert resolve_reference('analyses/greenwald_full_evaluation/scenario_cohort/a.csv') == expected
    with pytest.raises(ValueError):
        greenwald_evidence('../escape')


def test_csv_comparison_reports_unavailability_and_schema_changes(tmp_path):
    from analyses.greenwald_full.compare_csvs import compare
    before = tmp_path / 'before'; after = tmp_path / 'after'
    before.mkdir(); after.mkdir()
    pd.DataFrame({'value': [1.0]}).to_csv(before / 'values.csv', index=False)
    pd.DataFrame({'value': [float('nan')]}).to_csv(after / 'values.csv', index=False)
    pd.DataFrame({'value': [1, 2]}).to_csv(before / 'rows.csv', index=False)
    pd.DataFrame({'value': [1]}).to_csv(after / 'rows.csv', index=False)
    result = compare(before, after).set_index('file')
    assert result.loc['values.csv', 'status'] == 'values_changed'
    assert result.loc['rows.csv', 'status'] == 'shape_or_schema_changed'
