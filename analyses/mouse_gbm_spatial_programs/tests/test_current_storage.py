"""Protect the current execution path, without running TileScope."""
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


def test_runner_uses_supplied_score_bank_config_and_output(monkeypatch, tmp_path):
    from analyses.mouse_gbm_spatial_programs import phase6_harmonized_run_tilescope as runner
    inputs = tmp_path / 'download'; (inputs / 'scores').mkdir(parents=True)
    (inputs / 'pairs.csv').write_text('sample_dir,signature_a,signature_b\nMCT1_CTL,inflammation,myeloid_phagocytic\n')
    (inputs / 'signatures_harmonized.yaml').write_text('signatures:\n  inflammation:\n    kind: biological_program\n  myeloid_phagocytic:\n    kind: biological_program\n')
    (inputs / 'tilescope_run_config.json').write_text('invalid default; an override is supplied')
    bank = pd.DataFrame({'x': [3, 2], 'y': [5, 7], 'roi_id': [5, 2],
                         'inflammation__score': [-.2, .8], 'myeloid_phagocytic__score': [.5, -.4],
                         'inflammation__raw_mean': [100, 200], 'myeloid_phagocytic__raw_mean': [300, 400]})
    path = inputs / 'scores/MCT1_CTL.csv.gz'; bank.to_csv(path, index=False)
    before = {p: p.read_bytes() for p in inputs.rglob('*') if p.is_file()}
    outputs = tmp_path / 'results-by-choice'
    captured = {}
    def run(**kwargs):
        captured.update(kwargs)
        captured['frame'] = pd.read_csv(kwargs['input_csv'])
        (Path(kwargs['output_dir']) / 'run_metadata.json').write_text('{}')
        return {}
    config = tmp_path / 'override.json'; config.write_text(json.dumps({'technology': 'visium'}))
    monkeypatch.setattr(runner, "run_tilescope_from_csv", run)
    monkeypatch.setattr(runner, '_core_provenance', lambda: dict(core_git_sha='core', core_worktree_diff_sha256='clean'))
    monkeypatch.setenv('TILESCOPE_DATA_MOUSE_ROOT', str(tmp_path / 'must-not-be-read'))
    argv = ['run', '--all', '--input-dir', str(inputs), '--output-dir', str(outputs), '--config', str(config)]
    monkeypatch.setattr(sys, 'argv', [*argv, '--dry-run'])
    runner.main()
    assert not outputs.exists() and not captured
    monkeypatch.setattr(sys, 'argv', argv)
    runner.main()
    status = pd.read_csv(outputs / 'run_accounting/run_status.csv')
    assert status.status.tolist() == ['ok']
    assert captured["run_config"] == str(config)
    assert "profile" not in captured
    assert captured["feature_a_id"] == "inflammation"
    assert captured["feature_b_id"] == "myeloid_phagocytic"
    assert captured['output_dir'] == str(outputs / 'MCT1_CTL/inflammation__myeloid_phagocytic')
    expected = bank[['x', 'y', 'inflammation__score', 'myeloid_phagocytic__score', 'roi_id']].set_axis(
        ['x', 'y', 'gene1', 'gene2', 'roi_id'], axis=1)
    pd.testing.assert_frame_equal(captured['frame'], expected)
    assert not Path(captured['input_csv']).exists()
    assert {p: p.read_bytes() for p in inputs.rglob('*') if p.is_file()} == before
    monkeypatch.setattr(sys, 'argv', [a if a != '--all' else '--remainder' for a in argv])
    captured.clear()
    runner.main()
    assert not captured
    config.write_text(json.dumps({'technology': 'visium', 'seed': 11}))
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def test_evidence_does_not_silently_skip_missing_manifest_pair(monkeypatch, tmp_path):
    from analyses.mouse_gbm_spatial_programs import build_evidence
    root = tmp_path / "mouse"
    phase = root / "mouse_gbm_spatial_programs" / "phase6_harmonized"
    phase.mkdir(parents=True)
    pd.DataFrame([dict(sample_dir="MCT1_CTL", signature_a="inflammation",
                       signature_b="myeloid_phagocytic")]).to_csv(phase / "phase6_harmonized_pair_manifest.csv", index=False)
    pd.DataFrame(dict(roi_id=[2])).to_csv(phase / "MCT1_CTL_signature_score_bank.csv", index=False)
    monkeypatch.setenv("TILESCOPE_DATA_MOUSE_ROOT", str(root))
    with pytest.raises(FileNotFoundError, match="roi_correlations.csv"):
        build_evidence.main(tmp_path / "evidence")
