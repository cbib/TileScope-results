"""Identity checks retained when retiring the old scenario-browser reducer."""
import pandas as pd
import pytest
from analyses.gbm_spatial_repertoire import build_measurements as build


def test_scope_keeps_unavailable_rois_and_physical_program_order(tmp_path, monkeypatch):
    (tmp_path / 'sample/B_A').mkdir(parents=True)
    monkeypatch.setattr(build.prepare, 'RESULTS', tmp_path)
    class Inputs:
        def run(self, root, sample, pair):
            return root / sample / pair, {}
        def csv(self, path):
            return pd.DataFrame({'roi_id': ['1', '2', '7']})
    result, _ = build.prepare.reference_inventory(
        Inputs(), ['sample'], pd.DataFrame({'pair_id': ['A__B']}))
    assert result.roi.tolist() == ['1', '2', '7']
    assert result.pair.eq('B_A').all()


def test_equal_context_counts_do_not_excuse_swapped_roi_memberships():
    ledger = pd.DataFrame({'spot_index': [0, 1], 'roi_id': ['1', '2']})
    build.validate_context_membership(ledger.copy(), ledger)
    with pytest.raises(ValueError, match='membership differs'):
        build.validate_context_membership(ledger.assign(roi_id=['2', '1']), ledger)
