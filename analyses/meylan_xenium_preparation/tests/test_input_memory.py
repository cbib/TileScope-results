"""Validation inputs must not remain referenced during the core run."""
import gc
import weakref
import pandas as pd

from analyses.meylan_xenium_preparation import run


def test_validate_reads_once_and_releases_tables(tmp_path, monkeypatch):
    calls, refs = [], []
    source = tmp_path/'input.csv'
    def read(path):
        calls.append(path.name)
        if path.name == 'signature_gene_values.csv.gz':
            frame = pd.DataFrame({'cell_id':['s_0'], 'A':[2.], 'B':[3.]})
        else:
            frame = pd.DataFrame({'cell_id':['s_0'], 'gene1':[2.], 'gene2':[3.]})
        refs.append(weakref.ref(frame))
        return frame
    monkeypatch.setattr(run.pd, 'read_csv', read)
    monkeypatch.setattr(run, 'validate', lambda *args: None)
    run.validate_inputs(source, {}, {'genes_a':['A'], 'genes_b':['B']}, 's')
    gc.collect()
    assert calls == ['input.csv', 'cell_annotations.csv.gz', 'signature_gene_values.csv.gz']
    assert all(ref() is None for ref in refs)


def test_signature_validation_is_not_skipped(tmp_path, monkeypatch):
    import pytest
    def read(path):
        if path.name == 'signature_gene_values.csv.gz':
            return pd.DataFrame({'cell_id':['s_0'], 'A':[9.], 'B':[3.]})
        return pd.DataFrame({'cell_id':['s_0'], 'gene1':[2.], 'gene2':[3.]})
    monkeypatch.setattr(run.pd, 'read_csv', read)
    monkeypatch.setattr(run, 'validate', lambda *args: None)
    with pytest.raises(AssertionError):
        run.validate_inputs(tmp_path/'input.csv', {}, {'genes_a':['A'], 'genes_b':['B']}, 's')
