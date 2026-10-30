import json
import pytest
from analyses.meylan_xenium_preparation import performance_check as p


def test_reference_comparison_checks_complete_file_set(tmp_path):
    after, reference = tmp_path/'after', tmp_path/'reference'
    after.mkdir(); reference.mkdir()
    (reference/'a.csv').write_bytes(b'a\n1\n')
    (after/'a.csv').write_bytes(b'a\n1\n')
    record = dict(sample='test', peak_rss_bytes=10, elapsed_seconds=1)
    (after/'performance.json').write_text(json.dumps(record))
    assert p.compare_reference(after, reference)['identical_csvs'] == 1
    (reference/'extra.csv').write_bytes(b'a\n2\n')
    with pytest.raises(ValueError, match='differ'):
        p.compare_reference(after, reference)


def test_compare_rejects_different_dataset_seed_identity(tmp_path):
    before, after = tmp_path/'before', tmp_path/'after'
    before.mkdir(); after.mkdir()
    common = dict(sample='test', input_cells=10, table_rows={})
    (before/'performance.json').write_text(json.dumps(dict(common, dataset_id=0)))
    (after/'performance.json').write_text(json.dumps(dict(common, dataset_id=1)))
    with pytest.raises(ValueError, match='identity'):
        p.compare(before, after)


def test_comparison_reads_bytes_and_detects_equal_length_changes(tmp_path):
    before, after = tmp_path/'before', tmp_path/'after'
    before.mkdir(); after.mkdir()
    for path in [before/'a.csv', after/'a.csv']:
        path.write_bytes(b'a\n1\n')
    assert p.compare_csv_bytes(before, after) == (1, [])
    (after/'a.csv').write_bytes(b'a\n2\n')
    assert p.compare_csv_bytes(before, after) == (0, ['a.csv'])


def test_empty_results_cannot_pass_byte_comparison(tmp_path):
    with pytest.raises(ValueError, match='nonempty'):
        p.compare_csv_bytes(tmp_path, tmp_path)
