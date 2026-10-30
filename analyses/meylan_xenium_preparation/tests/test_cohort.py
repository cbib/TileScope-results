import numpy as np
import pandas as pd
import pytest

from analyses.meylan_xenium_preparation.cohort import SAMPLE_CELLS, same_definition
from analyses.meylan_xenium_preparation.cohort_analysis import (
    LA, INTERFACE, contrast_rows, enrichment_value, interface_rows, recurrence_table,
)
from analyses.meylan_xenium_preparation.run import validate


def roi_table():
    rows = []
    for roi, stem, interferon in [(LA, 3., 1.), (INTERFACE, 2., 2.)]:
        row = dict(roi_id=roi, cells=100, tessellation_eligible=80, tessellation_eligible_fraction=.8)
        for surface in ['cells', 'ensemble']:
            for gene, value in [('Stemlike', stem), ('Interferon', interferon)]:
                row[f'{gene}_enrichment_{surface}'] = value
                row[f'{gene}_enrichment_{surface}_status'] = 'finite'
        rows.append(row)
    return pd.DataFrame(rows)


def test_scope_is_sixteen_samples_from_eight_paired_patients():
    assert sum(SAMPLE_CELLS.values()) == 2401946
    patients = {s.split('_')[0] for s in SAMPLE_CELLS}
    assert len(patients) == 8
    assert set(SAMPLE_CELLS) == {f'{p}_{t}' for p in patients for t in ['pre', 'post']}


def test_definition_reuse_ignores_pilot_metadata_not_gene_choices():
    a = dict(pair='A_B', feature_a='A', feature_b='B', genes_a=['a'], genes_b=['b'], calculation='mean')
    assert same_definition(a, dict(a, sample='P28_post'))
    assert not same_definition(a, dict(a, genes_a=['another']))


def test_absent_region_is_not_a_failed_contrast_and_timepoints_stay_separate():
    full = roi_table()
    rows = contrast_rows(full, 'P28_post') + contrast_rows(full.iloc[:1], 'P28_pre')
    summary = recurrence_table(pd.DataFrame(rows)).set_index(['timepoint', 'representation', 'contrast'])
    post = summary.loc[('post', 'cells', 'both_contrasts')]
    pre = summary.loc[('pre', 'cells', 'both_contrasts')]
    assert (post.n_expected_direction, post.n_evaluable) == (1, 1)
    assert (pre.n_expected_direction, pre.n_evaluable, pre.n_absent_roi) == (0, 0, 1)
    assert post.expected_samples == 'P28_post'


def test_consensus_unavailability_does_not_erase_cell_contrast():
    frame = roi_table()
    frame.loc[0, 'Stemlike_enrichment_ensemble'] = np.nan
    frame.loc[0, 'Stemlike_enrichment_ensemble_status'] = np.nan
    result = pd.DataFrame(contrast_rows(frame, 'P28_post')).set_index(['representation', 'signature'])
    assert result.loc[('ensemble', 'Stemlike'), 'direction'] == 'unavailable'
    assert result.loc[('cells', 'Stemlike'), 'direction'] == 'expected_direction'


def test_opposite_and_tied_results_are_evaluable_not_unavailable():
    frame = roi_table()
    frame.loc[0, 'Stemlike_enrichment_cells'] = 1.
    frame.loc[0, 'Interferon_enrichment_cells'] = 2.
    rows = pd.DataFrame(contrast_rows(frame, 'P28_post')).set_index(['representation', 'signature'])
    assert rows.loc[('cells', 'Stemlike'), 'direction'] == 'opposite_direction'
    assert rows.loc[('cells', 'Interferon'), 'direction'] == 'equal'
    assert rows.evaluable.all()


def test_explicit_saturation_is_valid_but_missing_or_zero_over_zero_is_not():
    key = 'Stemlike_enrichment_cells'
    assert enrichment_value({key: np.nan, key+'_status': 'positive_over_zero_context'}, 'Stemlike', 'cells')[0] == np.inf
    assert enrichment_value({key: 0., key+'_status': 'finite'}, 'Stemlike', 'cells')[0] == 0.
    assert np.isnan(enrichment_value({key: 0., key+'_status': 'zero_over_zero_context'}, 'Stemlike', 'cells')[0])


def test_interface_direction_and_response_are_not_detached_from_endpoints():
    table = pd.DataFrame([dict(anchor_roi_id=LA, responder_roi_id=INTERFACE,
                               responder_feature_id='Interferon', surface='source_native',
                               interface_response=.4, availability_status='finite')])
    rows = pd.DataFrame(interface_rows(table, roi_table(), 'P28_post'))
    found = rows.loc[rows.availability_status.eq('finite')].iloc[0]
    assert (found.anchor_roi_id, found.responder_roi_id, found.responder_feature_id) == (LA, INTERFACE, 'Interferon')
    assert found.interface_response == .4
    assert rows.loc[rows.anchor_roi_id.eq(INTERFACE)].interface_response.isna().all()


def test_single_roi_sample_is_retained_for_coverage_without_inventing_a_context():
    sample = 'P51_pre'; n = SAMPLE_CELLS[sample]
    frame = pd.DataFrame(dict(cell_id=[f'{sample}_{i}' for i in range(n)], x=np.arange(n), y=0.,
                              gene1=1., gene2=1., roi_id='CN-Tumor'))
    validate(frame, frame.copy(), {'accepted': True, 'rounded_values_matching': 360}, sample)


def test_cohort_reader_handles_old_audit_and_empty_interface_table(tmp_path, monkeypatch):
    """Integration check on copies: never write to the installed scientific data."""
    import shutil
    from tilescope_results.paths import data_root
    from analyses.meylan_xenium_preparation import cohort_analysis as module
    selected = {s: SAMPLE_CELLS[s] for s in ['P51_pre', 'P05_post', 'P28_post']}
    original = data_root() / 'meylan_xenium'
    if any(not (original/f'analysis/{s}_Stemlike_Interferon/validation.json').exists() for s in selected):
        pytest.skip('Completed Meylan inputs not installed')
    for sample in selected:
        for folder, names in [
            (f'CurrentResults/{sample}/Stemlike_Interferon',
             ['runner_status.json', 'roi_enrichment.csv', 'roi_consensus_enrichment.csv', 'roi_pair_interface_response.csv']),
            (f'analysis/{sample}_Stemlike_Interferon', ['validation.json', 'roi_summary.csv']),
        ]:
            dest = tmp_path/folder; dest.mkdir(parents=True)
            for name in names:
                if (original/folder/name).exists():
                    shutil.copy2(original/folder/name, dest/name)
    monkeypatch.setattr(module, 'SAMPLE_CELLS', selected)
    comparisons, recurrence = module.build(tmp_path)
    p28 = comparisons.loc[comparisons['sample'].eq('P28_post')]
    assert p28.direction.eq('expected_direction').all()
    p51 = comparisons.loc[comparisons['sample'].eq('P51_pre')]
    assert p51.direction.eq('absent_roi').all()
    assert len(pd.read_csv(tmp_path/'analysis/cohort/ordered_interfaces.csv')) == 48
    row = recurrence.set_index(['timepoint','representation','contrast']).loc[('post','cells','both_contrasts')]
    assert (row.n_expected_direction, row.n_evaluable) == (1, 2)
