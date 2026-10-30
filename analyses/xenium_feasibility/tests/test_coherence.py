import pandas as pd
import pytest

from analyses.meylan_xenium_preparation.cohort import SAMPLE_CELLS
from analyses.meylan_xenium_preparation.cohort_analysis import (
    LA, INTERFACE, contrast_rows, recurrence_table, interface_rows,
)
from analyses.xenium_feasibility.coherence import check


@pytest.fixture
def evidence_and_figures(tmp_path):
    """Small tables check complete sample scope and plotted numerical values."""
    evidence, figure = [tmp_path/s for s in ['evidence','figure']]
    for p in [evidence/'cohort', figure/'source_data', figure.with_name('FigureS8')/'source_data', evidence/'statistics']:
        p.mkdir(parents=True)
    counts = {**SAMPLE_CELLS, 'sample1_rep1': 87770}
    samples, support, comparisons, contacts = [], [], [], []
    for sample, n in counts.items():
        study = 'Breast' if sample == 'sample1_rep1' else 'GBM'
        rois = [LA, INTERFACE, 'CN-Tumor'] if study == 'GBM' else ['DCIS_1', 'DCIS_2', 'Invasive']
        samples.append(dict(sample=sample, study=study, analysed_cells=n, n_rois=3,
                            tessellation_eligible_cells=n, core_commit='example-revision'))
        native = []
        for i, roi in enumerate(rois):
            cells = n//3 if i < 2 else n-2*(n//3)
            support.append(dict(study=study, sample=sample, roi_id=roi, n_cells=cells, tessellation_eligible_cells=cells))
            row = dict(roi_id=roi, cells=cells, tessellation_eligible=cells, tessellation_eligible_fraction=1.)
            for gene in ['Stemlike', 'Interferon']:
                for surface in ['cells','ensemble']:
                    row[f'{gene}_enrichment_{surface}'] = 1.5
                    row[f'{gene}_enrichment_{surface}_status'] = 'finite'
            native.append(row)
        if study == 'GBM':
            frame = pd.DataFrame(native)
            comparisons.extend(contrast_rows(frame, sample))
            contacts.extend(interface_rows(pd.DataFrame(), frame, sample))
    samples = pd.DataFrame(samples)
    samples.to_csv(evidence/'sample_inventory.csv', index=False)
    samples.to_csv(figure/'source_data/sample_inventory.csv', index=False)
    samples.to_csv(figure.with_name('FigureS8')/'source_data/sample_inventory.csv', index=False)
    pd.DataFrame(support).to_csv(evidence/'roi_cell_counts.csv', index=False)
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(evidence/'cohort/contrasts.csv', index=False)
    recurrence_table(comparisons).to_csv(evidence/'cohort/recurrence.csv', index=False)
    pd.DataFrame(contacts).to_csv(evidence/'cohort/ordered_interfaces.csv', index=False)
    plotted = []
    core = []
    for sample, genes, rois in [
        ('P28_post', ['Stemlike','Interferon'], [LA, INTERFACE, 'CN-Tumor']),
        ('sample1_rep1', ['ACTA2','FASN'], ['DCIS_1','DCIS_2','Invasive']),
    ]:
        for roi in rois:
            core.append(dict(sample=sample,roi_id=roi,g1_id=genes[0],g2_id=genes[1],g1_enrichment=1.5,g2_enrichment=1.5))
            for gene in genes:
                for surface in ['cells','ensemble']:
                    plotted.append(dict(sample=sample,roi_id=roi,gene=gene,representation=surface,enrichment=1.5))
    plotted = pd.DataFrame(plotted)
    plotted.loc[plotted['sample'].eq('P28_post')].to_csv(figure/'source_data/enrichment_contrasts.csv', index=False)
    plotted.loc[plotted['sample'].eq('sample1_rep1')].to_csv(figure.with_name('FigureS8')/'source_data/enrichment_contrasts.csv', index=False)
    for f in ['roi_enrichment','roi_consensus_enrichment']:
        pd.DataFrame(core).to_csv(evidence/f'statistics/{f}.csv.gz', index=False)
    return evidence, figure


def test_complete_scope_and_numerical_agreement(evidence_and_figures):
    result = check(*evidence_and_figures)
    assert result['samples'] == 17
    assert result['analysed_cells'] == 2489716
    assert result['complete_cohort_contrasts'] == 64
    assert result['interface_opportunities'] == 256


def test_figure_drift_prevents_publication(evidence_and_figures):
    evidence, figure = evidence_and_figures
    file = figure/'source_data/enrichment_contrasts.csv'
    data = pd.read_csv(file); data.loc[0,'enrichment'] = 9.
    data.to_csv(file, index=False)
    with pytest.raises(ValueError, match='Plotted enrichment'):
        check(evidence, figure)


def test_missing_sample_fails_cohort_check(evidence_and_figures):
    evidence, figure = evidence_and_figures
    file = evidence/'sample_inventory.csv'
    pd.read_csv(file).iloc[:-1].to_csv(file, index=False)
    with pytest.raises(ValueError, match='complete manuscript sample scope'):
        check(evidence, figure)
