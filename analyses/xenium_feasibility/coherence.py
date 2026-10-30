"""Check Xenium cohort coverage and agreement between figures and saved results."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analyses.meylan_xenium_preparation.cohort import SAMPLE_CELLS
from analyses.meylan_xenium_preparation.cohort_analysis import recurrence_table
from analyses.figures.scripts.xenium.build import enrichment_contrasts
from analyses.figures.scripts.xenium.layout import DISPLAY_PANELS
from analyses.figures.scripts.xenium.map_data import runs


def check_figure(figure, evidence):
    """Verify current panel membership and plotted values, without writing files."""
    figure = Path(figure)
    tables = []
    panel_count = 0
    for name in ['FigureS7', 'FigureS8']:
        current = figure if name == 'FigureS7' else figure.with_name(name)
        index = pd.read_csv(current / 'panel_index.csv')
        expected = set(DISPLAY_PANELS[name])
        if name == 'FigureS7':
            expected.update(['E1_GBM_Stemlike_cells_full', 'E2_GBM_Interferon_cells_full'])
        if set(index.panel) != expected or index.panel.duplicated().any():
            raise ValueError(f'Incorrect {name} panels')
        for extension in ['pdf', 'svg', 'png']:
            files = list((current / 'panels').glob(f'*.{extension}'))
            if {p.stem for p in files} != expected or any(p.stat().st_size == 0 for p in files):
                raise ValueError(f'Missing or obsolete {name} artwork: {extension}')
        tables.append(pd.read_csv(current / 'source_data/enrichment_contrasts.csv'))
        panel_count += len(index)
    actual = pd.concat(tables, ignore_index=True)
    expected = enrichment_contrasts(evidence)
    keys = ['study', 'sample', 'roi_id', 'gene', 'representation']
    pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),
                                  expected.sort_values(keys).reset_index(drop=True), check_exact=True)
    lookup = {r.sample: r for r in runs()}
    for (sample, representation), group in actual.groupby(['sample', 'representation']):
        run = lookup[sample]
        name = 'roi_enrichment' if representation == 'cells' else 'roi_consensus_enrichment'
        raw = pd.read_csv(Path(evidence) / 'statistics' / f'{name}.csv.gz')
        raw = raw.loc[raw['sample'].eq(sample)].set_index('roi_id')
        for row in group.itertuples():
            side = 1 if row.gene == run.feature_a else 2
            if row.enrichment != raw.loc[row.roi_id, f'g{side}_enrichment']:
                raise ValueError(f'Plotted enrichment differs from core: {sample}/{row.roi_id}')
    return dict(individual_figure_panels=panel_count, plotted_enrichment_matches_core=True)


def check(evidence, figure):
    evidence, figure = map(Path, (evidence, figure))
    samples = pd.read_csv(evidence/'sample_inventory.csv')
    expected = {**SAMPLE_CELLS, 'sample1_rep1': 87770}
    if samples['sample'].duplicated().any() or set(samples['sample']) != set(expected):
        raise ValueError('The evidence does not contain exactly the complete manuscript sample scope')
    for row in samples.itertuples(index=False):
        # Check package consistency, not membership in a core-commit whitelist.
        if row.analysed_cells != expected[row.sample]:
            raise ValueError(f'Unexpected cell count: {row.sample}')
    support = pd.read_csv(evidence/'roi_cell_counts.csv')
    if support.duplicated(['study','sample','roi_id']).any():
        raise ValueError('Duplicate sample/ROI in coverage inventory')
    for row in samples.itertuples(index=False):
        roi = support.loc[support['sample'].eq(row.sample)]
        if (len(roi) != row.n_rois or int(roi.n_cells.sum()) != row.analysed_cells
                or int(roi.tessellation_eligible_cells.sum()) != row.tessellation_eligible_cells):
            raise ValueError(f'Inconsistent complete-ROI coverage: {row.sample}')
    comparisons = pd.read_csv(evidence/'cohort/contrasts.csv')
    if len(comparisons) != 64 or comparisons.duplicated(['sample','signature','representation']).any():
        raise ValueError('Expected 16 samples × 2 signatures × 2 enrichment representations')
    if set(comparisons['sample']) != set(SAMPLE_CELLS):
        raise ValueError('Cohort comparisons have the wrong samples')
    recalculated = recurrence_table(comparisons).fillna('')
    saved = pd.read_csv(evidence/'cohort/recurrence.csv', keep_default_na=False)
    pd.testing.assert_frame_equal(saved, recalculated, check_dtype=False)
    interfaces = pd.read_csv(evidence/'cohort/ordered_interfaces.csv')
    keys = ['sample','anchor_roi_id','responder_roi_id','responder_feature_id','surface']
    if len(interfaces) != 256 or interfaces.duplicated(keys).any():
        raise ValueError('Incomplete cohort interface opportunity inventory')
    # The figure illustrates two samples, but carries the complete cohort inventory.
    figure_tables = []
    for current in [figure, figure.with_name('FigureS8')]:
        pd.testing.assert_frame_equal(pd.read_csv(current/'source_data/sample_inventory.csv'), samples)
        figure_tables.append(pd.read_csv(current/'source_data/enrichment_contrasts.csv'))
    plotted = pd.concat(figure_tables,ignore_index=True)
    if set(plotted['sample']) != {'P28_post','sample1_rep1'} or len(plotted) != 24:
        raise ValueError('Figures S7/S8 no longer describe their stated selected ROIs')
    for surface, name in [('cells','roi_enrichment'),('ensemble','roi_consensus_enrichment')]:
        complete = pd.read_csv(evidence/f'statistics/{name}.csv.gz')
        for row in plotted.loc[plotted.representation.eq(surface)].itertuples(index=False):
            match = complete.loc[complete['sample'].eq(row.sample) & complete.roi_id.eq(row.roi_id)]
            if len(match) != 1:
                raise ValueError('Figure enrichment cannot be identified in complete results')
            r = match.iloc[0]
            side = 1 if r.g1_id == row.gene else 2
            if r[f'g{side}_id'] != row.gene or not np.isclose(r[f'g{side}_enrichment'], row.enrichment, atol=1e-12, rtol=1e-12):
                raise ValueError('Plotted enrichment differs from deposited results')
    return dict(samples=len(samples), gbm_samples=len(SAMPLE_CELLS), gbm_patients=8,
                analysed_cells=int(samples.analysed_cells.sum()), rois=int(samples.n_rois.sum()),
                complete_cohort_contrasts=len(comparisons), interface_opportunities=len(interfaces),
                selected_figure_samples=['P28_post','sample1_rep1'],
                recurrence_reproduced=True, figure_values_match_complete_results=True)
