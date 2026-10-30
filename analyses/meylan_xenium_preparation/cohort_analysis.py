"""Describe the fixed lymphoid aggregate/interface contrast in all sixteen samples.

This module reads completed core results. It neither reruns TileScope nor assigns
new neighbourhoods. Counts are descriptive and use question-specific denominators.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tilescope_results.paths import data_root
from .cohort import PAIR, SAMPLE_CELLS

LA = 'CN-Lymphoid aggregate'
INTERFACE = 'CN-Tumor - T cell interface'
QUESTIONS = {
    'Stemlike': (LA, INTERFACE),
    'Interferon': (INTERFACE, LA),
}


def enrichment_value(row, gene, surface):
    """Use the core's per-gene status; unavailable is never zero/depletion."""
    if row is None:
        return np.nan, 'absent_roi'
    key = f'{gene}_enrichment_{surface}'
    status = row.get(key + '_status')
    if pd.isna(status):
        return np.nan, 'unavailable'
    if status == 'positive_over_zero_context':
        return np.inf, status
    value = row.get(key, np.nan)
    if status == 'finite' and np.isfinite(value):
        return float(value), status
    return np.nan, str(status)


def contrast_rows(rois, sample):
    """A row is one specified comparison, not an ROI or patient replicate."""
    if rois.roi_id.duplicated().any():
        raise ValueError(f'Duplicate ROI in {sample}')
    lookup = rois.set_index('roi_id').to_dict('index')
    records = []
    for surface in ('cells', 'ensemble'):
        for gene, (higher_roi, lower_roi) in QUESTIONS.items():
            higher, lower = lookup.get(higher_roi), lookup.get(lower_roi)
            high, hs = enrichment_value(higher, gene, surface)
            low, ls = enrichment_value(lower, gene, surface)
            present = higher is not None and lower is not None
            evaluable = present and not np.isnan(high) and not np.isnan(low)
            direction = ('absent_roi' if not present else 'unavailable' if not evaluable
                         else 'expected_direction' if high > low else 'opposite_direction'
                         if high < low else 'equal')
            record = dict(sample=sample, patient=sample.split('_')[0], timepoint=sample.split('_')[1],
                          signature=gene, representation=surface,
                          higher_expected_roi=higher_roi, lower_expected_roi=lower_roi,
                          higher_roi_present=higher is not None, lower_roi_present=lower is not None,
                          both_rois_present=present, evaluable=evaluable, direction=direction,
                          higher_roi_enrichment=high, lower_roi_enrichment=low,
                          higher_roi_status=hs, lower_roi_status=ls)
            for label, row in [('higher', higher), ('lower', lower)]:
                for field in ['cells', 'tessellation_eligible', 'tessellation_eligible_fraction']:
                    record[f'{label}_roi_{field}'] = row.get(field, np.nan) if row else np.nan
            records.append(record)
    return records


def recurrence_table(contrasts):
    records = []
    for (timepoint, surface), group in contrasts.groupby(['timepoint', 'representation'], sort=False):
        for question in ['Stemlike', 'Interferon', 'both_contrasts']:
            rows = group if question == 'both_contrasts' else group.loc[group.signature.eq(question)]
            states = {}
            for sample, d in rows.groupby('sample'):
                directions = d.direction.tolist()
                if 'absent_roi' in directions:
                    state = 'absent_roi'
                elif 'unavailable' in directions:
                    state = 'unavailable'
                elif all(x == 'expected_direction' for x in directions):
                    state = 'expected_direction'
                else:
                    state = 'other_evaluable'
                states[sample] = state
            expected = [s for s, state in states.items() if state == 'expected_direction']
            other = [s for s, state in states.items() if state == 'other_evaluable']
            absent = [s for s, state in states.items() if state == 'absent_roi']
            unavailable = [s for s, state in states.items() if state == 'unavailable']
            records.append(dict(timepoint=timepoint, representation=surface, contrast=question,
                                n_samples=len(states), n_both_rois_present=len(states)-len(absent),
                                n_evaluable=len(expected)+len(other), n_expected_direction=len(expected),
                                n_other_evaluable=len(other), n_absent_roi=len(absent),
                                n_unavailable=len(unavailable), expected_samples=';'.join(expected),
                                other_evaluable_samples=';'.join(other), absent_roi_samples=';'.join(absent),
                                unavailable_samples=';'.join(unavailable)))
    return pd.DataFrame(records)


def interface_rows(table, rois, sample):
    """Retain direction, responding gene and the actual near/far support."""
    present = {LA, INTERFACE}.issubset(set(rois.roi_id))
    records = []
    for anchor, responder in [(LA, INTERFACE), (INTERFACE, LA)]:
        for gene in QUESTIONS:
            for surface in ['source_native', 'tessellation', 'consensus_direct', 'consensus_ensemble']:
                row = dict(sample=sample, patient=sample.split('_')[0], timepoint=sample.split('_')[1],
                           anchor_roi_id=anchor, responder_roi_id=responder,
                           responder_feature_id=gene, surface=surface)
                if not present:
                    row.update(availability_status='absent_roi', interface_response=np.nan)
                else:
                    selected = table.loc[table.anchor_roi_id.eq(anchor) & table.responder_roi_id.eq(responder)
                                         & table.responder_feature_id.eq(gene) & table.surface.eq(surface)] if not table.empty else table
                    if len(selected) > 1:
                        raise ValueError(f'Duplicate ordered interface in {sample}')
                    if len(selected):
                        row.update(selected.iloc[0].drop(labels=['sample_id'], errors='ignore').to_dict())
                    else:
                        row.update(availability_status='no_contact_result', interface_response=np.nan)
                records.append(row)
    return records


def read_validated_rois(root, sample):
    """Read one completed, audited sample without writing cohort outputs."""
    root = Path(root)
    run = root / f'CurrentResults/{sample}/{PAIR}'
    audit = root / f'analysis/{sample}_{PAIR}'
    status = json.loads((run / 'runner_status.json').read_text())
    validation = json.loads((audit / 'validation.json').read_text())
    if status['status'] != 'complete':
        raise ValueError(f'Incomplete run: {sample}')
    if (validation['input_cells'] != SAMPLE_CELLS[sample]
            or not validation['source_enrichment_and_pearson_reproduced']
            or not validation['all_eligible_cells_represented_in_each_of_ten_tessellations']):
        raise ValueError(f'Missing validation: {sample}')
    rois = pd.read_csv(audit / 'roi_summary.csv')
    if int(rois.cells.sum()) != SAMPLE_CELLS[sample] or set(rois['sample']) != {sample}:
        raise ValueError(f'ROI inventory does not cover the complete sample: {sample}')
    if int(rois.tessellation_eligible.sum()) != validation['tessellation_eligible']:
        raise ValueError(f'Coverage audit disagrees with ROI counts: {sample}')
    raw = pd.read_csv(run / 'roi_enrichment.csv').set_index('roi_id')
    consensus = pd.read_csv(run / 'roi_consensus_enrichment.csv').set_index('roi_id')
    for num, gene in enumerate(QUESTIONS, 1):
        rois[f'{gene}_mean_cells'] = rois.roi_id.map(raw[f'sum_g{num}'] / raw.n_spots)
        # Restore precise statuses, including for the historical P28 audit.
        for surface, saved in [('cells', raw), ('ensemble', consensus)]:
            key = f'{gene}_enrichment_{surface}'
            rois[key] = rois.roi_id.map(saved[f'g{num}_enrichment'])
            statuses = saved[f'g{num}_enrichment_status'].copy()
            fallback = ('parent_enrichment_status' if surface == 'cells'
                        else 'parent_consensus_enrichment_status')
            if fallback in saved:
                statuses = statuses.fillna(saved[fallback])
            rois[key+'_status'] = rois.roi_id.map(statuses).fillna(
                'no_consensus_result' if surface == 'ensemble' else 'unavailable')
    rois['patient'], rois['timepoint'] = sample.split('_')
    return rois, status


def build(root):
    root = Path(root)
    roi_parts, comparisons, contacts, inventory = [], [], [], []
    for sample, expected_cells in SAMPLE_CELLS.items():
        run = root / f'CurrentResults/{sample}/{PAIR}'
        rois, status = read_validated_rois(root, sample)
        roi_parts.append(rois)
        comparisons.extend(contrast_rows(rois, sample))
        path = run / 'roi_pair_interface_response.csv'
        if not path.exists() and status.get('core_result_table_rows', {}).get('roi_pair_interface_response_df') != 0:
            raise ValueError(f'Undocumented missing interface table: {sample}')
        table = pd.read_csv(path) if path.exists() else pd.DataFrame()
        contacts.extend(interface_rows(table, rois, sample))
        inventory.append(dict(sample=sample, patient=sample.split('_')[0], timepoint=sample.split('_')[1],
                              cells=expected_cells, rois=len(rois),
                              tessellation_eligible=int(rois.tessellation_eligible.sum()),
                              eligible_fraction=rois.tessellation_eligible.sum()/expected_cells,
                              lymphoid_present=LA in set(rois.roi_id),
                              interface_present=INTERFACE in set(rois.roi_id)))
    contrasts = pd.DataFrame(comparisons)
    recurrence = recurrence_table(contrasts)
    samples = pd.DataFrame(inventory)
    out = root / 'analysis/cohort'; out.mkdir(parents=True, exist_ok=True)
    for name, table in [('cohort_roi_summary', pd.concat(roi_parts, ignore_index=True)),
                        ('contrasts', contrasts), ('recurrence', recurrence),
                        ('ordered_interfaces', pd.DataFrame(contacts)), ('sample_summary', samples)]:
        table.to_csv(out / f'{name}.csv', index=False)
    lines = ['# Complete Meylan cohort: regional contrasts', '',
             f'16 samples from 8 patients; {sum(SAMPLE_CELLS.values()):,} cells; {samples.rois.sum()} ROIs.',
             'Pre- and post-treatment samples are summarized separately, not treated as independent patients.', '',
             'The fixed questions are higher stem-like enrichment in lymphoid aggregates than interfaces, '
             'and higher interferon enrichment at interfaces than lymphoid aggregates. These are strict '
             'descriptive comparisons of complete parent-ROI enrichment ratios, not significance tests '
             'or estimates of treatment effects. Scores include all cells, unlike the paper’s T-cell-only comparison.', '',
             '| Time | Representation | Contrast | Expected / evaluable | Absent ROI | Unavailable |',
             '|:--|:--|:--|--:|--:|--:|']
    for r in recurrence.itertuples(index=False):
        lines.append(f'| {r.timepoint} | {r.representation} | {r.contrast} | '
                     f'{r.n_expected_direction}/{r.n_evaluable} | {r.n_absent_roi} | {r.n_unavailable} |')
    lines.extend(['', '## Coverage and interpretation', '',
                  f'Tessellation represents {samples.eligible_fraction.min():.1%}–{samples.eligible_fraction.max():.1%} '
                  'of complete-sample cells. See cohort_roi_summary.csv for the comparison regions separately.',
                  'contrasts.csv retains exact sample identities, both enrichment values, cell counts, '
                  'coverage and statuses. Equal ratios, opposite directions, absent ROIs and unavailable '
                  'statistics are not combined. recurrence.csv includes sample membership for each result.',
                  'ordered_interfaces.csv preserves both contact directions and signatures with core near/far '
                  'support and availability. A missing ROI or contact is not a zero boundary response.',
                  'Figure S7 remains a selected P28_post illustration, not a cohort-frequency display. '
                  'Component-level appearance must not be substituted for the complete parent-ROI contrasts.', ''])
    (out / 'FINDINGS.md').write_text('\n'.join(lines))
    print(recurrence.to_string(index=False))
    return contrasts, recurrence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=data_root()/'meylan_xenium')
    args = parser.parse_args()
    build(args.study_dir)


if __name__ == '__main__':
    main()
