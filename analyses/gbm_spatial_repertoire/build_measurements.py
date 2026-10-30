"""Current 30-section GBM measurements, without historical scenario browsers.

Reads saved core exports; never executes TileScope. Frozen cohort/pair/annotation
inputs declare the manuscript scope; historical 19-section membership is merely
an annotation. Run as python -m analyses.gbm_spatial_repertoire.build_measurements.
Use --output-root for a non-destructive parity build.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EVALUATION = HERE.parent / 'greenwald_full_evaluation'
sys.path.insert(0, str(EVALUATION))
sys.path.insert(0, str(HERE.parent / 'greenwald_full'))
from figure_common import Inputs, EXPECTED_CORE
from local_frequency_analysis import digest, csv, frequencies, contextual_collections
from statistical_frequency_overview import aggregate
from pair_catalog import load_pair_catalog
import prepare_configuration_inputs as prepare
import local_relationships as lr
from tilescope_results.paths import data_root, greenwald_evidence

INPUTS = HERE / 'inputs'
RESULTS = data_root() / 'heiland_tyrosh/CurrentResults'
KEY = ['sample', 'pair_id', 'roi']
PAIRS = ('Vasc_OPC', 'Mac_Inflammatory-Mac', 'MES-Hyp_MES-Ast',
         'AC_Reactive-Ast', 'Neuron_Oligo', 'MES_NPC', 'Chromatin-Reg_Prolif-Metab')

def roi_key(sample, pair, roi):
    return f'{sample}|{pair}|roi:{roi}'


def boundary_key(sample, pair, anchor, responder):
    return f'{sample}|{pair}|interface:{anchor}>{responder}'


def radial_wide(radial):
    keys = ['sample','pair_id','roi_id','component_id']
    if radial.duplicated(keys+['shell_bin_index']).any():
        raise ValueError('Duplicate component shell')
    out = []
    for key, g in radial.groupby(keys, sort=True):
        if g.component_rho_0.nunique(dropna=False) != 1:
            raise ValueError('Inconsistent component zero-lag value')
        r = dict(zip(keys, key));r['rho'] = g.component_rho_0.iloc[0]
        for s in g.itertuples():
            for name, value in [('value',s.shell_symmetric),('anchors',s.n_anchors_per_shell),
                                ('fraction',s.fraction_anchors_per_shell),('status',s.component_radial_shell_status)]:
                r[f'shell{s.shell_bin_index}_{name}'] = value
        out.append(r)
    return pd.DataFrame(out).rename(columns={'roi_id':'roi'})


def attach_parent_marginals(roi, marginal):
    for carrier in ['A','B']:
        mm=marginal[['sample','roi','program','high_spots','high_fraction','status']].rename(columns={
            'program':'program_'+carrier,'high_spots':'parent_high_spots_S_'+carrier,
            'high_fraction':'parent_high_fraction_S_'+carrier,'status':'parent_high_field_status_S_'+carrier})
        roi=roi.merge(mm,on=['sample','roi','program_'+carrier],how='left',validate='many_to_one',indicator=True)
        if not roi['_merge'].eq('both').all():raise ValueError('Parent marginal evidence record missing')
        roi=roi.drop(columns='_merge')
    return roi


def attach_locations(roi, radial, boundary):
    roi = roi.copy()
    roi['target_id'] = [roi_key(*x) for x in roi[KEY].itertuples(index=False, name=None)]
    comp = radial_wide(radial)
    comp['parent_target_id'] = [roi_key(*x) for x in comp[KEY].itertuples(index=False, name=None)]
    boundary = boundary.copy()
    keys = ['sample', 'pair_id', 'anchor_roi_id', 'responder_roi_id']
    boundary['target_id'] = [boundary_key(*x) for x in boundary[keys].itertuples(index=False, name=None)]
    boundary['reverse_target_id'] = [boundary_key(s, p, r, a) for s, p, a, r in boundary[keys].itertuples(index=False, name=None)]
    for role in ['anchor', 'responder']:
        boundary[role+'_target_id'] = [roi_key(*x) for x in
            boundary[['sample', 'pair_id', role+'_roi_id']].itertuples(index=False, name=None)]
    return roi, comp, boundary


def validate_context_membership(spots, ledger):
    """A matched row count is not enough to establish the same ROI partition."""
    s = spots.sort_values('spot_index')
    l = ledger.sort_values('spot_index')
    if not np.array_equal(s.spot_index, l.spot_index) or not np.array_equal(s.roi_id, l.roi_id):
        raise ValueError('Context annotation source-row/ROI membership differs from current ledger')


def marginal_tables(samples, inputs):
    index = load_pair_catalog().set_index('pair')
    parts = []
    for sample in samples:
        for pair in PAIRS:
            frame = inputs.csv(RESULTS / sample / pair / 'roi_enrichment.csv')
            for channel in (1, 2):
                part = frame[['roi_id', 'n_roi', 'n_ctx', f'n_g{channel}_high',
                              f'g{channel}_enrichment', f'CLES_g{channel}',
                              'parent_enrichment_status']].copy()
                part.columns = ['roi', 'source_spots', 'context_spots', 'high_spots',
                                'enrichment', 'cles', 'status']
                part['program'] = index.loc[pair, f'program_g{channel}']
                part['sample'] = sample
                part['high_fraction'] = part.high_spots / part.source_spots
                part['source_pair'] = pair
                parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    assert not result.duplicated(['sample', 'roi', 'program']).any()
    return result



def build(output_root=None):
    roots = ({name: Path(output_root) / name for name in ['cohort', 'numerical', 'frequencies']}
             if output_root else dict(cohort=greenwald_evidence('scenario_cohort'),
                 numerical=greenwald_evidence('two_level_evidence'),
                 frequencies=greenwald_evidence('gbm_spatial_repertoire') / 'statistical_frequencies'))
    inputs = Inputs()
    scope = inputs.csv(INPUTS / 'sample_scope.csv')
    pairs = inputs.csv(INPUTS / 'pair_scope.csv')
    context = pd.read_csv(INPUTS / 'roi_context.csv', dtype={'roi': str})
    inputs.paths.add(INPUTS / 'roi_context.csv')
    samples = scope.loc[scope.included_this_build, 'sample'].tolist()
    if len(samples) != 30 or len(pairs) != 91 or len(context) != 192:
        raise ValueError('Expected the declared 30-section / 192-ROI / 91-pair scope')
    status = inputs.run_status()
    if not set(samples) <= set(status.sample_id):
        raise ValueError('Cohort section missing from current run inventory')
    current_pairs = set(load_pair_catalog().unordered_pair_key.str.replace('||', '__', regex=False))
    if set(pairs.pair_id) != current_pairs:
        raise ValueError('Frozen pair scope differs from the input program definitions')
    products = {name: {} for name in roots}
    def save(group, name, table):
        path = roots[group] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        csv(table, path)
        products[group][name] = digest(path)
    save('cohort', 'sample_scope.csv', scope)
    save('cohort', 'pair_scope.csv', pairs)
    save('cohort', 'roi_context.csv', context)
    collections = []
    for i, sample in enumerate(samples, 1):
        print(f'[{i}/30] {sample}: current ROI, component and boundary measurements', flush=True)
        roi, boundary, radial, _ = prepare.load_data(inputs, [sample], pairs)
        roi, components, boundary = attach_locations(roi, radial, boundary)
        roi = roi.merge(context[context['sample'].eq(sample)], on=['sample', 'roi'], validate='many_to_one', how='left')
        if not roi.source_spots.eq(roi.n_input).all():
            raise ValueError('Author annotation / current ROI counts differ')
        # Verify original spot-to-ROI annotation membership, not only counts.
        spot_path = greenwald_evidence('context_matching') / 'spot_context_membership.csv.gz'
        spots = pd.read_csv(spot_path, dtype={'roi_id': str})
        inputs.paths.add(spot_path)
        ledger = inputs.csv(RESULTS / sample / 'AC_OPC/spot_ledger.csv')
        validate_context_membership(spots[spots.sample_id.eq(sample)], ledger)
        roi = attach_parent_marginals(roi, marginal_tables([sample], inputs))
        prefix = f'by_sample/{sample}/'
        for name, table in [('roi_evidence.csv.gz', roi), ('component_evidence.csv.gz', components),
                            ('boundary_evidence.csv.gz', boundary)]:
            save('cohort', prefix + name, table)
        local = pd.concat([pd.concat([lr.roi_queries(roi, policy), lr.component_queries(components, policy)],
                          ignore_index=True) for policy in lr.POLICIES], ignore_index=True)
        collection = pd.concat([lr.pair_collections(local[local.policy.eq(policy.name)], pairs.pair_id.tolist())
                                for policy in lr.POLICIES], ignore_index=True)
        collection['context'] = 'query_specific'
        collections.extend([collection, contextual_collections(local)])
        save('numerical', f'details/{sample}_local_properties.csv.gz', local)
        save('numerical', f'details/{sample}_boundary_values.csv.gz', lr.boundary_records(boundary, roi))
    save('numerical', 'display_policies.csv', lr.policy_table())
    save('numerical', 'numerical_contract.csv', pd.DataFrame(lr.CONTRACT,
         columns=['property', 'group', 'label', 'unit', 'definition']))
    freq = frequencies(pd.concat(collections, ignore_index=True))
    save('frequencies', 'section_local_frequencies.csv.gz', freq)
    save('frequencies', 'summary.csv.gz', aggregate(freq))
    inputs.paths.update([Path(__file__), Path(prepare.__file__), Path(lr.__file__)])
    for name, root in roots.items():
        manifest = dict(core=EXPECTED_CORE, n_tessellations=10, no_core_runs=True,
                        samples=samples, pair_ids=pairs.pair_id.tolist(),
                        input_sha256={str(p.resolve()): digest(p) for p in sorted(inputs.paths)},
                        output_sha256=products[name])
        (root / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    from analyses.gbm_spatial_repertoire.review import prepare as prepare_shared
    prepare_shared(roots['cohort'], roots['numerical'],
                   Path(output_root) / 'shared' if output_root else greenwald_evidence('gbm_spatial_repertoire'))
    print('Complete: numerical evidence only; no scenario classification or new visual judgment.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path)
    build(parser.parse_args().output_root)
