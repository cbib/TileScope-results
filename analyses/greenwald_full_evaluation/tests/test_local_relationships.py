"""Numerical local evidence is not selective visual-review coverage."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_relationships as lr


def roi_row(label='1', pair='A__B', **updates):
    r = dict(sample='s', pair_id=pair, roi=label, target_id=f's|{pair}|roi:{label}', source_spots=100)
    for name in ['enrichment_S_A', 'enrichment_S_B', 'enrichment_S_AB']:
        r[name], r[name + '__kind'] = 2., 'finite'
    for s in ['S', 'T', 'CD', 'CE']:
        for name in [f'correlation_{s}', f'moran_{s}_A|B', f'moran_{s}_B|A']:
            r[name], r[name + '__kind'] = .4, 'finite'
    for s in ['S', 'T', 'CE']:
        r['joint_low_' + s], r['joint_low_' + s + '__kind'] = .5, 'finite'
        for carrier in ['A', 'B']:
            for prefix, value in [('CLES', .7), ('log2FC', 1.)]:
                name = f'{prefix}_{s}_{carrier}'
                r[name], r[name + '__kind'] = value, 'finite'
    r['correlation_T__low'], r['correlation_T__high'] = .3, .5
    r.update(updates)
    return r


def states(frame):
    return lr.roi_queries(frame, lr.POLICIES[1]).set_index('property').state


def test_source_only_and_non_coenriched_do_not_lose_native_field_evidence():
    frame = pd.DataFrame([roi_row(enrichment_S_A=.5, correlation_CE=np.nan,
                                  correlation_CE__kind='unavailable')])
    s = states(frame)
    assert s.source_positive == 'meets'
    assert s.coenriched == 'other'
    assert s.coenriched_positive == 'outside_context'
    assert s.source_ensemble_positive == 'unavailable'


def test_statistic_combinations_cannot_be_assembled_across_rois():
    frame = pd.DataFrame([roi_row('1', correlation_S=.01),
                          roi_row('2', **{'moran_S_A|B': .01})])
    q = lr.roi_queries(frame, lr.POLICIES[1])
    assert q[q['property'].eq('source_positive')].state.eq('other').all()
    assert q[q['property'].eq('pearson_S_positive')].state.tolist() == ['other', 'meets']
    assert q[q['property'].eq('moran_S_A_B_positive')].state.tolist() == ['meets', 'other']


def test_mean_distribution_disagreement_and_joint_low_are_not_gated_by_pearson():
    frame = pd.DataFrame([roi_row(correlation_S=.01, log2FC_S_A=-1., joint_low_S=2.)])
    s = states(frame)
    assert s.source_positive == 'other'
    assert s.amplitude_disagreement_S_A == 'meets'
    assert s.joint_low_S_high == 'meets'


def test_territory_preserves_joint_deficit_saturation_and_unknown_context():
    s = states(pd.DataFrame([roi_row(enrichment_S_AB=.4)]))
    assert s.coenriched == 'meets'
    assert s.coenriched_joint_not_high == 'meets'
    assert s.coenriched_joint_high == 'other'
    s = states(pd.DataFrame([roi_row(enrichment_S_A=np.nan, enrichment_S_A__kind='saturated')]))
    assert s.coenriched == 'meets'
    s = states(pd.DataFrame([roi_row(enrichment_S_A=np.nan, enrichment_S_A__kind='unavailable')]))
    assert s.coenriched == 'unavailable' and s.coenriched_positive == 'unavailable'
    s = states(pd.DataFrame([roi_row(enrichment_S_B=.5, enrichment_S_A=np.nan, enrichment_S_A__kind='unavailable')]))
    assert s.coenriched == 'other' and s.coenriched_positive == 'outside_context'


def test_all_supports_quartile_is_not_ignored():
    s = states(pd.DataFrame([roi_row(correlation_T__low=.01)]))
    assert s.source_ensemble_positive == 'meets'
    assert s.correlation_all_positive == 'other'




def test_opposite_rois_coexist_without_becoming_a_dominant_label():
    a, b = roi_row('1'), roi_row('2')
    for k in b:
        if k.startswith(('correlation_', 'moran_')) and not k.endswith('__kind'):
            b[k] = -.4
    local = lr.roi_queries(pd.DataFrame([a, b]), lr.POLICIES[1])
    c = lr.coexistence(local, ['A__B']).iloc[0]
    assert c.coexistence_observed and c.can_assess_between_roi_variation
    assert json.loads(c.positive_roi_ids) == ['1'] and json.loads(c.negative_roi_ids) == ['2']


def component(**updates):
    r = dict(sample='s', pair_id='A__B', roi='1', component_id='c1', parent_target_id='s|A__B|roi:1', rho=.4)
    for i in range(4):
        r[f'shell{i}_value'], r[f'shell{i}_status'] = .2, 'finite'
    r.update(updates)
    return r


def test_radial_requires_actual_component_support_and_does_not_pool_shells():
    c = pd.DataFrame([component(shell3_status='insufficient_anchor_support'),
                      component(component_id='c2', shell0_value=-.2)])
    q = lr.component_queries(c, lr.POLICIES[1])
    g = q[q['property'].eq('radial_positive')].set_index('component_id')
    assert g.loc['c1', 'state'] == 'unavailable'
    assert g.loc['c2', 'state'] == 'other'
    assert q.target_id.str.contains('component:').all()


def boundary_frame():
    rows = []
    for surface in ['source_native', 'tessellation', 'consensus_direct', 'consensus_ensemble']:
        r = dict(sample='s', pair_id='A__B', anchor_roi_id='1', responder_roi_id='2', adjacent=True, surface=surface)
        for carrier in ['A', 'B']:
            r['interface_response_' + carrier] = .001 if carrier == 'A' else -.002
            r['availability_status_' + carrier] = 'finite'
            for col in ['source_near_mass', 'source_far_mass', 'represented_near_mass', 'represented_far_mass']:
                r[col + '_' + carrier] = 20.
            r['reachable_responder_fraction_' + carrier] = 1.
        rows.append(r)
    return pd.DataFrame(rows)


def test_boundary_keeps_tiny_raw_magnitudes_and_correct_responder_context():
    roi = pd.DataFrame([roi_row('1', correlation_S=.8), roi_row('2', correlation_S=-.03)])
    b = lr.boundary_records(boundary_frame(), roi)
    assert len(b) == 2
    assert b.anchor_correlation_S.eq(.8).all() and b.responder_correlation_S.eq(-.03).all()
    assert b[b.program.eq('A')].iloc[0].interface_response_S == .001
    assert set(b.source_ensemble_direction) == {'both_positive', 'both_negative'}
    assert b.interpretation.str.contains('not calibrated').all()
    assert 'scenario_assignment' not in b and 'strong_response' not in b
    with pytest.raises(ValueError, match='Duplicate interface'):
        lr.boundary_records(pd.concat([boundary_frame(), boundary_frame()]), roi)


def test_topology_is_not_multiplied_across_program_pairs():
    topo = pd.DataFrame([dict(sample='s', population_roi='1', source_spots=60, n_tessellation_eligible=0),
                         dict(sample='s', population_roi='1', source_spots=40, n_tessellation_eligible=0)])
    roi = pd.DataFrame([roi_row(), roi_row(pair='A__C')])
    result = lr.topology_inventory(topo, roi)
    assert len(result) == 1 and result.iloc[0].n_label_components == 2
    assert result.iloc[0].source_only


def test_higher_bands_nest_numerical_positive_witnesses_not_scenario_calls():
    frame = pd.DataFrame([roi_row('1', correlation_S=.15), roi_row('2', correlation_S=.25), roi_row('3')])
    positive = []
    for policy in lr.POLICIES:
        q = lr.roi_queries(frame, policy)
        positive.append(set(q.loc[q['property'].eq('source_positive') & q.state.eq('meets'), 'roi']))
    assert positive[2] < positive[1] < positive[0]
