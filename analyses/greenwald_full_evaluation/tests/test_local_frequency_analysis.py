"""Frequency estimands, missing opportunities and local linkage contracts."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_frequency_analysis as f


def collection(sample='s1', meets=1, other=4, unavailable=0, outside=0, **kw):
    return dict(sample=sample, pair_id='A__B', policy='reference', property='source_positive',
                context='query_specific', unit='roi', n_meets=meets, n_other=other,
                n_unavailable=unavailable, n_outside_context=outside,
                n_locations=meets+other+unavailable+outside, n_evaluable=meets+other,
                target_ids=json.dumps([f'{sample}|A__B|roi:{i}' for i in range(meets)]),
                roi_ids=json.dumps([str(i) for i in range(meets)])) | kw


def test_extent_is_not_any_hit_recurrence_and_pooling_is_explicit():
    d=f.frequencies(pd.DataFrame([collection('s1',1,9),collection('s2',5,0)]))
    assert d.frequency.tolist()==[.1,1.]
    r=f.summarize_sections(d,{'all':['s1','s2']}).iloc[0]
    assert r.frequency_section_mean==pytest.approx(.55)
    assert r.frequency_roi_pooled==pytest.approx(6/15)
    assert json.loads(r.section_frequencies)=={'s1':.1,'s2':1.}
    assert r.n_evaluable_sections==2  # not an any-hit 100% verdict


def test_unavailable_outside_and_no_locations_never_become_zero_frequencies():
    d=f.frequencies(pd.DataFrame([collection('zero',0,5),collection('unknown',0,0,5),
        collection('outside',0,0,0,5),collection('none',0,0,0,0)] )).set_index('sample')
    assert d.loc['zero','frequency']==0
    assert d.loc[['unknown','outside','none'],'frequency'].isna().all()
    assert d.opportunity_status.to_dict()==dict(zero='fully_evaluable',unknown='unavailable',
        outside='outside_context',none='no_locations')
    r=f.summarize_sections(d.reset_index(),{'all':list(d.index)}).iloc[0]
    assert r.n_evaluable_sections==1 and r.frequency_median==0


def test_missingness_envelope_is_separate_from_conditional_frequency():
    r=f.frequencies(pd.DataFrame([collection(meets=1,other=1,unavailable=2,outside=7)])).iloc[0]
    assert r.frequency==.5 and r.availability_fraction==.5
    assert r.missingness_lower_bound==.25 and r.missingness_upper_bound==.75


@pytest.mark.parametrize('update',[{'n_evaluable':100},{'n_locations':0},{'n_meets':-1},{'n_meets':.5}])
def test_invalid_counts_fail_closed(update):
    with pytest.raises(ValueError):f.frequencies(pd.DataFrame([collection(**update)]))


def test_duplicate_local_opportunity_sets_rejected():
    with pytest.raises(ValueError,match='Duplicate'):f.frequencies(pd.DataFrame([collection(),collection()]))


def test_context_is_same_roi_and_never_a_universal_gate():
    rows=[]
    for roi,ctx in [('1','other'),('2','meets'),('3','unavailable')]:
        for prop,state in [('coenriched',ctx),('source_positive','meets')]:
            rows.append(dict(sample='s',pair_id='A__B',roi=roi,policy='reference',
                target_id='s|A__B|roi:'+roi,unit='roi',property=prop,state=state))
    local=pd.DataFrame(rows)
    r=f.contextual_collections(local).iloc[0]
    assert (r.n_meets,r.n_unavailable,r.n_outside_context)==(1,1,1)
    assert json.loads(r.roi_ids)==['2']
    assert local[local['property'].eq('source_positive')].state.eq('meets').all()


def test_coexistence_not_collapsed_to_between_section_mean():
    rows=[]
    for s in ['s1','s2']:
        rows += [collection(s,1,1,roi_ids='["1"]'),
                 collection(s,1,1,property='source_negative',roi_ids='["2"]')]
    g=f.coexistence_coordinates(f.frequencies(pd.DataFrame(rows)))
    assert g.observed_coexistence.all()
    assert g.positive_frequency.eq(.5).all() and g.negative_frequency.eq(.5).all()
    rows=[collection('s1',2,0,roi_ids='["1","2"]'),
          collection('s1',0,2,property='source_negative',roi_ids='[]'),
          collection('s2',0,2,roi_ids='[]'),
          collection('s2',2,0,property='source_negative',roi_ids='["1","2"]')]
    g=f.coexistence_coordinates(f.frequencies(pd.DataFrame(rows)))
    assert not g.observed_coexistence.any()
    assert g.positive_frequency.mean()==.5 and g.negative_frequency.mean()==.5


def boundary_row(anchor='4',responder='2',**kw):
    r=dict(sample='s',pair_id='A__B',carrier='A',program='A',anchor_roi_id=anchor,responder_roi_id=responder)
    for surface in ['S','T','CD','CE']:
        r['interface_response_'+surface]=.001
        r['availability_status_'+surface]='finite'
    for side,program in [('anchor','A'),('anchor','B'),('responder','A'),('responder','B')]:
        name=f'{side}_enrichment_S_{program}'
        r[name]=2. if (side,program) in [('anchor','A'),('responder','B')] else .5
        r[name+'__kind']='finite'
    return r|kw


def test_boundary_retains_tiny_values_actual_responder_and_optional_context():
    b=pd.DataFrame([boundary_row(),boundary_row('3','1',anchor_enrichment_S_A=.5)])
    d=f.boundary_distributions(b)
    r=d[d.surface.eq('S')&d.context.eq('all_contacts')].iloc[0]
    assert json.loads(r['values'])==[.001,.001]
    assert json.loads(r.endpoints)==['4→2','3→1']
    c=d[d.surface.eq('S')&d.context.eq('cross_enriched')].iloc[0]
    assert c.n_evaluable==1 and c.n_outside_context==1
    assert json.loads(c.endpoints)==['4→2']
    assert 'strong_response' not in d and 'scenario' not in d


def test_boundary_missing_context_is_not_false_or_positive():
    b=pd.DataFrame([boundary_row(anchor_enrichment_S_A=np.nan,anchor_enrichment_S_A__kind='unavailable')])
    d=f.boundary_distributions(b)
    c=d[d.surface.eq('S')&d.context.eq('cross_enriched')].iloc[0]
    assert c.n_unavailable==1 and c.n_evaluable==0 and c.n_outside_context==0
    assert d[d.context.eq('all_contacts')].n_evaluable.eq(1).all()
