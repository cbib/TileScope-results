"""Keep the compact display faithful to the existing local-frequency estimand."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import statistical_frequency_overview as o
import local_relationships as lr


def row(sample='s1', prop='x_positive', meets=1, other=3, missing=0, outside=0, **kw):
    return dict(sample=sample, pair_id='A__B', policy='reference', property=prop,
                context='query_specific', n_meets=meets, n_other=other,
                n_unavailable=missing, n_outside_context=outside,
                n_locations=meets+other+missing+outside, n_evaluable=meets+other,
                target_ids=json.dumps([str(i) for i in range(meets)])) | kw


def test_mean_of_fractions_not_any_hit_or_mean_of_counts():
    s=o.aggregate(pd.DataFrame([row('s1',meets=1,other=9),row('s2',meets=5,other=0)])).iloc[0]
    assert s['mean']==pytest.approx(.55)
    assert s.pooled==pytest.approx(.4)
    assert s.n_available_sections==2


def test_no_opportunity_and_unavailable_are_not_negative_or_zero():
    d=pd.DataFrame([row('zero',meets=0),row('unavailable',meets=0,other=0,missing=5),
                    row('outside',meets=0,other=0,outside=7)])
    r=o.aggregate(d).iloc[0]
    assert r['mean']==0 and r.n_available_sections==1 and r.n_sections==3
    assert r.unavailable==5 and r.coverage==.5


def test_all_unavailable_stays_nan():
    r=o.aggregate(pd.DataFrame([row(meets=0,other=0,missing=2)])).iloc[0]
    assert np.isnan(r['mean']) and np.isnan(r.pooled) and r.coverage==0


def test_policies_and_contexts_are_never_averaged_together():
    d=pd.DataFrame([row(),row(policy='lower',meets=3,other=1),
                    row(context='coenriched',meets=1,other=0)])
    s=o.aggregate(d)
    assert len(s)==3 and set(s['mean'])=={.25,.75,1}


def test_all_sections_pass_through_no_reference_cohort_filter():
    d=pd.DataFrame([row('270_T'),row('243_T'),row('GBM_ZH916inf')])
    assert o.aggregate(d).iloc[0].n_sections==3


def test_invalid_denominator_rejected():
    with pytest.raises(ValueError,match='denominator'):
        o.aggregate(pd.DataFrame([row(n_evaluable=99)]))


V=[o.signed('x','X','Test','x')]


def test_stacked_states_keep_both_signs_with_disjoint_witnesses():
    d=pd.DataFrame([row(),row(prop='x_negative',target_ids='["2"]')])
    o.validate_opposites(d,V)


@pytest.mark.parametrize('change,message',[
    ({'target_ids':'["0"]'},'Overlapping'),
    ({'n_evaluable':5},'Different denominators'),
    ({'n_meets':4},'Nonexclusive'),
    ({'sample':'different'},'Unmatched'),
])
def test_invalid_stacking_rejected(change,message):
    d=pd.DataFrame([row(),row(prop='x_negative',target_ids='["2"]') | change])
    with pytest.raises(ValueError,match=message):o.validate_opposites(d,V)


def test_no_existing_statistical_query_disappears():
    assert {p for v in o.VIEWS for p in v['props']}=={r[0] for r in lr.CONTRACT}
    assert len({v['key'] for v in o.VIEWS})==len(o.VIEWS)
    assert all('scenario' not in p for v in o.VIEWS for p in v['props'])
