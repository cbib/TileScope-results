from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vet_shortlist import summarize, signed_summary, typed, triad_table, matched_changes, validate_locality, TRIAD


def test_frequency_equal_samples_not_pooled_rois():
    d = pd.DataFrame({'sample':['s1']*10+['s2'], 'pair_id':['A__B']*11})
    r = signed_summary(d, [0]*10+[.8], .2, metric='r', surface='S').set_index('sign')
    assert r.loc[1, 'frequency'] == .5
    assert r.loc[1, 'qualifying_median'] == .8


def test_unavailable_never_negative_or_zero_opportunity():
    d = pd.DataFrame({'sample':['s1','s2'], 'pair_id':['A__B']*2})
    r = signed_summary(d, [np.nan,-.5], .2, metric='r', surface='S').set_index('sign')
    assert r.loc[-1, 'frequency'] == 1
    assert r.loc[-1, 'n_unavailable'] == 1
    assert r.loc[-1, 'n_samples'] == 1


def test_signed_bands_are_strict_and_disjoint():
    d = pd.DataFrame({'sample':['s']*5, 'pair_id':['A__B']*5})
    r = signed_summary(d, [-.3,-.2,0,.2,.3], .2, metric='r', surface='S').set_index('sign')
    assert r.loc[1, 'n_qualifying'] == r.loc[-1, 'n_qualifying'] == 1


def test_typed_ratio_saturation_is_known_enrichment():
    d = pd.DataFrame({'enrichment_S_A':[np.nan,0,np.nan],
                      'enrichment_S_A__kind':['saturated','finite','unavailable']})
    np.testing.assert_array_equal(typed(d,'enrichment_S_A').to_numpy(), [np.inf,0,np.nan])


def make_roi():
    d = pd.DataFrame({'sample':['s']*6, 'roi':['1']*3+['2']*3,
                      'pair_id':TRIAD*2, 'source_spots':[100]*6})
    for s in ['S','T','CD','CE']:
        d['correlation_'+s] = [.4,.3,-.5,-.4,-.3,.5]
        d['correlation_'+s+'__kind'] = 'finite'
    return d


def test_triad_is_linked_to_same_roi_not_sample_average():
    t = triad_table(make_roi()); s = t[t.surface.eq('S')].set_index('roi')
    assert s.loc['1', 'positive_positive_negative_0.2'] == 1
    assert s.loc['2', 'positive_positive_negative_0.2'] == 0
    assert s.loc['1', 'minimum_signed_pattern'] == .3


def test_incomplete_triad_stays_unavailable():
    d = make_roi(); d.loc[1, 'correlation_S__kind'] = 'unavailable'
    t = triad_table(d); s = t[t.surface.eq('S')].set_index('roi')
    assert not s.loc['1','all_evaluable']
    assert np.isnan(s.loc['1','positive_positive_negative_0.2'])
    assert np.isnan(s.loc['1','minimum_signed_pattern'])


def test_support_changes_require_same_roi_available_on_both():
    d = make_roi(); d.loc[0, 'correlation_CE'] = .7
    d.loc[1, 'correlation_CE__kind'] = 'unavailable'
    m = matched_changes(d)
    assert m.loc[0,'delta_CE_minus_S'] == pytest.approx(.3)
    assert np.isnan(m.loc[1,'delta_CE_minus_S'])
    assert m.loc[1,'state_CE_0.2'] == 'unavailable'


def test_review_requires_real_roi_and_actual_ordered_interface():
    roi = make_roi()
    boundary = pd.DataFrame({'sample':['s'], 'pair_id':[TRIAD[0]],
                             'anchor_roi_id':['2'], 'responder_roi_id':['1']})
    note = dict(sample='s', roi='1', pairs=[TRIAD[0]], anchor_roi='2',
                status='reviewed', origin='test', observation='A field', caution='Not causal')
    validate_locality(note, TRIAD[0], roi, boundary)
    with pytest.raises(ValueError, match='exactly one ROI'):
        validate_locality({**note, 'roi':'99'}, TRIAD[0], roi, boundary)
    with pytest.raises(ValueError, match='ordered interface'):
        validate_locality({**note, 'roi':'2', 'anchor_roi':'1'}, TRIAD[0], roi, boundary)
    with pytest.raises(ValueError, match='Missing observation field'):
        validate_locality({**note, 'caution':''}, TRIAD[0], roi, boundary)
