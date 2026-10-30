"""Shared evidence must be verified, not silently made partner-independent."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('gbm_review', Path(__file__).with_name('review.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def marginal_fixture():
    rows = []
    for a, b in itertools.combinations(range(14), 2):
        row = dict(sample='s', pair_id=f'p{a}__p{b}', roi='1', source_spots=20,
                   support_class='constructed', coverage=1, program_A=f'p{a}', program_B=f'p{b}')
        for carrier, p in [('A', a), ('B', b)]:
            for metric in ['enrichment', 'CLES', 'log2FC']:
                for surface in ['S', 'T', 'CE']:
                    for suffix in ['', '__kind', '__status', '__low', '__high']:
                        row[f'{metric}_{surface}_{carrier}{suffix}'] = (
                            'finite' if suffix in ['__kind', '__status'] else p / 14)
            for part in ['spots', 'fraction', 'field_status']:
                row[f'parent_high_{part}_S_{carrier}'] = 'finite' if part == 'field_status' else 5
        rows.append(row)
    return pd.DataFrame(rows)


def test_marginal_equivalence_preserves_partner_dependent_statuses():
    d = marginal_fixture()
    d.loc[0, 'enrichment_T_A__status'] = 'partner_specific_diagnostic'
    result = r.marginal_records(d)
    assert len(result) == 14
    p = result[result.program.eq('p0')].iloc[0]
    assert p.n_verified_partner_copies == 13
    assert p.enrichment_T == 0
    assert p.n_partner_dependent_status_fields == 1
    assert p.enrichment_T__status == 'partner_dependent_consult_original_pair_status'
    details = json.loads(p.partner_dependent_statuses)['enrichment_T__status']
    assert len(details) == 13
    assert details['p0__p1'] == 'partner_specific_diagnostic'


@pytest.mark.parametrize('column,value', [('enrichment_T_A', 9), ('enrichment_CE_A__kind', 'unavailable'),
                                          ('parent_high_spots_S_A', 9)])
def test_differing_marginal_values_or_kinds_fail_closed(column, value):
    d = marginal_fixture(); d.loc[0, column] = value
    with pytest.raises(ValueError, match='depend on partner'):
        r.marginal_records(d)


def test_missing_marginal_copy_rejected():
    with pytest.raises(ValueError, match='13 partner'):
        r.marginal_records(marginal_fixture().iloc[1:])


def test_contact_identity_keeps_order_and_requires_all_copies():
    d = pd.DataFrame([dict(sample='s', anchor_roi_id=a, responder_roi_id=b, program='P', pair_id=f'P__{i}',
                          interface_response_S=value, availability_status_S='finite', source_near_mass_S=20)
                      for a, b, value in [('1', '2', .5), ('2', '1', -.2)] for i in range(13)])
    result = r.program_contacts(d)
    assert len(result) == 2
    assert result.interface_response_S.tolist() == [.5, -.2]
    with pytest.raises(ValueError, match='Missing contact'):
        r.program_contacts(d.iloc[1:])
    d.loc[0, 'interface_response_S'] = .6
    with pytest.raises(ValueError, match='differs across partners'):
        r.program_contacts(d)


def authored_fixture():
    return dict(inspected_views=[dict(sample='s', pair_id='A__B')], observations=[dict(
        sample='s', pair_id='A__B', target_id='s|A__B|roi:1', family='focal_pattern', mode='colocated',
        status='candidate', orientation='', observation='Matching local highs.', caution='Bounded patches unresolved.')])


def test_reviewed_overview_does_not_imply_a_scenario_assignment():
    authored = authored_fixture(); authored['observations'] = []
    r.validate_authored(authored, pd.DataFrame([dict(sample='s', pair_id='A__B')]), {('focal_pattern', 'colocated')})
    assert authored['observations'] == []


@pytest.mark.parametrize('fault', ['uninspected', 'duplicate', 'unknown_mode', 'unknown_status', 'missing_caution', 'unprepared'])
def test_review_decisions_require_specific_inspected_evidence(fault):
    a = authored_fixture()
    if fault == 'uninspected': a['inspected_views'] = []
    if fault == 'duplicate': a['observations'].append(copy.deepcopy(a['observations'][0]))
    if fault == 'unknown_mode': a['observations'][0]['mode'] = 'invented'
    if fault == 'unknown_status': a['observations'][0]['status'] = 'automatic_positive'
    if fault == 'missing_caution': a['observations'][0]['caution'] = ''
    if fault == 'unprepared': a['inspected_views'].append(dict(sample='another', pair_id='A__B'))
    with pytest.raises(ValueError):
        r.validate_authored(a, pd.DataFrame([dict(sample='s', pair_id='A__B')]), {('focal_pattern', 'colocated')})


def test_source_equivalence_checks_all_partners_and_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'DATA', tmp_path)
    (tmp_path / 'CSVs').mkdir()
    pairs = pd.DataFrame([dict(pair=f'p{a}_p{b}', program_g1=f'p{a}', program_g2=f'p{b}')
                          for a, b in itertools.combinations(range(14), 2)])
    for p in pairs.itertuples():
        pd.DataFrame(dict(x=[0, 1], y=[0, 0], roi_id=['1', '2'],
                          gene1=[int(p.program_g1[1:]), np.nan],
                          gene2=[int(p.program_g2[1:]), np.nan])).to_csv(
                              tmp_path / 'CSVs' / f's_expression_data_{p.pair}.csv', index=False)
    ledger = pd.DataFrame(dict(roi_id=['1', '2'])); hashes = {}
    checks, wide = r.input_equivalence('s', pairs, ledger, {}, hashes)
    assert len(checks) == 14 and checks.n_verified_partner_copies.eq(13).all()
    assert len(hashes) == 91 and len(wide) == 2
    path = tmp_path / 'CSVs/s_expression_data_p0_p2.csv'
    d = pd.read_csv(path); d.loc[0, 'gene1'] = 99; d.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Source program differs'):
        r.input_equivalence('s', pairs, ledger, {}, {})
    d.loc[0, 'gene1'] = 0; d.loc[0, 'x'] = 99; d.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Source rows/geometry differ'):
        r.input_equivalence('s', pairs, ledger, {}, {})
