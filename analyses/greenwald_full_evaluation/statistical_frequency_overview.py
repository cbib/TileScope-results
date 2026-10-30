"""Presentation of existing local frequencies; no core run or scenario calls.

Recover the legacy stability companion's estimand, not sign-only sample hits.
All inputs are checksum-verified. The original local-frequency browser remains
the detailed audit; this script only adds a simultaneous statistical overview.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from local_frequency_analysis import VerifiedReader, digest, csv, frequencies
from figure_common import EXPECTED_CORE
from tilescope_results.paths import greenwald_evidence

HERE = Path(__file__).resolve().parent
SOURCE = greenwald_evidence('local_frequencies')
OUT = greenwald_evidence('gbm_spatial_repertoire') / 'statistical_frequencies'
TEMPLATE = HERE / 'statistical_frequency_overview.html'
POS, NEG, OTHER = '#227d8d', '#c66b42', '#e8edef'


def signed(key, label, group, stem, main=False):
    return dict(key=key, label=label, group=group, main=main,
                props=[stem + '_positive', stem + '_negative'],
                colors=[POS, NEG], states=['Positive', 'Negative'])


def single(key, label, group, prop, main=False):
    return dict(key=key, label=label, group=group, main=main,
                props=[prop], colors=['#85709e'], states=['Meets query'])


VIEWS = [
    dict(key='territory', label='Both / one', group='Enrichment', main=True,
         props=['coenriched', 'one_enriched'], colors=['#8061a5', '#ccac52'],
         states=['Both enriched', 'Exactly one enriched']),
    dict(key='joint', label='Joint-high*', group='Enrichment', main=True,
         props=['coenriched_joint_high', 'coenriched_joint_not_high'],
         colors=['#8061a5', '#b8a6cf'], states=['Joint-high >1', 'Joint-high ≤1']),
    *[signed('r'+s, s, 'Pearson', 'pearson_'+s, s in ['S', 'CE'])
      for s in ['S', 'T', 'CD', 'CE']],
    *[signed('m'+s+c, s+' '+c.replace('_', '|'), 'Moran', 'moran_'+s+'_'+c,
             s in ['S', 'CE']) for s in ['S', 'T', 'CD', 'CE'] for c in ['A_B', 'B_A']],
    signed('stable', 'All four', 'Across supports', 'correlation_all', True),
    single('switch', 'Sign change', 'Across supports', 'correlation_support_change', True),
    signed('distance', 'All shells', 'Components', 'radial', True),
    single('turn', 'Near / far', 'Components', 'radial_sign_change', True),
    *[signed(k, label, 'Same-ROI combinations', k) for k, label in [
        ('source', 'S fields'), ('ensemble', 'CE fields'),
        ('source_ensemble', 'S + CE fields'), ('coenriched', 'Enriched + fields')]],
    *[single('low'+s, s+' joint-low', 'Joint-low', 'joint_low_'+s+'_high')
      for s in ['S', 'T', 'CE']],
    *[single('amplitude'+s+c, s+' '+c, 'Mean vs distribution',
             'amplitude_disagreement_'+s+'_'+c)
      for s in ['S', 'T', 'CE'] for c in ['A', 'B']],
]


def validate_opposites(frame, views=VIEWS):
    """Check matched counts/statuses and disjoint qualifying locality IDs.

    Common eligibility follows the paired query definitions in the verified
    producer. These aggregate exports cannot reconstruct every available ID.
    """
    keys = ['sample', 'pair_id', 'policy', 'context']
    if frame.duplicated(keys + ['property']).any():
        raise ValueError('Duplicate local-frequency rows')
    for view in views:
        if len(view['props']) != 2:
            continue
        a, b = [frame[frame['property'].eq(p)].set_index(keys).sort_index()
                for p in view['props']]
        if not a.index.equals(b.index):
            raise ValueError('Unmatched section opportunities: '+view['key'])
        for c in ['n_evaluable', 'n_unavailable', 'n_outside_context', 'n_locations']:
            if not a[c].equals(b[c]):
                raise ValueError('Different denominators: '+view['key'])
        if ((a.n_meets + b.n_meets) > a.n_evaluable).any():
            raise ValueError('Nonexclusive counts: '+view['key'])
        for x, y in zip(a.target_ids, b.target_ids):
            if set(json.loads(x)) & set(json.loads(y)):
                raise ValueError('Overlapping local witnesses: '+view['key'])


def aggregate(frame):
    """Equal-section mean of local fractions, NOT an any-ROI sample frequency."""
    d = frequencies(frame)
    d['coverage'] = d.n_evaluable / (d.n_locations-d.n_outside_context).replace(0, np.nan)
    keys = ['pair_id', 'policy', 'property', 'context']
    result = d.groupby(keys, sort=True).agg(
        mean=('frequency', 'mean'), q25=('frequency', lambda x: x.quantile(.25)),
        median=('frequency', 'median'), q75=('frequency', lambda x: x.quantile(.75)),
        n_sections=('sample', 'size'), n_available_sections=('frequency', 'count'),
        meets=('n_meets', 'sum'), evaluable=('n_evaluable', 'sum'),
        unavailable=('n_unavailable', 'sum'), coverage=('coverage', 'mean'),
    ).reset_index()
    result['pooled'] = result.meets / result.evaluable.replace(0, np.nan)
    return result
