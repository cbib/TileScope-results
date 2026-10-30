"""Local frequencies, not any-hit section labels or scenario prevalence.

Read checksum-verified current-core evidence already collected by
two_level_analysis. Recover the useful within-section frequency design from
the retired v5 analysis, NOT its thresholds, MAUP labels or biological results.
Keep the complete section distribution and local witnesses behind each summary.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import local_relationships as lr
from figure_common import EXPECTED_CORE
from tilescope_results.paths import greenwald_evidence, resolve_reference

HERE = Path(__file__).resolve().parent
EVIDENCE = greenwald_evidence('two_level_evidence')
COHORT = greenwald_evidence('scenario_cohort')
OUT = greenwald_evidence('local_frequencies')
TEMPLATE = HERE / 'local_frequency_view.html'
KEY = ['sample', 'pair_id', 'policy', 'property', 'context']
COUNT_COLUMNS = ['n_meets', 'n_other', 'n_unavailable', 'n_outside_context']

# Deliberately bounded interpretation-development set. Fixed before examining
# the new frequency rankings. Full locality/mode coverage within these views;
# not a random or representative cohort sample, and not a prevalence estimator.
PILOT_SAMPLES = ['243_T', '248_T', '260_T', '262_T', '270_T', 'GBM_ZH916bulk']
PILOT_PAIRS = ['AC__OPC', 'MES-Ast__MES-Hyp', 'AC__MES-Hyp']
PILOT_EXTRA = [('262_T', 'NPC__Neuron'), ('270_T', 'MES__Oligo')]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv(frame, path):
    frame.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0}
                 if path.suffix == '.gz' else None)


def pair_anchor(pair):
    return 'pair-' + hashlib.sha256(pair.encode()).hexdigest()[:16]


class VerifiedReader:
    def __init__(self, root):
        self.root = resolve_reference(root)
        path = self.root / 'provenance.json'
        self.manifest = json.loads(path.read_text())
        self.manifest['input_sha256'] = {
            str(resolve_reference(p)): value
            for p, value in self.manifest.get('input_sha256', {}).items()}
        self.inputs = {str(path.resolve()): digest(path)}

    def read(self, name):
        path = self.root / name
        actual = digest(path)
        if self.manifest['output_sha256'].get(name) != actual:
            raise ValueError(f'Stale or modified evidence: {path}')
        self.inputs[str(path.resolve())] = actual
        return pd.read_csv(path, low_memory=False, dtype={c: str for c in
            ['roi', 'population_roi', 'component_id', 'anchor_roi_id',
             'responder_roi_id', 'host_roi', 'anchor_roi', 'responder_roi']})


def frequencies(collections):
    """One fraction per section/pair/property. Zero opportunities are NOT zero."""
    d = collections.copy()
    if 'context' not in d:
        d['context'] = 'query_specific'
    if d.duplicated(KEY).any():
        raise ValueError('Duplicate section/property opportunity set')
    counts = d[COUNT_COLUMNS + ['n_locations', 'n_evaluable']]
    if counts.isna().any().any() or (counts < 0).any().any() or ((counts % 1) != 0).any().any():
        raise ValueError('Counts must be nonnegative integers')
    if not d[COUNT_COLUMNS].sum(axis=1).eq(d.n_locations).all():
        raise ValueError('Opportunity counts do not partition locations')
    if not (d.n_meets + d.n_other).eq(d.n_evaluable).all():
        raise ValueError('Evaluable denominator mismatch')
    d['n_potential_context'] = d.n_locations - d.n_outside_context
    d['frequency'] = d.n_meets / d.n_evaluable.replace(0, np.nan)
    d['availability_fraction'] = d.n_evaluable / d.n_potential_context.replace(0, np.nan)
    # Descriptive worst-case envelopes, NOT confidence intervals. Unknown
    # context may belong outside the opportunity set; these remain conservative.
    denom = d.n_potential_context.replace(0, np.nan)
    d['missingness_lower_bound'] = d.n_meets / denom
    d['missingness_upper_bound'] = (d.n_meets + d.n_unavailable) / denom
    d['opportunity_status'] = np.select(
        [d.n_locations.eq(0), d.n_potential_context.eq(0), d.n_evaluable.eq(0),
         d.n_unavailable.gt(0)],
        ['no_locations', 'outside_context', 'unavailable', 'partially_evaluable'],
        default='fully_evaluable')
    return d


def contextual_collections(local):
    """Optional shared-enrichment view, never a universal scenario gate.

    Reuse the actual SAME-ROI co-enrichment state. No cross-ROI multiplication.
    The unconditioned/source-only inventories remain the primary opportunity set.
    """
    keys = ['sample', 'pair_id', 'policy', 'target_id']
    eligible_props = [p for p, group, _, unit, _ in lr.CONTRACT
                      if unit == 'roi' and group not in
                      ['Territory', 'Linked territory / fields', 'Joint-low occupancy']]
    context = local[local['property'].eq('coenriched')][keys + ['state']]
    if context.duplicated(keys).any():
        raise ValueError('Duplicate co-enrichment context')
    g = local[local['property'].isin(eligible_props)].merge(
        context.rename(columns={'state': 'context_state'}), on=keys,
        how='left', validate='many_to_one')
    if g.context_state.isna().any():
        raise ValueError('Missing same-ROI context')
    g['state'] = np.where(g.context_state.eq('other'), 'outside_context',
                        np.where(g.context_state.eq('unavailable'), 'unavailable', g.state))
    rows = []
    for key, part in g.groupby(['sample', 'pair_id', 'policy', 'property'], sort=False):
        good = part[part.state.eq('meets')]
        counts = part.state.value_counts()
        rows.append(dict(zip(['sample', 'pair_id', 'policy', 'property'], key),
            unit='roi', context='coenriched', n_locations=len(part),
            **{'n_' + name: int(counts.get(name, 0)) for name in
               ['meets', 'other', 'unavailable', 'outside_context']},
            n_evaluable=int(part.state.isin(['meets', 'other']).sum()),
            target_ids=json.dumps(good.target_id.tolist()),
            roi_ids=json.dumps(sorted(good.roi.astype(str).unique()))))
    return pd.DataFrame(rows)


def summarize_sections(frame, scopes):
    """Section-balanced distributions; pooled ROI frequency is sensitivity only."""
    rows = []
    for scope, samples in scopes.items():
        d = frame[frame['sample'].isin(samples)]
        for key, g in d.groupby(['pair_id', 'policy', 'property', 'context'], sort=False):
            v = g.frequency.dropna()
            n = int(g.n_evaluable.sum())
            q = v.quantile([.25, .5, .75]).tolist() if len(v) else [np.nan] * 3
            rows.append(dict(zip(['pair_id', 'policy', 'property', 'context'], key),
                scope=scope, unit=g.unit.iloc[0], n_sections=len(g),
                n_evaluable_sections=len(v), n_sections_without_evaluable_locations=len(g)-len(v),
                n_sections_with_unavailable_locations=int(g.n_unavailable.gt(0).sum()),
                n_local_meets=int(g.n_meets.sum()), n_local_evaluable=n,
                n_local_unavailable=int(g.n_unavailable.sum()),
                frequency_q25=q[0], frequency_median=q[1], frequency_q75=q[2],
                frequency_section_mean=float(v.mean()) if len(v) else np.nan,
                frequency_min=float(v.min()) if len(v) else np.nan,
                frequency_max=float(v.max()) if len(v) else np.nan,
                frequency_roi_pooled=int(g.n_meets.sum())/n if n else np.nan,
                section_frequencies=json.dumps({r.sample: None if pd.isna(r.frequency) else r.frequency
                                               for r in g.itertuples()})))
    return pd.DataFrame(rows)


def coexistence_coordinates(frame):
    """Keep two frequencies and distinct ROI witnesses together, not their mean."""
    rows = []
    for stem in ['source', 'ensemble', 'source_ensemble', 'coenriched',
                 'pearson_S', 'pearson_CE']:
        p = frame[frame['property'].eq(stem + '_positive')]
        m = frame[frame['property'].eq(stem + '_negative')]
        g = p.merge(m, on=['sample', 'pair_id', 'policy', 'context'],
                    suffixes=('_positive', '_negative'), validate='one_to_one')
        for r in g.itertuples():
            pos, neg = json.loads(r.roi_ids_positive), json.loads(r.roi_ids_negative)
            if set(pos) & set(neg):
                raise ValueError('Opposite states attached to the same ROI')
            if r.n_evaluable_positive != r.n_evaluable_negative:
                raise ValueError('Opposite states use different opportunity sets')
            rows.append(dict(sample=r.sample, pair_id=r.pair_id, policy=r.policy,
                context=r.context, evidence=stem, positive_frequency=r.frequency_positive,
                negative_frequency=r.frequency_negative, n_evaluable=r.n_evaluable_positive,
                positive_rois=json.dumps(pos), negative_rois=json.dumps(neg),
                observed_coexistence=bool(pos and neg)))
    return pd.DataFrame(rows)


def boundary_distributions(boundary):
    """Actual contact values as empirical distributions; no universal cutoff.

    Each program/contact stays attached to its pair and ordered endpoints.
    These are different questions from internal co-enrichment. Independent
    program-contact recurrence would first need partner-copy deduplication.
    """
    key = ['sample', 'pair_id', 'carrier', 'anchor_roi_id', 'responder_roi_id']
    if boundary.duplicated(key).any():
        raise ValueError('Duplicate program/contact values')
    boundary = boundary.copy()
    ka, a = lr.ratio(boundary, 'anchor_enrichment_S_A', 1)
    kb, b = lr.ratio(boundary, 'anchor_enrichment_S_B', 1)
    kr_a, ra = lr.ratio(boundary, 'responder_enrichment_S_A', 1)
    kr_b, rb = lr.ratio(boundary, 'responder_enrichment_S_B', 1)
    cross = (ka & kr_b & a & rb) | (kb & kr_a & b & ra)
    not_cross = ((ka & ~a) | (kr_b & ~rb)) & ((kb & ~b) | (kr_a & ~ra))
    boundary['_cross'] = cross
    boundary['_context_known'] = cross | not_cross
    rows = []
    for (sample, pair, carrier), g in boundary.groupby(['sample', 'pair_id', 'carrier'], sort=False):
        for surface in ['S', 'T', 'CD', 'CE']:
            v = g['interface_response_' + surface]
            known = g['availability_status_' + surface].eq('finite') & np.isfinite(v)
            for context in ['all_contacts', 'cross_enriched']:
                # A on anchor/B on responder OR B on anchor/A on responder;
                # context is explicit, optional and independent of response.
                eligible = pd.Series(True, index=g.index) if context == 'all_contacts' else g['_cross']
                unknown_context = pd.Series(False, index=g.index) if context == 'all_contacts' else ~g['_context_known']
                good = g[known & eligible]
                rows.append(dict(sample=sample, pair_id=pair, carrier=carrier,
                    program=g.program.iloc[0], surface=surface, context=context,
                    n_contacts=len(g), n_evaluable=len(good),
                    n_unavailable=int((eligible & ~known | unknown_context).sum()),
                    n_outside_context=int((~eligible & ~unknown_context).sum()),
                    values=json.dumps(good['interface_response_' + surface].tolist()),
                    endpoints=json.dumps((good.anchor_roi_id.astype(str) + '→' + good.responder_roi_id.astype(str)).tolist()),
                    interpretation='empirical raw-response distribution; not classified contact scenarios'))
    return pd.DataFrame(rows)
