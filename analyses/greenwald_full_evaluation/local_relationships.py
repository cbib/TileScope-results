"""Level 1: explicit descriptive queries on mature core exports.

No scenario classifier, p-value inference, new descriptor, or universal
co-enrichment gate. Every query keeps its native ROI/component identity and
availability. Bands are display conventions with sensitivity, not synthetic
benchmark decision boundaries.
"""
from dataclasses import dataclass, asdict
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Bands:
    name: str
    pearson: float
    moran: float
    radial: float
    enrichment: float = 1.0
    cles: float = .10


POLICIES = [Bands('lower', .10, .05, .05, cles=.05), Bands('reference', .20, .10, .10),
            Bands('higher', .30, .15, .15, cles=.15)]

# IDs stay stable; labels never silently upgrade a numerical property to a
# morphological or biological interpretation.
CONTRACT = [
    ('coenriched', 'Territory', 'Both enriched', 'roi', 'A and B source enrichment > 1; saturated ratios retained.'),
    ('coenriched_joint_high', 'Territory', 'Both + joint high', 'roi', 'Within co-enriched ROIs: joint-high enrichment > 1.'),
    ('coenriched_joint_not_high', 'Territory', 'Both, joint not high', 'roi', 'Within co-enriched ROIs: joint-high enrichment <= 1; not a focality claim.'),
    ('one_enriched', 'Territory', 'One enriched', 'roi', 'Exactly one source marginal enrichment > 1.'),
    ('source_positive', 'Internal fields', 'Source +', 'roi', 'Source Pearson and both Moran directions above their bands in the SAME ROI.'),
    ('source_negative', 'Internal fields', 'Source −', 'roi', 'Source Pearson and both Moran directions below negative bands in the SAME ROI.'),
    ('ensemble_positive', 'Internal fields', 'Ensemble +', 'roi', 'Ensemble Pearson and both Moran directions above bands; no source requirement.'),
    ('ensemble_negative', 'Internal fields', 'Ensemble −', 'roi', 'Ensemble Pearson and both Moran directions below negative bands; no source requirement.'),
    ('source_ensemble_positive', 'Across supports', 'S/CE fields positive', 'roi', 'Source AND ensemble positive field combinations, same ROI.'),
    ('source_ensemble_negative', 'Across supports', 'S/CE fields negative', 'roi', 'Source AND ensemble negative field combinations, same ROI.'),
    ('coenriched_positive', 'Linked territory / fields', 'Both enriched / +', 'roi', 'Co-enrichment AND positive source/ensemble fields, all in the SAME ROI.'),
    ('coenriched_negative', 'Linked territory / fields', 'Both enriched / −', 'roi', 'Co-enrichment AND negative source/ensemble fields, all in the SAME ROI.'),
    ('correlation_all_positive', 'Across supports', 'S/T/CD/CE r +', 'roi', 'All four Pearson values and tessellation lower quartile > Pearson band.'),
    ('correlation_all_negative', 'Across supports', 'S/T/CD/CE r −', 'roi', 'All four Pearson values and tessellation upper quartile < negative Pearson band.'),
    ('correlation_support_change', 'Across supports', 'r changes sign', 'roi', 'All four Pearson values finite; some above positive and others below negative band in one ROI.'),
    ('radial_positive', 'Component distances', 'Positive across shells', 'component', 'Component zero-lag correlation > Pearson band and every recorded shell > radial band, all four shells finite.'),
    ('radial_negative', 'Component distances', 'Negative across shells', 'component', 'Component zero-lag correlation < negative Pearson band and every shell < negative radial band, all four finite.'),
    ('radial_sign_change', 'Component distances', 'Near/far sign change', 'component', 'First/last shell lie beyond opposite radial bands in the SAME component; neither a gradient nor a focus call.'),
]

# Independent channels matter: Pearson/Moran disagreement must not make both
# signals disappear behind an all-statistics-agree conjunction.
for _surface in ['S', 'T', 'CD', 'CE']:
    for _direction, _label in [('positive', '+'), ('negative', '−')]:
        CONTRACT.append((f'pearson_{_surface}_{_direction}', 'Pearson independently',
            f'{_surface} r {_label}', 'roi', f'Pearson at {_surface} beyond its signed band; no Moran or enrichment gate.'))
        for _channel in ['A|B', 'B|A']:
            _id = _channel.replace('|', '_')
            CONTRACT.append((f'moran_{_surface}_{_id}_{_direction}', f'Moran independently — {_surface}',
                f'{_channel} {_label}', 'roi', f'Moran {_channel} at {_surface} beyond its signed band; no Pearson or other-direction gate.'))
for _surface in ['S', 'T', 'CE']:
    CONTRACT.append((f'joint_low_{_surface}_high', 'Joint-low occupancy', f'{_surface} joint low >1', 'roi',
        f'Joint-low enrichment >1 at {_surface}; contextual occupancy, not program activation or a scenario.'))
    for _carrier in ['A', 'B']:
        CONTRACT.append((f'amplitude_disagreement_{_surface}_{_carrier}', 'Mean / distribution context',
            f'{_surface} {_carrier}: FC vs CLES', 'roi',
            f'Opposite signs of log2FC and CLES−0.5 for {_carrier} at {_surface}, with CLES beyond its display band. '
            'Mean/distribution disagreement, not a focality claim or calibrated test.'))


def policy_table():
    return pd.DataFrame([asdict(p) for p in POLICIES])


def finite(frame, names):
    mask = pd.Series(True, index=frame.index)
    for name in names:
        mask &= np.isfinite(frame[name]) & frame[name + '__kind'].eq('finite')
    return mask


def ratio(frame, name, floor):
    kind = frame[name + '__kind']
    known = kind.eq('saturated') | (kind.eq('finite') & np.isfinite(frame[name]))
    high = kind.eq('saturated') | (known & frame[name].gt(floor))
    return known, high


def roi_queries(roi, bands):
    """Return all local opportunities, including unknown and outside-context."""
    if roi.duplicated(['sample', 'pair_id', 'roi']).any():
        raise ValueError('Duplicate ROI identities')
    rows = []

    def add(name, known, meets, context=None, context_known=None):
        g = roi[['sample', 'pair_id', 'roi', 'target_id']].copy()
        g['unit'] = 'roi'
        g['property'] = name
        state = np.where(known, np.where(meets, 'meets', 'other'), 'unavailable')
        if context is not None:
            state = np.where(~context_known, 'unavailable', np.where(context, state, 'outside_context'))
        g['state'] = state
        rows.append(g)

    ka, a = ratio(roi, 'enrichment_S_A', bands.enrichment)
    kb, b = ratio(roi, 'enrichment_S_B', bands.enrichment)
    kj, j = ratio(roi, 'enrichment_S_AB', bands.enrichment)
    # Kleene conjunction: one known non-enriched marginal establishes that the
    # ROI is outside co-enriched context even if the other marginal is unknown.
    co_known = (ka & kb) | (ka & ~a) | (kb & ~b)
    co = ka & kb & a & b
    add('coenriched', co_known, co)
    add('coenriched_joint_high', kj, j, co, co_known)
    add('coenriched_joint_not_high', kj, ~j, co, co_known)
    add('one_enriched', ka & kb, a ^ b)
    fields = {}
    for surface, label in [('S', 'source'), ('CE', 'ensemble')]:
        names = [f'correlation_{surface}', f'moran_{surface}_A|B', f'moran_{surface}_B|A']
        known = finite(roi, names)
        values = roi[names].to_numpy()
        limits = np.array([bands.pearson, bands.moran, bands.moran])
        pos, neg = (values > limits).all(axis=1), (values < -limits).all(axis=1)
        fields[surface] = known, pos, neg
        add(label + '_positive', known, pos)
        add(label + '_negative', known, neg)
    both_known = fields['S'][0] & fields['CE'][0]
    for direction, index in [('positive', 1), ('negative', 2)]:
        meets = fields['S'][index] & fields['CE'][index]
        add('source_ensemble_' + direction, both_known, meets)
        add('coenriched_' + direction, both_known, meets, co, co_known)
    names = ['correlation_' + s for s in ['S', 'T', 'CD', 'CE']]
    known = finite(roi, names)
    v = roi[names].to_numpy()
    low, high = roi.correlation_T__low, roi.correlation_T__high
    add('correlation_all_positive', known & np.isfinite(low),
        (v > bands.pearson).all(axis=1) & low.gt(bands.pearson))
    add('correlation_all_negative', known & np.isfinite(high),
        (v < -bands.pearson).all(axis=1) & high.lt(-bands.pearson))
    add('correlation_support_change', known,
        (v > bands.pearson).any(axis=1) & (v < -bands.pearson).any(axis=1))
    for surface in ['S', 'T', 'CD', 'CE']:
        p = 'correlation_' + surface
        for direction, sign in [('positive', 1), ('negative', -1)]:
            add(f'pearson_{surface}_{direction}', finite(roi, [p]), sign * roi[p] > bands.pearson)
            for channel in ['A|B', 'B|A']:
                m = f'moran_{surface}_{channel}'
                add(f'moran_{surface}_{channel.replace("|", "_")}_{direction}', finite(roi, [m]),
                    sign * roi[m] > bands.moran)
    for surface in ['S', 'T', 'CE']:
        known, high = ratio(roi, 'joint_low_' + surface, bands.enrichment)
        add(f'joint_low_{surface}_high', known, high)
        for carrier in ['A', 'B']:
            cles, fc = f'CLES_{surface}_{carrier}', f'log2FC_{surface}_{carrier}'
            delta = roi[cles] - .5
            add(f'amplitude_disagreement_{surface}_{carrier}', finite(roi, [cles, fc]),
                delta.abs().gt(bands.cles) & (delta * roi[fc]).lt(0))
    return pd.concat(rows, ignore_index=True).assign(policy=bands.name)


def component_queries(components, bands):
    if components.duplicated(['sample', 'pair_id', 'roi', 'component_id']).any():
        raise ValueError('Duplicate component identities')
    rows = []
    values = components[[f'shell{i}_value' for i in range(4)]].to_numpy()
    known = np.isfinite(values).all(axis=1) & np.isfinite(components.rho)
    for i in range(4):
        known &= components[f'shell{i}_status'].eq('finite')
    criteria = {
        'radial_positive': components.rho.gt(bands.pearson) & (values > bands.radial).all(axis=1),
        'radial_negative': components.rho.lt(-bands.pearson) & (values < -bands.radial).all(axis=1),
        'radial_sign_change': ((values[:, 0] > bands.radial) & (values[:, 3] < -bands.radial))
                             | ((values[:, 0] < -bands.radial) & (values[:, 3] > bands.radial)),
    }
    for name, meets in criteria.items():
        g = components[['sample', 'pair_id', 'roi', 'component_id', 'parent_target_id']].copy()
        g['target_id'] = g.parent_target_id + '|component:' + g.component_id.astype(str)
        g['unit'] = 'component'
        g['property'] = name
        g['state'] = np.where(known, np.where(meets, 'meets', 'other'), 'unavailable')
        rows.append(g)
    return pd.concat(rows, ignore_index=True).assign(policy=bands.name)


def pair_collections(local, pairs):
    """Counts and full locality sets for each pair, not a strongest-ROI winner."""
    sample = local['sample'].unique()
    policy = local.policy.unique()
    if len(sample) != 1 or len(policy) != 1:
        raise ValueError('One sample/policy per collection')
    groups = dict(tuple(local.groupby(['pair_id', 'property'], sort=False)))
    empty = local.iloc[:0]
    rows = []
    for pair in pairs:
        for prop, _, _, unit, _ in CONTRACT:
            g = groups.get((pair, prop), empty)
            good = g[g.state.eq('meets')]
            rows.append(dict(sample=sample[0], policy=policy[0], pair_id=pair,
                property=prop, unit=unit, n_locations=len(g),
                n_meets=len(good), n_other=int(g.state.eq('other').sum()),
                n_unavailable=int(g.state.eq('unavailable').sum()),
                n_outside_context=int(g.state.eq('outside_context').sum()),
                n_evaluable=int(g.state.isin(['meets', 'other']).sum()),
                target_ids=json.dumps(good.target_id.tolist()),
                roi_ids=json.dumps(sorted(good.roi.astype(str).unique()))))
    return pd.DataFrame(rows)


def coexistence(local, pairs):
    """Opposite patterns require different ROI identities in the same pair/section."""
    rows = []
    sample, policy = local['sample'].iloc[0], local.policy.iloc[0]
    for pair in pairs:
        g = local[local.pair_id.eq(pair)]
        plus = g[g['property'].eq('source_ensemble_positive')]
        minus = g[g['property'].eq('source_ensemble_negative')]
        pos = set(plus.loc[plus.state.eq('meets'), 'roi'].astype(str))
        neg = set(minus.loc[minus.state.eq('meets'), 'roi'].astype(str))
        if pos & neg:
            raise ValueError('One ROI cannot meet both signed field combinations')
        evaluated = plus.loc[plus.state.isin(['meets', 'other']), 'roi'].nunique()
        rows.append(dict(sample=sample, pair_id=pair, policy=policy,
            n_evaluable_rois=evaluated, can_assess_between_roi_variation=evaluated >= 2,
            positive_roi_ids=json.dumps(sorted(pos)), negative_roi_ids=json.dumps(sorted(neg)),
            coexistence_observed=bool(pos and neg)))
    return pd.DataFrame(rows)


def boundary_records(boundary, roi):
    """Keep raw program-specific magnitudes and actual endpoints; no magnitude gate.

    Source/ensemble sign agreement is explicitly ONLY directional agreement.
    Linking a responder field reads the responder's ROI, never the anchor's.
    """
    b = boundary[boundary.adjacent].copy()
    keys = ['sample', 'pair_id', 'anchor_roi_id', 'responder_roi_id']
    if b.duplicated(keys + ['surface']).any():
        raise ValueError('Duplicate interface/surface identity')
    outputs = []
    for carrier in ['A', 'B']:
        parts = []
        columns = ['interface_response', 'availability_status', 'source_near_mass',
                   'source_far_mass', 'represented_near_mass', 'represented_far_mass',
                   'reachable_responder_fraction']
        for surface, short in [('source_native', 'S'), ('tessellation', 'T'),
                               ('consensus_direct', 'CD'), ('consensus_ensemble', 'CE')]:
            g = b[b.surface.eq(surface)]
            part = g[keys + [c + '_' + carrier for c in columns]].rename(
                columns={c + '_' + carrier: c + '_' + short for c in columns})
            parts.append(part.set_index(keys))
        joined = pd.concat(parts, axis=1).reset_index()
        joined['carrier'] = carrier
        joined['program'] = joined.pair_id.str.split('__').str[0 if carrier == 'A' else 1]
        joined['target_id'] = (joined['sample'] + '|' + joined.pair_id + '|interface:'
                               + joined.anchor_roi_id.astype(str) + '>' + joined.responder_roi_id.astype(str))
        outputs.append(joined)
    result = pd.concat(outputs, ignore_index=True)
    for role in ['anchor', 'responder']:
        wanted = ['sample', 'pair_id', 'roi', 'correlation_S', 'correlation_CE',
                  'moran_S_A|B', 'moran_S_B|A', 'moran_CE_A|B', 'moran_CE_B|A',
                  'enrichment_S_A', 'enrichment_S_B', 'enrichment_S_AB',
                  'enrichment_S_A__kind', 'enrichment_S_B__kind']
        context = roi[wanted].rename(columns={'roi': role + '_roi_id',
                   **{c: role + '_' + c for c in wanted if c not in ['sample', 'pair_id', 'roi']}})
        result = result.merge(context, on=['sample', 'pair_id', role + '_roi_id'],
                              validate='many_to_one', how='left', indicator=True)
        if not result['_merge'].eq('both').all():
            raise ValueError('Boundary endpoint has no matching ROI')
        result = result.drop(columns='_merge')
    valid = result.availability_status_S.eq('finite') & result.availability_status_CE.eq('finite')
    valid &= np.isfinite(result.interface_response_S) & np.isfinite(result.interface_response_CE)
    s, e = result.interface_response_S, result.interface_response_CE
    result['source_ensemble_direction'] = np.where(~valid, 'unavailable',
        np.where((s > 0) & (e > 0), 'both_positive',
        np.where((s < 0) & (e < 0), 'both_negative', 'zero_or_different_signs')))
    result['interpretation'] = 'raw program-score response; directional agreement is not calibrated effect evidence'
    return result


def topology_inventory(topology, roi):
    """Label geometry counted once per ROI, NEVER copied into pair detections."""
    rows = []
    for (sample, label), g in topology.groupby(['sample', 'population_roi']):
        parent = roi[roi['sample'].eq(sample) & roi.roi.eq(str(label))]
        if parent.empty or parent.source_spots.nunique() != 1:
            raise ValueError('Topology lacks a consistent parent ROI')
        if int(g.source_spots.sum()) != int(parent.source_spots.iloc[0]):
            raise ValueError('Topology source membership count differs from parent ROI')
        rows.append(dict(sample=sample, roi=str(label), source_spots=int(g.source_spots.sum()),
            n_label_components=len(g), largest_component_spots=int(g.source_spots.max()),
            n_tessellation_eligible=int(g.n_tessellation_eligible.sum()),
            source_only=bool(g.n_tessellation_eligible.sum() == 0),
            interpretation='label geometry inventory only; neither embedded nor dispersed pair phenotype established'))
    return pd.DataFrame(rows)
