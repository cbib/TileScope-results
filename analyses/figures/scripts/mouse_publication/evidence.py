"""Read the frozen mouse corpus; verify, but never rerun, scientific analyses."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from analyses.figures.common import rebuild_bands, spot_label_outlines
from tilescope_results.paths import input_data_root, supplementary_data_root
from tilescope_results.bundles import verify

FOCUS = {'MCT1_CTL': (2, 5), 'MCT1_KO': (4, 9), 'V4_WT': (10, 2)}
PRIMARY = 'cell_cycle_proliferation__myeloid_phagocytic'
IFN_PAIR = 'cell_cycle_proliferation__interferon_related'
V4_PAIR = 'myeloid_phagocytic__v4_reactive_interface'
MES_PAIR = 'mesenchymal_reactive_stromal__myeloid_phagocytic'
NAMES = {
    'cell_cycle_proliferation': 'Proliferation',
    'myeloid_phagocytic': 'Myeloid/phagocytic',
    'interferon_related': 'Interferon-related',
    'antigen_presenting_myeloid': 'Antigen presentation',
    'cytotoxic_t_nk': 'Cytotoxic T/NK',
    'mesenchymal_reactive_stromal': 'Mesenchymal/stromal',
    'v4_reactive_interface': 'Reactive interface',
    'gbm_tumour_compartment_locator': 'Tumour-associated',
    'ecm_remodelling': 'ECM remodelling',
    'hypoxia_integrated_stress': 'Stress-related',
    'oligodendroglial_neural_like': 'Oligodendroglial',
    'inflammation': 'Inflammation', 'innate_immune': 'Innate immune',
    'endothelial_vascular': 'Endothelial/vascular',
}
TABLE_KEYS = {
    'roi_correlations': ['roi_id'],
    'roi_tessellation_correlations': ['roi_id'],
    'roi_replicate_correlations': ['roi_id', 'replicate_index'],
    'roi_enrichment': ['roi_id'],
    'roi_tessellation_enrichment': ['roi_id'],
    'roi_replicate_enrichment': ['roi_id', 'replicate_index'],
    'roi_consensus_correlations': ['roi_id', 'value_surface'],
    'roi_consensus_enrichment': ['roi_id', 'value_surface'],
    'roi_source_moran': ['roi_id', 'moran_direction'],
    'roi_tessellation_moran': ['roi_id', 'moran_direction'],
    'roi_consensus_moran': ['roi_id', 'value_surface', 'moran_direction'],
    'component_radial_shell_correlogram': ['roi_id', 'component_id', 'shell_bin_index'],
    'roi_pair_interface_response': ['anchor_roi_id', 'responder_roi_id', 'responder_feature_id', 'surface'],
    'roi_cross_grain_sensitivity': ['roi_id', 'statistic', 'channel', 'moran_direction', 'reference_surface', 'target_surface'],
}


def close(a, b, label):
    try:
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10, equal_nan=True)
    except AssertionError as exc:
        raise ValueError(f'{label}: saved values disagree') from exc


def one(frame, **keys):
    selected = frame
    for key, value in keys.items():
        selected = selected.loc[selected[key].eq(value)]
    if len(selected) != 1:
        raise ValueError(f'Expected one row for {keys}; found {len(selected)}')
    return selected.iloc[0]


def pearson(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


class Evidence:
    def __init__(self):
        self.analysis = input_data_root() / 'mouse'
        self.evidence = supplementary_data_root(4)
        verify(input_data_root())
        verify(self.evidence)
        self.inputs = set()
        self.checks = []
        self.spatial_runs = {}
        self.band_runs = {}
        self.displays = []
        self.samples = self.read(self.analysis / 'samples.csv')
        self.roles = self.read(self.analysis / 'roi_roles.csv')
        contract_path = self.analysis / 'signatures_harmonized.yaml'
        self.inputs.add(contract_path)
        self.contract = yaml.safe_load(contract_path.read_text())
        self.manifest = self.read(self.analysis / 'pairs.csv')
        self.banks = {s: self.read(self.analysis / 'scores' / f'{s}.csv.gz') for s in FOCUS}
        self.selection = self.read(self.evidence / 'selected_pairs.csv')
        self.tables = {}
        self.metadata = self.json(self.evidence / 'original_run_metadata.json')
        self.config = self.json(self.analysis / 'resolved_scientific_config.json')
        self.run_inventory = []
        self._load()

    def read(self, path):
        path = Path(path)
        self.inputs.add(path)
        return pd.read_csv(path, float_precision='round_trip', low_memory=False)

    def json(self, path):
        self.inputs.add(Path(path))
        return json.loads(Path(path).read_text())

    def table(self, name, sample, pair):
        d = self.tables[name]
        return d.loc[d['sample'].eq(sample) & d.pair.eq(pair)]

    def _load(self):
        counts = self.manifest.groupby('sample_dir').size().to_dict()
        if counts != {'MCT1_CTL': 78, 'MCT1_KO': 78, 'V4_WT': 66}:
            raise ValueError(f'Unexpected pair corpus: {counts}')
        collected = {name: [] for name in TABLE_KEYS}
        saved = {name: self.read(self.evidence / 'measurements' / f'{name}.csv.gz') for name in TABLE_KEYS}
        rows = []
        for record in self.manifest.itertuples():
            sample = record.sample_dir
            a, b = record.signature_a, record.signature_b
            pair = f'{a}__{b}'
            run = f'{sample}/{pair}'
            meta = self.metadata[run]
            identity = meta['feature_identity']
            if (identity['feature_g1_id'], identity['feature_g2_id']) != (a, b):
                raise ValueError(f'Feature identity mismatch: {run}')
            frames = {}
            for name, keys in TABLE_KEYS.items():
                f = saved[name].loc[saved[name]['sample'].eq(sample) & saved[name].pair.eq(pair)].copy()
                if f.empty:
                    raise ValueError(f'Missing pair measurements: {run}/{name}')
                if f.duplicated(keys).any():
                    raise ValueError(f'Duplicate scientific row: {sample}/{pair}/{name}')
                if not f.empty and ('g1_id' in f) and (set(f.g1_id) != {a} or set(f.g2_id) != {b}):
                    raise ValueError(f'Wrong feature columns: {run}/{name}')
                frames[name] = f
                collected[name].append(f.assign(sample=sample, pair=pair))
            source = frames['roi_correlations']
            if len(frames['roi_cross_grain_sensitivity']) != 14 * len(source):
                raise ValueError(f'Cross-grain row closure: {run}')
            bank = self.banks[sample]
            if set(source.roi_id) != set(bank.roi_id):
                raise ValueError(f'Lost parent ROI: {run}')
            for src in source.itertuples():
                roi = src.roi_id
                spots = bank.loc[bank.roi_id.eq(roi)]
                if len(spots) != src.n_roi:
                    raise ValueError(f'Wrong source support count: {run}/{roi}')
                r = pearson(spots[a + '__score'], spots[b + '__score'])
                if src.parent_correlation_status == 'finite':
                    close(r, src.corr_obs, f'{sample}/{pair}/{roi} source Pearson')
                t = one(frames['roi_tessellation_correlations'], roi_id=roi)
                c = one(frames['roi_consensus_correlations'], roi_id=roi, value_surface='consensus_ensemble')
                e = one(frames['roi_enrichment'], roi_id=roi)
                rows.append(dict(sample=sample, pair=pair, roi=roi, program_A=a, program_B=b,
                    n_spots=len(spots), n_hexagons=c.sample_size, source_r=src.corr_obs,
                    source_status=src.parent_correlation_status, source_q=src.corr_q_perm,
                    tess_r=t.corr_median, tess_q25=t.corr_q25, tess_q75=t.corr_q75,
                    tess_status=t.parent_tessellation_correlation_status,
                    n_valid_tessellations=t.n_valid_replicates, n_planned_tessellations=t.n_planned_replicates,
                    cons_r=c.correlation_value, cons_status=c.parent_consensus_correlation_status,
                    cons_q_analytic=c.corr_q_analytic,
                    raw_source_r=pearson(spots[a + '__raw_mean'], spots[b + '__raw_mean']),
                    enrich_A=e.g1_enrichment, enrich_B=e.g2_enrichment,
                    enrich_A_status=e.g1_enrichment_status, enrich_B_status=e.g2_enrichment_status,
                    enrichment_status=e.parent_enrichment_status, joint_high=e.joint_high_enrichment_fc))
            self.run_inventory.append(dict(sample=sample, pair=pair, n_rois=len(source),
                core_version=meta['tilescope_version'], policy_version=meta['analysis_policy_version'],
                original_run_timestamp=meta['timestamp'], n_tessellations=meta['tessellation']['n_tessellations']))
        self.tables = {name: pd.concat(parts, ignore_index=True) for name, parts in collected.items()}
        self.roi = pd.DataFrame(rows)
        if len(self.roi) != 2064:
            raise ValueError('Unexpected full ROI inventory')
        old = self.read(self.evidence / 'all_pair_roi_summary.csv')
        keys = ['sample', 'pair', 'roi']
        new, old = self.roi.set_index(keys).sort_index(), old.set_index(keys).sort_index()
        if not new.index.equals(old.index):
            raise ValueError('Saved evidence and current corpus identities differ')
        columns = ['source_r', 'tess_r', 'cons_r', 'raw_source_r', 'enrich_A', 'enrich_B']
        close(new[columns], old[columns], 'All evidence-screen values')
        self.checks.append(dict(check='current_corpus_and_screen_parity', count=len(new), status='pass'))
        self.selected_roi = self.roi.merge(self.selection, on=['sample', 'pair'], validate='many_to_one')
        self.focus_roi = self.selected_roi.loc[
            [r.roi in (*FOCUS[r.sample], 1) if r.sample == 'V4_WT' else r.roi in FOCUS[r.sample]
             for r in self.selected_roi.itertuples()]].copy()

    def spatial(self, sample, pair):
        key = sample, pair
        if key in self.spatial_runs:
            return self.spatial_runs[key]
        folder = self.evidence / 'source_spots'
        a, b = pair.split('__')
        bank = self.banks[sample]
        frame = bank[['x', 'y', a + '__score', b + '__score', 'roi_id']].rename(
            columns={a + '__score': 'gene1', b + '__score': 'gene2'})
        ledger = self.read(folder / f'{sample}_spot_support.csv.gz').sort_values('spot_index').reset_index(drop=True)
        if not np.array_equal(ledger.spot_index, np.arange(len(frame))):
            raise ValueError('Source spot indices are not complete')
        if not np.array_equal(ledger.roi_id, frame.roi_id):
            raise ValueError('Spot/score/ledger identities differ')
        frame = frame.assign(spot_index=np.arange(len(frame)), spot_id=bank.spot_id,
            component_id=ledger.component_id, tessellation_eligible=ledger.tessellation_eligible)
        hexes = self.read(self.evidence / 'selected_consensus_maps' / f'{sample}__{pair}.csv.gz')
        geometry = self.read(folder / f'{sample}_component_geometry.csv.gz')
        if hexes.duplicated(['roi_id', 'component_id', 'hex_id']).any():
            raise ValueError('Duplicate consensus unit identity')
        for roi, h in hexes.groupby('roi_id'):
            c = one(self.table('roi_consensus_correlations', sample, pair), roi_id=roi, value_surface='consensus_ensemble')
            values = h[['gene1_density_ensemble', 'gene2_density_ensemble']].dropna()
            if c.parent_consensus_correlation_status == 'finite':
                if len(values) != c.sample_size:
                    raise ValueError('Consensus plotted support differs from statistic support')
                close(pearson(values.iloc[:, 0], values.iloc[:, 1]), c.correlation_value, 'Plotted consensus Pearson')
        radial = self.table('component_radial_shell_correlogram', sample, pair)
        for cid, r in radial.groupby('component_id'):
            if r.component_radial_shell_status.eq('finite').any():
                spots = frame.loc[frame.component_id.eq(cid)]
                close(pearson(spots.gene1, spots.gene2), r.component_rho_0.iloc[0], 'Component source Pearson')
        spacing = float(np.median(cKDTree(frame[['x', 'y']]).query(frame[['x', 'y']], k=2)[0][:, 1]))
        # Outlines are display footprints, not replacements for saved analysis domains.
        outlines = spot_label_outlines(frame, spacing, list(frame.roi_id.unique()))
        self.spatial_runs[key] = dict(sample=sample, pair=pair, names=(a, b), frame=frame,
            bank=bank, hexes=hexes, geometry=geometry, ledger=ledger, spacing=spacing, outlines=outlines,
            config=self.config)
        self.checks.append(dict(check='plotted_support_and_coefficients', sample=sample, pair=pair, status='pass'))
        return self.spatial_runs[key]

    def bands(self, sample, pair):
        key = sample, pair
        if key in self.band_runs:
            return self.band_runs[key]
        run = self.spatial(sample, pair)
        anchor, responder = FOCUS[sample]
        frame = run['frame']
        bands, hops, _ = rebuild_bands(frame.assign(roi_id=frame.roi_id.astype(str)),
                                     run['config'], str(anchor), str(responder))
        for i, program in enumerate(run['names'], 1):
            row = one(self.table('roi_pair_interface_response', sample, pair),
                anchor_roi_id=anchor, responder_roi_id=responder,
                responder_feature_id=program, surface='source_native')
            if row.availability_status != 'finite':
                raise ValueError('Selected boundary is unavailable')
            near, far = [frame.loc[bands == b, f'gene{i}'] for b in ('near', 'far')]
            close([len(near), len(far)], [row.source_near_mass, row.source_far_mass], 'Near/far spot counts')
            close(near.mean() - far.mean(), row.interface_response, 'Rebuilt boundary score difference')
        self.band_runs[key] = frame.assign(sample=sample, pair=pair, anchor_roi_id=anchor,
                                         responder_roi_id=responder, band=bands, hops=hops)
        self.checks.append(dict(check='boundary_bands_and_source_response', sample=sample, pair=pair, status='pass'))
        return self.band_runs[key]

    def boundaries(self):
        f = self.tables['roi_pair_interface_response'].merge(self.selection, on=['sample', 'pair'])
        f = f.loc[[r.anchor_roi_id == FOCUS[r.sample][0] and r.responder_roi_id == FOCUS[r.sample][1]
                   for r in f.itertuples()]].copy()
        keys = ['sample', 'anchor_roi_id', 'responder_roi_id', 'responder_feature_id', 'surface']
        # These are feature measurements repeated in pair-shaped exports. Verify before deduplicating.
        for _, group in f.groupby(keys):
            for col in ['interface_response', 'source_near_mass', 'source_far_mass', 'represented_near_mass', 'represented_far_mass']:
                close(group[col], np.repeat(group[col].iloc[0], len(group)), f'Pair-independent {col}')
            if group.availability_status.nunique() != 1:
                raise ValueError('Pair contexts disagree on feature availability')
        return f.drop_duplicates(keys).drop(columns=['pair', 'g1_id', 'g2_id'])

    def satellite(self):
        run = self.spatial('MCT1_KO', IFN_PAIR)
        bank = run['bank'].assign(component_id=run['ledger'].component_id)
        rows = []
        for cid, expected in [(6, 243), (7, 49)]:
            d = bank.loc[bank.component_id.eq(cid)]
            if len(d) != expected or set(d.roi_id) != {4}:
                raise ValueError('KO component identity changed')
            rows.append(dict(sample='MCT1_KO', roi_id=4, component_id=cid, n_spots=len(d),
                source_r=pearson(d.cell_cycle_proliferation__score, d.interferon_related__score),
                raw_source_r=pearson(d.cell_cycle_proliferation__raw_mean, d.interferon_related__raw_mean),
                mean_tumour_score=d.gbm_tumour_compartment_locator__score.mean(),
                mean_proliferation_score=d.cell_cycle_proliferation__score.mean()))
        return pd.DataFrame(rows)
