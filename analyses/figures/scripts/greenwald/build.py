"""Greenwald manuscript figures from the retained canonical exports.

The scope is 30 GBM samples, 91 pairs and 192 parent ROIs. Every comparison
keeps its parent ROI, component, or ordered-interface identity. Rendering does
not run TileScope. See README.md for the figure outputs and reproduction commands.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analyses/greenwald_full_evaluation'))
sys.path.insert(0, str(ROOT / 'analyses/gbm_spatial_repertoire'))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'tilescope_figure_mpl'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import Normalize, ListedColormap
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import wkt

from tilescope_results.paths import input_data_root, supplementary_data_root
from tilescope_results.bundles import verify
from tilescope.viz.painters import SpotSplatPainter, draw_polygon_boundaries
from figure_common import Inputs, one, polygon_patch
from analyses.figures.common import rebuild_bands, BANDS
from vet_shortlist import PAIRS, typed, signed_summary
from analyses.figures.common import spot_label_outlines, expression_colormap

RESULTS = supplementary_data_root(2)
INPUTS = input_data_root() / 'greenwald'
SURFACES = {'S': 'Spots', 'T': 'Tessellation', 'CE': 'Consensus'}
POS, NEG = '#B94B40', '#367FA1'
CASES = [
    ('259_T', 'AC_Reactive-Ast', ('2', '3')),
    ('259_T', 'Mac_Reactive-Ast', ('2', '3')),
    ('259_T', 'MES-Hyp_Reactive-Ast', ('2', '3')),
    ('262_T', 'Neuron_NPC', ('4', '5')),
    ('260_T', 'MES-Hyp_MES-Ast', ('2', '4')),
]


def check_close(actual, expected, context):
    try:
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
    except AssertionError as exc:
        raise ValueError(f'{context}: saved evidence differs') from exc


def read_csv(path):
    return pd.read_csv(path, float_precision='round_trip', low_memory=False,
                       dtype={c: str for c in ('roi', 'roi_id', 'component_id',
                                              'anchor_roi_id', 'responder_roi_id')})


def verify_boundary(run, bands, stored):
    """Match semantic feature identities to the corresponding source columns."""
    checks=[]
    for i, program in enumerate(run['names']):
        row=one(stored,surface='source_native',responder_feature_id=program)
        if row.availability_status!='finite':raise ValueError('Selected boundary is unavailable')
        near,far=[run['frame'].loc[bands==b,f'gene{i+1}'] for b in ('near','far')]
        if (len(near),len(far))!=(row.source_near_mass,row.source_far_mass):
            raise ValueError('Boundary bands disagree with stored support counts')
        value=float(near.mean()-far.mean())
        check_close(value,row.interface_response,'boundary near-minus-far')
        reachable=1-np.sum(bands=='unreachable')/np.isin(bands,['near','buffer','far','unreachable']).sum()
        check_close(reachable,row.reachable_responder_fraction,'boundary reachability')
        checks.append(dict(program=program,n_near=len(near),n_far=len(far),
                           rebuilt_source_response=value,stored_source_response=row.interface_response))
    return checks


class Evidence:
    def __init__(self):
        self.inputs = Inputs()
        verify(input_data_root())
        verify(RESULTS)
        scope = self.read(RESULTS / 'sample_scope.csv')
        self.samples = scope.loc[scope.included_this_build, 'sample'].tolist()
        self.roi = self.read(RESULTS / 'all_roi_measurements.csv.gz')
        if (len(self.samples), self.roi.pair_id.nunique(), len(self.roi)) != (30, 91, 17472):
            raise ValueError('Unexpected GBM scope')
        if self.roi.duplicated(['sample', 'pair_id', 'roi']).any():
            raise ValueError('Duplicate parent ROI measurements')
        self.summary = self.read(RESULTS / 'figure_evidence/statistical_summary.csv')
        self.matched = self.read(RESULTS / 'figure_evidence/matched_support.csv')
        self.runs = {}
        self.checks = []
        self.boundaries = {}
        self.map_settings = []

    def read(self, path):
        path = Path(path)
        self.inputs.paths.add(path.resolve())
        return read_csv(path)

    def validate_corpus(self):
        # Recreate the displayed Pearson frequencies from all saved ROI rows.
        selected = self.roi[self.roi.pair_id.isin(PAIRS)]
        for surface in ('S', 'CE'):
            for band in (.1, .2, .3):
                fresh = signed_summary(selected, typed(selected, 'correlation_' + surface),
                                       band, metric='correlation', surface=surface)
                old = self.summary[self.summary.metric.eq('correlation') &
                                   self.summary.surface.eq(surface) & self.summary.context.eq('all') &
                                   self.summary.threshold.abs().eq(band)]
                keys = ['pair_id', 'sign']
                a = fresh.set_index(keys).sort_index()
                b = old.set_index(keys).sort_index()
                if not a.index.equals(b.index):
                    raise ValueError('Cohort-summary identities differ')
                check_close(a[['frequency', 'qualifying_median', 'n_evaluable']],
                            b[['frequency', 'qualifying_median', 'n_evaluable']], 'cohort frequencies')
        self.checks.append(dict(check='rederived_display_frequencies', count=84, status='pass'))

    def run(self, sample, pair):
        if (sample, pair) in self.runs:
            return self.runs[sample, pair]
        folder = RESULTS / 'spatial' / sample / pair
        meta = self.inputs.json(folder / 'run_metadata.json')
        identities = meta['feature_identity']
        names = [identities['feature_g1_id'], identities['feature_g2_id']]
        if names != pair.split('_'):
            raise ValueError('Run folder and semantic feature identity differ')
        bank = self.read(INPUTS / 'scores' / f'{sample}.csv.gz')
        frame = bank[['x', 'y', names[0], names[1], 'roi_id']].rename(columns={names[0]: 'gene1', names[1]: 'gene2'})
        ledger = self.read(folder.parent / 'spot_ledger.csv.gz')
        if not np.array_equal(ledger.spot_index, np.arange(len(frame))) or not ledger.roi_id.equals(frame.roi_id):
            raise ValueError('Source rows and ledger membership differ')
        run = dict(sample=sample, pair=pair, names=names, folder=folder, frame=frame, ledger=ledger,
                   config=self.inputs.json(INPUTS / 'resolved_config.json'), meta=meta)
        run['geometry'] = self.read(folder.parent / 'component_geometry.csv.gz')
        for key, filename in [('hexes', 'consensus_hexes.csv.gz'),
                              ('radial', 'component_radial_shell_correlogram.csv.gz'),
                              ('interfaces', 'roi_pair_interface_response.csv.gz'),
                              ('source', 'roi_correlations.csv.gz'), ('consensus', 'roi_consensus_correlations.csv.gz'),
                              ('enrichment', 'roi_enrichment.csv.gz')]:
            run[key] = self.read(folder / filename)
        run['spacing'] = float(np.median(cKDTree(frame[['x', 'y']]).query(frame[['x', 'y']], k=2)[0][:, 1]))
        run['outlines'] = spot_label_outlines(frame, run['spacing'], frame.roi_id.unique().tolist())
        run['limits'] = {p: max(float(np.abs(frame[f'gene{i+1}']).quantile(.98)), 1e-12)
                         for i, p in enumerate(names)}
        pair_id = '__'.join(sorted(names))
        for roi in frame.roi_id.unique():
            record = one(self.roi, sample=sample, pair_id=pair_id, roi=roi)
            s = one(run['source'], roi_id=roi)
            c = one(run['consensus'], roi_id=roi, value_surface='consensus_ensemble')
            check_close(s.corr_obs, record.correlation_S, f'{sample}/{pair}/{roi} source')
            check_close(c.correlation_value, record.correlation_CE, f'{sample}/{pair}/{roi} consensus')
            spots = frame[frame.roi_id.eq(roi)]
            if s.parent_correlation_status == 'finite':
                check_close(np.corrcoef(spots.gene1, spots.gene2)[0, 1], s.corr_obs, 'source Pearson')
            if c.parent_consensus_correlation_status == 'finite':
                h = run['hexes'].loc[run['hexes'].roi_id.eq(roi),
                                    ['gene1_density_ensemble', 'gene2_density_ensemble']].dropna()
                if len(h) != c.sample_size:
                    raise ValueError('Plotted finite hex support differs from correlation support')
                check_close(np.corrcoef(h.to_numpy().T)[0, 1], c.correlation_value, 'consensus Pearson')
            enriched = one(run['enrichment'], roi_id=roi)
            if enriched.parent_enrichment_status == 'finite':
                for i, p in enumerate(names):
                    carrier = 'A' if p == record.program_A else 'B'
                    if enriched[f'g{i+1}_enrichment_status'] == 'finite':
                        check_close(enriched[f'g{i+1}_enrichment'], record[f'enrichment_S_{carrier}'], 'enrichment')
        for cid, radial in run['radial'].groupby('component_id'):
            finite = radial.component_radial_shell_status.eq('finite')
            if finite.any():
                mask = ledger.component_id.eq(cid)
                check_close(np.corrcoef(frame.loc[mask, ['gene1', 'gene2']].to_numpy().T)[0, 1],
                            radial.component_rho_0.iloc[0], 'component zero-lag Pearson')
                if ledger.loc[mask, 'roi_id'].nunique() != 1:
                    raise ValueError('Component spans parent ROIs')
        self.checks.append(dict(check='selected_run_values_and_geometry', sample=sample, pair=pair,
                                count=len(frame), status='pass'))
        self.runs[sample, pair] = run
        return run

    def row(self, sample, pair, roi):
        return one(self.roi, sample=sample, pair_id='__'.join(sorted(pair.split('_'))), roi=roi)

    def bands(self, sample, pair, anchor, responder):
        key = (sample, pair, anchor, responder)
        if key not in self.boundaries:
            run = self.run(sample, pair)
            bands, hops, edges = rebuild_bands(run['frame'], run['config'], anchor, responder)
            stored = run['interfaces']
            stored = stored[stored.anchor_roi_id.eq(anchor) & stored.responder_roi_id.eq(responder)]
            checks = verify_boundary(run, bands, stored)
            self.checks.extend(dict(check='boundary_bands', sample=sample, pair=pair,
                                    anchor=anchor, responder=responder, status='pass', **r) for r in checks)
            self.boundaries[key] = (bands, hops, edges, stored)
        return self.boundaries[key]

    def prepare(self):
        self.validate_corpus()
        for sample, pair, rois in CASES:
            self.run(sample, pair)
        # The same Reactive-Ast program is used across the three displayed pairs.
        runs = [self.run('259_T', p) for p in ('AC_Reactive-Ast', 'Mac_Reactive-Ast', 'MES-Hyp_Reactive-Ast')]
        for run in runs[1:]:
            pd.testing.assert_frame_equal(runs[0]['frame'][['x', 'y', 'roi_id', 'gene2']],
                                          run['frame'][['x', 'y', 'roi_id', 'gene2']])
            a = runs[0]['hexes'].set_index(['roi_id', 'component_id', 'hex_id']).sort_index()
            b = run['hexes'].set_index(['roi_id', 'component_id', 'hex_id']).sort_index()
            if not a.index.equals(b.index):
                raise ValueError('The displayed shared program has different hex supports')
            check_close(a.gene2_density_ensemble, b.gene2_density_ensemble, 'shared Reactive-Ast hex values')
        for a, r in [('4', '2'), ('2', '4')]:
            self.bands('260_T', 'MES-Hyp_MES-Ast', a, r)
        # Preserve the full numerical inventory without plotting 91 pair rows.
        self.all_pair_summary = self.read(RESULTS / 'all_pair_frequencies.csv.gz')
        print(f'Validated {len(self.roi):,} ROI-pair profiles and {len(self.runs)} selected runs.', flush=True)


def tidy(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=7, length=2)


def map_draw(e, sample, pair, program, surface, rois, *, title=None):
    def draw(fig, rect):
        run = e.run(sample, pair)
        f = run['frame']
        field = f'gene{run["names"].index(program)+1}'
        # Common full-sample viewport and program-specific scale across supports.
        xy = f[['x', 'y']].to_numpy()
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        pad = 2 * run['spacing']
        x, y, w, h = rect
        ax = fig.add_axes([x + .035*w, y + .20*h, .93*w, .69*h])
        ax.set(xlim=(lo[0]-pad, hi[0]+pad), ylim=(lo[1]-pad, hi[1]+pad), aspect='equal')
        ax.axis('off')
        fig.canvas.draw()
        limit = run['limits'][program]
        norm = Normalize(-limit, limit)
        cmap = expression_colormap()
        if surface == 'S':
            artist = SpotSplatPainter(.40).draw(ax, xy, f[field].to_numpy(), cmap=cmap, norm=norm,
                                               canvas=None, median_nn_override=run['spacing'])
            if not np.array_equal(artist.get_array(), f[field].to_numpy()):
                raise ValueError('Source painter changed expression values')
            ids, values = np.arange(len(f)), f[field].to_numpy()
        else:
            SpotSplatPainter(.32).draw(ax, xy, np.zeros(len(f)), cmap=ListedColormap(['#E4E7EA']),
                                      norm=Normalize(0, 1), canvas=None, median_nn_override=run['spacing'])
            hexes = run['hexes']
            patches, values, ids = [], [], []
            for i, r in hexes.iterrows():
                value = r[field + '_density_ensemble']
                if not np.isfinite(value):
                    continue
                geom = wkt.loads(r.hex_geometry_wkt)
                for poly in geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]:
                    patches.append(polygon_patch(poly)); values.append(value); ids.append(i)
            artist = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor='#9DA6AA', linewidth=.12)
            artist.set_array(np.asarray(values)); ax.add_collection(artist)
        outlines = {r: run['outlines'][r] for r in rois}
        draw_polygon_boundaries(ax, outlines, color='#363F45', linewidth=.55, alpha=.95)
        for roi, geom in outlines.items():
            parts = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
            centre = max(parts, key=lambda p: p.area).representative_point()
            ax.text(centre.x, centre.y, roi, ha='center', va='center', fontsize=6.6,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.87, pad=.7))
        fig.text(x+w/2, y+h*.955, title or program, ha='center', va='top', fontsize=8)
        fig.text(x+w/2, y+h*.08, SURFACES[surface], ha='center', fontsize=6.6, color='#56616A')
        cax = fig.add_axes([x+w*.20, y+h*.15, w*.60, h*.023])
        cb = fig.colorbar(artist, cax=cax, orientation='horizontal', ticks=[-limit, 0, limit])
        cb.ax.set_xticklabels([f'{-limit:.1f}', '0', f'{limit:.1f}'])
        cb.ax.tick_params(labelsize=5.8, length=1, pad=1)
        cb.outline.set_visible(False)
        e.map_settings.append(dict(sample=sample, pair=pair, program=program, surface=surface,
            rois=';'.join(rois), vmin=-limit, vmax=limit, displayed_values=len(ids),
            bounds=[float(lo[0]-pad), float(hi[0]+pad), float(lo[1]-pad), float(hi[1]+pad)]))
        return ax
    return draw


def axes(fig, rect, title, left=.20, bottom=.24):
    x, y, w, h = rect
    ax = fig.add_axes([x+left*w, y+bottom*h, w*(.96-left), h*(.86-bottom)])
    fig.text(x+w/2, y+h*.96, title, ha='center', va='top', fontsize=8)
    tidy(ax)
    return ax


def cohort_territory(e):
    def draw(fig, rect):
        ax = axes(fig, rect, 'Territorial enrichment', left=.42, bottom=.20)
        selected = e.summary[e.summary.context.eq('all') & e.summary.surface.eq('S') & e.summary.sign.eq(1)]
        for i, pair in enumerate(PAIRS):
            g = selected[selected.pair_id.eq(pair)].set_index('metric')
            both, one_only = [g.loc[k, 'frequency'] for k in ('both_enriched', 'one_enriched')]
            left = 0
            for val, color in [(both, '#487E96'), (one_only, '#CDA565'), (1-both-one_only, '#E5E9EB')]:
                ax.barh(i, val, left=left, color=color, height=.57, linewidth=0); left += val
            ax.text(both/2, i, f'{100*both:.0f}', ha='center', va='center', color='white', fontsize=6.7)
        ax.set(ylim=(6.7, -.7), xlim=(0,1), yticks=range(7),
               yticklabels=[p.replace('__',' / ') for p in PAIRS], xticks=[0,.5,1], xticklabels=['0','50','100%'])
        ax.tick_params(axis='y', length=0, labelsize=7)
        fig.text(rect[0]+rect[2]*.70, rect[1]+rect[3]*.045, 'Both / one / neither enriched', ha='center', fontsize=6.4)
    return draw


def cohort_association(e, specs=None, title='Within-ROI association', labels=None, show_names=False):
    specs = specs or [('correlation', 'S', .2, 1), ('correlation','S',-.2,-1),
                      ('correlation','CE',.2,1), ('correlation','CE',-.2,-1)]
    labels = labels or ['Spots\n+', 'Spots\n−', 'Consensus\n+', 'Consensus\n−']
    def draw(fig, rect):
        ax = axes(fig, rect, title, left=.42 if show_names else .10, bottom=.20)
        d = e.summary[e.summary.context.eq('all')]
        for j, (metric, surface, threshold, sign) in enumerate(specs):
            for i, pair in enumerate(PAIRS):
                r = one(d, pair_id=pair, metric=metric, surface=surface, threshold=threshold, sign=sign)
                if r.frequency > 0:
                    ax.scatter(j, i, s=155*r.frequency, color=POS if sign==1 else NEG)
        ax.set(ylim=(6.7,-.7), xlim=(-.6,len(specs)-.4),
               yticks=range(7) if show_names else [], xticks=range(len(specs)), xticklabels=labels)
        if show_names:
            ax.set_yticklabels([p.replace('__',' / ') for p in PAIRS], fontsize=6.3)
            ax.tick_params(axis='y',length=0)
        ax.spines[['left','bottom']].set_visible(False)
        for n in (0.1, 0.3, 0.6):
            ax.scatter([], [], s=155*n, color='#6B7B83', label=f'{100*n:.0f}%')
        ax.legend(frameon=False, fontsize=5.8, ncol=3, loc='upper center', bbox_to_anchor=(.5,-.25),
                  handletextpad=.2, columnspacing=.6, borderpad=0)
    return draw


def association_values(e, sample, pairs, rois, title):
    def draw(fig, rect):
        ax = axes(fig, rect, title, left=.34, bottom=.24)
        labels = []
        for i, (pair, roi) in enumerate(zip(pairs, rois)):
            row = e.row(sample, pair, roi)
            labels.append(f'ROI {roi}' if len(set(pairs))==1 and len(set(rois))>1
                          else pair.replace('_Reactive-Ast', '').replace('_',' / '))
            ax.plot([row.correlation_S, row.correlation_CE], [i,i], color='#B9C2C8', lw=1)
            for dy, value, marker, color in [(-.12,row.correlation_S,'o','#5F6E76'), (.12,row.correlation_CE,'s','#A33E54')]:
                ax.plot(value, i+dy, marker, ms=3.8, color=color)
                ax.annotate(f'{value:+.2f}', (value,i+dy), xytext=(-5 if value>=.3 else 5,0),
                            ha='right' if value>=.3 else 'left',
                            textcoords='offset points', fontsize=6.1, va='center')
        ax.axvline(0, color='#BFC6CB', lw=.65)
        ax.set(xlim=(-1,1), ylim=(len(labels)-.5,-.5), yticks=range(len(labels)), yticklabels=labels,
               xticks=[-1,0,1], xlabel='Pearson r')
        ax.tick_params(axis='y', labelsize=6.5)
        fig.text(rect[0]+rect[2]*.5, rect[1]+rect[3]*.025, '● Spots   ■ Consensus', ha='center', fontsize=6.2)
    return draw


def enrichment_values(e, sample, pair, rois):
    def draw(fig, rect):
        ax = axes(fig, rect, 'Both programs enriched', left=.28, bottom=.25)
        for j, roi in enumerate(rois):
            r = e.row(sample,pair,roi)
            for k,(name,color) in enumerate(zip([r.program_A,r.program_B],['#46798B','#C29851'])):
                value=r['enrichment_S_'+('A' if k==0 else 'B')]
                ax.bar(j+(k-.5)*.32,value,width=.3,color=color,label=name if j==0 else None)
                ax.text(j+(k-.5)*.32,value+.13,f'{value:.2f}',ha='center',fontsize=6.2)
        ax.axhline(1,color='#747E85',lw=.7,ls='--')
        ax.set(xticks=range(len(rois)),xticklabels=['ROI '+r for r in rois],ylabel='Enrichment ratio')
        ax.set_ylim(0,max(ax.get_ylim()[1],7))
        ax.legend(frameon=False,fontsize=6,loc='upper left')
    return draw


def bands_draw(e, anchor, responder, *, title=None):
    def draw(fig, rect):
        run=e.run('260_T','MES-Hyp_MES-Ast'); f=run['frame']
        bands,_,edges,_=e.bands('260_T','MES-Hyp_MES-Ast',anchor,responder)
        ax=axes(fig,rect,title or f'ROI {anchor} → {responder}',left=.04,bottom=.24)
        ax.scatter(f.x,f.y,c='#EDF0F2',s=2,linewidths=0)
        for band,color in BANDS.items():
            g=f[bands==band]
            ax.scatter(g.x,g.y,c=color,s=4,linewidths=0,label=band.title())
        ax.add_collection(LineCollection(f[['x','y']].to_numpy()[edges],colors='#2A3034',linewidths=.2))
        ax.set_aspect('equal');ax.axis('off')
        ax.legend(frameon=False,fontsize=5.5,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.01),
                  columnspacing=.2,handletextpad=.1,markerscale=1)
        fig.text(rect[0]+rect[2]*.5,rect[1]+rect[3]*.025,
                 f'Near {(bands=="near").sum()} / far {(bands=="far").sum()} spots',ha='center',fontsize=6.2)
    return draw


def response_draw(e, anchor, responder, *, title=None):
    def draw(fig,rect):
        run=e.run('260_T','MES-Hyp_MES-Ast')
        stored=e.bands('260_T','MES-Hyp_MES-Ast',anchor,responder)[3]
        ax=axes(fig,rect,title or f'Within ROI {responder}: near − far',left=.28,bottom=.26)
        for i,(p,color) in enumerate(zip(run['names'],['#A33E54','#347D8B'])):
            values=[one(stored,surface=s,responder_feature_id=p).interface_response
                    for s in ('source_native','consensus_ensemble')]
            ax.plot([0,1],values,'o-',color=color,ms=4,lw=1,label=p)
            for x,v in enumerate(values):
                ax.annotate(f'{v:+.3f}',(x,v),xytext=(0,6 if v>.1 else -11 if v<-.1 else 6 if i==1 and responder=='4' else -11),
                            textcoords='offset points',ha='center',fontsize=6)
        ax.axhline(0,color='#A6AEB4',lw=.65)
        ax.set(xlim=(-.25,1.25),ylim=(-.53,.78),xticks=[0,1],xticklabels=['Spots','Consensus'],ylabel='Score difference')
        ax.legend(frameon=False,fontsize=6,loc='upper right')
    return draw


def radial_draw(e,sample,pair,roi):
    def draw(fig,rect):
        run=e.run(sample,pair); r=run['radial']
        r=r[r.roi_id.eq(roi) & r.component_radial_shell_status.eq('finite')]
        ax=axes(fig,rect,f'{sample}: ROI {roi}\n{pair.replace("_", " / ")}',left=.23,bottom=.27)
        for cid,g in r.groupby('component_id'):
            g=g.sort_values('shell_bin_index')
            n=int(run['ledger'].component_id.eq(cid).sum())
            line,=ax.plot(g.shell_midpoint_normalized,g.shell_symmetric,'o-',ms=3,lw=.9,label=f'Component {cid} ({n} spots)')
            ax.plot(0,g.component_rho_0.iloc[0],'D',ms=3.5,color=line.get_color())
        ax.axhline(0,color='#AFB6BA',lw=.65)
        ax.set(ylim=(-1,1),xlim=(-.25,7),xticks=[0,1,2,3.5,6.5],xlabel='Distance / spot spacing',ylabel='Association')
        ax.legend(frameon=False,fontsize=5.6,loc='lower left')
    return draw


def matched_draw(e):
    def draw(fig,rect):
        ax=axes(fig,rect,'Same ROIs at both supports',left=.18,bottom=.20)
        g=e.matched[e.matched.matched_S_CE]
        colors=plt.get_cmap('tab10').colors
        for color,pair in zip(colors,PAIRS):
            d=g[g.pair_id.eq(pair)]
            ax.scatter(d.r_S,d.r_CE,s=7,color=color,alpha=.55,edgecolors='none',label=pair.replace('__',' / '))
        ax.plot([-1,1],[-1,1],color='#777',lw=.7,ls='--')
        ax.axhline(0,color='#DDD',lw=.5);ax.axvline(0,color='#DDD',lw=.5)
        ax.set(xlim=(-1,1),ylim=(-1,1),xticks=[-1,-.5,0,.5,1],yticks=[-1,-.5,0,.5,1],
               xlabel='Spot Pearson r',ylabel='Consensus Pearson r')
        # Program colours are shared with the adjacent cutoff panel's legend.
    return draw


def threshold_draw(e):
    def draw(fig,rect):
        ax=axes(fig,rect,'Sensitivity to reporting cutoff',left=.15,bottom=.20)
        d=e.summary[e.summary.context.eq('all') & e.summary.metric.eq('correlation') & e.summary.surface.eq('S')]
        for pair,color in zip(PAIRS,plt.get_cmap('tab10').colors):
            sign=1 if pair in PAIRS[:4] else -1
            g=d[d.pair_id.eq(pair)&d.sign.eq(sign)].copy().sort_values('threshold',key=lambda s:s.abs())
            ax.plot(g.threshold.abs(),g.frequency,'o-',ms=3,lw=1,color=color,label=pair.replace('__',' / '))
        ax.set(xticks=[.1,.2,.3],ylim=(0,.7),xlabel='|r| cutoff',ylabel='Mean within-section ROI fraction')
        ax.legend(frameon=False,fontsize=5.6,loc='upper right')
    return draw






def main_panels(e):
    return [
        ('A','Recurrent program relationships across 30 GBM samples',[
            ('territory',cohort_territory(e)),('association',cohort_association(e))]),
        ('B','259_T | Reactive–myeloid and hypoxia-associated territories',[
            ('reactive',map_draw(e,'259_T','AC_Reactive-Ast','Reactive-Ast','CE',['2','3'])),
            ('astrocytic',map_draw(e,'259_T','AC_Reactive-Ast','AC','CE',['2','3'])),
            ('macrophage',map_draw(e,'259_T','Mac_Reactive-Ast','Mac','CE',['2','3'])),
            ('hypoxic',map_draw(e,'259_T','MES-Hyp_Reactive-Ast','MES-Hyp','CE',['2','3'])),
            ('association',association_values(e,'259_T',['AC_Reactive-Ast','Mac_Reactive-Ast','MES-Hyp_Reactive-Ast'],['2']*3,'Reactive-Ast in ROI 2'))]),
        ('C','262_T | Shared territory, different internal organisation',[
            ('neuron',map_draw(e,'262_T','Neuron_NPC','Neuron','CE',['4','5'])),
            ('npc',map_draw(e,'262_T','Neuron_NPC','NPC','CE',['4','5'])),
            ('enrichment',enrichment_values(e,'262_T','Neuron_NPC',['4','5'])),
            ('association',association_values(e,'262_T',['Neuron_NPC']*2,['4','5'],'Within each ROI'))]),
        ('D','260_T | A boundary pattern within an almost uncorrelated ROI',[
            ('hypoxic',map_draw(e,'260_T','MES-Hyp_MES-Ast','MES-Hyp','S',['2','4'])),
            ('mes_astrocytic',map_draw(e,'260_T','MES-Hyp_MES-Ast','MES-Ast','S',['2','4'])),
            ('boundary_bands',bands_draw(e,'4','2')),('boundary_response',response_draw(e,'4','2'))]),
    ]


def supplementary_panels(e):
    """Only the evidence needed to check and extend main Figure 3."""
    neighbor=[(f'moran_{direction}','S',band,sign) for direction in ('A|B','B|A') for band,sign in ((.1,1),(-.1,-1))]
    distance=[(f'shell_{shell}','component',band,sign) for shell in (0,3) for band,sign in ((.1,1),(-.1,-1))]
    maps=[]
    for sample,programs,rois in [
        ('259_T',[('AC_Reactive-Ast','Reactive-Ast'),('AC_Reactive-Ast','AC'),
                  ('Mac_Reactive-Ast','Mac'),('MES-Hyp_Reactive-Ast','MES-Hyp')],['2','3']),
        ('262_T',[('Neuron_NPC','Neuron'),('Neuron_NPC','NPC')],['4','5'])]:
        for pair,program in programs:
            maps.append((f'{sample}_{program}',map_draw(e,sample,pair,program,'S',rois,
                                                        title=f'{sample} | {program}')))
    return [
        ('S4','Cohort checks: support, cutoffs and spatial association',[
            ('matched_rois',matched_draw(e)),('cutoffs',threshold_draw(e)),
            ('moran',cohort_association(e,neighbor,'Neighbour association: spot Moran',
                     ['A ← B\n+','A ← B\n−','B ← A\n+','B ← A\n−'],True)),
            ('distance',cohort_association(e,distance,'Nearest and farthest distance bands',
                        ['First\n+','First\n−','Last\n+','Last\n−'],True))]),
        ('S5','Spot-level program maps underlying Figure 3B–C',maps),
        ('S6','Component-distance profiles and the reverse boundary',[
            ('npc_roi4',radial_draw(e,'262_T','Neuron_NPC','4')),
            ('npc_roi5',radial_draw(e,'262_T','Neuron_NPC','5')),
            ('reverse_bands',bands_draw(e,'2','4',title='260_T | ROI 2 → ROI 4')),
            ('reverse_response',response_draw(e,'2','4',title='260_T | Within ROI 4: near − far'))]),
    ]
