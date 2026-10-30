"""Figure 4 and S9–S10 from Input Data and Supplementary Data 4.

No analysis reruns, score changes, manuscript drafts or automatic figure assemblies.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from shapely import wkt

from analyses.figures.common import (BANDS, expression_colormap, figure_dir,
                                     save_panel, write_index, write_sources)
from analyses.greenwald_full_evaluation.figure_common import polygon_patch
from tilescope.cols import (HEX_G1_ENRICHMENT, HEX_G1_ENRICHMENT_STATUS,
                            HEX_G2_ENRICHMENT, HEX_G2_ENRICHMENT_STATUS, HEX_GEOMETRY_WKT)
from tilescope.viz.consensus import (_NEUTRAL_LOG2_BAND, _SATURATION_LOG2,
                                     _pattern_style_by_status)
from tilescope.viz.painters import SpotSplatPainter, draw_polygon_boundaries
from .evidence import Evidence, FOCUS, PRIMARY, IFN_PAIR, MES_PAIR, V4_PAIR, NAMES, one, close

COLORS = {'MCT1_CTL': '#187b98', 'MCT1_KO': '#93427b', 'V4_WT': '#507a45'}
SHORT = {'MCT1_CTL': 'CTL', 'MCT1_KO': 'KO', 'V4_WT': 'V4'}
CORE_COLOR, RIM_COLOR = '#4a4a4a', '#738789'
MAIN_FEATURES = ['cell_cycle_proliferation', 'myeloid_phagocytic', 'interferon_related',
                 'antigen_presenting_myeloid', 'cytotoxic_t_nk']


def chart(title, *, size=(4.3, 3.3), left=.20, bottom=.22):
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=left, right=.96, bottom=bottom, top=.85)
    ax.set_title(title, loc='left', pad=12, fontsize=10)
    return fig, ax


def map_axes(run, title):
    frame = run['frame']
    core, rim = FOCUS[run['sample']]
    focus = frame.loc[frame.roi_id.isin([core, rim])]
    lo, hi = focus[['x', 'y']].min().to_numpy(), focus[['x', 'y']].max().to_numpy()
    pad = 1.5 * run['spacing']
    bounds = [lo[0]-pad, hi[0]+pad, lo[1]-pad, hi[1]+pad]
    fig = plt.figure(figsize=(3.7, 3.8))
    ax = fig.add_axes([.025, .23, .95, .64])
    ax.set(xlim=bounds[:2], ylim=bounds[2:], aspect='equal')
    ax.axis('off')
    fig.text(.5, .96, title, ha='center', va='top', fontsize=10)
    fig.canvas.draw()
    return fig, ax, bounds


def outline_rois(ax, run):
    core, rim = FOCUS[run['sample']]
    for roi, color in [(core, CORE_COLOR), (rim, RIM_COLOR)]:
        geom = run['outlines'][roi]
        draw_polygon_boundaries(ax, {str(roi): geom}, color=color, linewidth=.65, alpha=.95)
        parts = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
        centre = max(parts, key=lambda p: p.area).representative_point()
        ax.text(centre.x, centre.y, str(roi), ha='center', va='center', fontsize=8,
                bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=.8))


def score_map(e, sample, pair, program, figure, stem):
    run = e.spatial(sample, pair)
    core, rim = FOCUS[sample]
    f = run['frame']
    index = run['names'].index(program) + 1
    fig, ax, bounds = map_axes(run, f'{SHORT[sample]} · {NAMES[program]}')
    SpotSplatPainter(.38).draw(ax, f[['x', 'y']].to_numpy(), np.zeros(len(f)),
        cmap=ListedColormap(['#e0e4e6']), norm=Normalize(0, 1), canvas=None,
        median_nn_override=run['spacing'])
    selected = run['hexes'].loc[run['hexes'].roi_id.eq(rim)].copy()
    selected = selected.loc[np.isfinite(selected[f'gene{index}_density_ensemble'])]
    values = selected[f'gene{index}_density_ensemble'].to_numpy()
    limit = max(float(np.quantile(np.abs(values), .98)), 1e-6)
    norm, cmap = Normalize(-limit, limit), expression_colormap()
    patches, expanded = [], []
    for row in selected.itertuples():
        geom = wkt.loads(row.hex_geometry_wkt)
        for polygon in (geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]):
            patches.append(polygon_patch(polygon))
            expanded.append(getattr(row, f'gene{index}_density_ensemble'))
    artist = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor='#98a1a8', linewidth=.15)
    artist.set_array(np.asarray(expanded))
    ax.add_collection(artist)
    outline_rois(ax, run)
    cax = fig.add_axes([.20, .145, .60, .027])
    bar = fig.colorbar(artist, cax=cax, orientation='horizontal', ticks=[-limit, 0, limit], extend='both')
    bar.ax.set_xticklabels([f'{-limit:.2f}', '0', f'{limit:.2f}'])
    bar.ax.tick_params(labelsize=8, length=2, pad=2)
    bar.outline.set_visible(False)
    fig.text(.5, .043, f'Consensus · ROI {rim} signature score', ha='center', fontsize=8)
    fig.text(.5, .008, f'Core ROI {core} shown in grey', ha='center', fontsize=7, color='#555555')
    e.displays.append(dict(figure=figure, panel=stem, sample=sample, pair=pair, program=program,
        surface='Consensus', colored_roi=rim, context_roi=core, vmin=-limit, vmax=limit,
        n_colored_units=len(values), xmin=bounds[0], xmax=bounds[1], ymin=bounds[2], ymax=bounds[3],
        scale_rule='Symmetric 98th percentile of absolute displayed ROI scores; display clipping only; separate scales by sample/program/representation',
        coordinate_units='Seurat tissue-image coordinate units; not converted to micrometres'))
    return fig


def band_map(e, sample, pair, figure, stem):
    run = e.spatial(sample, pair)
    bands = e.bands(sample, pair)
    core, rim = FOCUS[sample]
    fig, ax, bounds = map_axes(run, f'{SHORT[sample]} · core ROI {core} / periphery ROI {rim}')
    labels = {'anchor': f'Core ROI {core}', 'near': 'Near core boundary', 'buffer': 'Buffer',
              'far': 'Farther within periphery', 'unreachable': 'No route to this boundary', 'outside': 'Other ROIs'}
    for band in ['outside', 'anchor', 'unreachable', 'far', 'buffer', 'near']:
        d = bands.loc[bands.band.eq(band)]
        if d.empty:
            continue
        color = BANDS.get(band, '#e5e8e9')
        SpotSplatPainter(.40).draw(ax, d[['x', 'y']].to_numpy(), np.zeros(len(d)),
            cmap=ListedColormap([color]), norm=Normalize(0, 1), canvas=None,
            median_nn_override=run['spacing'])
    outline_rois(ax, run)
    handles = [Line2D([], [], marker='o', color=BANDS[b], ls='none', markersize=5, label=labels[b])
               for b in ['anchor', 'near', 'buffer', 'far', 'unreachable']]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .005), ncol=2,
               frameon=False, fontsize=7, columnspacing=.7, handletextpad=.3)
    fig.text(.5, .205, f'{(bands.band == "near").sum()} near / {(bands.band == "far").sum()} far spots',
             ha='center', fontsize=8)
    e.displays.append(dict(figure=figure, panel=stem, sample=sample, pair=pair, surface='Source boundary bands',
        colored_roi=rim, context_roi=core, xmin=bounds[0], xmax=bounds[1], ymin=bounds[2], ymax=bounds[3],
        coordinate_units='Seurat tissue-image coordinate units', scale_rule='No expression transformation'))
    return fig


def primary_correlations(e):
    fig, ax = chart('Proliferation–myeloid correlation', size=(3.7, 3.1))
    for sample, label in [('MCT1_CTL', 'CTL ROI 5'), ('MCT1_KO', 'KO ROI 9')]:
        r = one(e.roi, sample=sample, pair=PRIMARY, roi=FOCUS[sample][1])
        ys = [r.source_r, r.cons_r]
        ax.plot([0, 1], ys, 'o-', color=COLORS[sample], lw=1.8, label=label, ms=5)
        for x, y in enumerate(ys):
            ax.annotate(f'{y:+.2f}', (x, y), xytext=(5, 7), textcoords='offset points', fontsize=9)
    ax.axhline(0, color='#a4a4a4', lw=.7)
    ax.set(xticks=[0, 1], xticklabels=['Spots', 'Consensus'], xlim=(-.15, 1.35), ylim=(-.85, .70), ylabel='Pearson r')
    ax.legend(frameon=False, loc='lower left', fontsize=8)
    return fig


def enrichment_pattern_map(e, sample, stem):
    """Core consensus enrichment colours in the manuscript map layout.

    Colours combine two marginal enrichments, not the joint-high ratio or
    Pearson correlation. Values and typed availability come from the saved CSV.
    """
    run = e.spatial(sample, PRIMARY)
    core, rim = FOCUS[sample]
    f = run['frame']
    fig, ax, bounds = map_axes(run, f'{SHORT[sample]} · combined enrichment')
    a, b = run['names']
    fig.text(.5, .905, f'A = {NAMES[a]}     B = {NAMES[b]}', ha='center', fontsize=7)
    SpotSplatPainter(.38).draw(ax, f[['x', 'y']].to_numpy(), np.zeros(len(f)),
        cmap=ListedColormap(['#e0e4e6']), norm=Normalize(0, 1), canvas=None,
        median_nn_override=run['spacing'])
    selected = run['hexes'].loc[run['hexes'].roi_id.eq(rim)]
    patches, colors, edges, widths = [], [], [], []
    for row in selected.itertuples():
        color, edge, width = _pattern_style_by_status(
            getattr(row, HEX_G1_ENRICHMENT), getattr(row, HEX_G1_ENRICHMENT_STATUS),
            getattr(row, HEX_G2_ENRICHMENT), getattr(row, HEX_G2_ENRICHMENT_STATUS))
        geom = wkt.loads(getattr(row, HEX_GEOMETRY_WKT))
        for polygon in (geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]):
            patches.append(polygon_patch(polygon))
            colors.append(color)
            edges.append('#98a1a8' if edge == 'black' else edge)
            widths.append(.15 if edge == 'black' else width)
    artist = PatchCollection(patches, facecolors=colors, edgecolors=edges, linewidths=widths)
    artist.set_gid('consensus_enrichment_pattern')
    ax.add_collection(artist)
    outline_rois(ax, run)
    # Reuse the production palette and saturation rule for the colour key too.
    up, down = 2.0**_SATURATION_LOG2, 2.0**(-_SATURATION_LOG2)
    patterns = [('A↑ B↑', up, up), ('A↑ B↓', up, down), ('A↓ B↑', down, up),
                ('A↓ B↓', down, down), ('Near neutral', 1., 1.)]
    handles = [Patch(facecolor=_pattern_style_by_status(x, 'finite', y, 'finite')[0],
                     edgecolor='#98a1a8', linewidth=.3, label=label)
               for label, x, y in patterns]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .085), ncol=3,
               frameon=False, fontsize=7, columnspacing=.9, handletextpad=.4, handlelength=1.2)
    fig.text(.5, .052, '↑ enriched   ↓ depleted, relative to other ROIs', ha='center', fontsize=7)
    fig.text(.5, .015, f'Consensus · ROI {rim}; core ROI {core} shown in grey',
             ha='center', fontsize=7, color='#555555')
    e.displays.append(dict(figure='FigureS9', panel=stem, sample=sample, pair=PRIMARY,
        program_A=a, program_B=b, surface='Consensus marginal-enrichment pattern',
        colored_roi=rim, context_roi=core, n_colored_units=len(selected),
        xmin=bounds[0], xmax=bounds[1], ymin=bounds[2], ymax=bounds[3],
        value_columns=f'{HEX_G1_ENRICHMENT}; {HEX_G2_ENRICHMENT}',
        status_columns=f'{HEX_G1_ENRICHMENT_STATUS}; {HEX_G2_ENRICHMENT_STATUS}',
        neutral_log2_band=_NEUTRAL_LOG2_BAND, saturation_abs_log2=_SATURATION_LOG2,
        scale_rule='Core consensus PDF palette and status handling. Hue combines marginal enrichment/depletion; '
                   'strength follows the weaker absolute log2 ratio. Near neutral if either marginal is within '
                   'the neutral band. Not the joint-high ratio or a correlation map.',
        coordinate_units='Seurat tissue-image coordinate units; not converted to micrometres'))
    return fig


def radial_plot(e, settings, title, *, ylim=None):
    fig, ax = chart(title, size=(4.4, 3.3), bottom=.26)
    for sample, pair, cid, label, color in settings:
        r = e.table('component_radial_shell_correlogram', sample, pair)
        r = r.loc[r.component_id.eq(cid)].sort_values('shell_bin_index')
        if len(r) != 4 or not r.component_radial_shell_status.eq('finite').all():
            raise ValueError('Incomplete selected distance profile')
        ax.plot(r.shell_midpoint_normalized, r.shell_symmetric, 'o-', color=color, lw=1.6, ms=4, label=label)
    ax.axhline(0, color='#aaaaaa', lw=.7)
    ax.set(xticks=[1, 2, 3.5, 6.5], xlabel='Distance / median nearest-neighbour spacing',
           ylabel='Spatial cross-correlation')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(frameon=False, fontsize=8, loc='best')
    fig.text(.5, .035, 'Four distance bands; one connected component per curve', ha='center', fontsize=7)
    return fig


def boundary_plot(e, sample, features):
    core, rim = FOCUS[sample]
    boundary = e.boundaries()
    fig, ax = chart(f'{SHORT[sample]} · within ROI {rim}, beside core ROI {core}',
                    size=(4.8, 3.4), left=.35, bottom=.27)
    for surface, offset, marker, color, label in [
        ('source_native', -.11, 'o', '#46596a', 'Spots'),
        ('consensus_ensemble', .11, 's', '#c47b35', 'Consensus')]:
        values = [one(boundary, sample=sample, responder_feature_id=p, surface=surface).interface_response
                  for p in features]
        ax.scatter(values, np.arange(len(features))+offset, c=color, marker=marker, s=30, label=label, zorder=3)
    ax.axvline(0, color='#999999', lw=.8)
    ax.set(yticks=np.arange(len(features)), yticklabels=[NAMES[p] for p in features],
           ylim=(len(features)-.5, -.5), xlabel='Near − far signature score')
    ax.set_xlim(-.70, .70) if sample != 'V4_WT' else ax.set_xlim(-.15, 1.35)
    ax.grid(axis='x', alpha=.15)
    ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.25), ncol=2)
    return fig


def supporting_correlations(e):
    pairs = [(IFN_PAIR, 'Interferon–proliferation'),
             ('antigen_presenting_myeloid__cytotoxic_t_nk', 'Antigen presentation–cytotoxic'),
             (MES_PAIR, 'Mesenchymal–myeloid')]
    fig, ax = chart('Related periphery comparisons', size=(5.2, 4.1), left=.47, bottom=.22)
    labels = []
    for i, (pair, label) in enumerate(pairs):
        for j, sample in enumerate(['MCT1_CTL', 'MCT1_KO']):
            y = 2*i+j
            r = one(e.roi, sample=sample, pair=pair, roi=FOCUS[sample][1])
            labels.append(f'{label}\n{SHORT[sample]} ROI {FOCUS[sample][1]}')
            ax.plot([r.source_r, r.cons_r], [y-.12, y+.12], color='#bbbbbb', lw=.7)
            ax.plot(r.source_r, y-.12, 'o', color=COLORS[sample], ms=4.5)
            ax.plot(r.cons_r, y+.12, 's', color=COLORS[sample], ms=4.5)
    ax.axvline(0, color='#aaaaaa', lw=.7)
    ax.set(xlim=(-1, 1), ylim=(5.7, -.7), yticks=range(6), yticklabels=labels, xlabel='Pearson r')
    ax.tick_params(axis='y', labelsize=8, length=0)
    fig.text(.70, .04, '● Spots    ■ Consensus', ha='center', fontsize=8)
    return fig


def tessellation_plot(e):
    fig, ax = chart('Alternative tessellations of the same tissue', size=(4.3, 3.3), left=.25)
    for i, sample in enumerate(['MCT1_CTL', 'MCT1_KO']):
        roi = FOCUS[sample][1]
        r = e.table('roi_replicate_correlations', sample, PRIMARY)
        r = r.loc[r.roi_id.eq(roi)].sort_values('replicate_index')
        if len(r) != 10 or not r.parent_replicate_correlation_status.eq('finite').all():
            raise ValueError('Selected zoning comparison is incomplete')
        summary = one(e.roi, sample=sample, pair=PRIMARY, roi=roi)
        close(np.quantile(r.correlation_value, [.25, .5, .75]),
              [summary.tess_q25, summary.tess_r, summary.tess_q75], 'Tessellation distribution')
        ax.scatter(r.correlation_value, i+np.linspace(-.13, .13, len(r)), s=24,
                   c=COLORS[sample], alpha=.8, zorder=3)
        ax.plot([summary.tess_q25, summary.tess_q75], [i+.24]*2, lw=3, color=COLORS[sample])
        ax.plot(summary.tess_r, i+.24, '|', ms=12, color='#222222')
    ax.axvline(0, color='#aaaaaa', lw=.7)
    ax.set(xlim=(-.85, .75), ylim=(1.55, -.45), yticks=[0, 1], yticklabels=['CTL ROI 5', 'KO ROI 9'],
           xlabel='Proliferation–myeloid Pearson r')
    fig.text(.5, .035, '10 partitions each; line = IQR, tick = median (not a confidence interval)', ha='center', fontsize=7)
    return fig


def satellite_map(e, figure, stem):
    run = e.spatial('MCT1_KO', IFN_PAIR)
    f = run['frame']
    fig, ax, bounds = map_axes(run, 'KO · two connected components within ROI 4')
    groups = [(f, '#e2e5e6', None), (f.loc[f.roi_id.eq(9)], '#bdd9e1', 'Periphery ROI 9'),
              (f.loc[f.component_id.eq(6)], '#aa6a48', 'Main core: 243 spots'),
              (f.loc[f.component_id.eq(7)], COLORS['MCT1_KO'], 'Detached focus: 49 spots')]
    for d, color, _ in groups:
        SpotSplatPainter(.41).draw(ax, d[['x', 'y']].to_numpy(), np.zeros(len(d)),
            cmap=ListedColormap([color]), norm=Normalize(0, 1), canvas=None,
            median_nn_override=run['spacing'])
    outline_rois(ax, run)
    focus = f.loc[f.component_id.eq(7), ['x', 'y']].median()
    ax.annotate('Component 7', xy=focus, xytext=(22, 22), textcoords='offset points', fontsize=8,
                arrowprops=dict(arrowstyle='-', color='#4a4a4a', lw=.7),
                bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
    fig.legend(handles=[Line2D([], [], marker='o', ls='none', color=c, label=l) for _, c, l in groups[1:]],
               loc='lower center', bbox_to_anchor=(.5, .005), frameon=False, fontsize=8)
    e.displays.append(dict(figure=figure, panel=stem, sample='MCT1_KO', pair=IFN_PAIR,
        surface='Source component membership', colored_roi=4, context_roi=9,
        xmin=bounds[0], xmax=bounds[1], ymin=bounds[2], ymax=bounds[3],
        coordinate_units='Seurat tissue-image coordinate units'))
    return fig


def satellite_means(e):
    summary = e.satellite()
    fig, ax = chart('Similar mean scores in the two components', size=(4.4, 3.2))
    for i, (cid, label, color) in enumerate([(6, 'Main (243 spots)', '#aa6a48'), (7, 'Focus (49 spots)', COLORS['MCT1_KO'])]):
        r = one(summary, component_id=cid)
        values = [r.mean_tumour_score, r.mean_proliferation_score]
        x = np.arange(2) + (i-.5)*.30
        ax.bar(x, values, width=.28, color=color, label=label)
        for j, v in zip(x, values):
            ax.text(j, v+.025, f'{v:.2f}', ha='center', fontsize=8)
    ax.set(xticks=[0, 1], xticklabels=['Tumour-associated', 'Proliferation'],
           ylim=(0, 1.18), ylabel='Mean signature score')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    return fig


def v4_correlations(e):
    fig, ax = chart('Two definitions recover the V4 relationship', size=(4.2, 3.2))
    for pair, label, color in [(V4_PAIR, 'Reactive–myeloid', '#507a45'), (MES_PAIR, 'Mesenchymal–myeloid', '#a77520')]:
        r = one(e.roi, sample='V4_WT', pair=pair, roi=2)
        ax.plot([0, 1], [r.source_r, r.cons_r], 'o-', lw=1.5, c=color, label=label)
    ax.set(xticks=[0, 1], xticklabels=['Spots', 'Consensus'], xlim=(-.12, 1.13), ylim=(0, 1), ylabel='Pearson r')
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    return fig


def panels(e):
    groups = {name: [] for name in ['Figure4', 'FigureS9', 'FigureS10']}
    def add(group, stem, draw, sources):
        groups[group].append((stem, draw, sources))
    for i, sample in enumerate(['MCT1_CTL', 'MCT1_KO'], 1):
        stem = f'A{i}_{SHORT[sample]}_core_boundary'
        add('Figure4', stem, lambda s=sample, k=stem: band_map(e, s, PRIMARY, 'Figure4', k), ['boundary_spots', 'display_settings', 'display_outlines'])
        for j, program in enumerate(['cell_cycle_proliferation', 'myeloid_phagocytic'], 1):
            stem = f'B{2*(i-1)+j}_{SHORT[sample]}_{program}_consensus'
            add('Figure4', stem, lambda s=sample, p=program, k=stem: score_map(e, s, PRIMARY, p, 'Figure4', k), ['hexes', 'spots', 'display_settings', 'display_outlines'])
        stem = f'A{i}_{SHORT[sample]}_enrichment_pattern'
        add('FigureS9', stem, lambda s=sample, k=stem: enrichment_pattern_map(e, s, k),
            ['hexes', 'spots', 'display_settings', 'display_outlines'])
    add('Figure4', 'C_proliferation_myeloid_correlations', lambda: primary_correlations(e), ['roi_measurements'])
    add('Figure4', 'D_proliferation_myeloid_distance', lambda: radial_plot(e, [
        ('MCT1_CTL', PRIMARY, 17, 'CTL ROI 5 · 202 spots', COLORS['MCT1_CTL']),
        ('MCT1_KO', PRIMARY, 13, 'KO ROI 9 · 409 spots', COLORS['MCT1_KO'])],
        'The contrast extends to neighbouring spots', ylim=(-.35, .25)), ['component_curves'])
    for i, sample in enumerate(['MCT1_CTL', 'MCT1_KO'], 1):
        add('Figure4', f'E{i}_{SHORT[sample]}_boundary_scores', lambda s=sample: boundary_plot(e, s, MAIN_FEATURES), ['boundary_measurements'])
    add('FigureS9', 'B_supporting_periphery_correlations', lambda: supporting_correlations(e), ['roi_measurements'])
    add('FigureS9', 'C_tessellation_variation', lambda: tessellation_plot(e), ['replicate_correlations', 'roi_measurements'])
    add('FigureS9', 'D1_KO_component_map', lambda: satellite_map(e, 'FigureS9', 'D1_KO_component_map'), ['spots', 'display_outlines', 'display_settings'])
    add('FigureS9', 'D2_KO_component_scores', lambda: satellite_means(e), ['component_summary'])
    add('FigureS9', 'D3_KO_component_distance', lambda: radial_plot(e, [
        ('MCT1_KO', IFN_PAIR, 6, 'Main: 243 spots; same-spot r = −0.09', '#aa6a48'),
        ('MCT1_KO', IFN_PAIR, 7, 'Focus: 49 spots; same-spot r = −0.45', COLORS['MCT1_KO'])],
        'Interferon–proliferation within core ROI 4', ylim=(-.55, .38)), ['component_curves', 'component_summary'])
    add('FigureS10', 'A_V4_core_boundary', lambda: band_map(e, 'V4_WT', V4_PAIR, 'FigureS10', 'A_V4_core_boundary'), ['boundary_spots', 'display_outlines', 'display_settings'])
    for i, (pair, program) in enumerate([(V4_PAIR, 'myeloid_phagocytic'), (V4_PAIR, 'v4_reactive_interface'), (MES_PAIR, 'mesenchymal_reactive_stromal')], 1):
        stem = f'B{i}_V4_{program}_consensus'
        add('FigureS10', stem, lambda q=pair, p=program, k=stem: score_map(e, 'V4_WT', q, p, 'FigureS10', k), ['hexes', 'spots', 'display_outlines', 'display_settings'])
    add('FigureS10', 'C_V4_correlations', lambda: v4_correlations(e), ['roi_measurements'])
    add('FigureS10', 'D_V4_boundary_scores', lambda: boundary_plot(e, 'V4_WT', ['myeloid_phagocytic', 'v4_reactive_interface', 'mesenchymal_reactive_stromal']), ['boundary_measurements'])
    add('FigureS10', 'E_V4_distance', lambda: radial_plot(e, [('V4_WT', V4_PAIR, 6, 'ROI 2 component · 262 spots', COLORS['V4_WT'])],
        'Reactive–myeloid pattern over distance', ylim=(-.05, .55)), ['component_curves'])
    return groups


def sources(e, group):
    keys = [('MCT1_CTL', PRIMARY), ('MCT1_KO', PRIMARY)] if group == 'Figure4' else (
        [('MCT1_CTL', PRIMARY), ('MCT1_KO', PRIMARY), ('MCT1_KO', IFN_PAIR)] if group == 'FigureS9' else
        [('V4_WT', V4_PAIR), ('V4_WT', MES_PAIR)])
    spots, hexes, outlines = [], [], []
    for sample, pair in keys:
        run = e.spatial(sample, pair)
        a, b = run['names']
        spots.append(run['frame'].rename(columns={'gene1': 'score_A', 'gene2': 'score_B'}).assign(sample=sample, pair=pair, program_A=a, program_B=b))
        hexes.append(run['hexes'].rename(columns={'gene1_density_ensemble': 'score_A_ensemble', 'gene2_density_ensemble': 'score_B_ensemble'}).assign(sample=sample, pair=pair, program_A=a, program_B=b))
        outlines += [dict(sample=sample, pair=pair, roi_id=roi, geometry_wkt=geom.wkt,
                         role='display footprint, not an analysis domain') for roi, geom in run['outlines'].items() if roi in FOCUS[sample]]
    samples = {s for s, _ in keys}
    radial = e.tables['component_radial_shell_correlogram'].merge(e.selection, on=['sample', 'pair'])
    return dict(roi_measurements=e.focus_roi.loc[e.focus_roi['sample'].isin(samples)],
        spots=pd.concat(spots, ignore_index=True), hexes=pd.concat(hexes, ignore_index=True),
        display_outlines=pd.DataFrame(outlines), display_settings=pd.DataFrame(e.displays).query('figure == @group'),
        boundary_spots=pd.concat([e.bands(s, p) for s, p in keys], ignore_index=True),
        boundary_measurements=e.boundaries().loc[lambda f: f['sample'].isin(samples)],
        component_curves=radial.loc[radial['sample'].isin(samples)],
        replicate_correlations=e.tables['roi_replicate_correlations'].merge(e.selection, on=['sample', 'pair']).loc[lambda f: f['sample'].isin(samples)],
        component_summary=e.satellite(), validation=pd.DataFrame(e.checks))




def build(*, figure=None):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False})
    e = Evidence()
    print('Verified 222 saved runs and all 2,064 ROI-pair measurements.', flush=True)
    groups = panels(e)
    if figure is not None:
        groups = {figure: groups[figure]}
    for group, items in groups.items():
        for stem, draw, _ in items:
            save_panel(draw(), figure_dir(group) / 'panels', stem)
        tables = sources(e, group)
        mapping = {stem: keys for stem, _, keys in items}
        used = {k for values in mapping.values() for k in values} | {'validation'}
        enrichment_note = (
            'S9A combines hex_g1_enrichment and hex_g2_enrichment using the core consensus PDF palette '
            'and typed statuses, not the joint-high ratio. Display thresholds and colour-strength '
            'rules are recorded in display_settings. ' if group == 'FigureS9' else '')
        write_sources(group, {k: tables[k] for k in used}, mapping, inputs=e.inputs,
            notes='One section per sample. Scores are unchanged Seurat AddModuleScore values. '
                  'Consensus uses the same spots, not independent biological replication. '
                  'enrich_A, enrich_B and joint_high are source-spot enrichment ratios relative to other ROIs '
                  'of the same sample; joint_high requires both scores above their sample-wide 75th percentiles. '
                  'A ratio of 1 indicates equal frequencies; these ratios are not Pearson correlations. '
                  'Colour limits are display-specific; absolute score magnitudes are not compared across sections. '
                  'Boundary differences are mean signature scores near minus far from the core boundary, '
                  'inside the same periphery ROI. Near is 0–1 source-graph steps from contact; '
                  '2 steps form the excluded buffer; 3 or more steps are far; unreachable spots are excluded. '
                  'Component distances are scaled by median nearest-neighbour spacing, with bands '
                  '0.5–1.5, 1.5–2.5, 2.5–4.5 and 4.5–8.5. '
                  + enrichment_note)
        print(f'{group}: {len(items)} independent SVG/PDF/PNG assets.', flush=True)
    if figure is not None:
        write_index()
        return e
    write_index()
    print(json.dumps({'checks': len(e.checks), 'panels': sum(map(len, groups.values()))}, indent=2), flush=True)
    return e


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', choices=['Figure4', 'FigureS9', 'FigureS10'],
                        help='Rebuild only this figure and its source data')
    args = parser.parse_args()
    build(figure=args.figure)


if __name__ == '__main__':
    main()
