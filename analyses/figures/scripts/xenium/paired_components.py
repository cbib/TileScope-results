"""Complete Meylan components for the paper's interface/lymphoid comparison."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from shapely import from_wkt

from analyses.figures.common import expression_colormap
from analyses.xenium_feasibility.analyse import read_csv, align_ledger
from .map_data import complete_area


# Fixed by neighbourhood, component size and tumour contact, not expression extrema.
COMPONENTS = (
    ("CN-Tumor - T cell interface", 784, "Tumour–T cell interface", "expression"),
    ("CN-Lymphoid aggregate", 0, "Lymphoid aggregate", "expression"),
    ("CN-Tumor", 501, "Tumour", "context only"),
)
REGION_COLOURS = {784: '#238A8D', 0: '#E69F00', 501: '#9A55A2'}


def combined_bounds(cells, hexes):
    """Contain all selected cells and whole saved hexagons, with a 3% margin."""
    low, high = cells[['x', 'y']].min().to_numpy(), cells[['x', 'y']].max().to_numpy()
    if len(hexes):
        bounds = np.asarray([g.bounds for g in hexes.geometry])
        low = np.minimum(low, bounds[:, :2].min(axis=0))
        high = np.maximum(high, bounds[:, 2:].max(axis=0))
    pad = .03 * max(high - low)
    return dict(xmin=low[0]-pad, ymin=low[1]-pad, xmax=high[0]+pad, ymax=high[1]+pad)


def prepare_comparison(run):
    cell_tables, hex_tables, summaries = [], [], []
    for roi, component, label, role in COMPONENTS:
        cells, hexes, settings = complete_area(run, roi, component)
        cell_tables.append(cells.assign(map_label=label, display_role=role,
                                       region_colour=REGION_COLOURS[component]))
        hex_tables.append(hexes.assign(map_label=label, display_role=role))
        summaries.append(dict(settings, map_label=label, display_role=role,
                              region_colour=REGION_COLOURS[component],
                              mean_feature_a=cells.gene1.mean(), mean_feature_b=cells.gene2.mean()))
    context = pd.concat(cell_tables, ignore_index=True)
    all_hexes = pd.concat(hex_tables, ignore_index=True)
    cells = context.loc[context.display_role.eq('expression')].copy()
    hexes = all_hexes.loc[all_hexes.display_role.eq('expression')].copy()
    geometry = read_csv(run.output / 'component_geometry.csv.gz')
    geometry = geometry.loc[geometry.component_id.isin([r[1] for r in COMPONENTS])].copy()
    if len(geometry) != len(COMPONENTS):
        raise ValueError('Missing or duplicated component polygons')
    geometry['map_label'] = geometry.component_id.map({r[1]: r[2] for r in COMPONENTS})
    geometry['geometry'] = from_wkt(geometry.geometry_wkt)
    # Simplification affects outlines only, never expression, cells or hexagons.
    geometry['outline_tolerance'] = .25 * geometry.sample_median_nn
    geometry['display_outline_wkt'] = [
        g.simplify(t, preserve_topology=True).wkt
        for g, t in zip(geometry.geometry, geometry.outline_tolerance)
    ]
    settings = dict(run.identity, display_unit='components', component_id=None,
                    component_ids='784;0', roi_id=';'.join(r[0] for r in COMPONENTS[:2]),
                    n_cells=len(cells), n_eligible_cells=int(cells.tessellation_eligible.sum()),
                    n_hexagons=len(hexes), **combined_bounds(context, all_hexes),
                    context_component_ids='501',
                    extent_rule='Complete components 784 and 0 coloured by expression; component 501 outlined as context; same local bounds as region map; 3% margin',
                    detail_component_id=0,
                    detail_rule='Complete aggregate plus surrounding interface; limits inherited from the full map for each representation',
                    outline_rule='Actual saved component polygons, simplified by 0.25 median neighbour spacing for display; no hull or joining across gaps')
    return cells, hexes, context, all_hexes, geometry, pd.DataFrame(summaries), settings


def polygon_rings(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'Polygon':
        return [np.asarray(r.coords) for r in [geometry.exterior, *geometry.interiors]]
    return [ring for part in geometry.geoms for ring in polygon_rings(part)]


def outline(ax, geometry, component, *, colour='#626a70', alpha=.55, width=.4):
    rows = geometry.loc[geometry.component_id.eq(component)]
    rings = [ring for value in rows.display_outline_wkt for ring in polygon_rings(from_wkt(value))]
    artist = LineCollection(rings, colors=colour, linewidths=width, alpha=alpha, zorder=4)
    artist.set_gid(f'roi-boundary-{component}')
    ax.add_collection(artist)


def frame_axes(ax, bounds):
    ax.set_xlim(bounds['xmin'], bounds['xmax'])
    ax.set_ylim(bounds['ymax'], bounds['ymin'])
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])


def paint_expression(ax, cells, hexes, side, representation, norm, patches_fn, *, detail=False):
    cmap = expression_colormap()
    if representation == 'cells':
        return ax.scatter(cells.x, cells.y, c=cells[f'gene{side}'], cmap=cmap, norm=norm,
                          s=8. if detail else 1.1, linewidths=0, rasterized=True)
    excluded = cells.loc[~cells.tessellation_eligible]
    ax.scatter(excluded.x, excluded.y, s=5 if detail else 1.6,
               c='#a7a7a7', linewidths=0, rasterized=True)
    patches, values = [], []
    for row in hexes.to_dict('records'):
        pieces = patches_fn(row['geometry'])
        patches.extend(pieces)
        values.extend([row[f'gene{side}_density_ensemble']] * len(pieces))
    artist = PatchCollection(patches, cmap=cmap, norm=norm, linewidths=.12, edgecolors='#ffffff55')
    artist.set_array(np.asarray(values)); ax.add_collection(artist)
    return artist


def detail_bounds(cells, hexes):
    bounds = combined_bounds(cells.loc[cells.component_id.eq(0)], hexes.loc[hexes.component_id.eq(0)])
    centre_x = (bounds['xmin'] + bounds['xmax']) / 2
    centre_y = (bounds['ymin'] + bounds['ymax']) / 2
    half_width = max(bounds['xmax'] - bounds['xmin'], bounds['ymax'] - bounds['ymin'])
    return dict(xmin=centre_x-half_width, xmax=centre_x+half_width,
                ymin=centre_y-half_width, ymax=centre_y+half_width)


def add_scale_bar(ax, bounds, length):
    width = bounds['xmax'] - bounds['xmin']
    height = bounds['ymax'] - bounds['ymin']
    start = bounds['xmin'] + .06 * width
    baseline = bounds['ymax'] - .045 * height
    ax.plot([start, start+length], [baseline, baseline], color='#25323b', lw=2, zorder=6)
    ax.text(start+length/2, baseline-.025*height, f'{length:g} µm',
            ha='center', va='bottom', fontsize=8, zorder=6,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1))


def expression_panel(cells, hexes, geometry, settings, patches_fn, *, detail=False):
    representation, gene = settings['representation'], settings['gene']
    side = 1 if gene == settings['feature_a'] else 2
    norm = Normalize(settings['vmin'], settings['vmax'], clip=True)
    fig, ax = plt.subplots(figsize=(2.0, 2.75) if detail else (4.15, 4.45))
    fig.subplots_adjust(left=.035, right=.97, top=.86, bottom=.25 if detail else .19)
    artist = paint_expression(ax, cells, hexes, side, representation, norm, patches_fn, detail=detail)
    frame_axes(ax, settings); ax.set_axis_off()
    title = 'Cells' if representation == 'cells' else 'Consensus'
    name = 'Stem-like markers' if gene == 'Stemlike' else 'Interferon-related markers'
    if detail:
        name = 'Stem-like' if gene == 'Stemlike' else 'Interferon-related'
    ax.set_title(f'{name}\n{title}', fontsize=9 if detail else 10, pad=9)
    for _, component, _, _ in COMPONENTS:
        outline(ax, geometry, component, colour=REGION_COLOURS[component],
                alpha=.9, width=.6 if detail else .4)
    if not detail:
        bounds = detail_bounds(cells, hexes)
        ax.add_patch(Rectangle((bounds['xmin'], bounds['ymin']), bounds['xmax']-bounds['xmin'],
                               bounds['ymax']-bounds['ymin'], fill=False, ec='#4a5158', lw=.8))
        aggregate = cells.loc[cells.component_id.eq(0), ['x','y']].median()
        ax.annotate('Lymphoid\naggregate', xy=aggregate, xytext=(.76,.78), textcoords='axes fraction',
                    fontsize=8, ha='center', zorder=7, arrowprops=dict(arrowstyle='-', color='#4a5158', lw=.7),
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2))
        ax.text(.01,.76,'Tumour–T cell interface', transform=ax.transAxes, fontsize=8, zorder=7,
                va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2))
        ax.text(.39,.94,'Tumour (context)', transform=ax.transAxes, fontsize=8, zorder=7,
                ha='center', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2))
    add_scale_bar(ax, settings, 100 if detail else 500)

    cax = fig.add_axes([.18, .17 if detail else .105, .64, .035])
    cbar = fig.colorbar(artist, cax=cax, orientation='horizontal',
                       ticks=[settings['vmin'], settings['vmax']],
                       extend='max' if settings['n_above_colour_limit'] else 'neither')
    cbar.ax.tick_params(labelsize=8, length=2)
    cbar.ax.set_xticklabels([f"{settings['vmin']:.3g}", f"{settings['vmax']:.3g}"])
    cbar.set_label('Marker score' if detail else 'Mean normalized marker expression', fontsize=8)
    return fig


def section_cells(run):
    cells = read_csv(run.input)
    ledger = align_ledger(cells, read_csv(run.ledger))
    return cells[['x','y','roi_id']].assign(input_row=np.arange(len(cells)), component_id=ledger.component_id)


def location_panel(cells, settings):
    """Show the illustrated components and neighbouring cells in a local window."""
    cells = cells.loc[cells.x.between(settings['xmin'], settings['xmax']) &
                      cells.y.between(settings['ymin'], settings['ymax'])]
    fig, ax = plt.subplots(figsize=(6.0, 4.25))
    fig.subplots_adjust(left=.03, right=.63, top=.88, bottom=.035)
    other = cells.loc[~cells.component_id.isin(REGION_COLOURS)]
    ax.scatter(other.x, other.y, s=1.1, c='#d5d8db', linewidths=0, rasterized=True)
    for _, component, label, _ in COMPONENTS:
        selected = cells.loc[cells.component_id.eq(component)]
        ax.scatter(selected.x, selected.y, s=2, c=REGION_COLOURS[component], linewidths=0, rasterized=True)
    frame_axes(ax, settings); ax.set_axis_off()
    aggregate = cells.loc[cells.component_id.eq(0), ['x', 'y']].median()
    ax.annotate('Lymphoid\naggregate', xy=aggregate, xytext=(.80,.73), textcoords='axes fraction',
                fontsize=8, ha='center', arrowprops=dict(arrowstyle='-', color='#4a5158', lw=.7),
                bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2))
    add_scale_bar(ax, settings, 500)
    ax.set_title('ROIs', fontsize=10, pad=9)
    handles = [Patch(facecolor=REGION_COLOURS[component], edgecolor='none', label=label)
               for _, component, label, _ in COMPONENTS]
    handles.append(Patch(facecolor='#d5d8db', edgecolor='none', label='Other cells'))
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.66, .5),
               frameon=False, fontsize=8, handlelength=1.1, handleheight=.9,
               labelspacing=.65, borderaxespad=0)
    return fig
