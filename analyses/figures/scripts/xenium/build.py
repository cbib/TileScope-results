"""A compact Xenium supplement: two cell/consensus examples and complete context."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from shapely.geometry.polygon import orient

from analyses.figures.common import ROOT, expression_colormap, figure_dir, save_panel, write_sources
from analyses.xenium_feasibility.analyse import read_csv
from .map_data import complete_area, evidence_root, runs
from tilescope_results.bundles import verify
from tilescope_results.paths import input_data_root
from .paired_components import prepare_comparison, expression_panel, location_panel, section_cells, detail_bounds, combined_bounds

CASES = (("GBM", "P28_post", "Stemlike_Interferon", "CN-Tumor - T cell interface", 784, "A"),
         ("Breast", "sample1_rep1", "ACTA2_FASN", "Invasive", 7, "B"))


def polygon_patches(geometry):
    """Preserve interior holes and multipart clipped hexagons."""
    if geometry.is_empty: return []
    if geometry.geom_type != "Polygon":
        return [p for part in geometry.geoms for p in polygon_patches(part)]
    geometry = orient(geometry, sign=1.)
    paths = []
    for ring in [geometry.exterior, *geometry.interiors]:
        vertices = np.asarray(ring.coords)
        codes = np.full(len(vertices), MplPath.LINETO, dtype=np.uint8)
        codes[0] = MplPath.MOVETO; codes[-1] = MplPath.CLOSEPOLY
        paths.append(MplPath(vertices, codes))
    return [PathPatch(MplPath.make_compound_path(*paths))]


def prepare_map(run, roi, gene, component=None):
    cells, hexes, settings = complete_area(run, roi, component)
    return cells, hexes, dict(settings, gene=gene)


def panel_settings(cells, hexes, settings, representation):
    """Local linear scale, with a disclosed upper percentile for sparse counts."""
    side = 1 if settings["gene"] == settings["feature_a"] else 2
    rois = settings["roi_id"].split(';')
    if representation == "cells":
        values = cells.loc[cells.roi_id.isin(rois), f"gene{side}"]
    elif representation == "ensemble":
        values = hexes.loc[hexes.roi_id.isin(rois), f"gene{side}_density_ensemble"]
    else:
        raise ValueError(f"Unknown map representation: {representation}")
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("No finite expression values in the displayed components")
    low, high = float(finite.min()), float(finite.max())
    vmin, vmax = low, float(np.quantile(finite, .99))
    if vmax == vmin and high > low:
        vmax = high
    if vmin == vmax:
        # Do not turn a uniform map into apparent variation or use a zero-width norm.
        vmin, vmax = min(0., low), max(0., high)
        if vmin == vmax:
            vmax = 1.
    return dict(settings, representation=representation, vmin=vmin, vmax=vmax,
                displayed_min=low, displayed_max=high, n_finite_coloured_units=len(finite),
                n_above_colour_limit=int((finite > vmax).sum()), upper_colour_percentile=99,
                colour_limit_rule="One linear scale across the complete selected components, from minimum to 99th percentile; detail views reuse it; cells and consensus retain separate limits; larger values saturate")


def map_panel(cells, hexes, settings, representation):
    settings = panel_settings(cells, hexes, settings, representation)
    fig, ax = plt.subplots(figsize=(3.35, 3.45))
    fig.subplots_adjust(left=.04, right=.97, top=.87, bottom=.20)
    roi, gene = settings["roi_id"], settings["gene"]
    side = 1 if gene == settings["feature_a"] else 2
    cmap, norm = expression_colormap(), Normalize(settings["vmin"], settings["vmax"], clip=True)
    other = cells.loc[~cells.roi_id.eq(roi)]
    ax.scatter(other.x, other.y, s=2, c="#dedede", linewidths=0, rasterized=True)
    d = cells.loc[cells.roi_id.eq(roi)]
    if representation == "cells":
        artist = ax.scatter(d.x, d.y, c=d[f"gene{side}"], cmap=cmap, norm=norm,
                            s=float(np.clip(5000/len(d), .7, 4.)), linewidths=0, rasterized=True)
        title = "Cells"
    else:
        excluded = d.loc[~d.tessellation_eligible]
        ax.scatter(excluded.x, excluded.y, s=3, c="#a7a7a7", linewidths=0, rasterized=True)
        patches, values = [], []
        for row in hexes.loc[hexes.roi_id.eq(roi)].to_dict("records"):
            pieces = polygon_patches(row["geometry"])
            patches.extend(pieces); values.extend([row[f"gene{side}_density_ensemble"]] * len(pieces))
        artist = PatchCollection(patches, cmap=cmap, norm=norm, linewidths=.12, edgecolors="#ffffff55")
        artist.set_array(np.asarray(values)); ax.add_collection(artist)
        title = "Ensemble consensus"
    ax.set_xlim(settings["xmin"], settings["xmax"]); ax.set_ylim(settings["ymax"], settings["ymin"])
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(f"{gene} · {title}", fontsize=10, pad=9)
    label = roi.replace("_", " ") + (f" · component {settings['component_id']}" if settings.get('component_id') is not None else '')
    ax.text(.02, .02, label, transform=ax.transAxes, fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2))
    cax = fig.add_axes([.18, .105, .64, .035])
    cbar = fig.colorbar(artist, cax=cax, orientation="horizontal", ticks=[settings["vmin"], settings["vmax"]],
                        extend='max' if settings['n_above_colour_limit'] else 'neither')
    cbar.ax.tick_params(labelsize=8, length=2)
    cbar.ax.set_xticklabels([f"{settings['vmin']:.3g}", f"{settings['vmax']:.3g}"])
    cbar.set_label(f"{gene} expression", fontsize=9)
    return fig


def enrichment_contrasts(evidence=None):
    """Read existing whole-ROI ratios; do not recompute them for selected components."""
    root = Path(evidence or evidence_root()) / 'statistics'
    keys = [('GBM', 'P28_post', 'Stemlike_Interferon', ['CN-Lymphoid aggregate', 'CN-Tumor - T cell interface', 'CN-Tumor']),
            ('Breast', 'sample1_rep1', 'ACTA2_FASN', ['DCIS_1', 'DCIS_2', 'Invasive'])]
    rows = []
    for representation, file in [('cells', 'roi_enrichment'), ('ensemble', 'roi_consensus_enrichment')]:
        data = read_csv(root / f'{file}.csv.gz')
        for study, sample, pair, rois in keys:
            selected = data.loc[data.study.eq(study) & data['sample'].eq(sample) & data.pair.eq(pair) & data.roi_id.isin(rois)]
            if len(selected) != len(rois):
                raise ValueError('Incomplete enrichment comparison')
            for row in selected.to_dict('records'):
                for side in [1, 2]:
                    rows.append(dict(study=study, sample=sample, pair=pair, roi_id=row['roi_id'],
                                     gene=row[f'g{side}_id'], representation=representation,
                                     enrichment=row[f'g{side}_enrichment'], status=row[f'g{side}_enrichment_status'],
                                     statistic_unit='whole ROI', reference='other labelled ROIs in the same sample'))
    return pd.DataFrame(rows)


def enrichment_panel(data, study):
    d = data.loc[data.study.eq(study)]
    rois = ['CN-Lymphoid aggregate','CN-Tumor - T cell interface','CN-Tumor'] if study == 'GBM' else ['DCIS_1','DCIS_2','Invasive']
    genes = ['Stemlike','Interferon'] if study == 'GBM' else ['ACTA2','FASN']
    keys = [(roi, gene) for roi in rois for gene in genes]
    fig, ax = plt.subplots(figsize=(4.7, 2.85) if study == 'GBM' else (3.8, 2.85))
    fig.subplots_adjust(left=.51 if study == 'GBM' else .38, right=.94, top=.81, bottom=.23)
    for i, (roi, gene) in enumerate(keys):
        for representation, offset, marker, colour, label in [
            ('cells', -.10, 'o', '#3e5263', 'Cells'),
            ('ensemble', .10, 's', '#bc4a55', 'Ensemble')]:
            row = d.loc[d.roi_id.eq(roi) & d.gene.eq(gene) & d.representation.eq(representation)].iloc[0]
            if row.status != 'finite' or not np.isfinite(row.enrichment):
                raise ValueError('Selected enrichment contrast is not finite')
            ax.plot(row.enrichment, i+offset, marker=marker, color=colour, ms=4.5, ls='none', label=label if i==0 else None)
    ax.axvline(1, ls='--', lw=.8, color='#a5a5a5')
    short = lambda roi: roi.replace('CN-', '').replace('Tumor - T cell interface', 'Tumour–T cell interface').replace('Tumor', 'Tumour')
    ax.set_yticks(range(len(keys)), [f'{short(roi).replace("_"," ")} · {gene}' for roi,gene in keys], fontsize=7 if study == 'GBM' else 8)
    ax.set_ylim(len(keys)-.5, -.5); ax.set_xlim(0, max(3.2,d.enrichment.max()*1.08))
    ax.set_xlabel('Whole-ROI enrichment',fontsize=9); ax.tick_params(axis='x',labelsize=8)
    ax.tick_params(axis='y',length=0)
    for spine in ['top','right','left']: ax.spines[spine].set_visible(False)
    ax.legend(loc='lower center',bbox_to_anchor=(.42,1.02),ncol=2,frameon=False,fontsize=8)
    return fig


def build(output=None, evidence=None):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    evidence = Path(evidence or evidence_root())
    verify(input_data_root())
    verify(evidence)
    corr, support = (read_csv(evidence / f"{name}.csv") for name in ("correlations", "roi_cell_counts"))
    all_runs = list(runs())
    out = Path(output or figure_dir("FigureS7"))
    breast_out = out.with_name('FigureS8')
    for target in [out, breast_out]:
        if target.exists() and any(target.iterdir()):
            raise ValueError(f"Figure output must be empty: {target}")
    enrich = enrichment_contrasts(evidence)
    for study, sample, pair, roi, component, letter in CASES:
        tables, mapping, paths, settings = {}, {}, [], []
        target = out if study == 'GBM' else breast_out
        figure_name = 'FigureS7' if study == 'GBM' else 'FigureS8'
        run = next(r for r in all_runs if (r.study, r.sample, r.pair) == (study, sample, pair))
        if study == 'GBM':
            cells, hexes, context, context_hexes, geometry, summary, base_setting = prepare_comparison(run)
            tables['A_context_cells'] = context.assign(study=study, sample=sample, pair=pair)
            tables['A_context_hexagons'] = context_hexes.drop(columns='geometry').assign(study=study, sample=sample, pair=pair)
            tables['A_component_geometry'] = geometry.drop(columns='geometry')
            tables['A_component_summary'] = summary
            tables['A_section_cells'] = section_cells(run)
            stem = 'A_GBM_location'
            location_setting = dict(run.identity, **combined_bounds(context, context_hexes))
            location_setting.update(panel=stem, display_unit='local context window',
                component_ids='784;0;501',
                extent_rule='Complete interface, aggregate and neighbouring tumour components; 3% margin; other cells in this window shown in grey')
            settings.append(location_setting)
            save_panel(location_panel(tables['A_section_cells'], location_setting), target / 'panels', stem)
            mapping[stem] = ['A_section_cells', 'A_component_summary', 'display_settings']
            paths.append(run.output / 'component_geometry.csv.gz')
        else:
            cells, hexes, base_setting = prepare_map(run, roi, run.feature_a, component)
        tables[f"{letter}_cells"] = cells.assign(study=study, sample=sample, pair=pair)
        tables[f"{letter}_hexagons"] = hexes.drop(columns="geometry").assign(study=study, sample=sample, pair=pair)
        for side, gene in enumerate([run.feature_a, run.feature_b]):
            setting = dict(base_setting, gene=gene)
            for number, representation in enumerate(("cells", "ensemble"), 1):
                rendered_setting = panel_settings(cells, hexes, setting, representation)
                if study == 'GBM':
                    stem = f'B{side+1}_GBM_{gene}_consensus' if representation == 'ensemble' else f'E{side+1}_GBM_{gene}_cells_full'
                else:
                    stem = f'A{2*side+number}_Breast_{gene}_{representation}'
                settings.append(dict(rendered_setting, panel=stem))
                fig = (expression_panel(cells, hexes, geometry, rendered_setting, polygon_patches)
                       if study == 'GBM' else map_panel(cells, hexes, setting, representation))
                save_panel(fig, target / "panels", stem)
                mapping[stem] = [f"{letter}_cells", "display_settings"] + ([f"{letter}_hexagons"] if representation == "ensemble" else [])
                if study == 'GBM':
                    mapping[stem] += ['A_component_geometry', 'A_component_summary']
                if study == 'GBM' and representation == 'ensemble':
                    detail_stem = f'C{side+1}_GBM_{gene}_{representation}_detail'
                    detail_setting = dict(rendered_setting, **detail_bounds(cells, hexes))
                    detail_setting.update(display_unit='display window',
                        extent_rule='Square around complete aggregate, twice its maximum extent; includes surrounding interface; no statistics recomputed')
                    settings.append(dict(detail_setting, panel=detail_stem))
                    save_panel(expression_panel(cells, hexes, geometry, detail_setting, polygon_patches, detail=True),
                               target/'panels', detail_stem)
                    mapping[detail_stem] = mapping[stem].copy()
        paths.extend([run.input, run.ledger, run.output / "consensus_hexes.csv.gz"])
        stem = 'D_GBM_enrichment' if study == 'GBM' else 'B_Breast_enrichment'
        save_panel(enrichment_panel(enrich, study), target / 'panels', stem)
        mapping[stem] = ['enrichment_contrasts']
        tables.update(display_settings=pd.DataFrame(settings), correlations=corr.loc[corr.study.eq(study)],
                      roi_cell_counts=support.loc[support.study.eq(study)],
                      sample_inventory=read_csv(evidence / "sample_inventory.csv"),
                      enrichment_contrasts=enrich.loc[enrich.study.eq(study)])
        paths.extend(evidence / f"{name}.csv" for name in ("correlations", "roi_cell_counts", "sample_inventory"))
        paths.extend(evidence / 'statistics' / f'{name}.csv.gz' for name in ['roi_enrichment','roi_consensus_enrichment'])
        write_sources(figure_name, tables, mapping, inputs=paths, output=target,
                      notes=f'Input expression units: {run.expression_units}. Enrichment divides the high-expression fraction in each complete parent ROI by that in the other labelled ROIs. '
                            'Separate cell and consensus colour limits preserve visible contrast; detail maps inherit the corresponding full-field limits. '
                            'Cell coordinates, expression values, full hexagon geometry and whole-ROI enrichment are unchanged. '
                            'GBM E1/E2 are additional full-field cell maps for inspection, outside the suggested page layout. '
                            'Layout previews are assembled from the individual panels; editable SVG/PDF panels remain available.')
        from .layout import build_preview
        build_preview(target, figure_name)
        print(f'{figure_name} generated in {target}', flush=True)
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    build(args.output, args.evidence)
