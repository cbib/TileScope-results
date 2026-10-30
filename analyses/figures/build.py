"""Build manuscript panels and source data from saved results only.

Run: python -m analyses.figures.build [all|overview|benchmark|greenwald|mouse|xenium]
This entry point never generates datasets or calls a TileScope analysis API.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from analyses.figures.common import figure_dir, save_panel, write_sources, write_index
from tilescope_results.paths import input_data_root, supplementary_data_root, resolve_reference


def overview():
    from .scripts.overview import build as a, plot_panel_b as b, plot_panel_c as c
    out=figure_dir('Figure1'); panels=out/'panels'
    run,original,frames=b.load_saved_run()
    a.render_panels(frames['source'],frames['geometry'],panels)
    b.render(frames,panels)
    _,_,e,validation=c.prepare_evidence()
    c.render(e,panels)
    tables={'spots':e['source'], 'component_geometry':frames['geometry'],
            'tiles':frames['tiles'].loc[frames['tiles'].replicate_index.isin(b.REPLICATE_INDICES)],
            'hexes':frames['hexes'], 'enrichment':e['display_enrichment'],
            'component_correlations':e['components'], 'moran':e['moran'].loc[e['moran'].roi_id.eq(1)],
            'distance':e['distance'].loc[e['distance'].component_id.eq(0)], 'boundary':e['boundary'],
            'replicate_correlations':e['aggregation_replicates'],
            'source_correlation':e['aggregation_source'].to_frame().T,
            'consensus_correlation':e['aggregation_consensus'].to_frame().T,
            'colour_scale':pd.DataFrame({'value':[0,.25,.5,.75,1], 'colour':[v for _,v in a.EXPRESSION_COLOR_STOPS]})}
    mapping={}
    for p in sorted(panels.glob('*.svg')):
        stem=p.stem
        if stem.startswith('A') or stem.startswith('B0'): keys=['spots','component_geometry','colour_scale']
        elif stem.startswith('B1'): keys=['tiles','component_geometry','colour_scale']
        elif stem.startswith('B2'): keys=['hexes','component_geometry','colour_scale']
        elif stem.startswith('B_'): keys=['colour_scale']
        else: keys={'C1_enrichment':['spots','enrichment'],
                    'C2_internal_association':['spots','component_correlations','moran'],
                    'C3_association_over_distance':['spots','distance'],
                    'C4_boundary_response':['spots','boundary'],
                    'C5_sensitivity_to_aggregation':['replicate_correlations','source_correlation','consensus_correlation']}[stem]
        mapping[stem]=keys
    notes=('Synthetic coordinates and continuous fields A/B use arbitrary units. Expression maps use the fixed 0–1 scale. '
           'The three displayed tessellations have indices 0, 1 and 2; consensus combines all ten. '
           'High-expression membership uses the full-input 75th percentile and strict >; enrichment divides '
           'ROI high-expression fractions by those in the other ROIs. Joint-high requires both genes high. '
           'C2 separates ROI 1 components; its Moran inset uses the complete ROI. C3 uses component 0, with '
           'distance bands 0.5–1.5, 1.5–2.5, 2.5–4.5 and 4.5–8.5 median nearest-neighbour spacings. '
           'C4 compares means inside responder ROI 2: near hops 0–1, buffer 2, far ≥3 from ROI 1 contact. '
           'C5 uses complete ROI 3 at spots, across ten tessellations and in ensemble consensus. '+json.dumps(validation))
    write_sources('Figure1',tables,mapping,
                  inputs=[input_data_root()/'overview/spots.csv', *run.glob('*.csv')],notes=notes)
    print('Figure1: 14 individual graphics from retained plotting data.',flush=True)


def benchmark():
    from .scripts.benchmark import build as maps, build_panel_b as b, build_panel_b_profiles as profiles, build_panel_c as c
    from analyses.scenario_discriminability.recognition.full_catalogue_clustering import MODES
    from scipy.cluster.hierarchy import dendrogram
    data=maps.prepare()
    b.style()
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
    map_tables={}; map_settings=[]; mappings={'Figure2':{},'FigureS1':{}}
    inputs=[]
    for d in data:
        spec=d['spec'];f=d['frame']; key=spec.slug
        map_tables[key]=f.loc[f.display_visible].copy()
        map_settings.append(dict(case=spec.case,slug=key,role=spec.role,colour_min=d['colour_limits'][0],
            colour_max=d['colour_limits'][1],x_min=d['bounds'][0],y_min=d['bounds'][1],
            x_max=d['bounds'][2],y_max=d['bounds'][3],median_spacing=d['spacing']))
        inputs.extend(d['source_paths'])
        for figure in (['Figure2','FigureS1'] if spec.main else ['FigureS1']):
            for feature,title in [('gene1','Gene A'),('gene2','Gene B')]:
                fig=plt.figure(figsize=(2.4,2.65),facecolor='none')
                fig.text(.5,.94,title,fontsize=11,va='center',ha='center')
                maps.map_axes(fig,[.045,.08,.91,.79],d,feature)
                stem=f'{key}_{feature}'
                save_panel(fig,figure_dir(figure)/'panels',stem)
                mappings[figure][stem]=[key,'display_settings']
    map_tables['display_settings']=pd.DataFrame(map_settings)
    write_sources('FigureS1',map_tables,mappings['FigureS1'],inputs=inputs,
                  notes='Ten specified benchmark modes. Rows are displayed source spots; context and expression masks are explicit. Coordinates and scores are unchanged.')
    p=profiles.prepare();q=b.prepare(); b.style()
    out=figure_dir('Figure2')
    save_panel(profiles.plot_profiles(p),out/'panels','B0_clustered_measurement_profiles')
    save_panel(b.plot_distance(q['tables']),out/'panels','B1_correlation_with_distance')
    keys={k for v in mappings['Figure2'].values() for k in v}
    tables={k:map_tables[k] for k in keys}
    tables.update(measurement_values=p['traces'],column_definitions=p['specs'],
                  clustering_matrix=p['primary']['matrix'].reset_index(),clustering_scaling=p['primary']['scale'].reset_index(),
                  display_order=p['cases'].loc[p['primary']['order']].reset_index(),
                  branch_sensitivity=p['audits'],
                  linkage=pd.DataFrame(p['primary']['tree'],columns=['left','right','dissimilarity','n_cases']),
                  distance_curves=b.combined_curve_values(q['tables']),distance_observations=q['tables']['radial'],
                  component_correlations=q['tables']['correlations'])
    mappings['Figure2']['B0_clustered_measurement_profiles']=['measurement_values','column_definitions','clustering_matrix','clustering_scaling','display_order','linkage','branch_sensitivity']
    mappings['Figure2']['B1_correlation_with_distance']=['distance_curves','distance_observations','component_correlations']
    numeric_inputs=[resolve_reference(k) for k in {*p['source_sha256'],*q['source_sha256']}]
    comparator_input=c.comparator_benchmark_root()/'comparator_values.csv'
    comparison=c.build_matrix(c.load_values(comparator_input), row_order=p['primary']['order'])
    c.save(c.plot(comparison),out/'panels','C0_statistics_by_method')
    tables.update(C0_comparator_values=comparison['raw'].reset_index(),
                  C0_standardised_matrix=comparison['standardised'].reset_index(),
                  C0_scaling=comparison['scale'].rename_axis('').reset_index(),
                  C0_column_definitions=pd.DataFrame([
                      dict(method=m,source=src,direction=d,label=label) for m,src,d,label in c.STATS]))
    mappings['Figure2']['C0_statistics_by_method']=[
        'C0_comparator_values','C0_standardised_matrix','C0_scaling','C0_column_definitions','display_order']
    numeric_inputs.append(comparator_input)
    write_sources('Figure2',tables,mappings['Figure2'],inputs=inputs+numeric_inputs,
        notes='Main numerical panels: 18 constructions and six modes. B0 uses 12 source statistics; C0 shows comparator methods in the same case order, with per-column population z-scores clipped to ±2 for display only. Raw values, display transforms, standardization, statistical units and row order are separate tables.')
    save_panel(profiles.plot_profiles(p,grains=True),figure_dir('Benchmark')/'panels','matching_measurements_across_grains')
    write_sources('Benchmark',{'measurements':p['grain_traces'],'column_definitions':p['grain_specs'],
                              'display_order':p['cases'].loc[p['primary']['order']].reset_index()},
        {'matching_measurements_across_grains':['measurements','column_definitions','display_order']},
        inputs=numeric_inputs,
        notes='Retained diagnostic, not an additional numbered manuscript figure. Matching statistics across spatial supports; saved benchmark uses three tessellations, not a convergence study.')
    out=figure_dir('FigureS3')
    save_panel(b.plot_boundary_heatmap(q['tables']),out/'panels','A_boundary_removal_controls')
    save_panel(b.plot_relocated_controls(q['tables']),out/'panels','B_boundary_relocated_controls')
    tables={k:v for k,v in q['tables'].items() if k in ['inventory','membership','interfaces','correlations']}
    tables['boundary_heatmap']=b.boundary_heatmap_values(q['tables'])
    write_sources('FigureS3',tables,{p:list(tables) for p in ['A_boundary_removal_controls','B_boundary_relocated_controls']},
                  inputs=numeric_inputs,notes='Within-responder measurements. Anchor/responder identities and source columns are retained. Panels are exported directly as vector graphics, not composed from PNGs.')
    full=supplementary_data_root(1)
    from tilescope_results.bundles import verify
    verify(full)
    cases=pd.read_csv(full/'cases.csv').set_index('case_id')
    tree=json.loads((full/'hierarchies.json').read_text())['primary']
    labels=[]
    for key in tree['case_order']:
        r=cases.loc[key];roi=str(r.a_roi_id) if r.a_roi_id==r.b_roi_id else f'{r.a_roi_id}/{r.b_roi_id}'
        labels.append(f'{MODES[r.class_label]} | {r.source_sample_id} | {roi}')
    fig,ax=plt.subplots(figsize=(7.3,8.2));fig.subplots_adjust(left=.08,right=.54,bottom=.07,top=.99)
    dendrogram(tree['linkage'],labels=labels,orientation='left',leaf_font_size=8.3,color_threshold=0,above_threshold_color='#526878',ax=ax)
    ax.invert_yaxis();ax.set_xlabel('Statistical dissimilarity')
    save_panel(fig,figure_dir('FigureS2')/'panels','A_full_catalogue_hierarchy')
    tables={p.stem:pd.read_csv(p) for p in [full/'cases.csv',full/'values.csv',full/'statistic_catalogue.csv']}
    tables['linkage']=pd.DataFrame(tree['linkage'],columns=['left','right','dissimilarity','n_cases'])
    tables['display_order']=pd.DataFrame({'case_id':tree['case_order'],'label':labels})
    write_sources('FigureS2',tables,{'A_full_catalogue_hierarchy':list(tables)},
        inputs=[full/'provenance.json',full/'hierarchies.json',full/'cases.csv',full/'values.csv',full/'statistic_catalogue.csv'],
        notes='Average-linkage hierarchy of 30 constructions using 160 statistical values. Scenario names annotate the hierarchy after clustering. The linkage and ordering are read from the saved full-catalogue analysis; they are not refitted by this renderer.')
    print('Figure2 and S1–S3: individual vector panels and linked numerical data.',flush=True)


def greenwald():
    from .scripts.greenwald import build as g
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':7,
        'axes.titlesize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    e=g.Evidence();e.prepare()
    groups=[('Figure3',[(f'{letter}{i}_{name}',draw) for letter,_,drawers in g.main_panels(e)
                        for i,(name,draw) in enumerate(drawers,1)])]
    groups += [('Figure'+number,[(f'{number}{chr(65+i)}_{key}',draw) for i,(key,draw) in enumerate(drawers)])
               for number,_,drawers in g.supplementary_panels(e)]
    for number,drawers in groups:
        for key,draw in drawers:
            size=(4.8,3.7) if number=='Figure3' and key.startswith('A') else ((3.2,3.3) if number=='Figure3' else (4.3,3.8))
            fig=plt.figure(figsize=size);draw(fig,(0,0,1,1));save_panel(fig,figure_dir(number)/'panels',key)
    selected=pd.DataFrame([e.row(s,p,r) for s,p,rs in g.CASES for r in rs])
    spots=[];hexes=[];outlines=[];bands=[];interfaces=[];radial=[]
    for (s,p),run in e.runs.items():
        f=run['frame'].copy(); f['spot_index']=range(len(f)); f['component_id']=run['ledger'].component_id
        f=f.rename(columns={'gene1':'score_A','gene2':'score_B'}).assign(sample=s,pair=p,program_A=run['names'][0],program_B=run['names'][1]);spots.append(f)
        hexes.append(run['hexes'].assign(sample=s,pair=p,program_A=run['names'][0],program_B=run['names'][1]))
        outlines.extend(dict(sample=s,pair=p,roi_id=r,geometry_wkt=geom.wkt) for r,geom in run['outlines'].items())
        radial.append(run['radial'].assign(sample=s,pair=p))
    for (s,p,a,r),(labels,hops,edges,stored) in e.boundaries.items():
        f=e.run(s,p)['frame'].copy();f['spot_index']=range(len(f))
        bands.append(f.assign(sample=s,pair=p,anchor=a,responder=r,band=labels,hops=hops))
        interfaces.append(stored.assign(sample=s,pair=p))
    settings=pd.DataFrame(e.map_settings)
    settings['bounds']=settings['bounds'].map(json.dumps)
    common={'roi_measurements':selected,'spots':pd.concat(spots),'hexes':pd.concat(hexes),
            'display_outlines':pd.DataFrame(outlines),'display_settings':settings.drop_duplicates(),
            'boundary_spots':pd.concat(bands),'boundary_responses':pd.concat(interfaces),'component_curves':pd.concat(radial)}
    components=e.read(g.RESULTS/'figure_evidence/details/component_evidence.csv.gz')
    for number,drawers in groups:
        if number=='FigureS4':
            tables={'summary':e.summary,'matched_support':e.matched,
                    'roi_observations':e.roi.loc[e.roi.pair_id.isin(g.PAIRS)],
                    'component_observations':components}
            mapping={k:list(tables) for k,_ in drawers}
        else:
            tables=common.copy();mapping={}
            for key,_ in drawers:
                if any(v in key for v in ['boundary_bands','reverse_bands']):keys=['boundary_spots','display_outlines']
                elif 'response' in key: keys=['boundary_responses','boundary_spots']
                elif key.startswith(('S6A','S6B')):keys=['component_curves']
                elif key.startswith('A'):
                    tables['cohort_summary']=e.summary;tables['cohort_observations']=e.roi.loc[e.roi.pair_id.isin(g.PAIRS)]
                    keys=['cohort_summary','cohort_observations']
                elif any(v in key for v in ['association','enrichment']):keys=['roi_measurements']
                else:keys=['spots','display_outlines','display_settings']+(['hexes'] if number=='Figure3' else [])
                mapping[key]=keys
            used={k for values in mapping.values() for k in values};tables={k:tables[k] for k in used}
        tables['validation']=pd.DataFrame(e.checks)
        write_sources(number,tables,mapping,inputs=e.inputs.paths,
            notes='30 IDH-wildtype GBM sections; 91 program pairs; 192 ROIs. Cohort frequencies give equal weight to sections and exclude unavailable measurements from their denominators. '
                  'Enrichment uses ratios >1. Pearson display cutoffs are ±0.20; the sensitivity panel uses 0.10, 0.20 and 0.30. Moran and distance-frequency display cutoffs are ±0.10; none is a significance threshold. '
                  'Spot, hex, component and ordered-interface measurements retain separate identities. Input Seurat scores are unchanged; per-program map limits are recorded in display_settings. '
                  'Distances use component median nearest-neighbour spacing. Boundary differences compare means inside the responder: near hops 0–1, excluded buffer 2, far ≥3; unreachable spots are excluded.')
    print('Figure3 and S4–S6: 29 individual panels; Supplementary Data 2 unchanged.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group',nargs='?',choices=['all','overview','benchmark','greenwald','mouse','xenium'],default='all')
    args=parser.parse_args()
    groups = ('overview', 'benchmark', 'greenwald', 'mouse', 'xenium') if args.group == 'all' else (args.group,)
    for group in groups:
        # Figure-specific styles must not depend on which group ran first.
        with matplotlib.rc_context():
            plt.rcdefaults()
            if group == 'mouse':
                from .scripts.mouse_publication.build import build
                build()
            elif group == 'xenium':
                from analyses.xenium_feasibility.publish import publish
                publish(replace=True)
            else:
                {'overview': overview, 'benchmark': benchmark, 'greenwald': greenwald}[group]()
    write_index()


if __name__=='__main__':main()
