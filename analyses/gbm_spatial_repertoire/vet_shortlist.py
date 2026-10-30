"""Vet a selected biological story against complete local exports and actual PDFs.

Not a core run, new descriptor, scenario classifier, or exhaustive visual census.
Prepared PDFs never imply completed review.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
EVALUATION = HERE.parent / 'greenwald_full_evaluation'
sys.path.insert(0, str(EVALUATION))
from local_frequency_analysis import VerifiedReader, csv, digest
from tilescope_results.paths import greenwald_evidence
from review import RESULTS, render_standard_pdf

OUT = greenwald_evidence('gbm_spatial_repertoire') / 'shortlist_vetting'
PAIRS = ['AC__OPC', 'NPC__Neuron', 'AC__Reactive-Ast', 'Mac__Reactive-Ast',
         'MES-Hyp__Reactive-Ast', 'AC__MES-Hyp', 'MES-Ast__MES-Hyp']
TRIAD = PAIRS[2:5]
SURFACES = ['S', 'T', 'CD', 'CE']


def weighted_quantile(v, w, q=.5):
    order = np.argsort(v)
    v, w = np.asarray(v)[order], np.asarray(w)[order]
    return float(v[np.searchsorted(np.cumsum(w), q*w.sum())]) if len(v) else np.nan


def typed(d, key):
    x = d[key].astype(float).copy()
    kind = key+'__kind'
    if kind in d:
        x = x.where(d[kind].eq('finite') & np.isfinite(x))
        if key.startswith(('enrichment_', 'joint_low_')):
            x = x.mask(d[kind].eq('saturated'), np.inf)
    return x


def summarize(d, x, threshold, *, metric, surface, context='all'):
    """Equal samples, equal local units per sample; unknown never becomes zero."""
    d = d[['sample', 'pair_id']].copy(); d['value'] = np.asarray(x)
    rows = []
    for pair, g in d.groupby('pair_id', sort=False):
        good = g[g.value.notna()]
        sample_n = good.groupby('sample').value.transform('size').to_numpy(float)
        weights = 1/sample_n if len(good) else np.array([])
        v = good.value.to_numpy(float)
        for sign in [1, -1]:
            hit = sign*v > sign*threshold
            # Negative direction means value<threshold, including ratio/CLES thresholds.
            per = good.assign(hit=hit).groupby('sample').hit.mean()
            rows.append(dict(pair_id=pair, metric=metric, surface=surface, context=context,
                threshold=threshold, sign=sign, n_total=len(g), n_evaluable=len(good),
                n_unavailable=len(g)-len(good), n_samples=good['sample'].nunique(),
                frequency=per.mean(), sample_frequency_q25=per.quantile(.25),
                sample_frequency_q75=per.quantile(.75), n_qualifying=int(hit.sum()),
                qualifying_median=weighted_quantile(v[hit], weights[hit]),
                all_median=weighted_quantile(v, weights)))
    return pd.DataFrame(rows)


def signed_summary(d, x, band, **kwargs):
    # Different thresholds for positive and negative signed effects.
    a = summarize(d, x, band, **kwargs)
    b = summarize(d, x, -band, **kwargs)
    return pd.concat([a[a.sign.eq(1)], b[b.sign.eq(-1)]], ignore_index=True)


def triad_table(roi):
    """Actual same-sample/same-ROI linkage, not a conjunction of averages."""
    parts = []
    for surface in SURFACES:
        d = roi[roi.pair_id.isin(TRIAD)].copy()
        d['value'] = typed(d, 'correlation_'+surface)
        w = d.pivot(index=['sample', 'roi'], columns='pair_id', values='value').reindex(columns=TRIAD)
        w.columns = ['AC_Reactive', 'Mac_Reactive', 'Hyp_Reactive']
        w['source_spots'] = d.groupby(['sample', 'roi']).source_spots.first()
        w['all_evaluable'] = w.iloc[:, :3].notna().all(axis=1)
        w['minimum_signed_pattern'] = pd.concat([w.AC_Reactive, w.Mac_Reactive, -w.Hyp_Reactive], axis=1).min(axis=1, skipna=False)
        w['surface'] = surface
        for band in [.1, .2, .3]:
            w[f'positive_positive_negative_{band}'] = np.where(w.all_evaluable,
                (w.AC_Reactive > band) & (w.Mac_Reactive > band) & (w.Hyp_Reactive < -band), np.nan)
            w[f'all_positive_{band}'] = np.where(w.all_evaluable,
                (w.AC_Reactive > band) & (w.Mac_Reactive > band) & (w.Hyp_Reactive > band), np.nan)
        parts.append(w.reset_index())
    return pd.concat(parts, ignore_index=True)


def matched_changes(roi):
    d = roi[['sample', 'pair_id', 'roi', 'source_spots']].copy()
    for s in SURFACES:
        d['r_'+s] = typed(roi, 'correlation_'+s)
    d['matched_S_CE'] = d[['r_S', 'r_CE']].notna().all(axis=1)
    d['delta_CE_minus_S'] = (d.r_CE-d.r_S).where(d.matched_S_CE)
    for band in [.1, .2, .3]:
        for s in ['S', 'CE']:
            d[f'state_{s}_{band}'] = np.select([d['r_'+s].isna(), d['r_'+s] > band,
                d['r_'+s] < -band], ['unavailable', 'positive', 'negative'], default='small')
    return d


def prepare():
    cohort = VerifiedReader(greenwald_evidence('scenario_cohort'))
    numerical = VerifiedReader(greenwald_evidence('two_level_evidence'))
    shared = VerifiedReader(greenwald_evidence('gbm_spatial_repertoire'))
    if len({cohort.manifest['core'], numerical.manifest['core'], shared.manifest['core']}) != 1:
        raise ValueError('Mixed core evidence')
    scope = cohort.read('sample_scope.csv'); pair_scope = cohort.read('pair_scope.csv')
    samples = scope.loc[scope.included_this_build, 'sample'].tolist()
    if len(samples) != 30: raise ValueError('Expected 30 GBM samples')
    rs, cs, bs = [], [], []
    for sample in samples:
        r = cohort.read(f'by_sample/{sample}/roi_evidence.csv.gz')
        c = cohort.read(f'by_sample/{sample}/component_evidence.csv.gz')
        b = numerical.read(f'details/{sample}_boundary_values.csv.gz')
        source = shared.read(f'details/{sample}_source_programs.csv.gz')
        rs.append(r[r.pair_id.isin(PAIRS)]); cs.append(c[c.pair_id.isin(PAIRS)])
        b = b[b.pair_id.isin(PAIRS)].copy()
        b['program_source_sd'] = b.program.map(source.std(numeric_only=True, ddof=0))
        for s in SURFACES:
            b['standardized_'+s] = (b['interface_response_'+s]/b.program_source_sd.where(b.program_source_sd > 1e-12)).where(b['availability_status_'+s].eq('finite'))
        bs.append(b)
    roi, component, boundary = [pd.concat(x, ignore_index=True) for x in [rs, cs, bs]]
    if roi.duplicated(['sample', 'pair_id', 'roi']).any(): raise ValueError('Duplicate ROI')
    summaries = []
    for s in SURFACES:
        for metric, bands in [('correlation', [.1, .2, .3]), ('moran', [.05, .1, .15])]:
            for carrier in ([''] if metric == 'correlation' else ['_A|B', '_B|A']):
                x = typed(roi, metric+'_'+s+carrier)
                for band in bands:
                    summaries.append(signed_summary(roi, x, band, metric=metric+carrier, surface=s))
                    co = (typed(roi, 'enrichment_S_A') > 1) & (typed(roi, 'enrichment_S_B') > 1)
                    summaries.append(signed_summary(roi[co], x[co], band, metric=metric+carrier,
                                                   surface=s, context='both_source_enriched'))
        if s != 'CD':
            for metric, threshold in [('enrichment', 1), ('CLES', .5), ('log2FC', 0)]:
                for carrier in ['A', 'B'] + (['AB'] if metric == 'enrichment' else []):
                    summaries.append(summarize(roi, typed(roi, f'{metric}_{s}_{carrier}'), threshold,
                                              metric=metric+'_'+carrier, surface=s))
    territory = roi[['sample', 'pair_id', 'roi']].copy()
    a, b, j = [typed(roi, 'enrichment_S_'+c) for c in ['A', 'B', 'AB']]
    known = a.notna() & b.notna()
    territory['both_enriched'] = np.where(known, (a > 1) & (b > 1), np.nan)
    territory['one_enriched'] = np.where(known, (a > 1) ^ (b > 1), np.nan)
    territory['joint_high_enriched'] = np.where(j.notna(), j > 1, np.nan)
    for name in ['both_enriched', 'one_enriched', 'joint_high_enriched']:
        summaries.append(summarize(territory, territory[name], .5, metric=name, surface='S'))
    for shell in range(4):
        x = component[f'shell{shell}_value'].where(component[f'shell{shell}_status'].eq('finite'))
        for band in [.05, .1, .15]:
            summaries.append(signed_summary(component, x, band, metric='shell_'+str(shell), surface='component'))
    boundary_summary = []
    for carrier, g in boundary.groupby('carrier'):
        for s in SURFACES:
            for band in [.1, .2, .3]:
                boundary_summary.append(signed_summary(g, g['standardized_'+s], band,
                    metric='response_'+carrier, surface=s, context='all_contacts'))
    triad = triad_table(roi); changes = matched_changes(roi)
    triad_counts = []
    for s, g in triad.groupby('surface'):
        for band in [.1, .2, .3]:
            for query in ['positive_positive_negative', 'all_positive']:
                x = g[f'{query}_{band}']; good = g[x.notna()]
                f = good.assign(hit=x.loc[good.index]).groupby('sample').hit.mean()
                triad_counts.append(dict(surface=s, band=band, query=query, n_evaluable=len(good),
                    n_hit=int(x.sum()), n_samples_evaluable=len(f),
                    n_samples_with_hit=g.loc[x.eq(1), 'sample'].nunique(), mean_roi_frequency=f.mean()))
    OUT.mkdir(exist_ok=True); details = OUT/'details'; details.mkdir(exist_ok=True)
    for name, d in [('roi_evidence', roi), ('component_evidence', component), ('boundary_evidence', boundary)]:
        csv(d, details/(name+'.csv.gz'))
    csv(pd.concat(summaries, ignore_index=True), OUT/'statistical_summary.csv')
    csv(pd.concat(boundary_summary, ignore_index=True), OUT/'boundary_summary.csv')
    csv(territory, OUT/'territory.csv'); csv(changes, OUT/'matched_support.csv')
    csv(triad, OUT/'reactive_triad_rois.csv'); csv(pd.DataFrame(triad_counts), OUT/'reactive_triad_summary.csv')
    # Preserve the numerical selection and include the strongest contrary locality.
    eligible = triad[triad.surface.eq('S') & triad.source_spots.ge(100) & triad.all_evaluable]
    chosen = eligible.sort_values('minimum_signed_pattern', ascending=False).drop_duplicates('sample').head(2).copy()
    chosen['selection'] = 'largest_minimum_signed_source_pattern'
    contrary = eligible.sort_values('minimum_signed_pattern').head(1).copy()
    contrary['selection'] = 'strongest_contrary_locality'
    chosen = pd.concat([chosen, contrary], ignore_index=True)
    context = roi.assign(typed_A=typed(roi, 'enrichment_S_A'), typed_B=typed(roi, 'enrichment_S_B'))
    m = context[context.pair_id.isin(TRIAD)].pivot(index=['sample','roi'], columns='pair_id', values='typed_A')
    m['Reactive'] = context[context.pair_id.eq(TRIAD[0])].set_index(['sample','roi']).typed_B
    good_context = m[(m[TRIAD[0]] > 1) & (m[TRIAD[1]] > 1) & (m.Reactive > 1)].index
    rich = eligible[eligible.set_index(['sample','roi']).index.isin(good_context)
                    & eligible.minimum_signed_pattern.gt(.1)].sort_values('source_spots', ascending=False).head(1).copy()
    rich['selection'] = 'post_inspection_largest_program_rich_control'
    chosen = pd.concat([chosen, rich], ignore_index=True)
    csv(chosen, OUT/'review_selection.csv')
    views = {(r['sample'], pair) for r in chosen.to_dict('records') for pair in TRIAD}
    views |= {('248_T', 'AC__OPC'), ('262_T', 'NPC__Neuron'),
              ('GBM_ZH916bulk', 'AC__MES-Hyp'), ('260_T', 'MES-Ast__MES-Hyp')}
    mapping = pair_scope.set_index('pair_id'); maps = []; jobs = []; run_checks = {}
    previews = details/'previews'; previews.mkdir(exist_ok=True)
    for sample, pair in sorted(views):
        p = mapping.loc[pair]
        for name in ['effective_config.json', 'run_metadata.json', 'roi_correlations.csv',
                     'roi_consensus_correlations.csv', 'spot_ledger.csv']:
            # The cohort records one shared label ledger per sample.
            sibling = RESULTS/sample/('AC_OPC' if name == 'spot_ledger.csv' else p['pair'])/name
            actual = digest(sibling)
            if cohort.manifest['input_sha256'].get(str(sibling)) != actual:
                raise ValueError(f'Current run differs from numerical audit: {sibling}')
            run_checks[str(sibling)] = actual
        for kind, filename in [('source', 'expression_summary.pdf'), ('consensus', 'consensus_tessellation_summary.pdf')]:
            pdf = RESULTS/sample/p['pair']/filename
            h = digest(pdf)
            # Recheck a prior PDF hash when one exists; new pair views get their
            # first recorded PDF hash, with the sibling run identity checked above.
            expected = shared.manifest['input_sha256'].get(str(pdf))
            if expected is not None and h != expected: raise ValueError(f'Changed PDF: {pdf}')
            preview = previews/f'{sample}__{pair}__{kind}.png'
            jobs.append((pdf, preview))
            maps.append(dict(sample=sample, pair_id=pair, kind=kind, pdf=str(pdf), pdf_sha256=h,
                pdf_program_A=p.program_g1, pdf_program_B=p.program_g2,
                page=1, preview=str(preview.relative_to(OUT)), status='prepared_not_reviewed',
                pdf_provenance='prior_hash_reverified' if expected else 'new_hash_with_verified_sibling_run'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(render_standard_pdf, jobs))
    for r in maps: r['preview_sha256'] = digest(OUT/r['preview'])
    csv(pd.DataFrame(maps), OUT/'maps.csv')
    inputs = cohort.inputs | numerical.inputs | shared.inputs | run_checks
    inputs.update({r['pdf']:r['pdf_sha256'] for r in maps})
    manifest = dict(core=cohort.manifest['core'], n_tessellations=10,
        cross_grain_backfill_provenance=numerical.manifest.get('cross_grain_backfill_provenance'),
        samples=samples, pairs=PAIRS, input_sha256=inputs, no_core_runs=True,
        automatic_scenario_calls=False, prepared_views=len(views),
        producer_sha256={str(p):digest(p) for p in [Path(__file__),
                                                   HERE/'review.py', EVALUATION/'local_frequency_analysis.py']})
    write_manifest(manifest)
    render_html()
    print('Prepared', len(roi), 'ROI profiles and', len(views), 'source/consensus map views')


def write_manifest(manifest):
    manifest['output_sha256'] = {str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*')
                               if p.is_file() and p.name != 'provenance.json'}
    (OUT/'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')


def render_html():
    maps = pd.read_csv(OUT/'maps.csv'); roi = pd.read_csv(OUT/'details/roi_evidence.csv.gz')
    authored_path = HERE/'shortlist_vetting/observations.json'
    authored = json.loads(authored_path.read_text()) if authored_path.exists() else {'observations':[]}
    cards = []
    for (sample, pair), group in maps.groupby(['sample', 'pair_id'], sort=False):
        rows = roi[roi['sample'].eq(sample) & roi.pair_id.eq(pair)]
        columns = ['roi', 'source_spots', 'enrichment_S_A', 'enrichment_S_B', 'enrichment_S_AB',
                   'correlation_S', 'correlation_T', 'correlation_CD', 'correlation_CE', 'moran_S_A|B', 'moran_S_B|A']
        notes = [r for r in authored['observations'] if r['sample']==sample and pair in r['pairs']]
        first = group.iloc[0]
        cards.append(f'<article id="{sample}--{pair}"><h2>{sample} · {pair.replace("__", " / ")}</h2>'
                     +f'<p>PDF A = {first.pdf_program_A}; B = {first.pdf_program_B}. Original unmasked ROIs.</p>')
        for n in notes:
            cards.append('<div class="note"><b>ROI '+html.escape(str(n['roi']))+' · '+html.escape(n['status'])+'</b><p>'
                         +html.escape(n['observation'])+'</p><p>'+html.escape(n['caution'])+'</p></div>')
        if not notes: cards.append('<p>No new local assessment recorded. Prepared images are not review decisions.</p>')
        for r in group.itertuples():
            cards.append(f'<a href="{html.escape(r.preview)}"><img loading="lazy" src="{html.escape(r.preview)}" '
                         +f'alt="{sample} {pair} {r.kind}, PDF page 1"></a>')
        cards.append('<details><summary>All original ROIs: numerical evidence</summary><div class="scroll">'
                     +rows[columns].round(3).to_html(index=False, border=0)+'</div></details></article>')
    nav = ' · '.join(f'<a href="#{s}--{p}">{s}: {p.replace("__", "/")}</a>' for s,p in maps[['sample','pair_id']].drop_duplicates().itertuples(index=False))
    (OUT/'review.html').write_text(('''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Seven-pair biological story: evidence review</title><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:15px/1.5 system-ui;max-width:1450px;margin:24px auto;padding:0 20px;color:#253a43}img{width:100%}article{margin:45px 0;border-top:2px solid #c8d6dd}.note{background:#edf4f6;padding:10px}.scroll{overflow:auto}td,th{padding:6px}a{color:#147089}nav{font-size:13px}summary{cursor:pointer}</style></head><body>
<h1>Biology read through TileScope measurements</h1><p>Seven selected pairs, complete numerical evidence from 30 GBM samples; targeted spatial review, not a scenario census.</p>
<p><a href="__AUTHOR_FINDINGS.md__">Scientific verdict</a> · <a href="__FIGURE_INDEX__">Manuscript panels and source data</a> · <a href="__AUTHOR_README.md__">Definitions and reproduction</a> · <a href="statistical_summary.csv">Complete statistical summaries</a> · <a href="reactive_triad_summary.csv">Same-ROI Reactive-Ast query</a> · <a href="boundary_summary.csv">Boundary summaries</a></p>
<p><b>The story:</b> shared neural/glial territory, partner-dependent Reactive-Ast relationships, and internal versus boundary-associated hypoxic expression. The overview summarizes tendencies; the maps retain the actual regional contexts and exceptions.</p>
<p>Figure 3 and S4–S6 are supplied in the manuscript workspace. This page retains the selected intact core PDF previews and their authored review notes, not an alternative figure assembly.</p>
<p>Frequency = mean within-sample proportion of evaluable local units, not the fraction of samples with any positive ROI. Territory and source correlation have 185/192 evaluable ROIs per pair; ensemble correlation has 168/192. Unavailable measurements are excluded from their respective denominators, never treated as absence.</p>
<p><b>Useful spatial entry points:</b> <a href="#259_T--AC__Reactive-Ast">259_T ROI 2</a> shows broad program-rich territory despite modest native correlations; <a href="#GBM_ZH881T1--MES-Hyp__Reactive-Ast">GBM_ZH881T1 ROIs 2 and 4</a> show a hypoxic relationship changing with regional context; <a href="#260_T--MES-Ast__MES-Hyp">260_T interface 4→2</a> shows a hypoxic boundary field despite near-zero responder association.</p>
<p>All local data, including weak/contrary ROIs, remain available. Scenario interpretations require the maps; positive correlation is not synonymous with enriched overlap. These selected spatial assessments are exploratory, not an exhaustive scenario inventory.</p><nav>'''
        +nav+'</nav>'+''.join(cards)+'</body></html>').replace("__AUTHOR_FINDINGS.md__", (HERE/"FINDINGS.md").as_uri()).replace("__AUTHOR_README.md__", (HERE/"README.md").as_uri()).replace('__FIGURE_INDEX__', (HERE.parent/'figures/index.html').as_uri()))
    manifest = json.loads((OUT/'provenance.json').read_text())
    write_manifest(manifest)


def validate_locality(observation, pair, roi, boundary):
    """A local interpretation must name a real ROI and, if used, ordered contact."""
    for key in ['sample', 'roi', 'pairs', 'status', 'origin', 'observation', 'caution']:
        if key not in observation or not observation[key]:
            raise ValueError(f'Missing observation field: {key}')
    sample, rid = observation['sample'], str(observation['roi'])
    match = roi['sample'].eq(sample) & roi.pair_id.eq(pair) & roi.roi.astype(str).eq(rid)
    if match.sum() != 1:
        raise ValueError(f'Assessment must identify exactly one ROI: {sample}/{pair}/{rid}')
    if 'anchor_roi' in observation:
        contact = (boundary['sample'].eq(sample) & boundary.pair_id.eq(pair)
                   & boundary.anchor_roi_id.astype(str).eq(str(observation['anchor_roi']))
                   & boundary.responder_roi_id.astype(str).eq(rid))
        if not contact.any():
            raise ValueError(f'Assessment has no actual ordered interface: {sample}/{pair}')


def record():
    manifest = json.loads((OUT/'provenance.json').read_text())
    maps = pd.read_csv(OUT/'maps.csv')
    authored = json.loads((HERE/'shortlist_vetting/observations.json').read_text())
    roi = pd.read_csv(OUT/'details/roi_evidence.csv.gz', dtype={'roi':str})
    boundary = pd.read_csv(OUT/'details/boundary_evidence.csv.gz',
                           dtype={'anchor_roi_id':str, 'responder_roi_id':str})
    prepared = set(zip(maps['sample'], maps.pair_id))
    rows = []
    for r in authored['observations']:
        for pair in r['pairs']:
            validate_locality(r, pair, roi, boundary)
            if (r['sample'], pair) not in prepared: raise ValueError('Assessment outside prepared maps')
            selected = maps[maps['sample'].eq(r['sample']) & maps.pair_id.eq(pair)]
            for m in selected.itertuples():
                if digest(Path(m.pdf)) != m.pdf_sha256 or digest(OUT/m.preview) != m.preview_sha256:
                    raise ValueError('Reviewed asset changed')
            rows.append({**r, 'pair_id':pair, 'pairs':json.dumps(r['pairs']),
                         'pdf_hashes':json.dumps(dict(zip(selected.kind, selected.pdf_sha256)))})
    csv(pd.DataFrame(rows), OUT/'local_assessments.csv')
    manifest['authored_observation_count'] = len(authored['observations'])
    manifest['reviewed_pair_localities'] = len(rows)
    manifest['reviewer'] = 'assistant exploratory visual assessment; requires author review before manuscript'
    write_manifest(manifest); render_html()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    record() if args.record else prepare()
