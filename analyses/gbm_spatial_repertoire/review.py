"""Verify shared source fields and marginals for the current 30-section cohort.

Authored historical observations remain review records, not automatic labels.
No TileScope runs, scenario classification or reconstruction of retired browsers.
"""
from __future__ import annotations
from tilescope_results.paths import resolve_reference

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
EVALUATION = HERE.parent / 'greenwald_full_evaluation'
sys.path.insert(0, str(EVALUATION))
from local_frequency_analysis import VerifiedReader, digest, csv
from tilescope_results.paths import data_root, greenwald_evidence

COHORT = greenwald_evidence('scenario_cohort')
NUMERICAL = greenwald_evidence('two_level_evidence')
DATA = data_root() / 'heiland_tyrosh'
RESULTS = DATA / 'CurrentResults'
OUT = greenwald_evidence('gbm_spatial_repertoire')
DETAILS = OUT / 'details'
STATUSES = {'scenario_compatible', 'candidate', 'ambiguous', 'not_supported_here', 'geometry_only'}


def validate_authored(authored, maps, modes):
    """Overview inspection is explicit and never implies all localities reviewed."""
    inspected=[(v['sample'],v['pair_id']) for v in authored['inspected_views']]
    prepared=set(zip(maps['sample'],maps.pair_id))
    if len(inspected)!=len(set(inspected)):raise ValueError('Duplicate inspected view')
    if not set(inspected)<=prepared:raise ValueError('Inspected view outside prepared batch')
    seen=set()
    for r in authored['observations']:
        view=(r['sample'],r['pair_id'])
        if view not in inspected:raise ValueError('Assessment outside explicitly inspected maps')
        if (r['family'],r['mode']) not in modes:raise ValueError('Unknown scenario mode')
        if r['status'] not in STATUSES:raise ValueError('Invalid review status')
        if not r['observation'].strip() or not r['caution'].strip():raise ValueError('Missing observation/caution')
        key=view+(r['target_id'],r['family'],r['mode'],r.get('orientation',''))
        if key in seen:raise ValueError('Duplicate authored decision')
        seen.add(key)


def marginal_records(roi):
    """Carry source/T/CE marginal outputs once per program/ROI IF identical."""
    parts=[]
    for carrier in ['A','B']:
        rename={f'{metric}_{surface}_{carrier}'+suffix:f'{metric}_{surface}'+suffix
            for metric in ['enrichment','CLES','log2FC'] for surface in ['S','T','CE']
            for suffix in ['', '__kind', '__status', '__low', '__high']}
        rename.update({f'parent_high_{part}_S_{carrier}':f'parent_high_{part}_S'
                       for part in ['spots','fraction','field_status']})
        columns=['sample','pair_id','roi','source_spots','support_class','coverage', 'program_'+carrier]+list(rename)
        g=roi[columns].rename(columns=rename|{'program_'+carrier:'program'})
        parts.append(g)
    all_copies=pd.concat(parts,ignore_index=True)
    keys=['sample','roi','program']
    status_columns=[c for c in all_copies if c.endswith('__status')]
    value_columns=[c for c in all_copies if c not in keys+['pair_id']+status_columns]
    counts=all_copies.groupby(keys).pair_id.nunique()
    if not counts.eq(13).all():raise ValueError('Expected all 13 partner copies per marginal ROI profile')
    equality=all_copies.groupby(keys)[value_columns].nunique(dropna=False)
    if equality.gt(1).any().any():
        bad=equality.where(equality.gt(1)).stack().index.tolist()[:8]
        raise ValueError(f'Marginal outputs depend on partner; do not deduplicate: {bad}')
    result=[]
    for _,g in all_copies.groupby(keys,sort=False):
        r=g.iloc[0].drop('pair_id').to_dict();different={}
        for column in status_columns:
            if g[column].nunique(dropna=False)>1:
                different[column]=dict(zip(g.pair_id,g[column].fillna('missing')))
                r[column]='partner_dependent_consult_original_pair_status'
        r['partner_dependent_statuses']=json.dumps(different)
        r['n_partner_dependent_status_fields']=len(different)
        r['n_verified_partner_copies']=13
        result.append(r)
    return pd.DataFrame(result)


def input_equivalence(sample,pairs,ledger,recorded_hashes,input_hashes):
    """Verify all 91 current source inputs: geometry, labels and each field copy."""
    reference=None;fields={};emissions={};rows=[]
    for pair in pairs.itertuples():
        path=DATA/'CSVs'/f'{sample}_expression_data_{pair.pair}.csv'
        current=digest(path);input_hashes[str(path)]=current
        if str(path) in recorded_hashes and recorded_hashes[str(path)]!=current:
            raise ValueError(f'Historical source geometry changed: {path}')
        src=pd.read_csv(path,dtype={'roi_id':str})
        geometry=src[['x','y','roi_id']]
        if reference is None:reference=geometry
        elif not geometry.equals(reference):raise ValueError(f'Source rows/geometry differ across pairs: {sample}')
        if not np.array_equal(src.roi_id.astype(str),ledger.roi_id.astype(str)):
            raise ValueError(f'Source labels do not match current ledger: {sample}')
        for program,col in [(pair.program_g1,'gene1'),(pair.program_g2,'gene2')]:
            values=src[col].to_numpy()
            if program in fields and not np.array_equal(fields[program],values,equal_nan=True):
                raise ValueError(f'Source program differs across partners: {sample}/{program}')
            fields[program]=values
            emissions.setdefault(program,[]).append(dict(pair=pair.pair,channel=col,path=str(path),sha256=current))
    wide=reference.copy()
    for program in sorted(fields):
        if len(emissions[program])!=13:raise ValueError('Incomplete source partner copies')
        wide[program]=fields[program]
        rows.append(dict(sample=sample,program=program,n_verified_partner_copies=13,
            source_rows=len(wide),source_vector_sha256=hashlib.sha256(fields[program].tobytes()).hexdigest(),
            all_input_copies=json.dumps(emissions[program]),
            claim='source vectors and row/ROI/coordinate identity match; not a scenario assignment'))
    wide.insert(0,'source_row',np.arange(len(wide)))
    return pd.DataFrame(rows),wide


def program_contacts(boundary):
    keys=['sample','anchor_roi_id','responder_roi_id','program']
    cols=[c for c in boundary if c.startswith(('interface_response_','availability_status_',
        'source_near_mass_','source_far_mass_','represented_near_mass_','represented_far_mass_',
        'reachable_responder_fraction_'))]
    if not boundary.groupby(keys).pair_id.nunique().eq(13).all():raise ValueError('Missing contact partner copies')
    if boundary.groupby(keys)[cols].nunique(dropna=False).gt(1).any().any():
        raise ValueError('Contact response differs across partners; cannot reuse')
    return boundary[keys+cols].drop_duplicates(keys).assign(n_verified_partner_copies=13)


def render_standard_pdf(job):
    pdf,preview=job
    # Use the standard core PDF intact: no reconstruction, interpolation or
    # new field scaling. Cache is keyed by the exact PDF plus rasterization.
    stamp=preview.with_suffix('.sha256');key=digest(pdf)+' scale_to=2800 page=1'
    if preview.exists() and stamp.exists() and stamp.read_text()==key:return
    subprocess.run(['pdftoppm','-f','1','-singlefile','-scale-to','2800','-png',str(pdf),str(preview.with_suffix(''))],
                   check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    stamp.write_text(key)


def prepare(cohort_root=COHORT, numerical_root=NUMERICAL, output=OUT):
    cohort = VerifiedReader(cohort_root)
    numerical = VerifiedReader(numerical_root)
    scope = cohort.read('sample_scope.csv')
    pairs = cohort.read('pair_scope.csv')
    samples = scope.loc[scope.included_this_build, 'sample'].tolist()
    if len(samples) != 30 or len(pairs) != 91:
        raise ValueError('Expected 30 GBM sections and 91 pairs')
    output = Path(output)
    details = output / 'details'
    details.mkdir(parents=True, exist_ok=True)
    hashes, products, checks, marginals, contacts = {}, {}, [], [], []
    for sample in samples:
        roi = cohort.read(f'by_sample/{sample}/roi_evidence.csv.gz')
        boundary = numerical.read(f'details/{sample}_boundary_values.csv.gz')
        path = RESULTS / sample / 'AC_OPC/spot_ledger.csv'
        hashes[str(path)] = digest(path)
        if cohort.manifest['input_sha256'][str(path)] != hashes[str(path)]:
            raise ValueError('Source membership changed')
        ledger = pd.read_csv(path, dtype={'roi_id': str})
        check, wide = input_equivalence(sample, pairs, ledger,
                                       cohort.manifest['input_sha256'], hashes)
        name = f'details/{sample}_source_programs.csv.gz'
        csv(wide, output / name)
        products[name] = digest(output / name)
        checks.append(check)
        marginals.append(marginal_records(roi))
        contacts.append(program_contacts(boundary))
    for name, parts in [('source_equivalence.csv.gz', checks),
                        ('details/program_roi_evidence.csv.gz', marginals),
                        ('details/program_contact_evidence.csv.gz', contacts)]:
        csv(pd.concat(parts, ignore_index=True), output / name)
        products[name] = digest(output / name)
    (output / 'provenance.json').write_text(json.dumps(dict(
        core=cohort.manifest['core'], samples=samples, no_core_runs=True,
        input_sha256=cohort.inputs | numerical.inputs | hashes,
        output_sha256=products), indent=2) + '\n')
    print('Verified all source program copies and program-level evidence.', flush=True)


if __name__ == '__main__':
    prepare()
