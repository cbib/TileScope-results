"""Prepare linked, unmasked ROI/component/interface evidence from stored core exports.

No TileScope runs. Program order, cohort, support identities and
cross-pair spot/cell-interface copies are checked by the same contracts
used by the downstream readers.
"""
from __future__ import annotations
from tilescope_results.paths import greenwald_evidence
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from figure_common import HERE, Inputs, EXPECTED_CORE
from tilescope_results.paths import data_root
from local_frequency_analysis import digest
import evidence_profiles as evidence
from property_evidence import KEY, metric_name, pair_boundaries, cross_enriched_contact

OUT = greenwald_evidence('configuration_inputs')
RESULTS = data_root() / 'heiland_tyrosh/CurrentResults'
SCOPE = Path(__file__).resolve().parent.parent / 'gbm_spatial_repertoire/inputs'


def reference_inventory(inputs, samples=None, ref=None):
    if samples is None and ref is None:
        scope = inputs.csv(SCOPE / 'sample_scope.csv')
        samples = scope.loc[scope.included_this_build, 'sample'].tolist()
        ref = inputs.csv(SCOPE / 'pair_scope.csv')
    elif samples is None or ref is None:
        raise ValueError('Provide both samples and pair reference, or neither')
    if not samples or len(samples) != len(set(samples)) or ref.empty or ref.pair_id.duplicated().any():
        raise ValueError('Empty or duplicate sample/pair selection')
    rows, universes = [], {}
    for sample in samples:
        for r in ref.itertuples():
            a, b = r.pair_id.split("__")
            options = [pair for pair in [a+"_"+b, b+"_"+a]
                       if (RESULTS/sample/pair).is_dir()]
            if len(options) != 1:
                raise ValueError(f"Ambiguous/missing pair orientation: {sample}/{r.pair_id}")
            pair = options[0]
            folder, _ = inputs.run(RESULTS, sample, pair)
            support = inputs.csv(folder/"roi_support_status.csv")
            universe = set(support.roi_id)
            if len(support) != len(universe):
                raise ValueError("Duplicate ROI support rows")
            if sample in universes and universe != universes[sample]:
                raise ValueError("ROI universe changes across pairs")
            universes[sample] = universe
            rows.extend(dict(sample=sample, pair_id=r.pair_id, pair=pair, roi=roi)
                        for roi in sorted(universe))
    return pd.DataFrame(rows), ref


def load_data(inputs, samples=None, ref=None):
    cached, ref = reference_inventory(inputs, samples, ref)
    rows, boundaries, radial = [], [], []
    shared_programs = {}
    for (sample, pair_id), group in cached.groupby(['sample', 'pair_id'], sort=True):
        actual_pair = group.pair.iloc[0]
        names = actual_pair.split('_')
        reverse = names != pair_id.split('__')
        if sorted(names) != sorted(pair_id.split('__')):
            raise ValueError('Unexpected program pair mapping')
        folder, _ = inputs.run(RESULTS, sample, actual_pair)
        tables = {name: inputs.csv(folder / (name+'.csv')) for name in evidence.FILES}
        profiles = evidence.roi_profiles(tables, sample, actual_pair, names)
        if {p['roi'] for p in profiles} != set(group.roi):
            raise ValueError('ROI inventory mismatch')
        for p in profiles:
            r = dict(sample=sample, pair_id=pair_id, pair=actual_pair, roi=p['roi'],
                     program_A=pair_id.split('__')[0], program_B=pair_id.split('__')[1],
                     source_spots=p['n'], support_class=p['support'], n_components=p['n_components'],
                     n_qualified_components=p['n_qualified_components'], coverage=p['coverage'],
                     valid_reps=p['valid_reps'], planned_reps=p['planned_reps'])
            for name, value in p['metrics'].items():
                name = metric_name(name, reverse)
                r[name] = value['value']
                for kind in ['kind', 'status', 'low', 'high']:
                    r[name+'__'+kind] = value[kind]
            rows.append(r)
        interface = tables['roi_pair_interface_response']
        # Verify duplicates across available reference-pair copies. This is not
        # an independent interaction measurement every time a program is reused.
        for program, g in evidence.program_interfaces(interface, sample, names).items():
            k = (sample, program)
            if k in shared_programs:
                pd.testing.assert_frame_equal(g, shared_programs[k], check_exact=True)
            else:
                shared_programs[k] = g
        boundaries.append(pair_boundaries(interface, sample, pair_id, names))
        radial.append(tables['component_radial_shell_correlogram'].assign(sample=sample, pair_id=pair_id))
    roi = pd.DataFrame(rows)
    if len(roi) != len(cached) or roi.duplicated(KEY).any():
        raise ValueError('Incomplete or duplicate ROI universe')
    for _, g in roi.groupby(['sample', 'roi']):
        for col in ['source_spots', 'support_class', 'n_components', 'n_qualified_components']:
            if g[col].nunique(dropna=False) != 1:
                raise ValueError('Pair-independent topology/support mismatch')
    boundary = pd.concat(boundaries, ignore_index=True)
    for role, column in [('anchor', 'anchor_roi_id'), ('responder', 'responder_roi_id')]:
        context_cols = ['enrichment_S_A', 'enrichment_S_B', 'enrichment_S_AB', 'support_class']
        context_cols += [f'enrichment_S_{k}__{v}' for k in ['A','B','AB'] for v in ['kind','status']]
        context = roi[KEY + context_cols].copy()
        context['roi'] = context.roi.astype(str)
        boundary[column] = boundary[column].astype(str)
        context = context.rename(columns={'roi': column, **{c: role+'_'+c for c in context if c not in KEY}})
        boundary = boundary.merge(context, on=['sample', 'pair_id', column], how='left', validate='many_to_one')
        if boundary[role+'_support_class'].isna().any():
            raise ValueError('Boundary lost parent ROI context')
    return roi, boundary, pd.concat(radial, ignore_index=True), ref


def build(output=OUT):
    inputs = Inputs()
    roi, boundary, radial, _ = load_data(inputs)
    # Historical descriptive annotation, not a universal eligibility gate.
    boundary['cross_program_enriched_contact'] = cross_enriched_contact(boundary, 1.)
    output.mkdir(parents=True, exist_ok=True)
    tables = {"native_roi_profiles.csv.gz": roi,
              "ordered_boundary_profiles.csv.gz": boundary,
              "native_component_radial.csv.gz": radial}
    for name, table in tables.items():
        table.to_csv(output/name, index=False, compression={"method": "gzip", "mtime": 0})
    paths = inputs.paths | {Path(__file__), Path(evidence.__file__),
                           HERE/"figure_common.py", HERE/"property_evidence.py"}
    (output/"provenance.json").write_text(json.dumps(dict(
        core=EXPECTED_CORE, n_tessellations=10, no_core_reruns=True,
        measurement_scope="original_unmasked_roi",
        inputs_sha256={str(p.resolve()): digest(p) for p in sorted(paths)},
        outputs_sha256={name: digest(output/name) for name in tables},
    ), indent=2)+"\n")
    print(f"Prepared {len(roi)} ROI, {len(boundary)} ordered-interface and {len(radial)} component-shell rows.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    build(parser.parse_args().output)
