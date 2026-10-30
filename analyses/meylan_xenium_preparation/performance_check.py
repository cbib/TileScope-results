"""Reproduce a real sample in an isolated core checkout and compare CSV bytes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import filecmp
import json
import os
from pathlib import Path
import resource
import sys
import time

from tilescope_results.paths import data_root
from .cohort import HERE, input_path


def run(core, output, sample, *, source=None, config_path=None, feature_a=None, feature_b=None, dataset_id=1):
    if output.exists():
        raise ValueError('Performance runs require a new output directory')
    output.mkdir(parents=True)
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        os.environ[name] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/tilescope-matplotlib')
    sys.path.insert(0, str(core/'python'))
    import pandas as pd
    import tilescope
    from tilescope.settings import load_run_config
    from tilescope.api import run_tilescope_from_csv
    from tilescope.reporters import export_results
    from tilescope.defaults import dump_effective_config
    from tilescope.__version__ import __version__
    if not Path(tilescope.__file__).resolve().is_relative_to((core/'python').resolve()):
        raise ValueError('Wrong core imported')
    source = source or input_path(data_root()/'meylan_xenium', sample)
    config_path = config_path or HERE/'run.config.json'
    config = load_run_config(str(config_path), profile=None)
    definition = json.loads((HERE/'pair.json').read_text())
    start = time.monotonic()
    result = run_tilescope_from_csv(source, run_config=config, output_dir=output, export=False,
                                   html_report=False, configure_logging=True, return_minimal=False,
                                   dataset_id=dataset_id, feature_a_id=feature_a or definition['feature_a'],
                                   feature_b_id=feature_b or definition['feature_b'])
    export_results(result, output, generate_visualizations=False)
    dump_effective_config(result['config'].model_dump(), output/'effective_config.json',
                          metadata={'tilescope_version': __version__, 'merge_sources': []})
    elapsed = time.monotonic()-start
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report = dict(sample=sample, input_cells=len(result['spot_ledger_df']),
                  dataset_id=dataset_id, feature_a=feature_a or definition['feature_a'],
                  feature_b=feature_b or definition['feature_b'],
                  input_csv=str(source), configuration=str(config_path), core_checkout=str(core),
                  elapsed_seconds=elapsed, cpu_seconds=usage.ru_utime+usage.ru_stime,
                  peak_rss_bytes=int(usage.ru_maxrss*(1 if sys.platform == 'darwin' else 1024)),
                  finished_at=datetime.now(timezone.utc).isoformat(),
                  table_rows={key: len(value) for key,value in result.items() if isinstance(value,pd.DataFrame)},
                  csv_files=sorted(p.name for p in output.glob('*.csv')))
    (output/'performance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['csv_files','table_rows']}, indent=2), flush=True)


def compare_csv_bytes(before, after):
    """Compare the files themselves, including additions and omissions."""
    old = {p.name for p in before.glob('*.csv')}
    new = {p.name for p in after.glob('*.csv')}
    if not old or not new:
        raise ValueError('CSV comparison requires nonempty result directories')
    filecmp.clear_cache()
    mismatched = sorted(name for name in old | new
                        if name not in old or name not in new
                        or not filecmp.cmp(before/name, after/name, shallow=False))
    return len((old | new) - set(mismatched)), mismatched


def compare(before, after, reference=None, *, require_memory_improvement=True):
    old = json.loads((before/'performance.json').read_text())
    new = json.loads((after/'performance.json').read_text())
    for key in ['sample', 'input_cells', 'table_rows']:
        if old[key] != new[key]:
            raise ValueError(f'Different comparison inputs or table shape: {key}')
    for key in ['dataset_id', 'feature_a', 'feature_b']:
        if key in old and key in new and old[key] != new[key]:
            raise ValueError(f'Different comparison identity: {key}')
    identical, mismatched = compare_csv_bytes(before, after)
    reference_mismatched = []
    if reference:
        _, reference_mismatched = compare_csv_bytes(before, reference)
    report = dict(sample=old['sample'], identical_csvs=identical,
                  mismatched_csvs=mismatched, baseline_vs_completed_run_mismatches=reference_mismatched,
                  before_peak_rss_bytes=old['peak_rss_bytes'], after_peak_rss_bytes=new['peak_rss_bytes'],
                  memory_reduction_fraction=1-new['peak_rss_bytes']/old['peak_rss_bytes'],
                  before_seconds=old['elapsed_seconds'], after_seconds=new['elapsed_seconds'],
                  speedup=old['elapsed_seconds']/new['elapsed_seconds'])
    (after/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if mismatched or reference_mismatched or (require_memory_improvement and new['peak_rss_bytes'] >= old['peak_rss_bytes']):
        raise ValueError('CSV identity or memory improvement requirement failed')
    return report


def compare_reference(after, reference):
    report = json.loads((after/'performance.json').read_text())
    identical, mismatched = compare_csv_bytes(reference, after)
    check = dict(sample=report['sample'], identical_csvs=identical,
                 mismatched_csvs=mismatched, peak_rss_bytes=report['peak_rss_bytes'],
                 elapsed_seconds=report['elapsed_seconds'])
    (after/'reference_comparison.json').write_text(json.dumps(check, indent=2)+'\n')
    print(json.dumps(check, indent=2))
    if mismatched:
        raise ValueError('Existing production CSVs differ')
    return check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--core', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--sample', default='P34_pre')
    r.add_argument('--input-csv', type=Path)
    r.add_argument('--config', type=Path)
    r.add_argument('--feature-a')
    r.add_argument('--feature-b')
    r.add_argument('--dataset-id', type=int, default=1)
    c = sub.add_parser('compare')
    c.add_argument('--before', type=Path, required=True)
    c.add_argument('--after', type=Path, required=True)
    c.add_argument('--reference', type=Path)
    c.add_argument('--allow-memory-noise', action='store_true',
                   help='Report memory without requiring a decrease on small fixed-overhead samples')
    v = sub.add_parser('compare-reference')
    v.add_argument('--after', type=Path, required=True)
    v.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        custom = [args.input_csv, args.config, args.feature_a, args.feature_b]
        if any(custom) and not all(custom):
            parser.error('Custom inputs require --input-csv, --config, --feature-a and --feature-b together')
        run(args.core.resolve(), args.output.resolve(), args.sample, source=args.input_csv,
            config_path=args.config, feature_a=args.feature_a, feature_b=args.feature_b, dataset_id=args.dataset_id)
    elif args.command == 'compare-reference':
        compare_reference(args.after, args.reference)
    else:
        compare(args.before, args.after, args.reference, require_memory_improvement=not args.allow_memory_noise)


if __name__ == '__main__':
    main()
