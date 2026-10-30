"""Run a complete Meylan sample using the frozen paper-based marker means."""
import argparse
from datetime import datetime, timezone
import json
import os
import resource
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
from tilescope_results.paths import core_root, data_root
from .cohort import SAMPLE_CELLS, input_path, same_definition

HERE = Path(__file__).resolve().parent

def validate(frame, annotation, reconstruction, sample="P28_post"):
    expected = SAMPLE_CELLS[sample]
    if len(frame) != expected or frame.cell_id.duplicated().any() or frame.cell_id.isna().any():
        raise ValueError(f"Expected all {expected:,} distinct {sample} cells")
    if not frame.cell_id.str.startswith(sample + '_').all():
        raise ValueError('Cell identifiers do not match the sample')
    cols=["cell_id","x","y","gene1","gene2","roi_id"]
    pd.testing.assert_frame_equal(frame[cols], annotation[cols], check_exact=True)
    if not np.isfinite(frame[["x","y","gene1","gene2"]]).all().all() or (frame[["gene1","gene2"]]<0).any().any():
        raise ValueError("Invalid expression or coordinates")
    if frame.roi_id.isna().any() or frame.roi_id.nunique() < 1:
        raise ValueError("Missing neighbourhood labels")
    if not reconstruction.get("accepted") or reconstruction.get("rounded_values_matching") != 360:
        raise ValueError("Neighbourhood recovery has not passed the published-profile check")

def validate_inputs(source, validation, definition, sample):
    """Read input once; release validation-only tables before core analysis."""
    frame = pd.read_csv(source)
    annotation = pd.read_csv(source.parent/'cell_annotations.csv.gz')
    validate(frame, annotation, validation, sample)
    del annotation
    genes = pd.read_csv(source.parent/'signature_gene_values.csv.gz')
    np.testing.assert_array_equal(frame.cell_id, genes.cell_id)
    for side, key in [(1, 'genes_a'), (2, 'genes_b')]:
        np.testing.assert_allclose(frame[f'gene{side}'], genes[definition[key]].mean(axis=1),
                                   rtol=1e-12, atol=1e-14)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir",type=Path,default=data_root()/"meylan_xenium")
    p.add_argument("--validate-only",action="store_true")
    p.add_argument("--sample", choices=list(SAMPLE_CELLS), default="P28_post")
    p.add_argument("--resume", action="store_true", help="Reuse a completed sample/pair run; never overwrite partial runs")
    p.add_argument("--tables-only", action="store_true", help="Export every numerical core table without large PDF reports")
    args=p.parse_args()
    definition=json.loads((HERE/"pair.json").read_text())
    pair=definition["pair"]
    source=input_path(args.study_dir, args.sample)
    if not same_definition(json.loads((source.parent/"pair.json").read_text()), definition):
        raise ValueError("Prepared signature definition differs from the frozen definition")
    validation=json.loads((args.study_dir/"preparation/validated/annotation_validation.json").read_text())
    validate_inputs(source, validation, definition, args.sample)
    os.environ.setdefault("MPLBACKEND","Agg")
    os.environ.setdefault("MPLCONFIGDIR","/tmp/tilescope-matplotlib")
    sys.path.insert(0,str(core_root()/"python"))
    from tilescope.settings import load_run_config
    from tilescope.api import run_tilescope_from_csv
    config_path=HERE/"run.config.json"
    config=load_run_config(str(config_path),profile=None)
    core_sha=subprocess.check_output(["git","-C",str(core_root()),"rev-parse","HEAD"],text=True).strip()
    if args.validate_only:
        print(f"Validated: complete {args.sample}, paper-matched neighbourhoods, ten tessellations")
        return
    out=args.study_dir/f"CurrentResults/{args.sample}/{pair}"
    if out.exists() and any(out.iterdir()):
        if args.resume and (out/'runner_status.json').exists():
            previous=json.loads((out/'runner_status.json').read_text())
            if (previous.get('status')=='complete'
                    and previous.get('sample')==args.sample
                    and same_definition(previous.get('definition',{}),definition)):
                print(f'Reusing completed run: {args.sample}',flush=True)
                return
        raise ValueError("Refusing to overwrite an existing run")
    out.mkdir(parents=True,exist_ok=True)
    status=dict(status="running",sample=args.sample,pair=pair,definition=definition,
                started_at=datetime.now(timezone.utc).isoformat(),
                core_git_sha=core_sha,
                core_worktree_status=subprocess.check_output(["git","-C",str(core_root()),"status","--short"],text=True),
                rendering='canonical tables only' if args.tables_only else 'tables, PDFs and HTML',
                annotation_origin="Author-workflow reconstruction matched to all 360 published Fig. S4D proportions")
    status_file=out/"runner_status.json"
    status_file.write_text(json.dumps(status,indent=2)+"\n")
    start=time.monotonic()
    try:
        if args.tables_only:
            from tilescope.reporters import export_results
            from tilescope.defaults import dump_effective_config
            from tilescope.__version__ import __version__
            result=run_tilescope_from_csv(source,run_config=config,output_dir=out,export=False,
                                         html_report=False,configure_logging=True,return_minimal=False,
                                         feature_a_id=definition["feature_a"],feature_b_id=definition["feature_b"])
            status['core_result_table_rows']={key:len(value) for key,value in result.items() if isinstance(value,pd.DataFrame)}
            export_results(result,out,generate_visualizations=False)
            dump_effective_config(result['config'].model_dump(),out/'effective_config.json',
                                  metadata={'tilescope_version':__version__,'merge_sources':[]})
        else:
            run_tilescope_from_csv(source,run_config=config,output_dir=out,export=True,
                                  html_report=True,configure_logging=True,return_minimal=True,
                                  feature_a_id=definition["feature_a"],feature_b_id=definition["feature_b"])
        status["status"]="complete"
    except Exception:
        status["status"]="failed"; status["traceback"]=traceback.format_exc()
        raise
    finally:
        status["seconds"]=time.monotonic()-start
        usage = resource.getrusage(resource.RUSAGE_SELF)
        status['peak_rss_bytes'] = int(usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
        status['cpu_seconds'] = usage.ru_utime + usage.ru_stime
        status["finished_at"]=datetime.now(timezone.utc).isoformat()
        status_file.write_text(json.dumps(status,indent=2)+"\n")

if __name__ == "__main__":
    main()
