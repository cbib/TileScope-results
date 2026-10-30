"""Prepare, run and validate the complete Meylan cohort, with resumable status."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
from tilescope_results.paths import data_root, REPOSITORY
from .cohort import HERE, SAMPLE_CELLS, input_directory
from .validate_mapping import match_profiles, AUTHOR_MAPPING


def prepare(root):
    folder=root/'preparation/cohort'; folder.mkdir(parents=True,exist_ok=True)
    published=pd.read_csv(HERE/'inputs/figS4D_reference.csv',index_col=0).T
    computed=pd.read_csv(root/'preparation/reconstruction/cluster_neighbour_composition.csv',index_col=0)*5
    mapping,report=match_profiles(computed,published)
    if not report['accepted']:
        raise ValueError('Published neighbourhood profiles are not recovered')
    previous=pd.read_csv(root/'preparation/validated/cluster_mapping.csv')
    pd.testing.assert_frame_equal(mapping,previous,check_exact=False,rtol=1e-12)
    inventory=pd.DataFrame([dict(sample=s,patient=s.split('_')[0],timepoint=s.split('_')[1],
                                n_cells=n,input_directory=str(input_directory(root,s).relative_to(root)))
                            for s,n in SAMPLE_CELLS.items()])
    inventory.to_csv(folder/'cohort_inventory.csv',index=False)
    pd.DataFrame([dict(published_cluster=k,roi_id=v) for k,v in AUTHOR_MAPPING.items()]).to_csv(
        folder/'published_neighbourhood_mapping.csv',index=False)
    subprocess.run(['Rscript',str(HERE/'prepare_cohort.R'),str(root),str(HERE),
                    str(folder/'cohort_inventory.csv')],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir',type=Path,default=data_root()/'meylan_xenium')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--skip-prepare',action='store_true')
    p.add_argument('--background',action='store_true')
    a=p.parse_args(); root=a.study_dir.resolve()
    logs=root/'analysis/cohort/logs'; logs.mkdir(parents=True,exist_ok=True)
    if a.background:
        command=[sys.executable,'-m','analyses.meylan_xenium_preparation.batch',
                 *[x for x in sys.argv[1:] if x!='--background']]
        with (logs/'batch.log').open('a') as stream:
            child=subprocess.Popen(command,cwd=REPOSITORY,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,log=str(logs/'batch.log'))),flush=True)
        return
    env=os.environ.copy()
    env.update(MPLBACKEND='Agg',MPLCONFIGDIR='/tmp/tilescope-matplotlib',
               OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    state=dict(status='preparing',started_at=datetime.now(timezone.utc).isoformat(),
               expected_samples=list(SAMPLE_CELLS),complete=[],failed=[],pid=os.getpid())
    status_path=logs.parent/'batch_status.json'
    def save(): status_path.write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        if not a.skip_prepare: prepare(root)
        if a.prepare_only:
            state['status']='inputs_prepared'; save(); return
        state['status']='running'; save()
        # Small samples first establish end-to-end validity before expensive runs.
        for sample in sorted(SAMPLE_CELLS,key=SAMPLE_CELLS.get):
            state['current_sample']=sample; save()
            with (logs/f'{sample}.log').open('a') as stream:
                for module, extra in [('run',['--resume','--tables-only']),('analyse',[])]:
                    command=[sys.executable,'-m',f'analyses.meylan_xenium_preparation.{module}',
                             '--study-dir',str(root),'--sample',sample,*extra]
                    subprocess.run(command,cwd=REPOSITORY,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            state['complete'].append(sample); save()
            print(f'Validated {sample}: {len(state["complete"])}/16',flush=True)
        state['status']='analysing_cohort'; save()
        subprocess.run([sys.executable,'-m','analyses.meylan_xenium_preparation.cohort_analysis',
                        '--study-dir',str(root)],cwd=REPOSITORY,env=env,check=True)
        state['status']='complete'; state['finished_at']=datetime.now(timezone.utc).isoformat(); save()
    except Exception as exc:
        state['status']='failed'; state['error']=repr(exc)
        state['failed'].append(state.get('current_sample','preparation_or_publication')); save()
        raise


if __name__=='__main__': main()
