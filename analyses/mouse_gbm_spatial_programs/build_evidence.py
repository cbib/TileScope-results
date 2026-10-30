"""Extract the mouse comparison screen and gene-detection audit from saved data."""
import argparse
from pathlib import Path
import subprocess

from .paths import locations


def main(output_root=None):
    data_root, analysis_root, output_root = locations(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    import numpy as np

    ROOT=data_root
    OUT=output_root
    OUT.mkdir(exist_ok=True)
    focus={'MCT1_CTL':[2,5], 'MCT1_KO':[4,9], 'V4_WT':[10,2,1]}
    manifest=pd.read_csv(ROOT/'mouse_gbm_spatial_programs/phase6_harmonized/phase6_harmonized_pair_manifest.csv')
    rows=[]
    for sample,rois in focus.items():
        bank=pd.read_csv(ROOT/'mouse_gbm_spatial_programs/phase6_harmonized'/f'{sample}_signature_score_bank.csv')
        sample_pairs=manifest.loc[manifest.sample_dir.eq(sample)]
        for record in sample_pairs.itertuples():
            run=ROOT/'CurrentResults'/sample/f'{record.signature_a}__{record.signature_b}'
            s=pd.read_csv(run/'roi_correlations.csv').set_index('roi_id')
            t=pd.read_csv(run/'roi_tessellation_correlations.csv').set_index('roi_id')
            c=pd.read_csv(run/'roi_consensus_correlations.csv').query('value_surface == "consensus_ensemble"').set_index('roi_id')
            e=pd.read_csv(run/'roi_enrichment.csv').set_index('roi_id')
            a,b=run.name.split('__')
            for roi,src in s.iterrows():
                obs=bank.loc[bank.roi_id.eq(roi)]
                row={'sample':sample,'pair':run.name,'roi':roi,'focus':roi in rois,'n_spots':src.n_roi,
                     'source_r':src.corr_obs,'source_q':src.corr_q_perm,'source_status':src.parent_correlation_status,
                     'tess_r':t.loc[roi,'corr_median'],'cons_r':c.loc[roi,'correlation_value'],
                     'cons_n':c.loc[roi,'sample_size'],'cons_q_analytic':c.loc[roi,'corr_q_analytic'],
                     'raw_source_r':obs[f'{a}__raw_mean'].corr(obs[f'{b}__raw_mean']),
                     'enrich_A':e.loc[roi,'g1_enrichment'],'enrich_B':e.loc[roi,'g2_enrichment'],
                     'joint_high':e.loc[roi,'joint_high_enrichment_fc'],
                     'source_csv':str(run/'roi_correlations.csv')}
                rows.append(row)
    d=pd.DataFrame(rows)
    d.to_csv(OUT/'all_pair_roi_screen.csv',index=False)
    f=d[d.focus & ~d.pair.str.contains('locator')].copy()
    f['strength']=f[['source_r','tess_r','cons_r','raw_source_r']].abs().min(axis=1)
    f['same_sign']=(np.sign(f[['source_r','tess_r','cons_r','raw_source_r']]).nunique(axis=1)==1)
    cols=['pair','roi','source_r','tess_r','cons_r','raw_source_r','enrich_A','enrich_B']
    for sample,rois in focus.items():
        print('\nTOP FOCUS',sample)
        print(f[(f['sample']==sample)&f.same_sign].sort_values('strength',ascending=False)[cols].head(16).round(3).to_string(index=False))
    ctl=f[(f['sample']=='MCT1_CTL')&f.roi.eq(5)].set_index('pair')
    ko=f[(f['sample']=='MCT1_KO')&f.roi.eq(9)].set_index('pair')
    m=ctl.join(ko,lsuffix='_ctl',rsuffix='_ko',how='inner')
    m['cons_diff']=m.cons_r_ko-m.cons_r_ctl
    m['raw_diff']=m.raw_source_r_ko-m.raw_source_r_ctl
    m.to_csv(OUT/'periphery_contrast_screen.csv')
    print('\nLARGEST PERIPHERY DIFFERENCES')
    print(m.sort_values('cons_diff',key=abs,ascending=False)[['source_r_ctl','source_r_ko','cons_r_ctl','cons_r_ko','raw_source_r_ctl','raw_source_r_ko','cons_diff']].head(24).round(3).to_string())
    print('\nTOTAL',len(d),'rows',d.groupby('sample').pair.nunique().to_dict())
    subprocess.run([
        "Rscript", str(analysis_root / "audit_gene_detection.R"),
        str(analysis_root / "inputs/samples.csv"), str(data_root),
        str(output_root / "immune_gene_detection.csv"),
    ], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Numerical evidence directory")
    main(parser.parse_args().output)
