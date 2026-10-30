"""Check a complete sample against input values and report all its ROIs."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from tilescope_results.paths import data_root
from .cohort import SAMPLE_CELLS, input_path

def eligible_mask(ledger):
    """Absent eligibility on excluded cells must not remove them from the denominator."""
    return ledger.tessellation_eligible.eq(True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir",type=Path,default=data_root()/"meylan_xenium")
    p.add_argument("--sample",choices=list(SAMPLE_CELLS),default="P28_post")
    a=p.parse_args(); root=a.study_dir
    definition=json.loads(Path(__file__).with_name("pair.json").read_text())
    pair=definition["pair"]
    features=[definition["feature_a"],definition["feature_b"]]
    run=root/f"CurrentResults/{a.sample}/{pair}"
    status=json.loads((run/"runner_status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Run is not complete")
    data=pd.read_csv(input_path(root,a.sample))
    ledger=pd.read_csv(run/"spot_ledger.csv").sort_values("spot_index")
    assert len(data)==len(ledger)==SAMPLE_CELLS[a.sample]
    np.testing.assert_array_equal(ledger.spot_index.to_numpy(),np.arange(len(data)))
    np.testing.assert_array_equal(ledger.roi_id.to_numpy(),data.roi_id.to_numpy())
    eligible=eligible_mask(ledger).to_numpy()
    tiles=pd.read_csv(run/"tessellation_tiles.csv",usecols=['roi_id','replicate_index','n_spots'])
    tile_counts=tiles.groupby(["roi_id","replicate_index"]).n_spots.sum()
    assert set(tiles.replicate_index)==set(range(10))
    enrich=pd.read_csv(run/"roi_enrichment.csv").set_index("roi_id")
    corr=pd.read_csv(run/"roi_correlations.csv").set_index("roi_id")
    cenrich=pd.read_csv(run/"roi_consensus_enrichment.csv").set_index("roi_id")
    ccorr=pd.read_csv(run/"roi_consensus_correlations.csv")
    rows=[]
    for roi,g in data.groupby("roi_id"):
        context=data[data.roi_id!=roi]
        row=dict(sample=a.sample,roi_id=roi,cells=len(g),
                 tessellation_eligible=int(eligible[data.roi_id==roi].sum()),
                 gene1_positive=int((g.gene1>0).sum()),gene2_positive=int((g.gene2>0).sum()))
        for num,gene in enumerate(features, 1):
            values=g[f"gene{num}"]
            threshold=data[f"gene{num}"].quantile(.75)
            numerator=(values>threshold).mean()
            denominator=(context[f"gene{num}"]>threshold).mean()
            expected=numerator/denominator if denominator>0 else (np.inf if numerator>0 else np.nan)
            observed=enrich.loc[roi,f"g{num}_enrichment"] if roi in enrich.index else np.nan
            state=enrich.loc[roi,f"g{num}_enrichment_status"] if roi in enrich.index else 'unavailable'
            if state=='finite':
                np.testing.assert_allclose(observed,expected,atol=1e-12,rtol=1e-12)
            elif state=='positive_over_zero_context':
                assert np.isposinf(expected)
            row[f"{gene}_enrichment_cells"]=observed
            row[f"{gene}_enrichment_cells_status"]=state
            row[f"{gene}_enrichment_ensemble"]=cenrich.loc[roi,f"g{num}_enrichment"] if roi in cenrich.index else np.nan
            row[f"{gene}_enrichment_ensemble_status"]=cenrich.loc[roi,f"g{num}_enrichment_status"] if roi in cenrich.index else 'unavailable'
            row[f"{gene}_positive_eligible"]=int(((data.roi_id==roi)&eligible&(data[f"gene{num}"]>0)).sum())
            # Diagnostic only: hold represented cells fixed before interpolation.
            retained_roi=data.loc[(data.roi_id==roi)&eligible,f"gene{num}"]
            retained_context=data.loc[(data.roi_id!=roi)&eligible,f"gene{num}"]
            denominator=(retained_context>threshold).mean()
            row[f"{gene}_enrichment_eligible_cells"]=float((retained_roi>threshold).mean()/denominator) if len(retained_roi) and denominator>0 else np.nan
        if roi in tile_counts.index.get_level_values("roi_id"):
            counts=tile_counts.loc[roi].reindex(range(10),fill_value=0)
            row["cells_in_tiles_min"]=int(counts.min())
            row["cells_in_tiles_max"]=int(counts.max())
            np.testing.assert_array_equal(counts.to_numpy(),np.repeat(row["tessellation_eligible"],10))
        else:
            row["cells_in_tiles_min"]=row["cells_in_tiles_max"]=0
        expected=g.gene1.corr(g.gene2) if g.gene1.std()>0 and g.gene2.std()>0 else np.nan
        if roi in corr.index and corr.loc[roi,'parent_correlation_status']=='finite':
            np.testing.assert_allclose(corr.loc[roi,"corr_obs"],expected,atol=1e-12,rtol=1e-12,equal_nan=True)
        row["pearson_cells"]=corr.loc[roi,"corr_obs"] if roi in corr.index else np.nan
        for surface in ("consensus_direct","consensus_ensemble"):
            c=ccorr[(ccorr.roi_id==roi)&(ccorr.value_surface==surface)]
            row[f"pearson_{surface}"]=c.correlation_value.iloc[0] if len(c) else np.nan
        row["tessellation_eligible_fraction"]=row["tessellation_eligible"]/row["cells"]
        rows.append(row)
    out=root/f"analysis/{a.sample}_{pair}"; out.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame(rows); result.to_csv(out/"roi_summary.csv",index=False)
    checks=dict(input_cells=len(data),ledger_cells=len(ledger),
                tessellation_eligible=int(eligible.sum()),
                tessellation_eligible_fraction=float(eligible.mean()),
                all_eligible_cells_represented_in_each_of_ten_tessellations=True,
                source_enrichment_and_pearson_reproduced=True,
                annotation_origin="Reconstructed using published workflow and Fig. S4D composition matching",
                core_git_sha=status["core_git_sha"], run_seconds=status["seconds"])
    (out/"validation.json").write_text(json.dumps(checks,indent=2)+"\n")
    print(json.dumps(checks,indent=2)); print(result.to_string(index=False))

if __name__ == "__main__":
    main()
