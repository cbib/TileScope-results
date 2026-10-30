"""Match reconstructed clusters to the published Fig. S4D compositions.

Labels are accepted only if ALL published, rounded cell-composition values
match. This checks reconstruction against the paper, not TileScope outcomes.
It does not assert cell-for-cell identity with an unavailable final label file.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
AUTHOR_MAPPING = {
    **{f"CN-{i}": "CN-Tumor" for i in (8,5,15,18,14,1,16)},
    "CN-9": "CN-Necrotic", "CN-17": "CN-Plasma cells",
    "CN-12": "CN-Vasculature", "CN-7": "CN-Lymphoid aggregate",
    **{f"CN-{i}": "CN-Peri-necrotic" for i in (6,11)},
    **{f"CN-{i}": "CN-Tumor - T cell interface" for i in (2,10,4,3,13)},
}

def match_profiles(computed, published):
    computed = computed.loc[:, published.columns]
    costs = np.abs(computed.to_numpy()[:,None,:] - published.to_numpy()[None,:,:]).mean(axis=2)
    i,j = linear_sum_assignment(costs)
    if len(i) != len(published) or len(computed) != len(published):
        raise ValueError("Different number of clusters")
    matched = published.iloc[j].to_numpy()
    delta = computed.iloc[i].to_numpy() - matched
    unique_nearest = len(set(costs.argmin(axis=1))) == len(computed)
    rounded_equal = np.rint(computed.iloc[i].to_numpy()) == matched
    report = {
        "clusters": len(i), "cell_types": len(published.columns),
        "rounded_values_matching": int(rounded_equal.sum()), "values_checked": int(rounded_equal.size),
        "mean_absolute_percentage_point_difference": float(np.abs(delta).mean()),
        "maximum_absolute_percentage_point_difference": float(np.abs(delta).max()),
        "all_matches_unique_nearest": unique_nearest,
        "accepted": bool(rounded_equal.all() and unique_nearest),
    }
    mapping = pd.DataFrame({"reconstructed_cluster": computed.index[i], "published_cluster": published.index[j], "mean_absolute_difference": costs[i,j]})
    return mapping, report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reconstruction", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a=p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    pub=pd.read_csv(HERE/"inputs/figS4D_reference.csv", index_col=0).T
    computed=pd.read_csv(a.reconstruction/"cluster_neighbour_composition.csv", index_col=0)*5
    mapping, report=match_profiles(computed,pub)
    mapping.to_csv(a.output/"cluster_mapping.csv",index=False)
    (a.output/"annotation_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if not report["accepted"]:
        raise ValueError("Published profiles not recovered: do not materialize biological labels")
    frame=pd.read_csv(a.reconstruction/"P28_post_candidates_UNVALIDATED.csv.gz")
    lookup=mapping.set_index("reconstructed_cluster").published_cluster
    frame["published_cluster"]=frame.reconstructed_cluster.map(lookup)
    frame["ann_CN"]=frame.published_cluster.map(AUTHOR_MAPPING)
    if frame.ann_CN.isna().any() or frame.cell_id.duplicated().any():
        raise ValueError("Incomplete or duplicated memberships")
    frame=frame.drop(columns="unvalidated_author_mapping")
    frame.to_csv(a.output/"P28_post_annotations.csv.gz",index=False)
    frame.ann_CN.value_counts().rename_axis("region").to_csv(a.output/"P28_post_region_counts.csv")

if __name__ == "__main__":
    main()
