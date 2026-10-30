from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from analyses.meylan_xenium_preparation.validate_mapping import match_profiles, AUTHOR_MAPPING
from analyses.meylan_xenium_preparation.run import validate
from analyses.meylan_xenium_preparation.analyse import eligible_mask


def test_mapping_recovers_arbitrary_cluster_numbers():
    pub=pd.DataFrame([[90.,10.],[10.,90.],[50.,50.]],index=["A","B","C"],columns=["x","y"])
    computed=pub.iloc[[2,0,1]].copy()
    computed.index=["CN-1","CN-2","CN-3"]
    computed.iloc[0,0] += .1
    mapping,report=match_profiles(computed,pub)
    assert report["accepted"]
    assert dict(zip(mapping.reconstructed_cluster,mapping.published_cluster)) == {"CN-1":"C","CN-2":"A","CN-3":"B"}


def test_mapping_rejects_unmatched_published_proportions():
    pub=pd.DataFrame([[90.,10.],[10.,90.]],index=["A","B"],columns=["x","y"])
    computed=pub.copy(); computed.iloc[0,0] += .6
    _,report=match_profiles(computed,pub)
    assert not report["accepted"]


def test_published_inventory_is_complete():
    source=Path(__file__).parents[1]/"inputs/figS4D_reference.csv"
    p=pd.read_csv(source,index_col=0)
    assert p.shape == (20,18)
    assert set(p.columns)==set(AUTHOR_MAPPING)
    assert len(set(AUTHOR_MAPPING.values()))==7


def valid_input():
    n=97488
    frame=pd.DataFrame({"cell_id":[f"P28_post_{i}" for i in range(n)],"x":np.arange(n,dtype=float),"y":0.,"gene1":0.,"gene2":1.,"roi_id":np.where(np.arange(n)%2,"A","B")})
    return frame,frame.copy(),{"accepted":True,"rounded_values_matching":360}


def test_full_sample_validation():
    validate(*valid_input())


def test_rejects_cell_selection():
    frame,ann,report=valid_input()
    with pytest.raises(ValueError,match="97,488"):
        validate(frame.iloc[:-1],ann.iloc[:-1],report)


def test_rejects_unvalidated_regions():
    frame,ann,report=valid_input(); report["accepted"]=False
    with pytest.raises(ValueError,match="published-profile"):
        validate(frame,ann,report)


def test_coverage_denominator_includes_cells_with_missing_eligibility():
    ledger=pd.DataFrame({"tessellation_eligible":[True,False,np.nan,True]})
    mask=eligible_mask(ledger)
    assert mask.tolist()==[True,False,False,True]
    assert mask.mean()==.5
