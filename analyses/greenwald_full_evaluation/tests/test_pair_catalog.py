"""Inspection metadata does not require a duplicate statistical corpus."""
import ast
from pathlib import Path
import sys

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE/"greenwald_full"))
sys.path.insert(0, str(HERE/"greenwald_full_evaluation"))
from pair_catalog import load_pair_catalog


def inventory():
    return pd.DataFrame([dict(sample_id=s, pair=p, status="complete", core_sha="frozen")
                         for s in ["01", "s2"] for p in ["B_A", "A_C"]])


def test_pair_metadata_keeps_physical_order_and_has_no_cache_paths(tmp_path):
    path = tmp_path/"run_status.csv"
    inventory().to_csv(path, index=False)
    result = load_pair_catalog(path).set_index("pair")
    assert result.loc["B_A", ["program_g1", "program_g2", "unordered_pair_key"]].tolist() == ["B", "A", "A||B"]
    assert result.n_samples.eq(2).all()
    assert "path" not in result and "value" not in result


@pytest.mark.parametrize("mutation", ["duplicate", "incomplete", "mixed_core", "missing_pair", "reverse_pair"])
def test_bad_or_ambiguous_run_inventory_is_rejected(tmp_path, mutation):
    frame = inventory()
    if mutation == "duplicate": frame = pd.concat([frame, frame.iloc[:1]])
    if mutation == "incomplete": frame.loc[0, "status"] = "failed"
    if mutation == "mixed_core": frame.loc[0, "core_sha"] = "other"
    if mutation == "missing_pair": frame = frame.iloc[1:]
    if mutation == "reverse_pair": frame = pd.concat([frame, frame[frame.pair.eq("B_A")].assign(pair="A_B")])
    path = tmp_path/"run_status.csv"; frame.to_csv(path, index=False)
    with pytest.raises(ValueError): load_pair_catalog(path)


def test_current_catalogue_covers_complete_corpus():
    pairs = load_pair_catalog()
    assert len(pairs) == 91 and pairs.n_samples.eq(36).all()
    assert len(set(pairs.program_g1) | set(pairs.program_g2)) == 14


def test_no_active_python_imports_retired_cache_modules():
    retired = {"aggregate", "canonical_loader", "cache_provenance", "migrate_canonical_cache"}
    for root in [HERE/"greenwald_full", HERE/"greenwald_full_evaluation"]:
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = [n.name for n in node.names] if isinstance(node, ast.Import) else (
                    [node.module] if isinstance(node, ast.ImportFrom) else [])
                assert not retired.intersection(names), path
