"""Rung-0 case-oriented materializer tests (v7)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scenario_benchmark import load_sample_scaffold
from scenario_benchmark.case_ids import derive_case_id
from scenario_benchmark.materialize import (
    CORPUS_INDEX_SCHEMA_VERSION,
    DISPATCH,
    PLAN_SCHEMA_VERSION,
    CaseReport,
    Plan,
    _build_roles_manifest,
    load_plan,
    load_source_inventory,
    process,
    resolve_case_background_policy,
    role_contract_for_case,
    write_corpus_index,
)


# --------------------------------------------------------------------------- #
# Fixture scaffolding.
# --------------------------------------------------------------------------- #


def _scaffold(sample_id: str = "qual"):
    """20x12 grid. ROIs: cols 0-9 = "1" (host A), 10-16 = "2", 17-19 = "3"."""
    rows, cols = 12, 20
    xs = np.repeat(np.arange(cols), rows).astype(float)
    ys = np.tile(np.arange(rows), cols).astype(float)
    roi = np.array(
        ["1"] * (10 * rows) + ["2"] * (7 * rows) + ["3"] * (3 * rows),
        dtype=object,
    )
    for c, r in [(4, 4), (3, 4), (5, 4), (4, 3), (4, 5)]:
        roi[c * rows + r] = "2"
    spot_ids = [f"s{i:04d}" for i in range(xs.size)]
    df = pd.DataFrame({
        "x": xs, "y": ys, "spot_id": spot_ids, "roi_id": roi.tolist(),
    })
    return load_sample_scaffold(df, sample_id=sample_id, radius_factor=1.5)


def _live_scenario_role_contract() -> dict:
    from scenario_benchmark.materialize import live_scenario_role_contract
    return live_scenario_role_contract()


def _canonical_background_policy() -> dict:
    return {
        "background_level": 0.0,
        "participant_amplitude": 0.05,
        "non_participant_amplitude": 0.12,
        "latent_abs_clip": 1.75,
    }


def _broad_case(sample_id: str = "qual") -> dict:
    return {
        "case_id": f"{sample_id}__roi1",
        "scenario_name": "broad_colocalization",
        "source_sample_id": sample_id,
        "dataset": "greenwald",
        "selection_basis": "feasibility_selected",
        "seed": 20260828,
        "signal_model": {"effect_delta": 1.5},
        "args": {
            "anchor_roi_id": "1",
        },
    }


def _contact_case(mode: str, sample_id: str = "qual") -> dict:
    return {
        "case_id": f"{sample_id}__A1_B2__{mode}",
        "scenario_name": "contact_interface",
        "source_sample_id": sample_id,
        "dataset": "greenwald",
        "selection_basis": "feasibility_selected",
        "seed": 20260828,
        "signal_model": {"effect_delta": 1.0},
        "args": {"a_roi_id": "1", "b_roi_id": "2", "periphery_hops": 2},
        "mode": mode,
    }


# --------------------------------------------------------------------------- #
# Case-id derivation covers all 7 scenarios.
# --------------------------------------------------------------------------- #


def test_derive_case_id_covers_all_active_scenarios():
    sid = "SAMPLE_X"
    assert derive_case_id(
        "broad_colocalization", {"anchor_roi_id": "3"}, source_sample_id=sid,
    ) == f"{sid}__roi3"
    assert derive_case_id(
        "focal_pattern", {"anchor_roi_id": "3"},
        focal_pattern_mode="colocated", source_sample_id=sid,
    ) == f"{sid}__roi3__colocated"
    assert derive_case_id(
        "spatial_gradient", {"anchor_roi_id": "3"},
        gradient_relationship="opposed", source_sample_id=sid,
    ) == f"{sid}__roi3__opposed"
    assert derive_case_id(
        "dispersed_infiltration",
        {"a_roi_id": "2", "b_roi_id": "1"}, source_sample_id=sid,
    ) == f"{sid}__A2_B1"
    assert derive_case_id(
        "embedded_domains",
        {"a_roi_id": "4", "b_roi_id": "1"}, source_sample_id=sid,
    ) == f"{sid}__A4_B1"
    assert derive_case_id(
        "nested_foci",
        {"a_roi_id": "3", "b_roi_id": "1"}, source_sample_id=sid,
    ) == f"{sid}__A3_B1"
    assert derive_case_id(
        "contact_interface", {"a_roi_id": "1", "b_roi_id": "2"},
        contact_interface_mode="anchor_fade", source_sample_id=sid,
    ) == f"{sid}__A1_B2__anchor_fade"
    with pytest.raises(ValueError):
        derive_case_id("bogus", {}, source_sample_id=sid)


# --------------------------------------------------------------------------- #
# Plan parsing rejects malformed input.
# --------------------------------------------------------------------------- #


def test_plan_parsing_rejects_missing_case_field(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "master_seed": 1,
        "background_policy": _canonical_background_policy(),
        "cases": [{"case_id": "x", "scenario_name": "broad_colocalization"}],
    }))
    with pytest.raises(ValueError, match="missing fields"):
        load_plan(plan_path)


def test_plan_parsing_rejects_conflicting_dataset_for_sample_id(tmp_path):
    """Two cases naming the same source_sample_id under different
    ``dataset`` values must trigger the plan-level validator."""
    plan_path = tmp_path / "plan.json"
    a = _broad_case("shared_sample")
    a["case_id"] = "shared_sample__roi1"
    a["dataset"] = "greenwald"
    b = dict(a)
    b["case_id"] = "shared_sample__roi2"
    b["args"] = dict(a["args"])
    b["args"]["anchor_roi_id"] = "2"
    b["dataset"] = "hcc"
    plan_path.write_text(json.dumps({
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "master_seed": 1,
        "background_policy": _canonical_background_policy(),
        "cases": [a, b],
    }))
    with pytest.raises(ValueError, match="maps to multiple datasets"):
        load_plan(plan_path)


def test_plan_parsing_rejects_duplicate_case_ids(tmp_path):
    plan_path = tmp_path / "plan.json"
    base = _broad_case("s")
    plan_path.write_text(json.dumps({
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "master_seed": 1,
        "background_policy": _canonical_background_policy(),
        "cases": [base, dict(base)],
    }))
    with pytest.raises(ValueError, match="duplicate"):
        load_plan(plan_path)


def test_process_reuses_sample_scaffold(monkeypatch, tmp_path):
    import scenario_benchmark.materialize as materialize

    calls = []
    scaffold = _scaffold("shared")
    source_path = tmp_path / "shared.csv"
    source_path.write_text("scaffold loader is mocked; this file supplies the source checksum\n")

    def load_once(*args, **kwargs):
        calls.append(1)
        return scaffold

    monkeypatch.setattr(materialize, "load_greenwald_sample_scaffold", load_once)
    cache = {}
    canonical = _canonical_background_policy()
    for case_id in ("shared__roi1_a", "shared__roi1_b"):
        case = _broad_case("shared")
        case["case_id"] = case_id
        report = process(
            case,
            output_root=tmp_path / "corpus",
            dry_run=False,
            source_csv=source_path,
            generator_commit_sha="test",
            generator_source_state="dirty",
            background_policy=resolve_case_background_policy(canonical, case),
            scaffold_cache=cache,
        )
        assert report.dispatched
        case_dir = tmp_path / "corpus" / "broad_colocalization" / case_id
        manifest = json.loads((case_dir / "manifest.json").read_text())
        for role in manifest["roles"]:
            if role["input_file"] is not None:
                assert role["input_file"] == f"{role['name']}.csv"
                assert (case_dir / role["input_file"]).is_file()
        assert not (case_dir / "inputs").exists()
    assert len(calls) == 1


def test_materialize_from_downloaded_geometry_keeps_inputs_unchanged(monkeypatch, tmp_path):
    import scenario_benchmark.materialize as materialize

    inputs = tmp_path / "input_data"
    benchmark = inputs / "benchmark"
    scores = inputs / "greenwald" / "scores"
    benchmark.mkdir(parents=True)
    scores.mkdir(parents=True)
    monkeypatch.setattr(materialize, "input_data_root", lambda: inputs)
    scaffold = _scaffold()
    pd.DataFrame({
        "x": scaffold.coords[:, 0], "y": scaffold.coords[:, 1],
        "roi_id": scaffold.source_parent_roi_id,
    }).to_csv(scores / "qual.csv.gz", index=False)
    sources = benchmark / "source_samples.csv"
    sources.write_text(
        "dataset,sample_id,input_file\n"
        "greenwald,qual,../greenwald/scores/qual.csv.gz\n"
    )
    plan = benchmark / "benchmark_plan.json"
    plan.write_text(json.dumps({
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "master_seed": 1,
        "background_policy": _canonical_background_policy(),
        "cases": [_broad_case()],
    }))
    before = {p: p.read_bytes() for p in inputs.rglob("*") if p.is_file()}
    output = tmp_path / "regenerated"
    args = ["--plan", str(plan), "--sources", str(sources), "--output", str(output)]
    assert materialize.main(args + ["--dry-run"]) == 0
    assert not output.exists()
    assert materialize.main(args) == 0
    index = json.loads((output / "corpus_index.json").read_text())
    assert index["cases"][0]["acceptance_gate_passed"]
    assert len(index["cases"][0]["role_inputs"]) == 2
    assert {p: p.read_bytes() for p in inputs.rglob("*") if p.is_file()} == before
    for override in (["--output", str(benchmark)], ["--index", str(plan)]):
        with pytest.raises(SystemExit) as error:
            materialize.main(args + override)
        assert error.value.code == 2


def test_source_inventory_rejects_duplicate_and_missing_sources(tmp_path):
    source = tmp_path / "sample.csv"
    source.write_text("x,y,roi_id\n0,0,1\n")
    inventory = tmp_path / "sources.csv"
    row = "greenwald,sample,sample.csv\n"
    inventory.write_text("dataset,sample_id,input_file\n" + row * 2)
    with pytest.raises(ValueError, match="duplicate source"):
        load_source_inventory(inventory)
    inventory.write_text("dataset,sample_id,input_file\n" + row)
    source.unlink()
    with pytest.raises(FileNotFoundError, match="not found"):
        load_source_inventory(inventory)


# --------------------------------------------------------------------------- #
# Case-level manifest builder.
# --------------------------------------------------------------------------- #


def test_case_manifest_writer_projects_roles(tmp_path):
    scaffold = _scaffold()
    case = _broad_case()
    scenario_set = DISPATCH["broad_colocalization"](scaffold, case)

    contract = _live_scenario_role_contract()["broad_colocalization"]
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    roles = _build_roles_manifest(
        scenario_set=scenario_set,
        contract=contract,
        case_dir=case_dir,
    )
    role_names = [r["name"] for r in roles]
    assert role_names == list(contract["required_roles"]) + list(
        contract["conditional_roles"]
    )
    feasible = [r for r in roles if r["disposition"] == "feasible"]
    assert feasible, "expected at least one feasible role"
    for r in feasible:
        assert r["input_file"] == f"{r['name']}.csv"
        assert (case_dir / r["input_file"]).exists()
        assert r["input_sha256"] and r["n_spots"] > 0


@pytest.mark.parametrize(("mode", "expected_roles"), [
    ("responder_activation", [
        "positive", "no_interface_increment",
        "unrelated_boundary",
    ]),
    ("anchor_fade", ["positive", "no_anchor_extension"]),
])
def test_contact_manifest_uses_only_mode_roles(tmp_path, mode, expected_roles):
    scaffold = _scaffold()
    case = _contact_case(mode)
    contract = role_contract_for_case(case)
    assert contract["required_roles"] == expected_roles[:2]
    scenario_set = DISPATCH["contact_interface"](scaffold, case)
    case_dir = tmp_path / mode
    case_dir.mkdir(parents=True)
    roles = _build_roles_manifest(
        scenario_set=scenario_set, contract=contract, case_dir=case_dir,
    )
    assert [role["name"] for role in roles] == expected_roles
    assert all(role["disposition"] != "not_emitted" for role in roles)


# --------------------------------------------------------------------------- #
# Corpus index writer.
# --------------------------------------------------------------------------- #


def test_corpus_index_writer_v2_shape(tmp_path):
    plan = Plan(
        plan_schema_version=PLAN_SCHEMA_VERSION,
        master_seed=1,
        background_policy=_canonical_background_policy(),
        cases=[_broad_case("s")],
    )
    report = CaseReport(
        case_id="s__roi1",
        scenario_name="broad_colocalization",
        source_sample_id="s",
        dispatched=True,
    )
    report.role_dispositions = {
        "positive_broad": "feasible", "b_independent_background": "feasible",
    }
    report.role_inputs_sha = {
        "positive_broad": "a" * 64, "b_independent_background": "b" * 64,
    }
    report.manifest_sha256 = "c" * 64
    report.truth_ledger_sha256 = "d" * 64

    index_path = tmp_path / "corpus_index.json"
    write_corpus_index(
        index_path=index_path, plan=plan, reports=[report],
        generator_commit_sha="deadbeef", generator_source_state="clean",
        materialized=True,
    )
    payload = json.loads(index_path.read_text())
    assert payload["corpus_index_schema_version"] == CORPUS_INDEX_SCHEMA_VERSION
    assert payload["plan_schema_version"] == PLAN_SCHEMA_VERSION
    assert len(payload["cases"]) == 1
    c = payload["cases"][0]
    assert c["case_path"] == "broad_colocalization/s__roi1"
    assert c["acceptance_gate_passed"] is True
    assert set(c["role_inputs"]) == {
        "positive_broad", "b_independent_background",
    }
    assert set(c["role_dispositions"]) == {
        "positive_broad", "b_independent_background",
    }


# --------------------------------------------------------------------------- #
# Plan-level canonical background_policy forwarding + nested override merging
# reach the real materialization adapter.
# --------------------------------------------------------------------------- #


def test_canonical_background_policy_reaches_dispatched_scenario_set():
    """A case with no override receives the canonical policy through the
    adapter. Non-participant bg field respects canonical clip * amplitude."""
    scaffold = _scaffold()
    case = _broad_case()
    canonical = _canonical_background_policy()
    resolved = resolve_case_background_policy(canonical, case)
    assert resolved.participant_amplitude == canonical["participant_amplitude"]
    assert resolved.non_participant_amplitude == canonical["non_participant_amplitude"]
    assert resolved.latent_abs_clip == canonical["latent_abs_clip"]
    sset = DISPATCH["broad_colocalization"](scaffold, case, resolved)
    inst = next(i for i in sset.instances
                if i.counterfactual_role == "positive_broad")
    rows = sorted(
        [r for r in sset.spot_truth_rows if r.counterfactual_role == "positive_broad"],
        key=lambda r: r.source_spot_id,
    )
    id_to_idx = {sid: i for i, sid in enumerate(scaffold.spot_ids.astype(str))}
    order = np.argsort([id_to_idx[r.source_spot_id] for r in rows])
    fb = np.array([r.feature_b_value for r in rows])[order]
    # Non-anchor spots (ROI "2" and "3") get non-participant amp with clip.
    nonpart = ~np.isin(scaffold.source_parent_roi_id.astype(str),
                       np.array(["1"]))
    cap = canonical["non_participant_amplitude"] * canonical["latent_abs_clip"]
    assert np.max(np.abs(fb[nonpart])) <= cap + 1e-12


def test_nested_override_merges_participant_amplitude_only():
    """Nested cases carry a partial override — participant_amplitude=0.03
    — merged on top of the canonical policy. Non-participant amp and clip
    inherit; participant amp swaps to 0.03."""
    canonical = _canonical_background_policy()
    nested_case = {
        "case_id": "x__nested",
        "scenario_name": "nested_foci",
        "background_policy_override": {"participant_amplitude": 0.03},
    }
    resolved = resolve_case_background_policy(canonical, nested_case)
    assert resolved.participant_amplitude == 0.03
    assert resolved.non_participant_amplitude == canonical["non_participant_amplitude"]
    assert resolved.latent_abs_clip == canonical["latent_abs_clip"]
    assert resolved.background_level == canonical["background_level"]


def test_plan_construction_rejects_invalid_background_policy_override_key():
    """Plan.__post_init__ fails fast on unknown override keys, so load_plan
    surfaces the error before materialization starts."""
    bad_case = dict(_broad_case("s"))
    bad_case["background_policy_override"] = {"unknown_field": 0.1}
    with pytest.raises(ValueError, match="background_policy_override"):
        Plan(
            plan_schema_version=PLAN_SCHEMA_VERSION,
            master_seed=1,
            background_policy=_canonical_background_policy(),
            cases=[bad_case],
        )


def test_manifest_payload_records_resolved_nested_policy():
    """Nested cases record participant_amplitude=0.03 in the manifest's
    background_policy_resolved block; other three fields inherit canonical."""
    from scenario_benchmark.materialize import (
        build_case_manifest_payload, resolve_case_background_policy,
    )
    canonical = _canonical_background_policy()
    nested_case = {
        "case_id": "x__nested",
        "scenario_name": "nested_foci",
        "source_sample_id": "x",
        "dataset": "greenwald",
        "selection_basis": "feasibility_selected",
        "seed": 1,
        "signal_model": {"effect_delta": 1.0},
        "args": {"a_roi_id": "1", "b_roi_id": "2"},
        "background_policy_override": {"participant_amplitude": 0.03},
    }
    resolved = resolve_case_background_policy(canonical, nested_case)
    payload = build_case_manifest_payload(
        case=nested_case, source_data_sha256="e" * 64,
        generator_commit_sha="dead", generator_source_state="clean",
        background_policy=resolved, roles_manifest=[],
    )
    assert payload["background_policy_resolved"] == {
        "background_level": canonical["background_level"],
        "participant_amplitude": 0.03,
        "non_participant_amplitude": canonical["non_participant_amplitude"],
        "latent_abs_clip": canonical["latent_abs_clip"],
    }
