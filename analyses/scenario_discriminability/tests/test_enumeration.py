"""Enumeration accounts for feasible + infeasible + not_applicable roles
under the case-oriented plan/index schema."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_atlas import PlannedRole, enumerate_planned_roles  # noqa: E402


PLAN = {
    "cases": [
        {
            "case_id": "X_T__roi1",
            "scenario_name": "broad_colocalization",
            "source_sample_id": "X_T",
            "dataset": "greenwald",
            "selection_basis": "feasibility_selected",
            "seed": 20260828,
            "signal_model": {},
            "args": {"anchor_roi_id": "1"},
        },
        {
            "case_id": "Y_T__roi3__opposed",
            "scenario_name": "spatial_gradient",
            "source_sample_id": "Y_T",
            "dataset": "greenwald",
            "selection_basis": "manually_reviewed_reference",
            "seed": 20260828,
            "signal_model": {},
            "args": {"anchor_roi_id": "3"},
            "gradient_relationship": "opposed",
        },
    ],
}
INDEX = {
    "cases": [
        {
            "scenario_name": "broad_colocalization",
            "case_id": "X_T__roi1",
            "case_path": "broad_colocalization/X_T__roi1",
            "role_dispositions": {
                "positive_broad": "feasible",
                "paired_randomized": "feasible",
            },
        },
        {
            "scenario_name": "spatial_gradient",
            "case_id": "Y_T__roi3__opposed",
            "case_path": "spatial_gradient/Y_T__roi3__opposed",
            "role_dispositions": {
                "positive_spatial_gradient": "feasible",
                "paired_randomized": "feasible",
                "some_control": "not_applicable",
            },
        },
    ],
}


def test_enumerate_covers_every_planned_role() -> None:
    planned = enumerate_planned_roles(PLAN, INDEX)
    keys = {(p.scenario_name, p.case_id, p.role, p.role_disposition) for p in planned}
    assert (
        "broad_colocalization", "X_T__roi1", "positive_broad", "feasible"
    ) in keys
    assert (
        "spatial_gradient", "Y_T__roi3__opposed", "some_control", "not_applicable"
    ) in keys
    assert len(planned) == 5


def test_scenario_mode_populated_from_gradient_relationship() -> None:
    planned = enumerate_planned_roles(PLAN, INDEX)
    modes = {
        (p.case_id, p.role): p.scenario_mode for p in planned
    }
    assert modes[("Y_T__roi3__opposed", "positive_spatial_gradient")] == "opposed"
    assert modes[("X_T__roi1", "positive_broad")] is None


def test_selection_basis_propagates_from_plan() -> None:
    planned = enumerate_planned_roles(PLAN, INDEX)
    basis = {p.case_id: p.selection_basis for p in planned}
    assert basis["X_T__roi1"] == "feasibility_selected"
    assert basis["Y_T__roi3__opposed"] == "manually_reviewed_reference"
