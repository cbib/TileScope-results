"""Contract tests for the roi_pair interface_conditioned_periphery feature.

Sister-side extraction + panel wiring for parent_roi_interface_response.
Real emitted schema: per (anchor_roi_id, responder_roi_id, surface,
responder_feature_id) with `availability_status` ∈ {finite, undefined};
undefined_reason drawn from the analyzer's status enum (e.g.
no_anchor_responder_interface, insufficient_reachable_responder).

Extraction is scenario-blind (no manifest / plan lookup); selection of
the case-declared A/B pair happens in recognition via
``roi_id`` (anchor) + ``responder_roi_id`` slot resolution.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractors import (  # noqa: E402
    ATLAS_COLUMNS,
    INTERFACE_RESPONSE_GRAIN,
    INTERFACE_RESPONSE_SOURCE,
    INTERFACE_RESPONSE_STATISTIC,
    MissingExportError,
    _extract_interface_response,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recognition"))


# --- Fixture: build a minimal CurrentResults/<scenario>/<case>/<role>/ tree ---


def _write_case_fixture(tmp_path, *, rows):
    """Materialise a role-level tilescope dir + the analyzer CSV.
    Extraction is scenario-blind so no manifest.json is needed."""
    role_dir = tmp_path / "CurrentResults" / "scen" / "case" / "positive"
    role_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(role_dir / INTERFACE_RESPONSE_SOURCE, index=False)
    return role_dir


CONTEXT = {"case_id": "case", "scenario_name": "scen", "scenario_mode": None,
           "sample_id": "sX", "dataset": "d",
           "selection_basis": "manually_reviewed_reference",
           "role": "positive", "role_disposition": "feasible"}


def _row(anchor, responder, feature, value, status,
         surface="source_native", g1_id="gene1", g2_id="gene2"):
    return {
        "sample_id": "0", "g1_id": g1_id, "g2_id": g2_id,
        "anchor_roi_id": anchor, "responder_roi_id": responder,
        "responder_feature_id": feature, "surface": surface,
        "availability_status": status, "interface_response": value,
        "source_near_mass": 10.0, "source_far_mass": 100.0,
        "represented_near_mass": 10.0, "represented_far_mass": 100.0,
        "reachable_responder_fraction": 0.9, "n_near_far_mixed_units": 0,
        "n_replicates_planned": pd.NA, "n_valid_replicates": pd.NA,
    }


# --- Extractor: participant filter + grain + channel mapping ---


def test_extractor_keeps_all_adjacent_pairs_no_participant_filter(tmp_path):
    """Extraction is scenario-blind: every ordered adjacent-pair row
    from the analyzer CSV lands in the atlas. Participant selection is
    the recognition layer's job, not the extractor's."""
    a, b, bg = "4", "3", "5"
    rows = [
        _row(a, b, "gene1", 0.30, "finite"),
        _row(a, b, "gene2", 0.55, "finite"),
        _row(b, a, "gene1", 0.02, "finite"),
        _row(b, a, "gene2", 0.03, "finite"),
        # Background pairs — MUST be kept (not filtered).
        _row(a, bg, "gene1", 0.10, "finite"),
        _row(bg, a, "gene2", 0.11, "finite"),
        _row(b, bg, "gene1", 0.12, "finite"),
    ]
    run_dir = _write_case_fixture(tmp_path, rows=rows)
    out = _extract_interface_response(run_dir, dict(CONTEXT))

    assert len(out) == 7
    kept_pairs = {(r["roi_id"], r["responder_roi_id"]) for r in out}
    assert kept_pairs == {(a, b), (b, a), (a, bg), (bg, a), (b, bg)}
    assert {r["grain"] for r in out} == {INTERFACE_RESPONSE_GRAIN}
    assert {r["statistic"] for r in out} == {INTERFACE_RESPONSE_STATISTIC}
    # gene1/gene2 mapped to g1/g2 (matching enrichment's channel convention).
    assert {r["channel"] for r in out} == {"g1", "g2"}
    # ATLAS_COLUMNS includes responder_roi_id.
    assert "responder_roi_id" in ATLAS_COLUMNS


def test_extractor_channel_maps_generic_feature_identities(tmp_path):
    """Channel is derived per-row against ``g1_id`` / ``g2_id``, not
    the string literals 'gene1'/'gene2'. Real samples carry semantic
    gene identifiers; the mapping must round-trip them into g1/g2."""
    rows = [
        _row("4", "3", "EGFR",   0.31, "finite", g1_id="EGFR", g2_id="KRAS"),
        _row("4", "3", "KRAS",   0.52, "finite", g1_id="EGFR", g2_id="KRAS"),
        _row("3", "4", "EGFR",   0.01, "finite", g1_id="EGFR", g2_id="KRAS"),
        _row("3", "4", "KRAS",   0.02, "finite", g1_id="EGFR", g2_id="KRAS"),
    ]
    run_dir = _write_case_fixture(tmp_path, rows=rows)
    out = _extract_interface_response(run_dir, dict(CONTEXT))
    by_ch = {r["channel"] for r in out}
    assert by_ch == {"g1", "g2"}
    # Match values to channels by (anchor, responder, channel).
    lookup = {(r["roi_id"], r["responder_roi_id"], r["channel"]): r["value"]
              for r in out}
    assert lookup[("4", "3", "g1")] == pytest.approx(0.31)  # EGFR mapped
    assert lookup[("4", "3", "g2")] == pytest.approx(0.52)  # KRAS mapped


def test_extractor_rejects_unknown_responder_identity(tmp_path):
    """An identity not equal to g1_id or g2_id is a mapping
    ambiguity — reject rather than silently forward."""
    rows = [
        _row("4", "3", "UNKNOWN_MARK", 0.4, "finite",
             g1_id="EGFR", g2_id="KRAS"),
    ]
    run_dir = _write_case_fixture(tmp_path, rows=rows)
    with pytest.raises(MissingExportError, match="not matching"):
        _extract_interface_response(run_dir, dict(CONTEXT))


def test_extractor_preserves_undefined_status_and_reason(tmp_path):
    a, b = "4", "3"
    rows = [
        _row(a, b, "gene1", None, "no_anchor_responder_interface"),
        _row(a, b, "gene2", 0.55, "finite"),
        _row(b, a, "gene1", 0.02, "finite"),
        _row(b, a, "gene2", 0.03, "finite"),
    ]
    run_dir = _write_case_fixture(tmp_path, rows=rows)
    out = _extract_interface_response(run_dir, dict(CONTEXT))
    by_key = {(r["roi_id"], r["responder_roi_id"], r["channel"]): r for r in out}
    undef = by_key[(a, b, "g1")]
    assert undef["availability_status"] == "undefined"
    assert undef["undefined_reason"] == "no_anchor_responder_interface"
    assert undef["value"] is None
    fin = by_key[(a, b, "g2")]
    assert fin["availability_status"] == "finite"
    assert fin["value"] == pytest.approx(0.55)


# --- _match_stat_row: responder_roi_id filter isolates the direction ---




# --- _panel_status: structural non-applicability + missing_feature ---






# --- _build_case_features + PANEL_SPECS: end-to-end fixture-driven ---
