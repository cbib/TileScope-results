"""Freeze no biology: emit the reviewed harmonized signature *proposal*.

The proposal is driven by the predeclared focus ROIs and the source-expression
and score-level audits.  TileScope pair outcomes are used only below to check
whether already-reported findings remain traceable; they do not choose genes.
"""
from __future__ import annotations

from pathlib import Path
from tilescope_results.paths import data_root

import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
DATA_ROOT = (data_root() / 'mouse/mouse_gbm_spatial_programs')
SCORE_ROOT = DATA_ROOT / "phase5_score_comparison"
RAW_ROOT = DATA_ROOT / "phase5_candidate_comparison"
CATALOGUE = HERE / "inputs" / "signature_score_candidates.yaml"
OUT_YAML = HERE / "inputs" / "signatures_harmonized_proposed.yaml"
OUT_TRACE = SCORE_ROOT / "finding_traceability.csv"


SELECTIONS = [
    {
        "program": "cell_cycle_proliferation",
        "candidate": "proliferation_current",
        "use_scope": "MCT1_CTL, MCT1_KO and V4_WT",
        "kind": "biological_program",
        "decision": "Keep the established five-gene panel; it is more leave-one-out robust than the compact panel and exactly preserves the previous IFN-proliferation analysis.",
    },
    {
        "program": "interferon_related",
        "candidate": "interferon_current",
        "use_scope": "MCT1_CTL, MCT1_KO and V4_WT",
        "kind": "biological_program",
        "decision": "Keep the established five-gene panel; the Zbp1 extension is coherent but adds no necessary spatial field, and the compact panel is less contrast-stable.",
    },
    {
        "program": "inflammation",
        "candidate": "inflammation_current",
        "use_scope": "MCT1_CTL, MCT1_KO and V4_WT",
        "kind": "biological_program",
        "decision": "Keep the four-gene panel; it retains complement plus chemokine biology and is more leave-one-out robust than the compact panel.",
    },
    {
        "program": "myeloid_phagocytic",
        "candidate": "myeloid_current",
        "use_scope": "MCT1_CTL, MCT1_KO and V4_WT",
        "kind": "biological_program",
        "decision": "Use one common general-myeloid panel. In V4, within-ROI association is interpretable, but cross-ROI AddModuleScore enrichment must be checked against the raw member-gene mean.",
    },
    {
        "program": "cytotoxic_t_nk",
        "candidate": "cytotoxic_current",
        "use_scope": "same genes in all samples; biologically supported in the MCT sections and evaluation-only in V4_WT",
        "kind": "biological_program",
        "decision": "Use the constructable common panel. It preserves the CTL APC-cytotoxic source association; V4 carries little cytotoxic signal and is not used to define the program.",
    },
    {
        "program": "ecm_remodelling",
        "candidate": "ecm_col11a1_extension",
        "use_scope": "MCT1_CTL, MCT1_KO and V4_WT",
        "kind": "biological_program",
        "decision": "Add Col11a1: it improves leave-one-out robustness and provides a GBM-relevant tumour-associated collagen while preserving the reference contrast signs.",
    },
    {
        "program": "mesenchymal_reactive_stromal",
        "candidate": "mesenchymal_current",
        "use_scope": "MCT1_CTL, MCT1_KO and V4_WT",
        "kind": "biological_program",
        "decision": "Keep the three-gene panel; the two-gene alternative changes one focus contrast and cannot be leave-one-gene-out tested as a program.",
    },
    {
        "program": "innate_immune",
        "candidate": "innate_current",
        "use_scope": "identical in MCT1_CTL and MCT1_KO; unavailable in V4_WT because the required genes are absent",
        "kind": "biological_program",
        "decision": "Keep focal Il1b with Casp4 and Ripk2; the compact pair is cleaner but narrows the intended innate-inflammatory biology.",
    },
    {
        "program": "hypoxia_integrated_stress",
        "candidate": "hypoxia_current",
        "use_scope": "identical in MCT1_CTL and MCT1_KO; unavailable as a multi-gene program in V4_WT",
        "kind": "biological_program",
        "decision": "Keep the four-gene MCT panel; it is more robust than the compact panel. Trib3 remains a V4-compatible single-gene stress marker, not a substitute program.",
    },
    {
        "program": "endothelial_vascular",
        "candidate": "vascular_compact",
        "use_scope": "MCT1_CTL, MCT1_KO and V4_WT",
        "kind": "biological_program",
        "decision": "Use Cldn5 and Vwf as a portable endothelial/vascular panel. Do not call it angiogenesis: adding Angpt2 is MCT-only and less score/raw stable.",
    },
    {
        "program": "oligodendroglial_neural_like",
        "candidate": "neural_olig_portable",
        "use_scope": "MCT1_CTL, MCT1_KO and V4_WT",
        "kind": "biological_program",
        "decision": "Use the portable Olig1/Olig2 lineage pair. The MCT-only Sox2 panel is less portable and mixes lineage with stem-like biology.",
    },
    {
        "program": "antigen_presenting_myeloid",
        "candidate": "apc_myeloid_current",
        "use_scope": "MCT-defined and evaluated unchanged in MCT1_CTL, MCT1_KO and V4_WT",
        "kind": "biological_program",
        "decision": "Use the coherent four-gene MHC-II panel; it separates antigen presentation from the general-myeloid panel and preserves the CTL source finding.",
    },
    {
        "program": "v4_reactive_interface",
        "candidate": "v4_reactive_current",
        "use_scope": "V4_WT only",
        "kind": "experiment_specific_program",
        "decision": "Keep the V4-specific Pdpn/A2m/Serping1 panel. It preserves the prior V4 reactive-myeloid source association without duplicating S100a4 from the common mesenchymal panel.",
    },
    {
        "program": "gbm_tumour_compartment_locator",
        "candidate": "tumour_anchor_gbm_supported",
        "use_scope": "technical reference in MCT1_CTL, MCT1_KO and V4_WT",
        "kind": "technical_locator_not_biological_program",
        "decision": "Replace the legacy locator with a better-supported GBM-associated panel. It locates the experimental tumour compartment; it is not a universal malignant-cell signature.",
    },
]


TRACE_CASES = [
    {
        "finding": "MCT1_CTL ROI5 interferon-proliferation",
        "sample_dir": "MCT1_CTL",
        "roi_id": 5,
        "old_a": "interferon_current",
        "old_b": "proliferation_current",
        "new_a": "interferon_current",
        "new_b": "proliferation_current",
    },
    {
        "finding": "MCT1_KO ROI9 interferon-proliferation",
        "sample_dir": "MCT1_KO",
        "roi_id": 9,
        "old_a": "interferon_current",
        "old_b": "proliferation_current",
        "new_a": "interferon_current",
        "new_b": "proliferation_current",
    },
    {
        "finding": "MCT1_CTL ROI5 APC-cytotoxic",
        "sample_dir": "MCT1_CTL",
        "roi_id": 5,
        "old_a": "apc_myeloid_ctl_legacy",
        "old_b": "cytotoxic_mct1_legacy",
        "new_a": "apc_myeloid_current",
        "new_b": "cytotoxic_current",
    },
    {
        "finding": "MCT1_CTL ROI2 APC-cytotoxic",
        "sample_dir": "MCT1_CTL",
        "roi_id": 2,
        "old_a": "apc_myeloid_ctl_legacy",
        "old_b": "cytotoxic_mct1_legacy",
        "new_a": "apc_myeloid_current",
        "new_b": "cytotoxic_current",
    },
    {
        "finding": "V4_WT ROI2 reactive-myeloid",
        "sample_dir": "V4_WT",
        "roi_id": 2,
        "old_a": "v4_reactive_legacy",
        "old_b": "v4_myeloid_legacy",
        "new_a": "v4_reactive_current",
        "new_b": "myeloid_current",
    },
    {
        "finding": "V4_WT ROI1 reactive-myeloid",
        "sample_dir": "V4_WT",
        "roi_id": 1,
        "old_a": "v4_reactive_legacy",
        "old_b": "v4_myeloid_legacy",
        "new_a": "v4_reactive_current",
        "new_b": "myeloid_current",
    },
]


def _finite_corr(frame: pd.DataFrame, a: str, b: str, method: str) -> float:
    if a == b:
        values = pd.to_numeric(frame[a], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        return 1.0 if len(values) >= 3 and values.nunique() >= 2 else np.nan
    values = frame[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 3 or values[a].nunique() < 2 or values[b].nunique() < 2:
        return np.nan
    return float(values[a].corr(values[b], method=method))


def _build_traceability(scores: pd.DataFrame) -> pd.DataFrame:
    records = []
    for case in TRACE_CASES:
        subset = scores[
            scores["sample_dir"].eq(case["sample_dir"])
            & scores["roi_id"].astype(str).eq(str(case["roi_id"]))
        ]
        wide = subset.pivot(index="spot_id", columns="candidate", values="score")
        required = [case["old_a"], case["old_b"], case["new_a"], case["new_b"]]
        missing = sorted(set(required) - set(wide.columns))
        if missing:
            raise RuntimeError(f"traceability case {case['finding']} lacks {missing}")
        old_r = _finite_corr(wide, case["old_a"], case["old_b"], "pearson")
        new_r = _finite_corr(wide, case["new_a"], case["new_b"], "pearson")
        old_rho = _finite_corr(wide, case["old_a"], case["old_b"], "spearman")
        new_rho = _finite_corr(wide, case["new_a"], case["new_b"], "spearman")
        records.append(
            {
                **case,
                "n_spots": int(len(wide)),
                "old_pearson": old_r,
                "proposed_pearson": new_r,
                "delta_pearson": new_r - old_r,
                "pearson_sign_preserved": bool(np.sign(new_r) == np.sign(old_r)),
                "old_spearman": old_rho,
                "proposed_spearman": new_rho,
                "delta_spearman": new_rho - old_rho,
                "a_old_vs_proposed_spearman": _finite_corr(wide, case["old_a"], case["new_a"], "spearman"),
                "b_old_vs_proposed_spearman": _finite_corr(wide, case["old_b"], case["new_b"], "spearman"),
                "audit_scope": "source spots only; consensus/tessellation preservation requires a targeted TileScope rerun",
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    catalogue = yaml.safe_load(CATALOGUE.read_text())
    by_candidate = {row["candidate"]: row for row in catalogue["candidates"]}
    score = pd.read_csv(SCORE_ROOT / "candidate_score_comparison.csv").set_index("candidate")
    raw = pd.read_csv(RAW_ROOT / "candidate_panel_comparison.csv").set_index("candidate")

    programs = []
    for decision in SELECTIONS:
        name = decision["candidate"]
        if name not in by_candidate or name not in score.index or name not in raw.index:
            raise RuntimeError(f"selected candidate is not audit-closed: {name}")
        spec = by_candidate[name]
        score_row = score.loc[name]
        raw_row = raw.loc[name]
        programs.append(
            {
                **decision,
                "topic": spec["topic"],
                "genes": spec["genes"],
                "definition_scope": spec["scope"],
                "score_method": spec["score_method"],
                "audit": {
                    "n_focus_contexts": int(score_row["n_focus_contexts"]),
                    "minimum_score_raw_spearman": float(score_row["minimum_score_raw_spearman"]),
                    "minimum_loo_spearman": None if pd.isna(score_row["minimum_loo_spearman"]) else float(score_row["minimum_loo_spearman"]),
                    "score_raw_contrast_sign_agreement": float(score_row["score_raw_contrast_sign_agreement"]),
                    "minimum_gene_detection": float(raw_row["minimum_gene_detection"]),
                    "mean_pairwise_codetection": float(raw_row["mean_pairwise_codetection"]),
                    "minimum_component_coverage": float(raw_row["minimum_component_coverage"]),
                },
            }
        )

    payload = {
        "status": "proposed_not_frozen",
        "selection_authority": "predeclared focus ROIs plus source-expression and score-level audits; no TileScope pair outcome selected genes",
        "scoring_contract": catalogue["scoring_contract"],
        "scope_rule": {
            "definition_focus": {
                "MCT1_CTL": [2, 5],
                "MCT1_KO": [4, 9],
                "V4_WT": [10, 2, 1],
            },
            "portability": "CTL and KO always use identical genes per topic; V4 uses the same panel where constructable, plus one explicitly V4-specific reactive-interface program.",
        },
        "programs": programs,
        "single_gene_controls": {
            "rule": "direct LogNormalized expression; marker/control only, never named a program",
            "genes": ["Mki67", "Isg15", "Cxcl10", "Tyrobp", "Nkg7", "Mmp2", "Fn1", "Casp4", "Ddit4", "Trib3", "Cldn5", "Olig2", "Cd74", "Pdpn", "Hmga2"],
        },
    }
    OUT_YAML.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))

    spot_scores = pd.read_csv(SCORE_ROOT / "candidate_spot_scores.csv")
    trace = _build_traceability(spot_scores)
    trace.to_csv(OUT_TRACE, index=False)
    print(f"wrote {OUT_YAML} ({len(programs)} programs/locators)")
    print(f"wrote {OUT_TRACE} ({len(trace)} finding checks)")


if __name__ == "__main__":
    main()
