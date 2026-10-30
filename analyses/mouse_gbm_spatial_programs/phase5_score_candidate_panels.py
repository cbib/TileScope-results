"""Prepare and summarize the reduced mouse-GBM score-level comparison."""
from __future__ import annotations

import argparse
from pathlib import Path
from tilescope_results.paths import data_root

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
DATA_ROOT = (data_root() / 'mouse')
OUT_ROOT = DATA_ROOT / "mouse_gbm_spatial_programs" / "phase5_score_comparison"
MASTER = HERE / "inputs" / "signature_panel_candidates.yaml"
OUT_YAML = HERE / "inputs" / "signature_score_candidates.yaml"

SCORE_IDS = {
    "proliferation_current", "proliferation_compact", "proliferation_single_mki67",
    "interferon_current", "interferon_compact", "interferon_zbp1_extension", "interferon_single_isg15",
    "inflammation_current", "inflammation_compact", "inflammation_single_cxcl10",
    "myeloid_current", "myeloid_compact", "myeloid_single_tyrobp", "v4_myeloid_legacy",
    "cytotoxic_current", "cytotoxic_mct1_legacy", "cytotoxic_mct1_tcell_extension", "cytotoxic_single_nkg7",
    "ecm_current", "ecm_col11a1_extension", "ecm_single_mmp2",
    "mesenchymal_current", "mesenchymal_compact", "mesenchymal_single_fn1",
    "innate_current", "innate_compact", "innate_single_casp4",
    "hypoxia_current", "hypoxia_compact", "hypoxia_single_ddit4", "hypoxia_single_trib3",
    "vascular_current", "vascular_compact", "vascular_single_cldn5",
    "neural_current", "neural_compact", "neural_olig_portable", "neural_single_olig2",
    "apc_myeloid_current", "apc_myeloid_ctl_legacy", "apc_myeloid_compact", "apc_myeloid_single_cd74",
    "v4_reactive_current", "v4_reactive_legacy", "v4_reactive_s100a4_extension", "v4_reactive_single_pdpn",
    "tumour_anchor_legacy", "tumour_anchor_gbm_supported", "tumour_anchor_single_hmga2",
}

REFERENCE_BY_TOPIC = {
    "Cell-cycle and proliferation": "proliferation_current",
    "Interferon-related gene program": "interferon_current",
    "Inflammation": "inflammation_current",
    "Myeloid / phagocytic": "myeloid_current",
    "Cytotoxic T/NK-cell": "cytotoxic_current",
    "ECM remodelling and signalling": "ecm_current",
    "Mesenchymal / reactive stromal": "mesenchymal_current",
    "Innate immune": "innate_current",
    "Hypoxia / stress": "hypoxia_current",
    "Vascular / angiogenic": "vascular_current",
    "Neural-like / differentiation": "neural_current",
    "Antigen-presenting myeloid state": "apc_myeloid_current",
    "Reactive tumour-brain interface": "v4_reactive_current",
    "GBM-associated tumour-compartment anchor": "tumour_anchor_legacy",
}


def prepare() -> None:
    master = yaml.safe_load(MASTER.read_text())
    by_name = {row["candidate"]: row for row in master["candidates"]}
    missing = sorted(SCORE_IDS - by_name.keys())
    if missing:
        raise RuntimeError(f"score candidate IDs absent from master: {missing}")
    candidates = [row for row in master["candidates"] if row["candidate"] in SCORE_IDS]
    if len(candidates) != len(SCORE_IDS):
        raise RuntimeError("score candidate selection is not closed")
    payload = {
        "status": "score_level_comparison_not_frozen",
        "source_catalogue": str(MASTER.relative_to(HERE)),
        "scoring_contract": {
            "multi_gene": {
                "method": "Seurat::AddModuleScore",
                "assay": "Spatial",
                "layer": "data",
                "nbin": 24,
                "ctrl": 50,
                "seed": 1234,
                "pool": "full Spatial assay feature set",
                "invocation": "each panel scored separately",
            },
            "single_gene": "LogNormalized Spatial/data expression; never called a programme",
            "raw_panel_control": "unweighted mean of LogNormalized member-gene expression",
            "comparison_scope": "within sample and focus ROI only; no absolute cross-sample score comparisons",
        },
        "reference_by_topic": REFERENCE_BY_TOPIC,
        "candidates": candidates,
    }
    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    OUT_YAML.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    print(f"wrote {OUT_YAML} ({len(candidates)} candidates)")


def _finite_min(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.min()) if len(values) else np.nan


def summarize() -> None:
    roi = pd.read_csv(OUT_ROOT / "candidate_score_roi_audit.csv")
    loo = pd.read_csv(OUT_ROOT / "candidate_score_loo_audit.csv")
    similarity = pd.read_csv(OUT_ROOT / "candidate_score_reference_similarity.csv")

    roi_summary = roi.groupby(
        ["candidate", "topic", "architecture", "scope"], sort=False
    ).agg(
        n_focus_contexts=("roi_id", "size"),
        minimum_score_raw_spearman=("score_raw_spearman", _finite_min),
        median_score_raw_spearman=("score_raw_spearman", "median"),
        mean_abs_score_contrast=("score_contrast", lambda x: x.abs().mean()),
        mean_abs_raw_contrast=("raw_mean_contrast", lambda x: x.abs().mean()),
        score_raw_contrast_sign_agreement=("score_raw_contrast_sign_agreement", "mean"),
    ).reset_index()

    if loo.empty:
        loo_summary = pd.DataFrame(columns=["candidate"])
    else:
        loo_summary = loo.groupby("candidate", sort=False).agg(
            minimum_loo_spearman=("full_vs_loo_spearman", _finite_min),
            median_loo_spearman=("full_vs_loo_spearman", "median"),
            loo_contrast_sign_preserved_fraction=("contrast_sign_preserved", "mean"),
        ).reset_index()

    similarity_summary = similarity.groupby("candidate", sort=False).agg(
        reference_candidate=("reference_candidate", "first"),
        minimum_reference_spearman=("candidate_reference_spearman", _finite_min),
        median_reference_spearman=("candidate_reference_spearman", "median"),
        reference_contrast_sign_agreement=("contrast_sign_agreement", "mean"),
        mean_abs_contrast_difference=("contrast_difference", lambda x: x.abs().mean()),
    ).reset_index()

    out = (
        roi_summary.merge(loo_summary, on="candidate", how="left")
        .merge(similarity_summary, on="candidate", how="left")
    )
    out.to_csv(OUT_ROOT / "candidate_score_comparison.csv", index=False)

    sample = roi.groupby(
        ["candidate", "topic", "architecture", "scope", "sample_dir"],
        sort=False,
    ).agg(
        n_focus_contexts=("roi_id", "size"),
        minimum_score_raw_spearman=("score_raw_spearman", _finite_min),
        median_score_raw_spearman=("score_raw_spearman", "median"),
        mean_abs_score_contrast=("score_contrast", lambda x: x.abs().mean()),
        mean_abs_raw_contrast=("raw_mean_contrast", lambda x: x.abs().mean()),
        score_raw_contrast_sign_agreement=("score_raw_contrast_sign_agreement", "mean"),
    ).reset_index()
    sample.to_csv(OUT_ROOT / "candidate_score_sample_summary.csv", index=False)
    print(f"wrote candidate_score_comparison.csv ({len(out)} candidates)")
    print(f"wrote candidate_score_sample_summary.csv ({len(sample)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "summarize"])
    args = parser.parse_args()
    prepare() if args.command == "prepare" else summarize()


if __name__ == "__main__":
    main()
