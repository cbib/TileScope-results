"""Prepare and summarize competing mouse-GBM signature candidates.

The biologist-defined focus ROIs remain the definition authority:

* MCT1_CTL ROI 2 (core) and ROI 5 (periphery);
* MCT1_KO ROI 4 (core) and ROI 9 (periphery);
* V4_WT ROI 10 (core), ROI 2 (reactive interface), and ROI 1
  (proliferative intratumour subregion).

Marker discovery is broad but bounded: enriched Seurat markers down to rank
1,000 in each retained focus view. A predeclared literature-supported gene
below that window can be rescued, but the rescue is explicit in the output.
Other ROIs and all TileScope pair outcomes are evaluation data and cannot
alter candidate composition.

This module compares a small number of interpretable architectures per topic:
the current harmonized panel, a compact alternative, an optional established
or experimental extension, and a single-gene spatial-marker comparator. A
single gene is never labelled a biological programme and is scored from its
LogNormalized expression rather than AddModuleScore.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from tilescope_results.paths import data_root
from typing import Any

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
DATA_ROOT = (data_root() / 'mouse')
PHASE_ROOT = DATA_ROOT / "mouse_gbm_spatial_programs"
OUT_YAML = HERE / "inputs" / "signature_panel_candidates.yaml"
OUT_GENE_EVIDENCE = PHASE_ROOT / "phase5_candidate_comparison" / "signature_candidate_gene_evidence.csv"
AUDIT_ROOT = PHASE_ROOT / "phase5_candidate_comparison"
AUDIT_ROWS = AUDIT_ROOT / "candidate_panel_focus_audit.csv"
OUT_COMPARISON = AUDIT_ROOT / "candidate_panel_comparison.csv"
OUT_SAMPLE_COMPARISON = AUDIT_ROOT / "candidate_panel_sample_summary.csv"

ALL_SAMPLES = ["MCT1_CTL", "MCT1_KO", "V4_WT"]
MCT1_SAMPLES = ["MCT1_CTL", "MCT1_KO"]

SOURCES = {
    "neftel_2019": "https://www.broadinstitute.org/publications/broad613186",
    "ravi_2022": "https://pubmed.ncbi.nlm.nih.gov/35700707/",
    "pombo_antunes_2021": "https://www.nature.com/articles/s41593-020-00789-y",
    "cloughesy_2021": "https://www.nature.com/articles/s41467-021-26940-2",
    "gbm_ecm_2023": "https://www.nature.com/articles/s41525-023-00381-w",
    "gbm_hypoxia_2020": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7563149/",
    "mct4_gbm_2014": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4087088/",
    "hmga2_gbm": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5091648/",
    "cd109_gbm": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8262342/",
    "emp3_gbm": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5362410/",
}


def _candidate(
    name: str,
    topic: str,
    architecture: str,
    scope: str,
    samples: list[str],
    genes: list[str],
    rationale: str,
    relevance: str = "high",
    novelty: str = "established",
) -> dict[str, Any]:
    return {
        "candidate": name,
        "topic": topic,
        "architecture": architecture,
        "scope": scope,
        "intended_samples": samples,
        "genes": genes,
        "rationale": rationale,
        "biological_relevance": relevance,
        "novelty": novelty,
        "score_method": (
            "LogNormalized single-gene expression"
            if architecture == "single_gene"
            else "Seurat::AddModuleScore after panel freeze"
        ),
    }


CANDIDATES = [
    # Cross-experiment core programmes. Their genes must be unchanged across
    # MCT1_CTL, MCT1_KO and V4_WT.
    _candidate("proliferation_current", "Cell-cycle and proliferation", "current", "all_samples", ALL_SAMPLES, ["Cdk1", "Birc5", "Top2a", "Mki67", "Mcm5"], "Existing harmonized candidate."),
    _candidate("proliferation_compact", "Cell-cycle and proliferation", "compact", "all_samples", ALL_SAMPLES, ["Cdk1", "Birc5", "Top2a"], "Three canonical cycling genes with strong focus-ROI support."),
    _candidate("proliferation_extended", "Cell-cycle and proliferation", "established_extension", "all_samples", ALL_SAMPLES, ["Cdk1", "Birc5", "Top2a", "Mki67", "Mcm5", "Ube2c"], "Adds a well-supported mitotic effector."),
    _candidate("proliferation_single_mki67", "Cell-cycle and proliferation", "single_gene", "all_samples", ALL_SAMPLES, ["Mki67"], "Canonical single-gene proliferation comparator."),

    _candidate("interferon_current", "Interferon-related gene program", "current", "all_samples", ALL_SAMPLES, ["Gbp2", "Ifi47", "Ifit3", "Isg15", "Irf7"], "Existing harmonized candidate."),
    _candidate("interferon_compact", "Interferon-related gene program", "compact", "all_samples", ALL_SAMPLES, ["Gbp2", "Ifit3", "Isg15"], "Compact interferon-induced panel."),
    _candidate("interferon_stat1_extension", "Interferon-related gene program", "established_extension", "all_samples", ALL_SAMPLES, ["Gbp2", "Ifi47", "Ifit3", "Isg15", "Irf7", "Stat1"], "Adds the central interferon transcriptional mediator STAT1."),
    _candidate("interferon_zbp1_extension", "Interferon-related gene program", "experimental_extension", "all_samples", ALL_SAMPLES, ["Gbp2", "Ifi47", "Ifit3", "Isg15", "Irf7", "Zbp1"], "Tests whether an interferon-linked nucleic-acid sensing gene adds spatial information.", novelty="exploratory_refinement"),
    _candidate("interferon_single_isg15", "Interferon-related gene program", "single_gene", "all_samples", ALL_SAMPLES, ["Isg15"], "Canonical interferon-induced single-gene comparator."),

    _candidate("inflammation_current", "Inflammation", "current", "all_samples", ALL_SAMPLES, ["C3", "Ccl5", "Cxcl10", "Ccl2"], "Existing harmonized candidate."),
    _candidate("inflammation_compact", "Inflammation", "compact", "all_samples", ALL_SAMPLES, ["Ccl5", "Cxcl10", "Ccl2"], "Chemokine-centred compact panel."),
    _candidate("inflammation_ccl12_extension", "Inflammation", "experimental_extension", "all_samples", ALL_SAMPLES, ["C3", "Ccl5", "Cxcl10", "Ccl2", "Ccl12"], "Adds a mouse monocyte-recruiting chemokine with strong MCT1 support.", novelty="model_relevant_refinement"),
    _candidate("inflammation_single_cxcl10", "Inflammation", "single_gene", "all_samples", ALL_SAMPLES, ["Cxcl10"], "Strongly detected inflammatory chemokine comparator."),

    _candidate("myeloid_current", "Myeloid / phagocytic", "current", "all_samples", ALL_SAMPLES, ["Fcer1g", "Fcgr2b", "Lyz2", "Tyrobp"], "Existing harmonized candidate."),
    _candidate("myeloid_compact", "Myeloid / phagocytic", "compact", "all_samples", ALL_SAMPLES, ["Fcer1g", "Lyz2", "Tyrobp"], "Compact general-myeloid panel, separate from MHC-II state."),
    _candidate("myeloid_single_tyrobp", "Myeloid / phagocytic", "single_gene", "all_samples", ALL_SAMPLES, ["Tyrobp"], "Broad myeloid single-gene comparator."),
    _candidate("v4_myeloid_legacy", "Myeloid / phagocytic", "legacy_comparison", "legacy_v4_comparison", ["V4_WT"], ["Ly86", "Fcgr3", "Fcer1g", "Fcgr2b"], "Exact legacy V4 myeloid panel retained only to audit the previously reported V4 finding."),

    _candidate("cytotoxic_current", "Cytotoxic T/NK-cell", "current", "all_samples", ALL_SAMPLES, ["Nkg7", "Gzmb", "Cd3g"], "Existing harmonized candidate."),
    _candidate("cytotoxic_mct1_legacy", "Cytotoxic T/NK-cell", "legacy_comparison", "legacy_mct1_comparison", MCT1_SAMPLES, ["Nkg7", "Gzmb", "Cd3d"], "Exact legacy cytotoxic panel, evaluated unchanged in CTL and KO for finding traceability."),
    _candidate("cytotoxic_single_nkg7", "Cytotoxic T/NK-cell", "single_gene", "all_samples", ALL_SAMPLES, ["Nkg7"], "Sensitive cytotoxic-lymphocyte comparator."),
    _candidate("cytotoxic_mct1_tcell_extension", "Cytotoxic T/NK-cell", "experimental_extension", "mct1_shared", MCT1_SAMPLES, ["Nkg7", "Gzmb", "Cd3g", "Cd3d"], "MCT1-specific review candidate adding a second T-cell receptor component.", novelty="model_relevant_refinement"),
    _candidate("cytotoxic_mct1_effector_extension", "Cytotoxic T/NK-cell", "experimental_extension", "mct1_shared", MCT1_SAMPLES, ["Nkg7", "Gzmb", "Cd3g", "Prf1"], "MCT1-specific review candidate adding perforin; expected to be focal.", novelty="model_relevant_refinement"),

    _candidate("ecm_current", "ECM remodelling and signalling", "current", "all_samples", ALL_SAMPLES, ["Tgfbi", "Col18a1", "Mmp2", "Pcolce"], "Existing harmonized candidate."),
    _candidate("ecm_compact", "ECM remodelling and signalling", "compact", "all_samples", ALL_SAMPLES, ["Tgfbi", "Col18a1", "Mmp2"], "Compact basement-membrane/remodelling panel."),
    _candidate("ecm_col11a1_extension", "ECM remodelling and signalling", "experimental_extension", "all_samples", ALL_SAMPLES, ["Tgfbi", "Col18a1", "Mmp2", "Pcolce", "Col11a1"], "Adds a tumour-associated collagen with strong focus-ROI support.", novelty="gbm_context_refinement"),
    _candidate("ecm_single_mmp2", "ECM remodelling and signalling", "single_gene", "all_samples", ALL_SAMPLES, ["Mmp2"], "Matrix-remodelling single-gene comparator."),

    _candidate("mesenchymal_current", "Mesenchymal / reactive stromal", "current", "all_samples", ALL_SAMPLES, ["Fn1", "S100a4", "Vim"], "Existing harmonized candidate."),
    _candidate("mesenchymal_compact", "Mesenchymal / reactive stromal", "compact", "all_samples", ALL_SAMPLES, ["Fn1", "S100a4"], "Compact stromal/mesenchymal pair."),
    _candidate("mesenchymal_gfap_extension", "Mesenchymal / reactive stromal", "experimental_extension", "all_samples", ALL_SAMPLES, ["Fn1", "S100a4", "Vim", "Gfap"], "Explicitly tests a reactive-astroglial extension; host leakage is a review risk.", relevance="medium", novelty="contextual_refinement"),
    _candidate("mesenchymal_single_fn1", "Mesenchymal / reactive stromal", "single_gene", "all_samples", ALL_SAMPLES, ["Fn1"], "ECM-associated mesenchymal single-gene comparator."),

    # Shared MCT1 extensions. Composition is identical in CTL and KO.
    _candidate("innate_current", "Innate immune", "current", "mct1_shared", MCT1_SAMPLES, ["Casp4", "Il1b", "Ripk2"], "Existing shared MCT1 candidate."),
    _candidate("innate_compact", "Innate immune", "compact", "mct1_shared", MCT1_SAMPLES, ["Casp4", "Ripk2"], "Compact innate-sensing pair."),
    _candidate("innate_single_casp4", "Innate immune", "single_gene", "mct1_shared", MCT1_SAMPLES, ["Casp4"], "Inflammatory-caspase single-gene comparator."),

    _candidate("hypoxia_current", "Hypoxia / stress", "current", "mct1_shared", MCT1_SAMPLES, ["Ddit4", "Ddit3", "Trib3", "Atf4"], "Existing shared MCT1 candidate."),
    _candidate("hypoxia_compact", "Hypoxia / stress", "compact", "mct1_shared", MCT1_SAMPLES, ["Ddit4", "Ddit3", "Trib3"], "Compact hypoxia/integrated-stress panel."),
    _candidate("hypoxia_slc16a3_extension", "Hypoxia / stress", "experimental_extension", "mct1_shared", MCT1_SAMPLES, ["Ddit4", "Ddit3", "Trib3", "Atf4", "Slc16a3"], "Adds MCT4, a GBM hypoxia/lactate-axis gene directly relevant to the MCT1 experiment.", novelty="mct_axis_refinement"),
    _candidate("hypoxia_single_ddit4", "Hypoxia / stress", "single_gene", "mct1_shared", MCT1_SAMPLES, ["Ddit4"], "Hypoxia/stress single-gene comparator."),
    _candidate("hypoxia_single_trib3", "Hypoxia / stress", "single_gene", "all_samples", ALL_SAMPLES, ["Trib3"], "Portable integrated-stress marker; retained as a single-gene comparator, not a cross-sample hypoxia programme."),

    _candidate("vascular_current", "Vascular / angiogenic", "current", "mct1_shared", MCT1_SAMPLES, ["Cldn5", "Vwf", "Angpt2"], "Existing shared MCT1 candidate."),
    _candidate("vascular_compact", "Vascular / angiogenic", "compact", "all_samples", ALL_SAMPLES, ["Cldn5", "Vwf"], "Portable endothelial-identity pair, defined from the MCT focus and evaluated unchanged in V4."),
    _candidate("vascular_single_cldn5", "Vascular / angiogenic", "single_gene", "all_samples", ALL_SAMPLES, ["Cldn5"], "Portable endothelial single-gene comparator."),

    _candidate("neural_current", "Neural-like / differentiation", "current", "mct1_shared", MCT1_SAMPLES, ["Olig1", "Olig2", "Sox2"], "Existing shared MCT1 candidate."),
    _candidate("neural_compact", "Neural-like / differentiation", "compact", "mct1_shared", MCT1_SAMPLES, ["Olig2", "Sox2"], "Compact oligodendroglial/stem-like pair."),
    _candidate("neural_olig_portable", "Neural-like / differentiation", "portable", "all_samples", ALL_SAMPLES, ["Olig1", "Olig2"], "Portable oligodendroglial-lineage pair; avoids Sox2, which is absent from the V4 feature set."),
    _candidate("neural_single_olig2", "Neural-like / differentiation", "single_gene", "all_samples", ALL_SAMPLES, ["Olig2"], "Portable canonical GBM lineage-state comparator."),

    _candidate("apc_myeloid_current", "Antigen-presenting myeloid state", "current", "all_samples", ALL_SAMPLES, ["Cd74", "H2-Aa", "H2-Ab1", "H2-Eb1"], "MCT-defined MHC-II state panel, evaluated unchanged in V4."),
    _candidate("apc_myeloid_ctl_legacy", "Antigen-presenting myeloid state", "legacy_comparison", "legacy_ctl_comparison", ["MCT1_CTL"], ["Ly86", "H2-Eb1", "Lyz2", "H2-Aa"], "Exact legacy CTL APC panel retained only to audit the result already shared with biologists."),
    _candidate("apc_myeloid_compact", "Antigen-presenting myeloid state", "compact", "all_samples", ALL_SAMPLES, ["Cd74", "H2-Aa", "H2-Eb1"], "Portable compact MHC-II antigen-presentation panel."),
    _candidate("apc_myeloid_single_cd74", "Antigen-presenting myeloid state", "single_gene", "all_samples", ALL_SAMPLES, ["Cd74"], "Portable antigen-presentation single-gene comparator."),

    # V4-only extension; it does not redefine the common mesenchymal panel.
    _candidate("v4_reactive_current", "Reactive tumour-brain interface", "current", "v4_specific", ["V4_WT"], ["Pdpn", "A2m", "Serping1"], "Existing V4-specific extension."),
    _candidate("v4_reactive_legacy", "Reactive tumour-brain interface", "legacy_comparison", "legacy_v4_comparison", ["V4_WT"], ["S100a4", "Pdpn", "A2m"], "Exact legacy V4 reactive panel retained only to audit the result already shared with biologists."),
    _candidate("v4_reactive_compact", "Reactive tumour-brain interface", "compact", "v4_specific", ["V4_WT"], ["Pdpn", "A2m"], "Compact V4 interface pair."),
    _candidate("v4_reactive_s100a4_extension", "Reactive tumour-brain interface", "experimental_extension", "v4_specific", ["V4_WT"], ["Pdpn", "A2m", "Serping1", "S100a4"], "Tests whether the common stromal marker sharpens the V4 interface."),
    _candidate("v4_reactive_single_pdpn", "Reactive tumour-brain interface", "single_gene", "v4_specific", ["V4_WT"], ["Pdpn"], "Interface-associated single-gene comparator."),

    # Technical locator candidates. These are never biological pair-analysis
    # programmes.
    _candidate("tumour_anchor_legacy", "GBM-associated tumour-compartment anchor", "legacy", "technical_all_samples", ALL_SAMPLES, ["Cdkn2a", "Tcim", "Il11", "Emp3"], "Old dataset-derived locator retained for comparison.", relevance="medium"),
    _candidate("tumour_anchor_gbm_supported", "GBM-associated tumour-compartment anchor", "gbm_supported", "technical_all_samples", ALL_SAMPLES, ["Hmga2", "Cd109", "Cd44", "Emp3"], "More conventional GBM-associated locator supported in all three sections."),
    _candidate("tumour_anchor_compact", "GBM-associated tumour-compartment anchor", "compact", "technical_all_samples", ALL_SAMPLES, ["Hmga2", "Cd109", "Cd44"], "Compact GBM-associated locator."),
    _candidate("tumour_anchor_single_hmga2", "GBM-associated tumour-compartment anchor", "single_gene", "technical_all_samples", ALL_SAMPLES, ["Hmga2"], "Model-line single-gene locator; not a universal malignant-cell marker."),
]


TOPIC_SOURCE = {
    "Cell-cycle and proliferation": "neftel_2019",
    "Interferon-related gene program": "ravi_2022",
    "Inflammation": "ravi_2022",
    "Myeloid / phagocytic": "pombo_antunes_2021",
    "Cytotoxic T/NK-cell": "cloughesy_2021",
    "ECM remodelling and signalling": "gbm_ecm_2023",
    "Mesenchymal / reactive stromal": "neftel_2019",
    "Innate immune": "pombo_antunes_2021",
    "Hypoxia / stress": "gbm_hypoxia_2020",
    "Vascular / angiogenic": "gbm_ecm_2023",
    "Neural-like / differentiation": "neftel_2019",
    "Antigen-presenting myeloid state": "cloughesy_2021",
    "Reactive tumour-brain interface": "ravi_2022",
    "GBM-associated tumour-compartment anchor": "neftel_2019",
}

SPECIAL_GENE_EVIDENCE = {
    "Slc16a3": ("gbm_established", "MCT4; GBM hypoxia/lactate-axis candidate", "mct4_gbm_2014"),
    "Hmga2": ("gbm_established", "GBM stemness and tumour growth support", "hmga2_gbm"),
    "Cd109": ("gbm_established", "GBM stem-like-cell and tumour-growth support", "cd109_gbm"),
    "Emp3": ("gbm_supported", "GBM mesenchymal/tumour-associated support", "emp3_gbm"),
    "Zbp1": ("process_established", "interferon-linked nucleic-acid sensing; exploratory spatial refinement", "ravi_2022"),
    "Ccl12": ("model_relevant", "mouse monocyte-recruiting chemokine; experimental refinement", "pombo_antunes_2021"),
    "Col11a1": ("gbm_supported", "tumour-associated ECM candidate", "gbm_ecm_2023"),
    "Gfap": ("context_dependent", "reactive astroglial marker with host-leakage risk", "ravi_2022"),
}

# Most selected mouse symbols have a directly conserved human symbol. These
# exceptions are biologically important: murine Ifi47 lacks a strict human
# orthologue, Ccl12 maps to a multi-gene human chemokine group, and mouse MHC
# class-II genes map to HLA families rather than by capitalization. Sources:
# MGI/Alliance homology pages.
ORTHOLOGY_EXCEPTIONS = {
    "Ifi47": ("", "no_strict_human_ortholog", "https://www.informatics.jax.org/marker/MGI%3A99448"),
    "Ccl12": ("CCL2|CCL7|CCL8", "one_to_many_homology", "https://www.informatics.jax.org/marker/MGI%3A108224"),
    "H2-Aa": ("HLA-DQA1|HLA-DQA2", "one_to_many_homology", "https://www.informatics.jax.org/marker/MGI%3A95895"),
    "H2-Ab1": ("HLA-DQB1|HLA-DQB2", "one_to_many_homology", "https://www.informatics.jax.org/marker/MGI%3A103070"),
    "H2-Eb1": ("HLA-DRB1|HLA-DRB3|HLA-DRB4|HLA-DRB5", "one_to_many_homology", "https://www.informatics.jax.org/marker/MGI%3A95901"),
}


def _load_marker_rows() -> pd.DataFrame:
    from analyses.mouse_gbm_spatial_programs.phase4_harmonized_candidates import _load_marker_rows as load_rows

    rows = load_rows()
    return rows[rows["marker_direction"].str.endswith("enriched")].copy()


def _candidate_gene_rows(marker_rows: pd.DataFrame) -> pd.DataFrame:
    selected = sorted({gene for candidate in CANDIDATES for gene in candidate["genes"]})
    records: list[dict[str, Any]] = []
    for gene in selected:
        candidates_for_gene = [c for c in CANDIDATES if gene in c["genes"]]
        topics = sorted({str(c["topic"]) for c in candidates_for_gene})
        evidence = marker_rows[marker_rows["gene"].eq(gene)].copy()
        in_window = evidence[evidence["rank_within_direction"].le(1000)]
        chosen = in_window if not in_window.empty else evidence
        retrieval_class = (
            "top_1000_enriched_marker"
            if not in_window.empty
            else "targeted_literature_rescue_below_1000"
            if not evidence.empty
            else "not_retrieved"
        )
        for topic in topics:
            special = SPECIAL_GENE_EVIDENCE.get(gene)
            source_key = special[2] if special else TOPIC_SOURCE[topic]
            orthology = ORTHOLOGY_EXCEPTIONS.get(
                gene,
                (
                    gene.upper(),
                    "provisional_symbol_conserved",
                    "verify against MGI/Alliance before manuscript use",
                ),
            )
            records.append(
                {
                    "topic": topic,
                    "mouse_gene": gene,
                    "human_ortholog_symbol": orthology[0],
                    "orthology_status": orthology[1],
                    "orthology_source_or_note": orthology[2],
                    "literature_tier": special[0] if special else "process_established",
                    "literature_note": special[1] if special else f"Established component/marker of {topic}.",
                    "primary_source": SOURCES[source_key],
                    "retrieval_class": retrieval_class,
                    "best_enriched_marker_rank": chosen["rank_within_direction"].min() if not chosen.empty else pd.NA,
                    "n_enriched_marker_views": chosen["source_contrast"].nunique() if not chosen.empty else 0,
                    "retrieval_samples": "|".join(sorted(chosen["source_sample"].astype(str).unique())) if not chosen.empty else "",
                    "retrieval_contrasts": "|".join(sorted(chosen["source_contrast"].astype(str).unique())) if not chosen.empty else "",
                }
            )
    return pd.DataFrame(records).sort_values(["topic", "mouse_gene"]).reset_index(drop=True)


def prepare() -> None:
    names = [candidate["candidate"] for candidate in CANDIDATES]
    if len(names) != len(set(names)):
        raise RuntimeError("candidate names are not unique")
    for candidate in CANDIDATES:
        samples = candidate["intended_samples"]
        if candidate["scope"] == "mct1_shared" and samples != MCT1_SAMPLES:
            raise RuntimeError(f"{candidate['candidate']} is not identical across CTL and KO")
        if candidate["scope"] == "all_samples" and samples != ALL_SAMPLES:
            raise RuntimeError(f"{candidate['candidate']} does not cover all samples")

    marker_rows = _load_marker_rows()
    evidence = _candidate_gene_rows(marker_rows)
    missing = evidence[evidence["retrieval_class"].eq("not_retrieved")]
    if not missing.empty:
        raise RuntimeError(
            "candidate genes lack enriched Seurat marker support: "
            + ", ".join(missing["mouse_gene"].drop_duplicates())
        )

    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    catalogue = {
        "status": "candidate_comparison_not_frozen",
        "marker_search": {
            "authority": "Seurat enriched markers from predeclared focus views",
            "rank_window_per_direction": 1000,
            "targeted_rescue": "allowed only for a predeclared literature-supported gene; reported explicitly",
        },
        "scope_contract": {
            "all_samples": ALL_SAMPLES,
            "mct1_shared": MCT1_SAMPLES,
            "v4_specific": ["V4_WT"],
            "definition_rois": {
                "MCT1_CTL": ["2", "5"],
                "MCT1_KO": ["4", "9"],
                "V4_WT": ["10", "2", "1"],
            },
            "held_out_rule": "all other ROIs and all TileScope pair outcomes are evaluation-only",
        },
        "selection_rule": {
            "hard_gates": [
                "biological relevance",
                "complete gene availability in every intended sample",
                "focus-ROI detection and spatial-component coverage",
            ],
            "comparison_axes": [
                "same-spot co-detection",
                "within-ROI gene coherence",
                "cross-sample portability",
                "literature maturity",
                "novelty",
            ],
            "no_composite_score": True,
            "extension_rule": "compare with and without the added gene; the extension must add spatial information without dominating the canonical panel",
            "single_gene_rule": "report as a spatial marker, never as a biological programme",
        },
        "candidates": CANDIDATES,
    }
    with OUT_YAML.open("w") as handle:
        yaml.safe_dump(catalogue, handle, sort_keys=False, allow_unicode=True)
    OUT_GENE_EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(OUT_GENE_EVIDENCE, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"wrote {OUT_YAML} ({len(CANDIDATES)} candidates)")
    print(f"wrote {OUT_GENE_EVIDENCE} ({len(evidence)} topic-gene rows)")


def summarize() -> None:
    catalogue = yaml.safe_load(OUT_YAML.read_text())
    specs = pd.DataFrame(catalogue["candidates"])
    audit = pd.read_csv(AUDIT_ROWS)
    evidence = pd.read_csv(OUT_GENE_EVIDENCE)

    comparable = audit[audit["candidate_status"].eq("complete")].copy()
    grouped = comparable.groupby("candidate", sort=False).agg(
        n_focus_contexts=("roi_id", "size"),
        minimum_gene_detection=("minimum_gene_detection", "min"),
        mean_gene_detection=("mean_gene_detection", "mean"),
        mean_abs_gene_detection_delta=("mean_abs_gene_detection_delta", "mean"),
        mean_abs_raw_gene_mean_contrast=("raw_gene_mean_contrast", lambda x: x.abs().mean()),
        maximum_abs_raw_gene_mean_contrast=("raw_gene_mean_contrast", lambda x: x.abs().max()),
        minimum_fraction_spots_at_least_half_genes=("fraction_spots_at_least_half_genes", "min"),
        mean_fraction_spots_at_least_half_genes=("fraction_spots_at_least_half_genes", "mean"),
        minimum_fraction_spots_all_genes=("fraction_spots_all_genes", "min"),
        mean_fraction_spots_all_genes=("fraction_spots_all_genes", "mean"),
        mean_pairwise_codetection=("mean_pairwise_both_detected_fraction", "mean"),
        minimum_pairwise_codetection=("minimum_pairwise_both_detected_fraction", "min"),
        mean_pairwise_spearman=("mean_pairwise_spearman", "mean"),
        minimum_pairwise_spearman=("minimum_pairwise_spearman", "min"),
    ).reset_index()
    statuses = audit.groupby("candidate")["candidate_status"].apply(
        lambda x: "|".join(sorted(set(x)))
    ).rename("candidate_statuses").reset_index()

    evidence_by_gene = evidence.drop_duplicates(["mouse_gene", "topic"])
    gene_diagnostics = pd.read_csv(
        PHASE_ROOT / "phase4_harmonized" / "focus_gene_diagnostics.csv"
    ).drop_duplicates(["sample_dir", "roi_id", "gene"])
    gene_rows = []
    for candidate in catalogue["candidates"]:
        topic = candidate["topic"]
        rows = evidence_by_gene[
            evidence_by_gene["mouse_gene"].isin(candidate["genes"])
            & evidence_by_gene["topic"].eq(topic)
        ]
        coverage = gene_diagnostics[
            gene_diagnostics["gene"].isin(candidate["genes"])
            & gene_diagnostics["sample_dir"].isin(candidate["intended_samples"])
        ]
        gene_rows.append(
            {
                "candidate": candidate["candidate"],
                "genes_csv": ", ".join(candidate["genes"]),
                "n_genes": len(candidate["genes"]),
                "n_genes_top_1000": int(rows["retrieval_class"].eq("top_1000_enriched_marker").sum()),
                "n_targeted_rescues": int(rows["retrieval_class"].eq("targeted_literature_rescue_below_1000").sum()),
                "maximum_best_marker_rank": rows["best_enriched_marker_rank"].max(),
                "literature_tiers": "|".join(sorted(rows["literature_tier"].unique())),
                "minimum_component_coverage": coverage[
                    "component_coverage_fraction"
                ].min(),
                "mean_component_coverage": coverage[
                    "component_coverage_fraction"
                ].mean(),
            }
        )
    gene_summary = pd.DataFrame(gene_rows)

    output = (
        specs.merge(gene_summary, on="candidate", how="left")
        .merge(statuses, on="candidate", how="left")
        .merge(grouped, on="candidate", how="left")
    )
    output["intended_samples"] = output["intended_samples"].apply(
        lambda value: "|".join(value) if isinstance(value, list) else value
    )
    output["genes"] = output["genes"].apply(
        lambda value: "|".join(value) if isinstance(value, list) else value
    )
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT_COMPARISON, index=False)
    sample_summary = comparable.groupby(
        ["candidate", "topic", "architecture", "scope", "sample_dir"],
        sort=False,
    ).agg(
        n_focus_contexts=("roi_id", "size"),
        minimum_gene_detection=("minimum_gene_detection", "min"),
        mean_gene_detection=("mean_gene_detection", "mean"),
        mean_abs_gene_detection_delta=("mean_abs_gene_detection_delta", "mean"),
        mean_abs_raw_gene_mean_contrast=("raw_gene_mean_contrast", lambda x: x.abs().mean()),
        minimum_fraction_spots_at_least_half_genes=("fraction_spots_at_least_half_genes", "min"),
        mean_fraction_spots_at_least_half_genes=("fraction_spots_at_least_half_genes", "mean"),
        mean_pairwise_codetection=("mean_pairwise_both_detected_fraction", "mean"),
        mean_pairwise_spearman=("mean_pairwise_spearman", "mean"),
    ).reset_index()
    sample_summary.to_csv(OUT_SAMPLE_COMPARISON, index=False)
    print(f"wrote {OUT_COMPARISON} ({len(output)} candidates)")
    print(
        f"wrote {OUT_SAMPLE_COMPARISON} "
        f"({len(sample_summary)} candidate-sample rows)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "summarize"])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        summarize()


if __name__ == "__main__":
    main()
