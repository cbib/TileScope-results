"""Assign biological topics to candidate mouse GBM marker genes.

Consumes the Phase-1 one-vs-rest and Phase-2 compartment-conditioned marker
tables under $TILESCOPE_DATA_MOUSE_ROOT/mouse_gbm_spatial_programs/, applies
a curated topic-gene dictionary across the agreed vocabulary + technical
anchors, and emits phase3/topic_annotations.csv under the external mouse analysis data.

The dictionary is intentionally curated: canonical markers are assigned
with high confidence, functional-supporting markers with moderate confidence,
context-dependent markers flagged, host-tissue markers marked as contextual
host anchors (not first-class biological signatures). Genes not in the
dictionary remain `unassigned` rather than receiving an unsupported label.

Every emitted row keeps its provenance: source_view / sample / contrast /
direction / rank_within_direction / gene / topic / annotation_confidence /
evidence. Multiple topic assignments per gene are allowed if evidence
supports it (one row per (source_view, sample, contrast, direction, gene,
topic)).
"""
from __future__ import annotations

import os
from pathlib import Path
from tilescope_results.paths import data_root
from typing import Iterable

import pandas as pd

DATA_ROOT = Path(
    os.environ.get("TILESCOPE_DATA_MOUSE_ROOT", str((data_root() / 'mouse')))
)
PHASE1_ROOT = DATA_ROOT / "mouse_gbm_spatial_programs" / "phase1"
PHASE2_MARKERS = DATA_ROOT / "mouse_gbm_spatial_programs" / "phase2" / "phase2_markers.csv"
OUT_CSV = (
    DATA_ROOT / "mouse_gbm_spatial_programs" / "phase3" / "topic_annotations.csv"
)

TOP_N_PER_DIRECTION = 100

# ---------------------------------------------------------------------------
# Curated topic-gene dictionary
# ---------------------------------------------------------------------------
#
# Confidence levels:
#   canonical           textbook marker for the topic
#   supporting          functionally consistent with the topic, less specific
#   context-dependent   fits multiple topics; interpretation depends on context
#   contextual_host     host-tissue anchor, not a first-class biological signature
#
# A gene may appear under multiple topics (e.g. Hmga2 as tumour identity anchor
# AND as context-dependent proliferation). Every assignment emits its own row.

TOPIC_MAP: dict[str, dict[str, list[str]]] = {
    "Proliferation": {
        "canonical": [
            "Mki67", "Birc5", "Cdk1", "Plk1", "Top2a", "Cenpw", "Cenpa",
            "Cenpe", "Cenpf", "Cenpk", "Cenpo", "Pimreg", "Hmmr", "Ccna2",
            "Ccnb1", "Ccnb2", "Ccne1", "Ccne2", "Uhrf1", "Atad2", "Pclaf",
            "Ckap2", "Ckap2l", "Mcm2", "Mcm3", "Mcm4", "Mcm5", "Mcm6",
            "Mcm7", "Kif11", "Kif20a", "Kif22", "Kif23", "Aurka", "Aurkb",
            "Bub1", "Bub1b", "Tk1", "Tyms", "Rrm1", "Rrm2",
            "Cdt1", "Espl1", "Spc25", "Ube2c", "Cdc20", "Tpx2",
            "Cdca8", "Cks2", "Foxm1",
        ],
        "supporting": ["H19", "Npm3", "Ezh2"],
        "context-dependent": ["Myc", "Hmga1", "Hmga2", "Cdkn2b"],
    },
    "Interferon / antiviral response": {
        "canonical": [
            "Rsad2", "Ifit1", "Ifit2", "Ifit3", "Ifit3b", "Ifi44", "Ifi44l",
            "Ifi204", "Ifi205", "Ifi206", "Ifi207", "Ifi27l2a", "Ifi27l2b",
            "Isg15", "Isg20", "Mx1", "Mx2", "Oas1a", "Oas1b", "Oas1g",
            "Oas2", "Oas3", "Oasl1", "Oasl2", "Stat1", "Stat2", "Irf7",
            "Irf9", "Ifih1", "Ddx58", "Ddx60", "Ddx60l",
            "Gbp2", "Gbp2b", "Gbp3", "Gbp4", "Gbp5", "Gbp6", "Gbp7",
            "Gbp8", "Gbp9", "Gbp10", "Gbp11",
            "Apol9a", "Apol9b", "Cmpk2", "Zbp1", "Trim30a", "Trim30d",
            "Usp18", "Xaf1",
        ],
        "supporting": [
            "Ifi47", "Batf2", "Ube2l6", "Tgtp1", "Mndal", "Nmi", "Irf1",
            "Ubd",
        ],
    },
    "Cytotoxic lymphocyte": {
        # Canonical cytotoxic-T / NK markers only. Mouse granzyme paralogues
        # Gzmd / Gzmf / Gzmg and mast-cell proteases Mcpt1 / Mcpt2 are NOT
        # cytotoxic-lymphocyte evidence — they land unassigned and carry an
        # evidence flag (see PHASE4_FLAG_NOTES) so Phase 4 disambiguates.
        "canonical": [
            "Cd3d", "Cd3e", "Cd3g", "Cd8a", "Cd8b1", "Prf1", "Gzmb",
            "Nkg7", "Klrd1", "Klrb1", "Klrk1", "Ncr1", "Trbc1", "Trbc2",
            "Trac", "Trdc", "Zap70", "Lck", "Ctla4", "Pdcd1",
        ],
    },
    "Myeloid / phagocytic": {
        "canonical": [
            "Cd68", "Cd163", "Lyz2", "Aif1", "Csf1r", "Trem2", "Ms4a7",
            "Fcgr1", "Fcgr2b", "Fcgr3", "Fcgr4", "C1qa", "C1qb", "C1qc",
            "Tyrobp", "Cd74", "H2-Aa", "H2-Ab1", "H2-Eb1", "Fcer1g",
            "Itgam", "Itgax", "Marco", "Msr1",
        ],
        "supporting": ["Ly86"],
    },
    "Inflammation": {
        "canonical": [
            "Cxcl9", "Cxcl10", "Cxcl11", "Cxcl1", "Cxcl2",
            "Ccl2", "Ccl3", "Ccl4", "Ccl5", "Ccl7", "Ccl8", "Ccl12",
            "Il6", "Il1b", "Il1a", "Tnf", "Ifng", "Nfkbia", "Nfkbiz",
            "C3", "C4b", "Lcn2",
        ],
        "supporting": ["Serping1", "Cfh", "A2m", "Cxcl14"],
    },
    "Innate immune response": {
        "canonical": ["Nlrp3", "Aim2", "Casp1", "Casp4", "Il18", "Il1b", "Naip1", "Naip2"],
        "supporting": ["Pycard"],
    },
    "ECM remodelling / signalling": {
        "canonical": [
            "Col1a1", "Col1a2", "Col3a1", "Col4a1", "Col4a2", "Col4a3",
            "Col5a1", "Col5a2", "Col5a3", "Col6a1", "Col6a2", "Col6a3",
            "Col7a1", "Col8a1", "Col8a2", "Col9a1", "Col9a2", "Col9a3",
            "Col10a1", "Col11a1", "Col11a2", "Col12a1", "Col14a1", "Col15a1",
            "Col16a1", "Col18a1", "Col2a1",
            "Mmp2", "Mmp9", "Mmp13", "Mmp14", "Mmp3", "Mmp8", "Mmp11",
            "Timp1", "Timp2", "Timp3",
            "Tgfbi", "Fbn1", "Fbn2", "Serpinh1", "Tnc",
            "Sparc", "Sparcl1", "Postn",
            "Lox", "Loxl1", "Loxl2", "Loxl3", "Loxl4",
            "Lum", "Bgn", "Dcn", "Fmod",
            "Thbs2", "Adamts1", "Pcolce", "P4ha3",
        ],
        "supporting": ["Dkk3", "Ccn3"],
    },
    "Mesenchymal / reactive stromal": {
        "canonical": [
            "Twist1", "Twist2", "Snai1", "Snai2", "Zeb1", "Zeb2",
            "Vim", "Cdh2", "Fn1", "S100a4", "Acta2", "Tagln",
        ],
        "supporting": ["Aqp4", "Gfap", "S100b", "Pdpn"],
    },
    "Hypoxia / stress": {
        "canonical": [
            "Ddit4", "Ddit3", "Trib3", "Vegfa", "Slc2a1", "Ldha",
            "Hif1a", "Bnip3", "Bnip3l", "Aldoa", "Pgk1", "Eno1",
            "Atf3", "Atf4", "Chac1", "Ndrg1", "P4ha1", "P4ha2",
            "Fam162a", "Slc16a3",
        ],
        "supporting": ["Psat1", "Phgdh", "Crabp2", "Areg"],
    },
    "Vascular / angiogenic": {
        # No canonical endothelial program is convincingly present in the
        # current corpus. Mcam / Cd109 / Plagl1 are GBM-aggressiveness-
        # associated but not endothelial-canonical, so they land as
        # context-dependent vascular candidates for Phase 4 to arbitrate.
        "canonical": [
            "Pecam1", "Cdh5", "Vwf", "Kdr", "Flt1", "Flt4", "Tek",
            "Tie1", "Angpt1", "Angpt2", "Cldn5", "Emcn", "Cd93",
            "Egfl7", "Esam", "Robo4",
        ],
        "context-dependent": [
            "Mcam", "Cd109", "Plagl1",
            "Hbb-bt", "Hbb-bs", "Hba-a1", "Hba-a2", "Alas2",
        ],
    },
    "Neural-like / differentiation": {
        # Tumour-cell neural-differentiation programs (not host tissue — host
        # goes under "normal brain / host identity").
        "canonical": [
            "Olig1", "Olig2", "Sox2", "Nes", "Pax6", "Ascl1",
            "Dcx", "Neurod1", "Neurog2",
        ],
    },
    "malignant-compartment / model-line anchor": {
        # Technical anchor. Captures malignant-cell identity signal so
        # downstream analysis can separate it from real biological
        # programmes. Renamed from "tumour identity / invasion" because
        # Rhox2g/h, Tcim, and Fam110c help LOCATE the implanted tumour
        # compartment but do NOT establish invasion biology; invasion is a
        # spatial question for TileScope, not a marker-list question.
        "canonical": [
            "Cdkn2a", "Cdkn2b",
            "Egfr", "Pdgfra", "Cd44", "Nes",
            "Emp3", "Cd109",
        ],
        "supporting": [
            "Hmga1", "Hmga2", "Il11", "Trib3", "Twist1",
            "Fam110c", "Rhox2g", "Rhox2h", "Tcim", "Greb1",
        ],
    },
    "normal brain / host identity": {
        # Contextual host anchor per user direction: neural / striatal /
        # oligodendroglial / neuroendocrine / choroid sets are the peritumour
        # ecosystem, not equal-status biological signatures.
        "contextual_host": [
            # Cortical excitatory
            "Slc17a7", "Neurod6", "Fezf2", "Tbr1", "Nptx1", "Nptx2",
            "Satb2", "Rorb", "Cux2", "Bcl11a", "Bhlhe22",
            # Cortical / general neuronal
            "Vip", "Bdnf", "Nrn1", "Cck", "Lamp5", "Pvalb", "Sst",
            "Snap25", "Rab3a", "Rab3b", "Rab3c", "Stmn2", "Nrsn2",
            "Cplx1", "Ly6h", "Ndn", "Snrpn",
            # Striatal
            "Adora2a", "Drd1", "Drd2", "Pde10a", "Ppp1r1b", "Gpr88",
            "Penk", "Tac1", "Slc32a1",
            # GABAergic
            "Gad2", "Gad1",
            # Ion channels / interneuronal
            "Kcnab3", "Kcnh7", "Kcnc3", "Kcns1", "Kcnab1", "Rims3",
            "Syt2", "Syt17",
            # Neurodevelopmental
            "Zic1", "Zic3", "Sox1", "Dlx1", "Dlx2", "Sp8", "Sp9",
            "Prdm12", "Pnoc",
            # Choroid plexus / ependymal
            "Ttr", "Ecrg4", "Glp1r",
            # Neuroendocrine / hypothalamic
            "Gal", "Gnrh1", "Trh", "Crh", "Agt", "Cartpt", "Oxt",
            "Nts", "Calcr", "Igsf1", "Calb2", "Grm2", "Cpne9",
            # Oligodendroglial / white-matter
            "Mog", "Mal", "Mobp", "Opalin", "Myrf", "Fa2h", "Ugt8a",
            "Cldn11", "Plp1", "Plp2", "Gjc2", "Gjb1", "Gal3st1", "Mag",
            "Enpp2",
            # Astrocytic host-side (reactive gets Mesenchymal / reactive)
            "Aldh1l1", "Slc1a2", "Gja1", "Gjb6", "Aqp4",
            # Ependymal / non-coding neural
            "Bc1",
            # Neuronal support / misc host
            "Fxyd1", "Arpp21", "Necab2", "Spock3", "Gng7", "Nnat",
            "Klk6", "Crym", "Hpca",
            "Chgb", "Nkain4", "Gabra1", "Gabrg1", "Atp2b2", "Cadps",
            "Pgm2l1", "Igfbp5", "Igfbp2", "Rell2", "Adcy1", "Rasgrp1",
            "Peg3", "Cbarp", "Rgs4", "Rnf112", "Btbd3", "Slc24a3",
            "Dpp6", "Efnb3", "Sfrp5", "Unc5d", "Tmem255a", "Ncdn",
            "Atp1a1", "Atp1b1", "Ppp3ca", "Ppp3r1", "Cnmd", "Neb",
            "Prph", "Tafa1", "Tafa2", "Ptgds", "Cxcl14", "Bex4",
            "Csn3", "Chil1", "Serping1", "A2m", "Cfh", "Ly86",
            "Tinagl1", "Igf1", "Cib2",
        ],
    },
}

# Genes that should be surfaced as flagged candidates for Phase 4 to disambiguate
# rather than assigned to a biological topic outright.
PHASE4_FLAG_NOTES: dict[str, str] = {
    # Hemoglobin — vascular contamination candidate
    "Hbb-bt": "hemoglobin — vascular contamination candidate; Phase 4 to disambiguate",
    "Hbb-bs": "hemoglobin — vascular contamination candidate; Phase 4 to disambiguate",
    "Hba-a1": "hemoglobin — vascular contamination candidate; Phase 4 to disambiguate",
    "Hba-a2": "hemoglobin — vascular contamination candidate; Phase 4 to disambiguate",
    "Alas2": "erythroid heme biosynthesis — vascular / plasma contamination candidate",
    # Immunoglobulin / plasma-cell
    "Igkc": "immunoglobulin light chain — plasma-cell candidate; Phase 4 to disambiguate",
    "Igha": "immunoglobulin heavy chain — plasma-cell candidate; Phase 4 to disambiguate",
    "Ighm": "immunoglobulin heavy chain — plasma-cell candidate; Phase 4 to disambiguate",
    "Ighg": "immunoglobulin heavy chain — plasma-cell candidate; Phase 4 to disambiguate",
    "Jchain": "immunoglobulin J chain — plasma-cell candidate; Phase 4 to disambiguate",
    # Mast-cell / granzyme paralogue disambiguation. These are NOT
    # interchangeable evidence for cytotoxic lymphocytes.
    "Gzmd": "unresolved mast-cell / cytotoxic; Gzmd directly supported as a murine mast-cell protease (PMC3676009); Phase 4 must separate Cd3d/Prf1/Nkg7 from Cpa3/Kit/Mcpt co-expression",
    "Gzmf": "unresolved mast-cell / cytotoxic; Gzmf can occur in T-cell states (PMC10984789) but is not cytotoxic-canonical alone; Phase 4 must separate Cd3d/Prf1/Nkg7 from Cpa3/Kit/Mcpt co-expression",
    "Gzmg": "unresolved mast-cell / cytotoxic; mouse granzyme paralogue not cytotoxic-canonical alone; Phase 4 must separate Cd3d/Prf1/Nkg7 from Cpa3/Kit/Mcpt co-expression",
    "Mcpt1": "mast-cell protease (canonical mast-cell marker); NOT cytotoxic-lymphocyte evidence",
    "Mcpt2": "mast-cell protease (canonical mast-cell marker); NOT cytotoxic-lymphocyte evidence",
}


def build_gene_topic_index() -> dict[str, list[tuple[str, str]]]:
    """gene -> list of (topic, confidence)."""
    index: dict[str, list[tuple[str, str]]] = {}
    for topic, buckets in TOPIC_MAP.items():
        for confidence, genes in buckets.items():
            for g in genes:
                index.setdefault(g, []).append((topic, confidence))
    return index


def load_phase1(samples: Iterable[str]) -> pd.DataFrame:
    frames = []
    for sample in samples:
        path = PHASE1_ROOT / sample / "one_vs_rest_markers.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing Phase 1 output: {path}")
        df = pd.read_csv(path)
        df["source_view"] = "phase1_one_vs_rest"
        df["sample"] = sample
        df["contrast"] = df.apply(
            lambda r: f"{sample}_cluster{r['cluster']}_one_vs_rest", axis=1
        )
        df["direction"] = df["marker_direction"]
        frames.append(df[[
            "source_view", "sample", "contrast", "direction", "gene",
            "avg_log2FC", "pct.1", "pct.2", "retrieval_score",
            "rank_within_direction",
        ]])
    return pd.concat(frames, ignore_index=True)


def load_phase2() -> pd.DataFrame:
    if not PHASE2_MARKERS.exists():
        raise FileNotFoundError(f"missing Phase 2 output: {PHASE2_MARKERS}")
    df = pd.read_csv(PHASE2_MARKERS)
    df["source_view"] = "phase2_contrast"
    df["sample"] = df["sample_scope"]
    df["contrast"] = df["contrast_label"]
    df["direction"] = df["marker_direction"]
    return df[[
        "source_view", "sample", "contrast", "direction", "gene",
        "avg_log2FC", "pct.1", "pct.2", "retrieval_score",
        "rank_within_direction",
    ]]


def filter_top_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return (
        df[df["rank_within_direction"] <= n]
        .copy()
        .reset_index(drop=True)
    )


def annotate(df: pd.DataFrame, gene_topic: dict[str, list[tuple[str, str]]]) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        gene = r["gene"]
        base = {
            "source_view": r["source_view"],
            "sample": r["sample"],
            "contrast": r["contrast"],
            "direction": r["direction"],
            "gene": gene,
            "rank_within_direction": int(r["rank_within_direction"]),
            "retrieval_score": float(r["retrieval_score"]),
            "pct_1": float(r["pct.1"]),
            "pct_2": float(r["pct.2"]),
            "avg_log2FC": float(r["avg_log2FC"]),
        }
        assignments = gene_topic.get(gene, [])
        phase4_note = PHASE4_FLAG_NOTES.get(gene, "")
        if not assignments:
            rows.append({
                **base,
                "topic": "unassigned",
                "annotation_confidence": "",
                "evidence": phase4_note,
            })
        else:
            for topic, confidence in assignments:
                rows.append({
                    **base,
                    "topic": topic,
                    "annotation_confidence": confidence,
                    "evidence": phase4_note,
                })
    return pd.DataFrame(rows)


def main() -> None:
    samples = ["MCT1_CTL", "MCT1_KO", "V4_WT"]
    gene_topic = build_gene_topic_index()

    p1 = load_phase1(samples)
    p2 = load_phase2()
    combined = pd.concat([p1, p2], ignore_index=True)
    pool = filter_top_n(combined, TOP_N_PER_DIRECTION)

    out = annotate(pool, gene_topic)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"[phase3] wrote {OUT_CSV} ({len(out)} rows)")
    print()
    print("=== rows per topic ===")
    print(out["topic"].value_counts().to_string())
    print()
    print("=== assigned genes per topic (unique gene count) ===")
    print(
        out[out["topic"] != "unassigned"]
        .groupby("topic")["gene"]
        .nunique()
        .sort_values(ascending=False)
        .to_string()
    )
    print()
    print("=== confidence distribution over assigned rows ===")
    print(
        out[out["topic"] != "unassigned"]["annotation_confidence"]
        .value_counts()
        .to_string()
    )
    print()
    print("=== unassigned gene count per contrast (top 20 contrasts) ===")
    ua = out[out["topic"] == "unassigned"]
    print(
        ua.groupby("contrast")["gene"]
        .nunique()
        .sort_values(ascending=False)
        .head(20)
        .to_string()
    )


if __name__ == "__main__":
    main()
