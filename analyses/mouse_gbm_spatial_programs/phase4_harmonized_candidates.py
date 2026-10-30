"""Build marker-derived candidate pools for harmonized mouse-GBM programs.

Seurat marker tables are the retrieval authority.  The topic dictionary only
labels genes that Seurat retrieved in the predeclared biological focus views;
it never introduces a gene solely because it is biologically plausible.

Selection views:
  * MCT1_CTL ROI 2 and ROI 5, one versus rest and ROI 2 versus ROI 5;
  * MCT1_KO ROI 4 and ROI 9, one versus rest and ROI 4 versus ROI 9;
  * matched CTL/KO core and periphery contrasts;
  * V4_WT ROI 10, ROI 2 and ROI 1, one versus rest and their three
    predeclared contrasts.

There is deliberately no top-N cutoff here.  The marker files already encode
the Seurat retrieval gates, and some legitimate topics (notably neural-like
differentiation) occur below an arbitrary top-200 review window.  Marker rank
is retained as evidence for review.
"""
from __future__ import annotations

import os
from pathlib import Path
from tilescope_results.paths import data_root

import pandas as pd

from analyses.mouse_gbm_spatial_programs.phase3_annotate_topics import TOPIC_MAP

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(
    os.environ.get(
        "TILESCOPE_DATA_MOUSE_ROOT",
        str((data_root() / 'mouse')),
    )
)
PHASE_ROOT = DATA_ROOT / "mouse_gbm_spatial_programs"
OUT_CANDIDATES = PHASE_ROOT / "phase4_harmonized" / "harmonized_topic_candidates.csv"
OUT_PROVENANCE = PHASE_ROOT / "phase4_harmonized" / "harmonized_candidate_provenance.csv"

FOCUS_ROIS = {
    "MCT1_CTL": {"2", "5"},
    "MCT1_KO": {"4", "9"},
    "V4_WT": {"1", "2", "10"},
}

TOPIC_ALIASES = {
    "Interferon / antiviral response": "Interferon-related gene program",
}

# Ripk2 was retrieved by Seurat in every MCT1 focus compartment and is a
# canonical NOD-signalling adaptor.  It expands the otherwise two-gene shared
# CTL/KO innate set (Casp4, Il1b) without borrowing a gene from the final
# myeloid or interferon panels.
CURATED_TOPIC_ADDITIONS = {
    "Ripk2": [
        (
            "Innate immune response",
            "supporting",
            "NOD-signalling adaptor; Seurat-retrieved in CTL and KO focus ROIs",
        )
    ],
}


def _topic_index() -> dict[str, list[tuple[str, str, str]]]:
    index: dict[str, list[tuple[str, str, str]]] = {}
    for source_topic, buckets in TOPIC_MAP.items():
        if source_topic == "normal brain / host identity":
            continue
        topic = TOPIC_ALIASES.get(source_topic, source_topic)
        for confidence, genes in buckets.items():
            for gene in genes:
                index.setdefault(gene, []).append(
                    (topic, confidence, "phase3 curated topic dictionary")
                )
    for gene, assignments in CURATED_TOPIC_ADDITIONS.items():
        index.setdefault(gene, []).extend(assignments)
    return index


def _load_marker_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sample, roi_ids in FOCUS_ROIS.items():
        path = PHASE_ROOT / "phase1" / sample / "one_vs_rest_markers.csv"
        table = pd.read_csv(path)
        table = table[table["cluster"].astype(str).isin(roi_ids)].copy()
        table["source_view"] = "phase1_one_vs_rest"
        table["source_sample"] = sample
        table["source_contrast"] = (
            sample + "_roi" + table["cluster"].astype(str) + "_one_vs_rest"
        )
        frames.append(table)

    phase2 = pd.read_csv(PHASE_ROOT / "phase2" / "phase2_markers.csv")
    phase2["source_view"] = "phase2_focus_contrast"
    phase2["source_sample"] = phase2["sample_scope"]
    phase2["source_contrast"] = phase2["contrast_label"]
    frames.append(phase2)

    keep = [
        "gene",
        "source_view",
        "source_sample",
        "source_contrast",
        "marker_direction",
        "rank_within_direction",
        "retrieval_score",
        "avg_log2FC",
        "pct.1",
        "pct.2",
    ]
    return pd.concat([frame[keep] for frame in frames], ignore_index=True)


def build_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    marker_rows = _load_marker_rows()
    topic_index = _topic_index()
    annotated: list[dict[str, object]] = []
    for row in marker_rows.to_dict("records"):
        for topic, confidence, annotation_basis in topic_index.get(row["gene"], []):
            annotated.append(
                {
                    **row,
                    "topic": topic,
                    "annotation_confidence": confidence,
                    "annotation_basis": annotation_basis,
                }
            )
    provenance = pd.DataFrame(annotated).drop_duplicates()
    if provenance.empty:
        raise RuntimeError("no focus-view Seurat markers matched the topic dictionary")

    candidates = (
        provenance[
            ["topic", "annotation_confidence", "annotation_basis", "gene"]
        ]
        .drop_duplicates()
        .sort_values(["topic", "gene"])
        .reset_index(drop=True)
    )
    return candidates, provenance.sort_values(
        ["topic", "gene", "source_view", "source_contrast", "rank_within_direction"]
    ).reset_index(drop=True)


def main() -> None:
    candidates, provenance = build_candidates()
    OUT_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(OUT_CANDIDATES, index=False)
    provenance.to_csv(OUT_PROVENANCE, index=False)
    print(f"wrote {OUT_CANDIDATES} ({len(candidates)} topic-gene rows)")
    print(f"wrote {OUT_PROVENANCE} ({len(provenance)} evidence rows)")


if __name__ == "__main__":
    main()
