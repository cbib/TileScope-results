"""Author-context/ROI annotation and barcode-checked cohort preparation.

Preparation validates original barcodes, coordinates, ROI identity and current
spot-ledger order. Intermediate/unannotated observations are never assigned a
structured/disorganized identity. Expression inputs and existing outputs remain
untouched. This module does not recompute statistics on masked observations.
"""
from __future__ import annotations
from tilescope_results.paths import greenwald_evidence

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from tilescope_results.paths import core_root, data_root, input_data_root

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'analyses/greenwald_full'))
from pair_catalog import load_pair_catalog
OUT = greenwald_evidence('context_matching')
DATA = data_root() / "heiland_tyrosh"
CORE = core_root()
BASELINE_CORE = "91ff89600b48be38f4d90dbcd0716cdfc65dca10"
CONTEXTS = ("structured", "disorganized")
ZONE_CLASS = {"st_mal": "structured", "st_norm": "structured", "dis_mal": "disorganized",
              "dis_norm": "disorganized", "Intermediate": "intermediate", "": "unannotated"}


def read(path, **kwargs):
    # Match the production core's pd.read_csv precision, not a different parser.
    return pd.read_csv(path, dtype={"roi_id": str, "source_roi_id": str, "parent_roi_id": str}, **kwargs)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_prepared_source(source, membership):
    """A mask may only be reused against its original observation ordering."""
    if len(source) != len(membership) or not np.array_equal(membership.spot_index, np.arange(len(source))):
        raise ValueError("Prepared mask row/order mismatch")
    if not np.array_equal(source.roi_id.astype(str), membership.roi_id.astype(str)):
        raise ValueError("Prepared mask ROI identity mismatch")
    if not np.allclose(source[["x", "y"]].to_numpy(float), membership[["x", "y"]].to_numpy(float),
                       rtol=0, atol=1e-7):
        raise ValueError("Prepared mask coordinate identity mismatch")


def official_to_current(name):
    # This alias is accepted only AFTER barcode/coordinate/ROI validation.
    if name == "ZH881inf":
        return "GBM_ZH881_inf"
    if name.startswith("UKF") and name[3:].isdigit():
        return f"{name[3:]}_T"
    if name == "MGH258":
        return "GBM_MGH258"
    if name.startswith("ZH"):
        return f"GBM_{name}"
    raise ValueError(f"Unknown official sample alias: {name}")


def validate_annotation_join(source, ledger, annotation, *, atol=1e-7):
    """Reject reordered, mismatched or ambiguous identity; never nearest-match."""
    n = len(source)
    if len(ledger) != n or len(annotation) != n:
        raise ValueError("CSV / ledger / Seurat row count mismatch")
    expected = np.arange(n)
    if not np.array_equal(ledger.spot_index, expected) or not np.array_equal(annotation.spot_index, expected):
        raise ValueError("Noncanonical spot order")
    if annotation.spot_barcode.duplicated().any():
        raise ValueError("Duplicate barcode")
    for ids in (ledger.roi_id, annotation.source_roi_id):
        if not np.array_equal(source.roi_id.astype(str), ids.astype(str)):
            raise ValueError("ROI identity mismatch")
    delta = np.abs(source[["x", "y"]].to_numpy(float) - annotation[["x", "y"]].to_numpy(float))
    if not np.isfinite(delta).all() or delta.max(initial=0) > atol:
        raise ValueError("Coordinate/order mismatch")
    return float(delta.max(initial=0))


def roi_context_inventory(spots, cohorts):
    records = []
    eligible = {(r.context, r.sample_id): bool(r.in_author_context_cohort)
                for r in cohorts.itertuples()}
    for (sample, roi), g in spots.groupby(["sample_id", "roi_id"], sort=True):
        counts = g.structure_class.value_counts()
        annotated = bool(g.annotation_available.iloc[0])
        for context in CONTEXTS:
            keep = g.structure_class.eq(context)
            n = int(keep.sum())
            status = ("annotation_unavailable" if not annotated else
                      "no_context_spots" if n == 0 else
                      "whole_roi_in_context" if n == len(g) else "roi_context_intersection")
            ids = g.loc[keep, "spot_index"].astype(int).to_numpy()
            records.append(dict(
                sample_id=sample, roi_id=roi, context=context,
                in_author_context_cohort=eligible[(context, sample)],
                annotation_available=annotated, mask_status=status,
                n_source_spots_original=len(g), n_context_spots=n,
                n_other_context_spots=int(counts.get("disorganized" if context == "structured" else "structured", 0)),
                n_intermediate_spots=int(counts.get("intermediate", 0)),
                n_unannotated_spots=int(counts.get("unannotated", 0)),
                n_annotation_unavailable_spots=int(counts.get("annotation_unavailable", 0)),
                fraction_original_roi_in_context=n / len(g),
                source_spot_indices_sha256=hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
                constructed_status=("not_in_author_context_cohort" if not eligible[(context, sample)]
                                    else "masked_rerun_required" if n else "no_context_spots"),
            ))
    return pd.DataFrame(records)


def prepare(output=OUT, data=DATA):
    output.mkdir(parents=True, exist_ok=True)
    original = data / "Original/Inputs"
    zones_path = original / "Spatial_coh_zones/final_zones.rds"
    official = (original / "general/GBM_samples.txt").read_text().split()
    # Production task inventory, not the obsolete Iteration-C cache directory.
    task_path = DATA / "CurrentResults/run_accounting/run_status.csv"
    tasks = pd.read_csv(task_path)
    samples = sorted(tasks.sample_id.unique())
    cohort_path = input_data_root() / 'greenwald/author_context_samples.csv'
    declared = pd.read_csv(cohort_path)
    cohort_set = set(zip(declared.context, declared.author_sample_id))
    mapping = pd.DataFrame([dict(author_sample_id=s, sample_id=official_to_current(s),
                                 in_current_corpus=official_to_current(s) in samples) for s in official])
    if mapping.sample_id.duplicated().any():
        raise ValueError("Ambiguous author sample mapping")
    mapping.to_csv(output / "author_sample_mapping.csv", index=False)
    raw = output / "author_spot_annotations.csv"
    subprocess.run(["Rscript", str(HERE / "extract_context_annotations.R"), str(zones_path),
                    str(output / "author_sample_mapping.csv"), str(data / "RDSs"), str(raw)], check=True)
    ann = read(raw, keep_default_na=False)
    if not set(ann.author_zone) <= set(ZONE_CLASS):
        raise ValueError("Unexpected author zone; no implicit relabelling allowed")
    mapping_by_sample = mapping.set_index("sample_id").author_sample_id.to_dict()
    spot_rows, sample_rows, inputs = [], [], []
    for sample in samples:
        source_path = data / "CSVs" / f"{sample}_expression_data_AC_OPC.csv"
        ledger_path = data / "CurrentResults" / sample / "AC_OPC/spot_ledger.csv"
        source, ledger = read(source_path), read(ledger_path)
        if not np.array_equal(ledger.spot_index, np.arange(len(source))) or not np.array_equal(source.roi_id, ledger.roi_id):
            raise ValueError(f"Current CSV/ledger identity mismatch: {sample}")
        g = ann.loc[ann.sample_id == sample].copy().reset_index(drop=True)
        available = sample in mapping_by_sample
        official_id = mapping_by_sample.get(sample, "")
        maximum = np.nan
        if available:
            maximum = validate_annotation_join(source, ledger, g)
            g["structure_class"] = g.author_zone.map(ZONE_CLASS)
            g["roi_id"] = source.roi_id
            inputs.append(dict(path=str(data / "RDSs" / f"seuratObject_denoised_clustered_{sample}_500.RDS"),
                               sha256=sha256(data / "RDSs" / f"seuratObject_denoised_clustered_{sample}_500.RDS")))
        else:
            g = source[["x", "y", "roi_id"]].copy()
            g["sample_id"], g["author_sample_id"] = sample, ""
            g["spot_index"] = np.arange(len(g))
            g["spot_barcode"], g["author_zone"] = "", ""
            g["structure_class"] = "annotation_unavailable"
        g["annotation_available"] = available
        spot_rows.append(g[["sample_id", "author_sample_id", "spot_index", "spot_barcode", "x", "y", "roi_id",
                            "author_zone", "structure_class", "annotation_available"]])
        inputs.extend(dict(path=str(p), sha256=sha256(p)) for p in (source_path, ledger_path))
        for context in CONTEXTS:
            member = (context, official_id) in cohort_set
            sample_rows.append(dict(
                sample_id=sample, author_sample_id=official_id, context=context, in_current_corpus=True,
                in_author_context_cohort=member, annotation_available=available,
                cohort_match_status=("matched" if member and available else
                                     "not_in_author_context_cohort" if available else "annotation_unavailable"),
                n_input_spots=len(g), n_context_spots=int(g.structure_class.eq(context).sum()),
                max_coordinate_difference=maximum,
            ))
    for r in declared.itertuples():
        sample = official_to_current(r.author_sample_id)
        if sample not in samples:
            sample_rows.append(dict(sample_id=sample, author_sample_id=r.author_sample_id, context=r.context,
                                    in_current_corpus=False, in_author_context_cohort=True, annotation_available=False,
                                    cohort_match_status="author_sample_absent_from_current_corpus", n_input_spots=0,
                                    n_context_spots=0, max_coordinate_difference=np.nan))
    spots = pd.concat(spot_rows, ignore_index=True)
    cohorts = pd.DataFrame(sample_rows)
    inventory = roi_context_inventory(spots, cohorts)
    spots.to_csv(output / "spot_context_membership.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    cohorts.to_csv(output / "cohort_eligibility.csv", index=False)
    inventory.to_csv(output / "roi_context_support.csv", index=False)
    for p in (zones_path, cohort_path, original / "general/GBM_samples.txt", task_path):
        inputs.append(dict(path=str(p), sha256=sha256(p)))
    provenance = dict(contract="exact author-barcode/coordinate/ROI join; no purity threshold or inferred labels",
                      cohort_authority=str(cohort_path), baseline_core=BASELINE_CORE,
                      n_current_samples=len(samples), n_author_samples_with_validated_annotations=int(spots.loc[spots.annotation_available, "sample_id"].nunique()),
                      matched_sample_counts={c: int(((cohorts.context == c) & (cohorts.cohort_match_status == "matched")).sum()) for c in CONTEXTS},
                      inputs=inputs,
                      outputs_sha256={n: sha256(output / n) for n in ("author_sample_mapping.csv", "author_spot_annotations.csv", "cohort_eligibility.csv", "roi_context_support.csv", "spot_context_membership.csv.gz")},
                      scripts_sha256={p.name: sha256(p) for p in (Path(__file__), HERE / "extract_context_annotations.R")})
    (output / "preparation_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({k: v for k, v in provenance.items() if k.startswith("n_") or k == "matched_sample_counts"}, indent=2), flush=True)
    return spots, cohorts, inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    prepare(args.output)
