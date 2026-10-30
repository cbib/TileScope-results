"""Validate the complete Meylan cohort and breast run and collect their evidence.

No TileScope run or new spatial statistic is computed here. Correlations are
reproduced from the exact inputs solely to validate the saved cell-level values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from tilescope_results.paths import data_root, reference
from .scope import TABLES, Run, evidence_root, runs
from analyses.meylan_xenium_preparation.cohort import SAMPLE_CELLS


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def read_csv(path):
    return pd.read_csv(path, low_memory=False, dtype={"roi_id": str, "anchor_roi_id": str,
                                                   "responder_roi_id": str})


def csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0}
                 if path.suffix == ".gz" else None)


def align_ledger(frame, ledger):
    """Null eligibility is false, not an excluded row in the denominator."""
    ledger = ledger.sort_values("spot_index").reset_index(drop=True)
    if len(ledger) != len(frame) or not np.array_equal(ledger.spot_index, np.arange(len(frame))):
        raise ValueError("Cell ledger is not complete and uniquely indexed")
    if not np.array_equal(frame.roi_id.astype(str), ledger.roi_id.astype(str)):
        raise ValueError("Cell ledger and input ROI identities differ")
    return ledger


def support_summary(frame, ledger):
    ledger = align_ledger(frame, ledger)
    rows = []
    for roi, d in ledger.groupby("roi_id", sort=True):
        n = len(d)
        eligible = int(d.tessellation_eligible.eq(True).sum())
        rows.append(dict(roi_id=roi, n_cells=n, tessellation_eligible_cells=eligible,
                         eligible_fraction=eligible/n,
                         cells_in_retained_components=int(d.component_status.eq("retained_component").sum()),
                         n_retained_components=d.loc[d.component_status.eq("retained_component"), "component_id"].nunique()))
    return pd.DataFrame(rows)


def annotations(run, frame):
    if run.study == "GBM":
        path = run.input.parent / "cell_annotations.csv.gz"
        all_cells = read_csv(path)
        cells = all_cells.copy()
        columns = ["cell_id", "x", "y", "roi_id", "gene1", "gene2"]
        pd.testing.assert_frame_equal(cells[columns], frame[columns], check_exact=True)
    elif run.study == "Breast":
        path = data_root() / "Janesick/sample1_rep1/roi/cell_roi_assignments.csv.gz"
        all_cells = read_csv(path)
        published = all_cells.loc[all_cells.roi_id.isin(["DCIS_1", "DCIS_2", "Invasive"])].copy()
        # The preparation explicitly sorts the workbook's numeric cell IDs.
        # Verify that exact ordering; tolerate only CSV decimal serialization.
        cells = published.assign(_order=pd.to_numeric(published.cell_id)).sort_values("_order").drop(columns="_order").reset_index(drop=True)
        if len(cells) != len(frame) or not np.array_equal(cells.roi_id, frame.roi_id):
            raise ValueError("Breast cell identifier order/ROI mismatch")
        np.testing.assert_allclose(cells[["x", "y"]], frame[["x", "y"]], rtol=0, atol=1e-8)
    else:
        raise ValueError(f"No annotation adapter for {run.study}")
    if cells.cell_id.duplicated().any(): raise ValueError("Duplicate cell identifiers")
    cells.insert(0, "input_row", np.arange(len(frame)))
    return cells, all_cells, path


def validate_run(run):
    status = json.loads(run.status.read_text())
    if status["status"] != "complete": raise ValueError(f"Run is not complete: {run.sample}/{run.pair}")
    effective = json.loads((run.output / "effective_config.json").read_text())
    config = effective["run_config"]
    for key, value in {"n": 10, "target_observations_per_tile": 15,
                       "min_observations_per_tile": 9, "min_component_observations": 50}.items():
        if config["tessellation"][key] != value: raise ValueError(f"Unexpected Xenium setting {key}")
    if config["technology"] != "xenium": raise ValueError("Not a Xenium run")
    if effective["resolved_policy"]["consensus"]["min_tess_per_hex"] != 7:
        raise ValueError("Unexpected consensus coverage threshold")
    return status, effective


def read_core_table(run, name, status):
    path=run.output/f'{name}.csv'
    if path.exists():
        return read_csv(path)
    rows=status.get('core_result_table_rows',{})
    if rows.get(name)==0 or rows.get(name+'_df')==0:
        return pd.DataFrame()
    raise ValueError(f'Missing core table without a recorded empty result: {run.sample}/{name}')


def correlation_rows(run, tables):
    rows = []
    for name, value, status, surface in [
        ("roi_correlations", "corr_obs", "parent_correlation_status", "cells"),
        ("roi_tessellation_correlations", "corr_median", "parent_tessellation_correlation_status", "tessellations"),
        ("roi_consensus_correlations", "correlation_value", "parent_consensus_correlation_status", None),
    ]:
        for r in tables[name].to_dict("records"):
            rows.append(dict(**run.identity, roi_id=r["roi_id"], representation=surface or r["value_surface"],
                             pearson=r[value], status=r[status],
                             n_units=r.get("sample_size", r.get("n_tiles_contributed_median")),
                             q25=r.get("corr_q25"), q75=r.get("corr_q75"),
                             valid_tessellations=r.get("n_valid_replicates")))
    return rows


def build(output=None, *, include_breast=True):
    out = Path(output or evidence_root())
    if out.exists() and any(out.iterdir()):
        raise ValueError("Evidence output must be empty; build a fresh directory before replacing a package")
    manifest, samples, support, correlations, availability, checks, sources = [], [], [], [], [], [], []
    collected = {t: [] for t in TABLES}
    seen = {}
    selected_runs = [r for r in runs() if include_breast or r.study == "GBM"]
    for run in selected_runs:
        status, effective = validate_run(run)
        frame = read_csv(run.input)
        ledger = align_ledger(frame, read_csv(run.output / "spot_ledger.csv"))
        tables = {t: read_core_table(run,t,status) for t in TABLES}
        n_rois = frame.roi_id.nunique()
        for r in tables["roi_correlations"].itertuples(index=False):
            f = frame.loc[frame.roi_id.eq(r.roi_id)]
            if r.parent_correlation_status == "finite":
                observed = np.corrcoef(f.gene1, f.gene2)[0, 1]
                if not np.isclose(observed, r.corr_obs, rtol=1e-10, atol=1e-10):
                    raise ValueError(f"Cell Pearson does not reproduce: {run.sample}/{run.pair}/{r.roi_id}")
        checks.append(dict(**run.identity, check="ledger_config_and_cell_pearson", passed=True))
        for name, table in tables.items():
            if "g1_id" in table and (set(table.g1_id.dropna()) != {run.feature_a} or set(table.g2_id.dropna()) != {run.feature_b}):
                raise ValueError(f"Feature identity mismatch: {run.sample}/{run.pair}/{name}")
            tagged = table.rename(columns={"sample_id": "run_sample_id"}).assign(**run.identity)
            collected[name].append(tagged)
            sources.append(dict(**run.identity, table=name, rows=len(table),
                                source=reference(run.output / f"{name}.csv"),
                                sha256=sha(run.output / f"{name}.csv") if (run.output/f'{name}.csv').exists() else '',
                                file_status='present' if (run.output/f'{name}.csv').exists() else 'empty_core_result'))
            if table.empty:
                availability.append(dict(**run.identity,table=name,status_column='table',status='empty_core_result',n_rows=0))
            for col in table:
                if col == "status" or col.endswith("_status"):
                    for label, count in table[col].fillna("not_reported").value_counts().items():
                        availability.append(dict(**run.identity, table=name, status_column=col, status=label, n_rows=count))
        correlations.extend(correlation_rows(run, tables))
        core = status["core_git_sha"]
        manifest.append(dict(**run.identity, input_csv=reference(run.input), run_directory=reference(run.output),
                             input_sha256=sha(run.input), core_commit=core, n_cells=len(frame), n_rois=n_rois,
                             genes_a=run.genes_a, genes_b=run.genes_b, expression_units=run.expression_units))
        key = (run.study, run.sample)
        if key in seen:
            pd.testing.assert_frame_equal(ledger, seen[key][1], check_exact=True)
            pd.testing.assert_frame_equal(frame[["x", "y", "roi_id"]], seen[key][0], check_exact=True)
        else:
            seen[key] = (frame[["x", "y", "roi_id"]], ledger)
            per_roi = support_summary(frame, ledger).assign(study=run.study, sample=run.sample, patient=run.patient)
            support.append(per_roi)
            cell_map, all_cells, path = annotations(run, frame)
            csv(cell_map, out / "cells" / run.study / run.sample / "cell_identifiers.csv.gz")
            csv(ledger, out / "cells" / run.study / run.sample / "cell_aggregation_status.csv.gz")
            n_eligible = int(ledger.tessellation_eligible.eq(True).sum())
            samples.append(dict(study=run.study, sample=run.sample, patient=run.patient,
                                deposited_cells=len(all_cells), analysed_cells=len(frame),
                                cells_outside_selected_rois=len(all_cells)-len(frame), n_rois=n_rois,
                                n_pairs=sum(r.study == run.study and r.sample == run.sample for r in selected_runs),
                                tessellation_eligible_cells=n_eligible, eligible_fraction=n_eligible/len(frame),
                                core_commit=core, tessellations=10,
                                roi_definition="author neighbourhood workflow reconstructed and matched to published cell-type compositions" if run.study == "GBM" else "three published parent ROIs",
                                coordinate_units="micrometres"))
        print(f"Validated {run.study}/{run.sample}/{run.pair}: {len(frame):,} cells, {n_rois} ROIs", flush=True)
    for name, parts in collected.items(): csv(pd.concat(parts, ignore_index=True), out / "statistics" / f"{name}.csv.gz")
    result = {"run_inventory": pd.DataFrame(manifest), "sample_inventory": pd.DataFrame(samples),
              "roi_cell_counts": pd.concat(support, ignore_index=True), "correlations": pd.DataFrame(correlations),
              "statistic_availability": pd.DataFrame(availability), "validation": pd.DataFrame(checks),
              "statistic_files": pd.DataFrame(sources)}
    for name, table in result.items(): csv(table, out / f"{name}.csv")
    n_expected = len(SAMPLE_CELLS) + int(include_breast)
    expected = (n_expected, n_expected)
    if (len(manifest), len(samples)) != expected: raise ValueError("Incomplete Xenium scope")
    (out / "README.md").write_text(
        "# Xenium feasibility evidence\n\nGenerated from completed current runs only. "
        "All sixteen complete Meylan GBM samples from eight paired patients, and breast Sample 1 Replicate 1 when included. "
        "`run_inventory.csv` defines the exact analysed pairs and inputs. `statistics/` retains "
        "all numerical ROI, component, interface and aggregation statistics, including unavailable rows. "
        "Cell coordinates and original identifiers are in `cells/`; exact expression inputs remain in "
        "their study directories and are included in Supplementary Data 3. Geometry/tiles/hexagons "
        "remain in CurrentResults; no second core result tree is created.\n\n"
        "Eligibility fractions use every analysed cell in the denominator. Missing eligibility "
        "does not mean a cell may be dropped from that denominator. Counts of units in correlation "
        "tables refer to cells, tiles or hexagons, not patients.\n")
    cohort=data_root()/'meylan_xenium/analysis/cohort'
    for name in ['contrasts.csv','recurrence.csv','ordered_interfaces.csv','cohort_roi_summary.csv','sample_summary.csv','FINDINGS.md']:
        if not (cohort/name).exists():
            raise ValueError(f'Missing complete-cohort analysis: {name}')
        (out/'cohort').mkdir(exist_ok=True)
        shutil.copy2(cohort/name,out/'cohort'/name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gbm-only", action="store_true", help="Development check; omit breast")
    args = parser.parse_args()
    build(args.output, include_breast=not args.gbm_only)


if __name__ == "__main__": main()
