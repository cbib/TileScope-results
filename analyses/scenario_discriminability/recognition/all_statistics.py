"""Extract saved benchmark measurements; no classifiers or TileScope reruns.

The full-catalogue clustering is a separate consumer of these tables.
Scenario labels identify constructions, not predictors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from tilescope_results.paths import atlas_root, corpus_root, current_results_root, reference
from scenario_benchmark.registry import positive_role_for_scenario
from analyses.scenario_discriminability import extractors

PLAN_PATH = corpus_root() / "benchmark_plan.json"
OUT = atlas_root() / "recognition/all_statistics"
GROUPS = ["marginal_enrichment", "joint_enrichment", "prevalence", "log2fc", "cles",
          "correlation", "moran", "distance", "boundary", "aggregation"]
TITLES = {"marginal_enrichment": "Marginal enrichment", "joint_enrichment": "Joint enrichment",
          "prevalence": "High/low prevalence", "log2fc": "Log2 fold change", "cles": "CLES",
          "correlation": "Pearson", "moran": "Bivariate Moran", "distance": "Distance profile",
          "boundary": "Boundary response", "aggregation": "Aggregation sensitivity",
          "support_diagnostic": "Component support (QC)", "all": "All effect families"}
MODE_NAMES = {"broad_colocalization": "Broad overlap", "focal_pattern::colocated": "Colocated foci",
              "focal_pattern::separated": "Separated foci", "spatial_gradient::concordant": "Concordant gradient",
              "spatial_gradient::opposed": "Opposed gradient", "nested_foci": "Nested foci",
              "embedded_domains": "Embedded domains", "dispersed_infiltration": "Dispersed population",
              "contact_interface::responder_activation": "Responder activation",
              "contact_interface::anchor_fade": "Anchor extension"}
ROWS = []
FILES = set()
COLUMN_USE = {}



def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _class_label(scenario_name: str, scenario_mode: Optional[str]) -> str:
    if scenario_mode:
        return f"{scenario_name}::{scenario_mode}"
    return scenario_name


def _case_ab_rois(case: Dict) -> Tuple[str, str, bool]:
    scenario = case["scenario_name"]
    args = case["args"]
    if scenario in {
        "broad_colocalization", "focal_pattern",
        "spatial_gradient",
    }:
        anchor = str(args["anchor_roi_id"])
        return anchor, anchor, True
    a = str(args["a_roi_id"])
    b = str(args["b_roi_id"])
    return a, b, (a == b)


def _diagnostic_scenario_mode(case: Dict) -> Optional[str]:
    """Mode read from the plan for the diagnostic scenario_mode column
    (retains realization route on every stratified scenario). Do not
    parse case_id suffixes."""
    if case["scenario_name"] == "spatial_gradient":
        rel = case.get("gradient_relationship")
        if not rel:
            raise ValueError(
                f"spatial_gradient case {case.get('case_id')!r} "
                f"missing top-level gradient_relationship"
            )
        return str(rel)
    if case["scenario_name"] == "contact_interface":
        mode = case.get("mode")
        if not mode:
            raise ValueError(
                f"contact_interface case {case.get('case_id')!r} "
                f"missing top-level mode"
            )
        return str(mode)
    if case["scenario_name"] == "focal_pattern":
        mode = case.get("mode")
        if not mode:
            raise ValueError(
                f"focal_pattern case {case.get('case_id')!r} "
                f"missing top-level mode"
            )
        return str(mode)
    return None


def _class_scenario_mode(case: Dict) -> Optional[str]:
    """Mode used to construct the recognition class_label. Every
    scenario_name is its own class; modes further split classes only
    on the discrimination axes (gradient / contact / focal_pattern)."""
    return _diagnostic_scenario_mode(case)


def _load_cases() -> pd.DataFrame:
    plan = json.loads(PLAN_PATH.read_text())
    rows = []
    for case in plan["cases"]:
        a_roi, b_roi, same_parent = _case_ab_rois(case)
        rows.append({
            "case_id": case["case_id"],
            "scenario_name": case["scenario_name"],
            "scenario_mode": _diagnostic_scenario_mode(case) or "",
            "class_label": _class_label(
                case["scenario_name"], _class_scenario_mode(case),
            ),
            "source_sample_id": case["source_sample_id"],
            "a_roi_id": a_roi,
            "b_roi_id": b_roi,
            "same_parent_roi": same_parent,
            "b_expression_profile": str(
                case.get("args", {}).get("b_expression_profile", "")
            ),
            "positive_role": positive_role_for_scenario(case["scenario_name"]),
        })
    return pd.DataFrame(rows)


def select_largest_component(geometry, roi_id):
    """Size-only selection; identical rule for every case, before reading effects."""
    selected = geometry.loc[geometry.roi_id.astype(str).eq(str(roi_id))].copy()
    selected["_tie"] = selected.component_id.astype(str)
    return selected.sort_values(
        ["n_component_spots", "_tie"], ascending=[False, True]
    ).iloc[0]


def validate_runs():
    accounting_path = atlas_root() / "run_accounting.csv"
    FILES.add(accounting_path)
    accounting = pd.read_csv(accounting_path)
    config = current_results_root() / "tilescope_run_config.json"
    FILES.add(config)
    config_hash = digest(config)
    for _, run in accounting.iterrows():
        status_path = (atlas_root() /
                       run.scenario_name / run.case_id / run.role / "run_status.json")
        input_path = corpus_root() / run.scenario_name / run.case_id / f"{run.role}.csv"
        FILES.update([status_path, input_path])
        status = json.loads(status_path.read_text())
        assert status["execution_status"] == run.execution_status == "complete"
        assert status["input_hash"] == run.input_hash == digest(input_path)
        assert status["config_hash"] == run.config_hash == config_hash
        assert status["core_sha"] == run.core_sha
    assert accounting.core_sha.nunique() == 1
    return {"n_verified_runs": len(accounting), "saved_core_sha": accounting.core_sha.iloc[0],
            "config_sha256": config_hash}


def clean(x):
    return "" if pd.isna(x) else str(x)


def read(path):
    FILES.add(path)
    frame = pd.read_csv(path)
    for col in frame:
        COLUMN_USE.setdefault((path.name, col), set())
    return frame


def register(case, slot, surface, statistic, channel, group, value, status, filename,
             column, *, use="effect", grain="roi", component="", replicate="", extra=""):
    key = "|".join(map(str, [slot, surface, statistic, channel, extra]))
    if replicate != "":
        key += "|replicate=" + str(replicate)
    raw = float(value) if value is not None and pd.notna(value) else np.nan
    transformed = raw if status == "finite" and np.isfinite(raw) else np.nan
    transform = "identity"
    if statistic.startswith("enrichment"):
        transform = "ratio/(1+ratio)"
        if status == "positive_over_zero_context":
            transformed = 1.
        elif np.isfinite(transformed) and transformed >= 0:
            transformed = transformed / (1. + transformed)
    if use != "effect":
        COLUMN_USE[(filename, column)].add(use)
    else:
        COLUMN_USE[(filename, column)].add(group)
    a, b = str(case.a_roi_id), str(case.b_roi_id)
    roi_id = a if slot == "A_ROI" else b if slot == "B_ROI" else ""
    ROWS.append({"case_id": case.case_id, "class_label": case.class_label,
                 "source_sample_id": case.source_sample_id, "feature": key, "slot": slot,
                 "surface": surface, "statistic": statistic, "channel": channel, "group": group,
                 "use": use, "raw_value": raw, "value": transformed, "status": status,
                 "transform": transform, "source_file": filename, "source_column": column,
                 "grain": grain, "roi_id": roi_id, "component_id": component,
                 "anchor_roi_id": a if slot == "A_to_B" else b if slot == "B_to_A" else "",
                 "responder_roi_id": b if slot == "A_to_B" else a if slot == "B_to_A" else "",
                 "replicate_index": replicate})


def extraction():
    ROWS.clear()
    FILES.clear()
    COLUMN_USE.clear()
    FILES.add(PLAN_PATH)
    cases = _load_cases()
    for _, case in cases.iterrows():
        directory = current_results_root() / case.scenario_name / case.case_id / case.positive_role
        cache = {}
        for spec in extractors.REGISTRY:
            frame = cache.setdefault(spec.filename, read(directory / spec.filename))
            for slot, roi in [("A_ROI", case.a_roi_id), ("B_ROI", case.b_roi_id)]:
                for _, r in frame.loc[frame.roi_id.astype(str).eq(str(roi))].iterrows():
                    surface = clean(r.get("value_surface")) or spec.surface
                    if surface == "source":
                        surface = "source_native"
                    for value_spec in spec.values:
                        stat, channel = value_spec.statistic, value_spec.channel or ""
                        direction = clean(r.get("moran_direction"))
                        status, reason = extractors._resolve_availability(r, value_spec)
                        status = "finite" if status == "finite" else reason
                        base = stat.removesuffix("_median").removesuffix("_q25").removesuffix("_q75")
                        group = ("marginal_enrichment" if channel in ("g1", "g2") else "joint_enrichment") if base == "enrichment" else {"pearson": "correlation"}.get(base, base)
                        use = "replicate_measurement" if spec.grain == "roi_replicate" else "quantile_endpoint" if stat.endswith(("_q25", "_q75")) else "effect"
                        register(case, slot, surface, stat, channel, group, r[value_spec.value_column], status,
                                 spec.filename, value_spec.value_column, use=use, extra=direction,
                                 replicate=clean(r.get("replicate_index")), grain=spec.grain)
                    # Inference columns remain available but do not classify scenarios.
                    for col in frame.columns:
                        if "_p_" in col or "_q_" in col or col in ("corr_p_perm", "corr_q_perm"):
                            register(case, slot, surface, col, "", "inference_diagnostic", r[col],
                                     "finite" if pd.notna(r[col]) else "unavailable", spec.filename, col,
                                     use="inference_diagnostic", replicate=clean(r.get("replicate_index")))
        for slot, roi in [("A_ROI", case.a_roi_id), ("B_ROI", case.b_roi_id)]:
            e = cache["roi_enrichment.csv"].loc[lambda x: x.roi_id.astype(str).eq(str(roi))].iloc[0]
            for col in ("n_g1_high", "n_g2_high", "n_both_high", "n_g1_low", "n_g2_low", "n_both_low"):
                register(case, slot, "source_native", "prevalence", col.removeprefix("n_"), "prevalence",
                         e[col] / e.n_spots, "finite", "roi_enrichment.csv", col)
            # CLES/log2FC ensemble dispersion: arithmetic differences of exported quartiles.
            e = cache["roi_tessellation_enrichment.csv"].loc[lambda x: x.roi_id.astype(str).eq(str(roi))].iloc[0]
            for base in ("CLES_g1", "CLES_g2", "log2fc_g1", "log2fc_g2"):
                value = e[base + "_q75"] - e[base + "_q25"]
                register(case, slot, "tessellation", "iqr_" + base, "", "aggregation", value,
                         "finite" if np.isfinite(value) else "unavailable", "roi_tessellation_enrichment.csv", base + "_q75")

        sensitivity = read(directory / "roi_cross_grain_sensitivity.csv")
        for slot, roi in [("A_ROI", case.a_roi_id), ("B_ROI", case.b_roi_id)]:
            for _, r in sensitivity.loc[sensitivity.roi_id.astype(str).eq(str(roi))].iterrows():
                channel = clean(r.channel) or clean(r.moran_direction)
                register(case, slot, r.target_surface, "delta_" + r.statistic, channel, "aggregation", r.delta_effect,
                         "finite" if r.comparison_status == "both_comparable" else r.comparison_status,
                         "roi_cross_grain_sensitivity.csv", "delta_effect")
                if r.target_surface == "tessellation_median":
                    register(case, slot, r.target_surface, "iqr_" + r.statistic, channel, "aggregation", r.tessellation_effect_iqr,
                             "finite" if np.isfinite(r.tessellation_effect_iqr) else "unavailable",
                             "roi_cross_grain_sensitivity.csv", "tessellation_effect_iqr")
                for col in ("abs_delta_effect", "sign_preserved"):
                    register(case, slot, r.target_surface, col + "_" + r.statistic, channel, "aggregation", r[col],
                             "finite" if pd.notna(r[col]) else "unavailable", "roi_cross_grain_sensitivity.csv", col,
                             use="deterministic_companion")

        radial = read(directory / "component_radial_shell_correlogram.csv")
        geometry = read(directory / "component_geometry.csv")
        support = read(directory / "roi_support_status.csv")
        for slot, roi in [("A_ROI", case.a_roi_id), ("B_ROI", case.b_roi_id)]:
            gg = geometry.loc[geometry.roi_id.astype(str).eq(str(roi))]
            chosen = select_largest_component(geometry, roi).component_id if len(gg) else None
            rr = radial.loc[radial.roi_id.astype(str).eq(str(roi))]
            for i in range(4):
                matches = rr.loc[rr.component_id.eq(chosen) & rr.shell_bin_index.eq(i)]
                r = matches.iloc[0] if len(matches) else None
                register(case, slot, "largest_component", "shell_" + str(i), "", "distance",
                         r.shell_symmetric if r is not None else np.nan,
                         r.component_radial_shell_status if r is not None else "no_retained_component",
                         "component_radial_shell_correlogram.csv", "shell_symmetric", grain="component", component=chosen)
            matches = rr.loc[rr.component_id.eq(chosen)]
            register(case, slot, "largest_component", "rho_zero", "", "correlation",
                     matches.component_rho_0.iloc[0] if len(matches) else np.nan,
                     "finite" if len(matches) and pd.notna(matches.component_rho_0.iloc[0]) else "unavailable",
                     "component_radial_shell_correlogram.csv", "component_rho_0", grain="component", component=chosen)
            r = support.loc[support.roi_id.astype(str).eq(str(roi))].iloc[0]
            for col in support.select_dtypes(include="number").columns:
                if col != "roi_id":
                    register(case, slot, "source_native", col, "", "support_diagnostic", r[col], "finite",
                             "roi_support_status.csv", col, use="support_diagnostic")

        interfaces = read(directory / "roi_pair_interface_response.csv")
        for slot, a, b in [("A_to_B", case.a_roi_id, case.b_roi_id), ("B_to_A", case.b_roi_id, case.a_roi_id)]:
            part = interfaces.loc[interfaces.anchor_roi_id.astype(str).eq(str(a)) & interfaces.responder_roi_id.astype(str).eq(str(b))]
            for surface in ("source_native", "tessellation", "consensus_direct", "consensus_ensemble"):
                for gene in ("A", "B"):
                    match = part.loc[part.surface.eq(surface) & part.responder_feature_id.eq(gene)]
                    r = match.iloc[0] if len(match) else None
                    register(case, slot, surface, "interface_response", gene, "boundary",
                             r.interface_response if r is not None else np.nan,
                             r.availability_status if r is not None else "no_specified_interface",
                             "roi_pair_interface_response.csv", "interface_response", grain="ordered_interface")
        # Every export column receives a disposition, including maps and geometry.
        for path in directory.glob("*.csv"):
            if path.name not in {p.name for p in FILES if p.parent == directory}:
                for col in pd.read_csv(path, nrows=0):
                    COLUMN_USE.setdefault((path.name, col), set())
    long = pd.DataFrame(ROWS)
    assert not long.duplicated(["case_id", "feature"]).any()
    catalog = long.drop_duplicates("feature").drop(columns=["case_id", "class_label", "source_sample_id", "value", "raw_value", "status", "roi_id", "component_id", "anchor_roi_id", "responder_roi_id"])
    wide = long.pivot(index="case_id", columns="feature", values="value").reindex(cases.case_id)
    catalog["n_available"] = catalog.feature.map(wide.notna().sum())
    wide.index.name = "case_id"
    return cases.set_index("case_id"), wide, catalog, long


def column_inventory():
    records = []
    maps = {"consensus_hexes.csv", "tessellation_tiles.csv"}
    for (filename, column), uses in sorted(COLUMN_USE.items()):
        if uses:
            disposition = ";".join(sorted(uses))
        elif filename in maps:
            disposition = "map_or_underlying_tile_hex_measurement_not_an_independent_case_statistic"
        elif "geometry" in filename or "domain_regions" in filename:
            disposition = "geometry_or_support_provenance_not_an_expression_effect"
        elif column.endswith("status") or "reason" in column:
            disposition = "availability_status"
        elif column.endswith("id") or column in ("replicate_index", "moran_direction", "statistic", "channel", "value_surface", "surface"):
            disposition = "identity_or_measurement_definition"
        else:
            disposition = "support_count_inference_provenance_or_redundant_companion"
        records.append({"source_file": filename, "column": column, "disposition": disposition})
    return pd.DataFrame(records)



def main(output=OUT):
    output = Path(output)
    cases, wide, catalog, long = extraction()
    provenance = validate_runs()
    output.mkdir(parents=True, exist_ok=True)
    tables = {"measurements.csv": long, "features.csv": wide.reset_index(),
              "feature_catalogue.csv": catalog, "cases.csv": cases.reset_index(),
              "export_column_inventory.csv": column_inventory()}
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    FILES.update([Path(__file__).resolve(), Path(extractors.__file__).resolve()])
    provenance.update({"n_positive_cases": len(cases),
                       "n_effect_features": int(catalog.use.eq("effect").sum()),
                       "input_hashes": {reference(p): digest(p) for p in sorted(FILES)},
                       "output_hashes": {name: digest(output / name) for name in tables}})
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Verified {provenance['n_verified_runs']} runs; extracted {len(cases)} cases and {len(catalog)} columns.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    main(parser.parse_args().output)
