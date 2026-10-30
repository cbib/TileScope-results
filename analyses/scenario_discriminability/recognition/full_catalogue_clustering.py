"""Cluster all 30 benchmark constructions without imputing unavailable effects.

Run with ``python -m analyses.scenario_discriminability.recognition.full_catalogue_clustering``.
Only saved outputs are read. No core imports, classifier fitting or core reruns.
See README.md for the analysis specification.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, leaves_list, linkage, to_tree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from tilescope_results.paths import atlas_root, current_results_root, reference, resolve_reference

FAMILIES = ["marginal_enrichment", "joint_enrichment", "prevalence", "log2fc", "cles",
            "correlation", "moran", "distance", "boundary", "aggregation"]
TITLES = dict(zip(FAMILIES, ["Each-gene enrichment", "Joint enrichment", "Expression prevalence",
    "Mean-expression log2 fold change", "Common-language effect size", "Pearson correlation",
    "Bivariate Moran", "Cross-correlation over distance", "Boundary response", "Aggregation sensitivity"]))
TITLES["support_diagnostic"] = "Component support (diagnostic)"
ABUNDANCE = FAMILIES[:5]
SUPPORTS = {"source_native": "Spots", "tessellation": "Tessellations",
            "consensus_direct": "Direct consensus", "consensus_ensemble": "Ensemble consensus"}
MODES = {"broad_colocalization": "Broad overlap", "focal_pattern::colocated": "Colocated foci",
    "focal_pattern::separated": "Separated foci", "spatial_gradient::concordant": "Concordant gradient",
    "spatial_gradient::opposed": "Opposed gradient", "nested_foci": "Nested foci",
    "embedded_domains": "Embedded domains", "dispersed_infiltration": "Dispersed population",
    "contact_interface::responder_activation": "Responder activation",
    "contact_interface::anchor_fade": "Anchor extension"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_saved(directory):
    """Data must match. Historical Python code changes are disclosed, not hidden."""
    record = json.loads((directory / "provenance.json").read_text())
    checks = []
    for section in ("input_hashes", "output_hashes"):
        for name, expected in record[section].items():
            path = resolve_reference(name) if section == "input_hashes" else directory / name
            actual = sha(path) if path.is_file() else None
            code = section == "input_hashes" and path.suffix == ".py"
            if actual != expected and not code:
                raise ValueError(f"Saved numerical evidence changed or missing: {path}")
            checks.append(dict(section=section, path=reference(path), expected_sha256=expected,
                               actual_sha256=actual, status="match" if actual == expected else "historical_code_changed"))
    return pd.DataFrame(checks)


def recover_tessellation_enrichment(cases, values, *, runs_root):
    """Use the saved median bounded effect, including explicitly saturated ratios.

    Core uses E/(1+E)-0.5; the analysis stores E/(1+E). Apply to ALL entries,
    not just missing ones, so a finite-only median is never mixed with this one.
    """
    values = values.copy()
    changes = []
    for case_id, case in cases.iterrows():
        path = runs_root / case.scenario_name / case_id / case.positive_role / "roi_cross_grain_sensitivity.csv"
        table = pd.read_csv(path)
        for slot, roi in (("A_ROI", case.a_roi_id), ("B_ROI", case.b_roi_id)):
            rows = table.loc[table.roi_id.astype(str).eq(str(roi)) & table.statistic.eq("enrichment") &
                             table.target_surface.eq("tessellation_median")]
            for channel in ("g1", "g2", "joint_high", "joint_low"):
                row = rows.loc[rows.channel.eq(channel)]
                if len(row) != 1:
                    raise ValueError(f"Missing/duplicate core enrichment summary: {case_id}/{slot}/{channel}")
                row = row.iloc[0]
                value = float(row.target_effect_value) + .5 if pd.notna(row.target_effect_value) else np.nan
                if np.isfinite(value) and not -1e-12 <= value <= 1 + 1e-12:
                    raise ValueError("Invalid bounded enrichment effect")
                key = f"{slot}|tessellation|enrichment_median|{channel}|"
                old = values.loc[case_id, key]
                values.loc[case_id, key] = value
                changes.append(dict(case_id=case_id, feature=key, old_value=old, value=value,
                    changed=not np.isclose(old, value, rtol=0, atol=1e-12, equal_nan=True),
                    recovered_from_missing=pd.isna(old) and pd.notna(value),
                    n_comparable_replicates=row.n_comparable_replicates,
                    status=row.target_raw_status, source_file=reference(path)))
    return values, pd.DataFrame(changes)


def standardize(values, *, scaling="sd"):
    mean = values.mean()
    scale = values.std(ddof=0) if scaling == "sd" else values.max() - values.min()
    active = values.notna().sum().ge(2) & scale.gt(1e-12)
    audit = pd.DataFrame(dict(mean=mean, scale=scale, n_available=values.notna().sum(), active=active))
    audit.index.name = "feature"
    return (values.loc[:, active] - mean[active]) / scale[active], audit


def distances(z, meta, *, balancing="family_support", allow_missing=False):
    """Pairwise available-data distances; neither missingness nor labels are values."""
    meta = meta.set_index("feature").loc[z.columns]
    blocks = []
    for (family, support), ids in meta.groupby(["group", "support"], sort=True).groups.items():
        blocks.append((family, support, z.loc[:, ids].to_numpy(float)))
    n = len(z)
    matrix = np.zeros((n, n))
    coverage = []
    for i in range(n):
        for j in range(i + 1, n):
            family_blocks = {}
            all_sums, all_n, nblocks = 0., 0, 0
            used = []
            for family, support, x in blocks:
                mask = np.isfinite(x[i]) & np.isfinite(x[j])
                count = int(mask.sum())
                if not count:
                    continue
                ss = float(np.square(x[i, mask] - x[j, mask]).sum())
                family_blocks.setdefault(family, []).append((ss, count))
                all_sums += ss
                all_n += count
                nblocks += 1
                used.append(f"{family}/{support}:{count}")
            if not all_n:
                d = np.nan
                if not allow_missing:
                    raise ValueError(f"No common statistics: {z.index[i]} / {z.index[j]}")
            elif balancing == "column":
                d = np.sqrt(all_sums / all_n)
            else:
                effects = []
                for pieces in family_blocks.values():
                    effects.append(np.mean([s / c for s, c in pieces]) if balancing == "family_support"
                                   else sum(s for s, _ in pieces) / sum(c for _, c in pieces))
                d = np.sqrt(np.mean(effects))
            matrix[i, j] = matrix[j, i] = d
            coverage.append(dict(case_i=z.index[i], case_j=z.index[j], distance=d,
                                 n_common_values=all_n, n_common_families=len(family_blocks),
                                 n_common_blocks=nblocks, blocks=";".join(used)))
    return matrix, pd.DataFrame(coverage)


def cluster(values, meta, *, method="average", balancing="family_support", scaling="sd"):
    z, scale = standardize(values, scaling=scaling)
    if z.empty:
        raise ValueError("No varying statistics")
    d, coverage = distances(z, meta, balancing=balancing)
    if method == "ward":
        if z.isna().any().any():
            raise ValueError("Ward requires a complete, fixed Euclidean representation")
        m = meta.set_index("feature").loc[z.columns]
        weights = pd.Series(1., index=z.columns)
        for _, fam in m.groupby("group"):
            for _, block in fam.groupby("support"):
                weights.loc[block.index] = 1 / np.sqrt(m.group.nunique() * fam.support.nunique() * len(block))
        euclidean = pdist(z.mul(weights, axis=1))
        np.testing.assert_allclose(euclidean, squareform(d), atol=1e-12)
        tree = linkage(euclidean, method="ward", optimal_ordering=True)
    else:
        tree = linkage(squareform(d), method=method, optimal_ordering=True)
    return dict(tree=tree, distances=d, coverage=coverage, z=z, scale=scale,
                order=values.index[leaves_list(tree)].tolist(), method=method)


def branch_audit(tree, cases):
    """No chosen cut: count outsiders in each mode's smallest containing branch."""
    _, nodes = to_tree(tree, rd=True)
    leaves = {node.id: set(node.pre_order(lambda leaf: leaf.id)) for node in nodes}
    rows = []
    for mode, group in cases.groupby("class_label"):
        ids = set(np.flatnonzero(cases.class_label.eq(mode)))
        node = min((n for n in nodes if leaves[n.id] >= ids), key=lambda n: len(leaves[n.id]))
        members = cases.iloc[sorted(leaves[node.id])]
        rows.append(dict(mode=mode, n_cases=len(group), n_branch_cases=len(members),
                         n_other_cases=len(members)-len(group), pure=len(members) == len(group),
                         height=node.dist, branch_case_ids=";".join(members.index),
                         other_modes=";".join(sorted(set(members.class_label)-{mode}))))
    return pd.DataFrame(rows)


def redundancy(values, meta):
    ids = values.columns.tolist()
    rho = values.corr(method="spearman", min_periods=6)
    n = values.notna().astype(int).T @ values.notna().astype(int)
    groups = meta.set_index("feature").group
    return pd.DataFrame([dict(feature_A=a, feature_B=b, family_A=groups[a], family_B=groups[b],
        n_common_cases=int(n.loc[a, b]), spearman=rho.loc[a, b])
        for i, a in enumerate(ids) for b in ids[i+1:]])


def prepare(input_dir=None, *, runs_root=None, verify=True):
    directory = Path(input_dir or atlas_root() / "recognition/all_statistics")
    verification = verify_saved(directory) if verify else pd.DataFrame()
    cases = pd.read_csv(directory / "cases.csv").set_index("case_id").sort_index()
    if len(cases) != 30 or set(cases.class_label) != set(MODES) or not cases.groupby("class_label").size().eq(3).all():
        raise ValueError("Expected ALL 30 constructions: three for every one of ten modes")
    catalogue = pd.read_csv(directory / "feature_catalogue.csv")
    ledger = pd.read_csv(directory / "measurements.csv")
    raw = pd.read_csv(directory / "features.csv").set_index("case_id").loc[cases.index]
    rebuilt = ledger.pivot(index="case_id", columns="feature", values="value").loc[cases.index, raw.columns]
    np.testing.assert_allclose(raw, rebuilt, rtol=0, atol=1e-12, equal_nan=True)
    meta = catalogue.loc[catalogue.use.eq("effect")].copy()
    if len(meta) != 160 or set(meta.group) != set(FAMILIES):
        raise ValueError("Unexpected effect inventory; review it explicitly before changing the contract")
    meta["support"] = meta.surface.replace({"largest_component": "source_native", "tessellation_median": "tessellation"})
    numeric, corrections = recover_tessellation_enrichment(cases, raw.loc[:, meta.feature],
        runs_root=Path(runs_root or current_results_root()))
    variants, skips = {}, []
    def add(name, vals=numeric, description="", **kwargs):
        try:
            r = cluster(vals, meta if not name.startswith("with_component") else diagmeta, **kwargs)
            r["description"] = description
            variants[name] = r
        except ValueError as error:
            if name == "primary":
                raise
            skips.append(dict(variant=name, reason=str(error), n_cases=len(vals)))
    add("primary", description="All effects; average linkage; available paired values; family/representation balance")
    add("complete_linkage", method="complete")
    add("equal_values", balancing="column")
    add("equal_families", balancing="family")
    add("range_scaling", scaling="range")
    add("original_enrichment", raw.loc[:, meta.feature])
    for family in FAMILIES:
        add("without_"+family, numeric.loc[:, meta.loc[meta.group.ne(family), "feature"]])
    add("without_abundance", numeric.loc[:, meta.loc[~meta.group.isin(ABUNDANCE), "feature"]])
    add("target_ROI_only", numeric.loc[:, meta.loc[meta.slot.eq("A_ROI"), "feature"]])
    for support in SUPPORTS:
        add("representation_"+support, numeric.loc[:, meta.loc[meta.support.eq(support), "feature"]])
        portable = meta.support.eq(support) & meta.group.isin(["correlation", "moran"]) & meta.slot.isin(["A_ROI", "B_ROI"]) & meta.surface.ne("largest_component")
        add("matched_Pearson_Moran_"+support, numeric.loc[:, meta.loc[portable, "feature"]])
    complete = numeric.loc[:, numeric.notna().all()]
    add("complete_values_average", complete)
    add("complete_values_ward", complete, method="ward")
    # Support counts are separate geometry diagnostics, not expression or proof of enclosure.
    support_meta = catalogue.loc[catalogue.use.eq("support_diagnostic")].copy()
    support_meta["support"] = "source_native"
    diagmeta = pd.concat([meta, support_meta], ignore_index=True)
    support_values = raw.loc[:, support_meta.feature].copy()
    for key in support_values:
        if "fraction" not in key:
            support_values[key] = np.log1p(support_values[key])
    add("with_component_support_DIAGNOSTIC", pd.concat([numeric, support_values], axis=1))
    availability = numeric.notna().astype(float)
    avd = squareform(pdist(availability, metric="hamming"))
    variants["availability_only_DIAGNOSTIC"] = dict(tree=linkage(squareform(avd), method="average", optimal_ordering=True),
        distances=avd, order=cases.index[leaves_list(linkage(squareform(avd), method="average", optimal_ordering=True))].tolist(),
        method="average", description="Missingness only: NOT expression evidence", z=availability)
    audits, comparisons = [], []
    primary_coph = cophenet(variants["primary"]["tree"])
    for name, result in variants.items():
        audit = branch_audit(result["tree"], cases)
        audit["variant"] = name
        audits.append(audit)
        comparisons.append(dict(variant=name, n_cases=len(cases), n_values=result["z"].shape[1],
            n_pure_modes=int(audit.pure.sum()), method=result["method"],
            cophenetic_spearman_to_primary=float(spearmanr(primary_coph, cophenet(result["tree"])).statistic)))
    # Descriptive family contrasts retain missing family comparisons instead of inventing zeros.
    family_contrasts = []
    for family in FAMILIES:
        z, _ = standardize(numeric.loc[:, meta.loc[meta.group.eq(family), "feature"]])
        d, _ = distances(z, meta, allow_missing=True)
        for mode in MODES:
            ix = np.flatnonzero(cases.class_label.eq(mode))
            within = d[np.ix_(ix, ix)][np.triu_indices(3, 1)]
            between = []
            for other in MODES:
                if other == mode:
                    continue
                jx = np.flatnonzero(cases.class_label.eq(other))
                vals = d[np.ix_(ix, jx)]
                if np.isfinite(vals).all():
                    between.append((float(vals.mean()), other))
            valid = np.isfinite(within).all() and bool(between)
            nearest = min(between) if between else (np.nan, "")
            family_contrasts.append(dict(family=family, mode=mode,
                within_mode_mean_distance=float(within.mean()) if np.isfinite(within).all() else np.nan,
                nearest_other_mode=nearest[1], nearest_other_mean_distance=nearest[0],
                n_comparable_other_modes=len(between), all_nine_other_modes_comparable=len(between) == 9,
                descriptive_separation_margin=nearest[0]-float(within.mean()) if valid else np.nan))
    meta["n_available"] = meta.feature.map(numeric.notna().sum())
    meta["primary_active"] = meta.feature.isin(variants["primary"]["z"].columns)
    return dict(cases=cases, catalogue=catalogue, meta=meta, numeric=numeric, variants=variants,
        corrections=corrections, verification=verification, ledger=ledger,
        audits=pd.concat(audits, ignore_index=True), comparisons=pd.DataFrame(comparisons),
        skips=pd.DataFrame(skips, columns=["variant", "reason", "n_cases"]),
        family_contrasts=pd.DataFrame(family_contrasts), redundancy=redundancy(numeric, meta),
        export_inventory=pd.read_csv(directory / "export_column_inventory.csv"), input_dir=directory)


def plot_tree(data, name="primary"):
    result = data["variants"][name]
    labels = [f"{MODES[r.class_label]} | {r.source_sample_id} | ROI {r.a_roi_id}" +
              (f" / {r.b_roi_id}" if r.a_roi_id != r.b_roi_id else "") for r in data["cases"].itertuples()]
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(left=.06, right=.56, bottom=.07, top=.94)
    dendrogram(result["tree"], labels=labels, orientation="left", leaf_font_size=8,
               color_threshold=0, above_threshold_color="#506473", ax=ax)
    ax.invert_yaxis()  # Same top-to-bottom order as every profile page.
    ax.set_xlabel("Statistical dissimilarity" if name == "primary" else name.replace("_", " "))
    ax.set_title("All 30 datasets · ten scenario modes", loc="left", fontsize=12)
    return fig


def short_label(row):
    stat = row.statistic.replace("enrichment_median", "enrichment").replace("pearson_median", "Pearson")
    channel = "" if pd.isna(row.channel) else str(row.channel)
    extra = row.feature.split("|")[4]
    text = f"{row.surface}: {stat} {channel or extra}"
    for a, b in (("source_native", "S"), ("largest_component", "S component"),
                 ("consensus_direct", "CD"), ("consensus_ensemble", "CE"),
                 ("tessellation_median", "T"), ("tessellation", "T"),
                 ("g1_lag_g2", "A→B"), ("g2_lag_g1", "B→A"), ("g1", "A"), ("g2", "B")):
        text = text.replace(a, b)
    if row.slot in ("A_to_B", "B_to_A"):
        text = row.slot.replace("_to_", "→") + " ROI · " + text
    elif row.group == "aggregation":
        text = row.slot.replace("_ROI", " ROI") + " · " + text
    return text


def plot_profiles(data, selector, title):
    order = data["variants"]["primary"]["order"]
    meta = data["meta"].loc[selector].copy()
    meta["family_order"] = meta.group.map({g: i for i, g in enumerate(FAMILIES)})
    meta = meta.sort_values(["family_order", "slot", "support", "feature"])
    z, _ = standardize(data["numeric"].loc[:, meta.feature])
    # Constants remain visible as zero standardized variation, explicitly marked.
    displayed = z.reindex(columns=meta.feature)
    for key in meta.loc[~meta.feature.isin(z.columns), "feature"]:
        displayed[key] = data["numeric"][key].where(data["numeric"][key].isna(), 0.)
    fig, ax = plt.subplots(figsize=(max(14, len(meta)*.24+5), 12))
    fig.subplots_adjust(left=.255, right=.935, bottom=.32, top=.92)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#d7d7d7")
    limit = max(3., float(np.ceil(np.nanmax(np.abs(displayed.to_numpy())))))
    im = ax.imshow(displayed.loc[order], cmap=cmap, vmin=-limit, vmax=limit, aspect="auto", interpolation="none")
    ax.set_xticks(range(len(meta)), [short_label(r) + (" *" if not r.primary_active else "") for r in meta.itertuples()],
                  rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)), [f"{MODES[data['cases'].loc[c, 'class_label']]} | " +
        f"{data['cases'].loc[c, 'source_sample_id']} A{data['cases'].loc[c, 'a_roi_id']}/B{data['cases'].loc[c, 'b_roi_id']}" for c in order], fontsize=8)
    ax.tick_params(length=0)
    start = 0
    for family, group in meta.groupby("group", sort=False):
        end = start+len(group)
        ax.axvline(start-.5, color="white", lw=2)
        ax.text((start+end-1)/2, -1.1, TITLES[family].replace(" ", "\n"), ha="center", va="bottom", fontsize=7)
        start = end
    fig.suptitle(title, fontsize=12, y=.995)
    cb = fig.colorbar(im, ax=ax, fraction=.017, pad=.01); cb.set_label("Standardized value")
    fig.text(.02, .015, "Same all-30 hierarchy on every page. Grey: unavailable, never zero-imputed. "
             "S: spots; T: tessellations; CD: direct consensus; CE: ensemble consensus. *: constant, not used for distances.", fontsize=8)
    return fig


def export(data, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})
    tables = {"cases": data["cases"], "values": data["numeric"], "statistic_catalogue": data["meta"],
        "full_extraction_catalogue": data["catalogue"], "branch_audit": data["audits"],
        "sensitivity_summary": data["comparisons"], "unavailable_sensitivities": data["skips"],
        "enrichment_extraction_changes": data["corrections"], "input_verification": data["verification"],
        "pairwise_coverage": data["variants"]["primary"]["coverage"], "family_contrasts": data["family_contrasts"],
        "statistic_redundancy": data["redundancy"], "all_export_dispositions": data["export_inventory"],
        "scaling": data["variants"]["primary"]["scale"]}
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=name in ("cases", "values", "scaling"))
    hierarchies = {name: dict(method=r["method"], case_order=list(data["cases"].index), leaf_order=r["order"], linkage=r["tree"].tolist())
                   for name, r in data["variants"].items()}
    (output / "hierarchies.json").write_text(json.dumps(hierarchies, indent=2)+"\n")
    for name, result in data["variants"].items():
        folder = output / "variants"; folder.mkdir(exist_ok=True)
        pd.DataFrame(result["distances"], index=data["cases"].index, columns=data["cases"].index).to_csv(folder / f"{name}_distances.csv")
    for name, stem in (("primary", "hierarchy"), ("availability_only_DIAGNOSTIC", "availability_hierarchy")):
        fig = plot_tree(data, name)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(output / f"{stem}.{ext}", dpi=160)
        plt.close(fig)
    meta = data["meta"]
    blocks = [(meta.slot.eq("A_ROI") & meta.group.ne("aggregation"), "ROI A: expression statistics"),
              (meta.slot.eq("B_ROI") & meta.group.ne("aggregation"), "ROI B: expression statistics"),
              (meta.group.eq("boundary"), "Boundary responses: both genes and both directions"),
              (meta.group.eq("aggregation"), "Variation across tessellations and differences after aggregation")]
    with PdfPages(output / "profiles.pdf") as pdf:
        for index, (selector, title) in enumerate(blocks, 1):
            fig = plot_profiles(data, selector, title)
            pdf.savefig(fig)
            fig.savefig(output / f"profiles_{index}.png", dpi=160)
            plt.close(fig)
    gallery = "".join(f'<h2>{html.escape(title)}</h2><a href="profiles_{i}.png"><img src="profiles_{i}.png"></a>'
                      for i, (_, title) in enumerate(blocks, 1))
    (output / "overview.html").write_text('<!doctype html><meta charset="utf-8"><title>All 30 benchmark datasets</title>'
        '<style>body{font:15px system-ui;margin:2em;max-width:1600px}img{max-width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>Full catalogue: 30 datasets, ten modes</h1>'
        '<p><a href="profiles.pdf">Statistical profiles</a> · <a href="branch_audit.csv">Branch audit</a> · '
        '<a href="sensitivity_summary.csv">Sensitivity summary</a> · <a href="pairwise_coverage.csv">Pairwise coverage</a></p>'
        '<img src="hierarchy.png">' + gallery + '<h2>Availability-only diagnostic</h2><img src="availability_hierarchy.png">'
        '<h2>Sensitivity summary</h2>' + data["comparisons"].to_html(index=False, float_format=lambda v: f"{v:.3f}"))
    sources = {reference(Path(__file__)): sha(__file__), reference(Path(__file__).with_name("README.md")): sha(Path(__file__).with_name("README.md"))}
    record = dict(n_datasets=30, n_modes=10, n_effect_quantities=160, primary_linkage="average",
        tilescope_rerun=False, main_figure_modified=False, live_manuscript_modified=False,
        input_dir=reference(data["input_dir"]), source_sha256=sources,
        outputs={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "provenance.json"})
    (output / "provenance.json").write_text(json.dumps(record, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=atlas_root()/"recognition/all_statistics")
    parser.add_argument("--runs-dir", type=Path, default=current_results_root())
    parser.add_argument("--output-dir", type=Path, default=atlas_root()/"recognition/full_catalogue")
    args = parser.parse_args()
    data = prepare(args.input_dir, runs_root=args.runs_dir)
    export(data, args.output_dir)
    print(data["comparisons"].to_string(index=False))
    print(f"Wrote all 30 datasets to {args.output_dir}")


if __name__ == "__main__":
    main()
