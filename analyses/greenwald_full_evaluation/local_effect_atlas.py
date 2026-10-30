"""Sample-resolved source-ROI artwork; no pooled effects or scenario calls.

Both main and supplementary pages use the same data table and renderer. All
eligibility decisions come from core. Support bins affect marker size only.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PAGE_SIZE = 8
SUPPORT_LABELS = ("<100 spots", "100–399 spots", "≥400 spots")
SUPPORT_AREAS = (11.0, 23.0, 39.0)


def identity_key(value):
    value = str(value)
    return (0, int(value)) if value.isdigit() else (1, value)


def prepare_source_rows(correlations, support, *, sample, pair, features):
    """Keep one row per source ROI, including typed unavailability."""
    for frame in (correlations, support):
        if frame.roi_id.isna().any() or frame.roi_id.duplicated().any():
            raise ValueError(f"Missing/duplicate ROI identity: {sample}/{pair}")
    if set(correlations.roi_id) != set(support.roi_id):
        raise ValueError(f"Correlation/support ROI universes differ: {sample}/{pair}")
    for column, expected in zip(("g1_id", "g2_id"), features):
        if not correlations[column].eq(expected).all():
            raise ValueError(f"Feature identity mismatch: {sample}/{pair}/{column}")
    rows = correlations.merge(support, on="roi_id", validate="one_to_one")
    if not np.array_equal(rows.n_roi, rows.source_spots):
        raise ValueError(f"Source count mismatch: {sample}/{pair}")
    n = rows.source_spots.to_numpy(float)
    if not (np.isfinite(n) & (n > 0) & (n == np.floor(n))).all():
        raise ValueError(f"Invalid source counts: {sample}/{pair}")
    finite = rows.parent_correlation_status.eq("finite")
    values = rows.corr_obs.to_numpy(float)
    if rows.parent_correlation_status.isna().any():
        raise ValueError(f"Missing availability status: {sample}/{pair}")
    if not np.array_equal(finite, np.isfinite(values)):
        raise ValueError(f"Status/value disagreement: {sample}/{pair}")
    if (np.abs(values[finite]) > 1).any():
        raise ValueError(f"Correlation outside [-1, 1]: {sample}/{pair}")
    order = sorted(rows.roi_id, key=identity_key)
    rows = rows.set_index("roi_id").loc[order].reset_index()
    # This is layout, not spatial distance; never jitter correlation values.
    rows["display_y_offset"] = np.linspace(-0.27, 0.27, len(rows)) if len(rows) > 1 else 0.0
    rows["display_status"] = np.where(rows.parent_correlation_status.eq("finite"), "finite", "unavailable")
    rows["display_support_bin"] = np.select(
        [rows.source_spots < 100, rows.source_spots < 400],
        SUPPORT_LABELS[:2], default=SUPPORT_LABELS[2],
    )
    rows["sample_id"] = sample  # Folder identity; core may export sample_id=0.
    rows["pair"] = pair
    rows["program_g1"], rows["program_g2"] = features
    rows["display_pair"] = f"{features[0]} × {features[1]}"
    return rows


def load_source_atlas(inputs, results_root, status, *, expected_core):
    """Read current raw core exports, not sample-averaged evaluation shards."""
    pieces, reference_support = [], {}
    pairs = sorted(status.pair.unique())
    for index, pair in enumerate(pairs):
        pair_features = None
        for sample in sorted(status.loc[status.pair.eq(pair), "sample_id"]):
            folder = Path(results_root) / sample / pair
            meta = inputs.json(folder / "run_metadata.json")
            if meta.get("core_sha") != expected_core:
                raise ValueError(f"Unexpected core SHA: {folder}")
            features = tuple(meta["feature_identity"][f"feature_g{i}_id"] for i in (1, 2))
            if "_".join(features) != pair or (pair_features is not None and features != pair_features):
                raise ValueError(f"Pair identity mismatch: {folder}")
            pair_features = features
            corr = inputs.csv(folder / "roi_correlations.csv")
            support = inputs.csv(folder / "roi_support_status.csv").sort_values("roi_id").reset_index(drop=True)
            if sample in reference_support:
                pd.testing.assert_frame_equal(reference_support[sample], support, check_exact=True)
            else:
                reference_support[sample] = support
            if (len(support) != meta["population"]["biological_roi_count"]
                    or support.source_spots.sum() != meta["population"]["source_spot_count"]):
                raise ValueError(f"ROI inventory differs from metadata: {folder}")
            rows = prepare_source_rows(corr, support, sample=sample, pair=pair, features=features)
            rows["source_csv"] = str((folder / "roi_correlations.csv").resolve())
            rows["support_csv"] = str((folder / "roi_support_status.csv").resolve())
            rows["core_sha"] = expected_core
            pieces.append(rows)
        if (index + 1) % 15 == 0 or index + 1 == len(pairs):
            print(f"Read source ROI atlas: {index + 1}/{len(pairs)} pairs", flush=True)
    return pd.concat(pieces, ignore_index=True)


def availability_table(rows):
    """Accounting only: these are not counts of biological detections."""
    records = []
    for (sample, pair), group in rows.groupby(["sample_id", "pair"], sort=True):
        finite = group.display_status.eq("finite")
        records.append(dict(
            sample_id=sample, pair=pair, n_rois_total=len(group),
            n_rois_finite=int(finite.sum()), n_rois_unavailable=int((~finite).sum()),
            n_source_spots_total=int(group.source_spots.sum()),
            n_source_spots_finite_rois=int(group.loc[finite, "source_spots"].sum()),
            n_source_spots_unavailable_rois=int(group.loc[~finite, "source_spots"].sum()),
            n_finite_scattered_rois=int((finite & group.spatial_support_class.eq("scattered")).sum()),
            unavailable_statuses=";".join(sorted(group.loc[~finite, "parent_correlation_status"].unique())),
        ))
    return pd.DataFrame(records)


def pair_pages(all_pairs, main_pairs, page_size=PAGE_SIZE):
    """Selected main pairs first; the complete remainder in lexical order."""
    if len(set(all_pairs)) != len(all_pairs) or len(set(main_pairs)) != len(main_pairs):
        raise ValueError("Duplicate pair selection")
    if not set(main_pairs).issubset(all_pairs) or page_size < 1:
        raise ValueError("Invalid pair selection/page size")
    order = list(main_pairs) + sorted(set(all_pairs) - set(main_pairs))
    return [order[i:i + page_size] for i in range(0, len(order), page_size)]


def plot_source_atlas(rows, pairs, samples):
    """One unchanged coefficient per ROI; one identically ordered row/sample."""
    if not pairs or not samples or len(set(pairs)) != len(pairs) or len(set(samples)) != len(samples):
        raise ValueError("Expected unique, nonempty pair/sample lists")
    selected = rows[rows.pair.isin(pairs) & rows.sample_id.isin(samples)]
    if selected.duplicated(["sample_id", "pair", "roi_id"]).any():
        raise ValueError("Duplicate plotted ROI identity")
    expected = {(s, p) for s in samples for p in pairs}
    if set(zip(selected.sample_id, selected.pair)) != expected:
        raise ValueError("Missing sample/pair inventory, not typed unavailability")
    # A short final supplementary page retains exactly the same facet width.
    n_columns = max(PAGE_SIZE, len(pairs))
    fig = plt.figure(figsize=(18.2, 10.8))
    gs = fig.add_gridspec(1, n_columns, left=0.135, right=0.992,
                         bottom=0.14, top=0.93, wspace=0.12)
    cmap, norm = plt.get_cmap("RdBu_r"), Normalize(-1, 1)
    support_sizes = dict(zip(SUPPORT_LABELS, SUPPORT_AREAS))
    for column, pair in enumerate(pairs):
        ax = fig.add_subplot(gs[0, column])
        group = selected[selected.pair.eq(pair)]
        features = group[["program_g1", "program_g2"]].drop_duplicates()
        if len(features) != 1:
            raise ValueError(f"Ambiguous feature identity: {pair}")
        g1, g2 = features.iloc[0]
        for index, sample in enumerate(samples):
            if index % 2 == 0:
                ax.axhspan(index - 0.5, index + 0.5, color="#F3F5F7", zorder=-3)
            roi = group[group.sample_id.eq(sample)]
            finite = roi.display_status.eq("finite")
            y = index + roi.display_y_offset
            ax.scatter(roi.loc[finite, "corr_obs"], y[finite],
                       s=roi.loc[finite, "display_support_bin"].map(support_sizes),
                       c=roi.loc[finite, "corr_obs"], cmap=cmap, norm=norm,
                       edgecolors="#555D68", linewidths=0.3, zorder=3)
            if (~finite).any():
                ax.scatter(np.full(int((~finite).sum()), 1.25), y[~finite],
                           marker="x", s=13, c="#767E89", linewidths=0.7, zorder=3)
        ax.axvspan(1.12, 1.39, color="#F0F1F3", zorder=-2)
        ax.axvline(0, color="#9DA4AE", linewidth=0.65, zorder=-1)
        for x in (-0.5, 0.5):
            ax.axvline(x, color="#DFE3E8", linewidth=0.45, zorder=-1)
        # Separate the explicitly IDHm-prefixed samples; no inferred patient IDs.
        for index in range(1, len(samples)):
            if samples[index].startswith("IDHm_") != samples[index - 1].startswith("IDHm_"):
                ax.axhline(index - 0.5, color="#A4ABB4", linewidth=1.0)
        ax.set(xlim=(-1.06, 1.39), ylim=(len(samples) - 0.5, -0.5),
               xticks=[-1, 0, 1, 1.25], xticklabels=["−1", "0", "+1", "NA"],
               yticks=range(len(samples)), yticklabels=samples if column == 0 else [])
        ax.set_title(f"{g1}\n× {g2}", fontsize=9, pad=10)
        ax.tick_params(axis="y", length=0, labelsize=8.3, pad=7)
        ax.tick_params(axis="x", length=3, labelsize=7.5)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_bounds(-1, 1)
    fig.supxlabel("Within-ROI source Pearson r", y=0.085, fontsize=10)
    handles = [Line2D([], [], linestyle="", marker="o", markersize=math.sqrt(area),
                      markerfacecolor="#C9CDD3", markeredgecolor="#555D68", markeredgewidth=0.4,
                      label=label) for label, area in zip(SUPPORT_LABELS, SUPPORT_AREAS)]
    handles.append(Line2D([], [], linestyle="", marker="x", color="#767E89",
                          markersize=4, label="Unavailable ROI (NA column)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.56, 0.023),
               ncol=4, frameon=False, fontsize=9, handletextpad=0.6, columnspacing=2)
    return fig
