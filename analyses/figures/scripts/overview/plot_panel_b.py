"""Separate Figure 1B artwork from the existing Figure 1 example core run.

Read-only with respect to inputs and TileScope results. No analysis, invented
tile geometry, manually assembled consensus or expression rescaling occurs.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analyses.figures.scripts.overview.build import expression_colormap
from analyses.figures.common import save_panel as save_artwork
from tilescope_results.paths import input_data_root, supplementary_data_root
from tilescope_results.bundles import verify

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from shapely import wkt

from tilescope.viz.geometry_helpers import iter_polygons, patch_from_polygon
from tilescope.viz.painters import SpotSplatPainter, draw_polygon_boundaries


REPLICATE_INDICES = (0, 1, 2)
CANVAS_SIZE = (4.2, 3.35)
AXES_RECT = (0.035, 0.19, 0.93, 0.70)
XLIM, YLIM = (-15.5, 15.5), (-10.5, 10.5)
EXPRESSION_NORM = Normalize(0.0, 1.0)


def load_saved_run():
    """Read the retained figure evidence; the original full core run is not included."""
    verify(input_data_root())
    verify(supplementary_data_root(1))
    run = supplementary_data_root(1) / 'overview'
    provenance = json.loads((run / 'original_figure_provenance.json').read_text())
    frames = {
        "source": pd.read_csv(input_data_root() / 'overview/spots.csv', float_precision='round_trip'),
        "geometry": pd.read_csv(run / "component_geometry.csv", dtype={"component_id": str, "roi_id": str}),
        "tiles": pd.read_csv(run / "tiles.csv", dtype={"component_id": str, "roi_id": str}),
        "hexes": pd.read_csv(run / "hexes.csv", dtype={"component_id": str, "roi_id": str}),
    }
    tiles, hexes = frames["tiles"], frames["hexes"]
    if set(tiles.replicate_index) != set(REPLICATE_INDICES):
        raise ValueError("Expected geometry for the three displayed tessellations")
    if not hexes.n_tessellations_covered.eq(10).all():
        raise ValueError("Consensus coverage changed; review the legend/support description")
    if not np.allclose(tiles.g1_density, tiles.sum_g1 / tiles.n_spots, rtol=1e-12, atol=1e-12):
        raise ValueError("Exported tile means disagree with expression sums / spot counts")
    if not np.allclose(hexes.gene1_density_ensemble,
                       hexes.obs_equiv_g1_sum_mass / hexes.obs_equiv_spot_mass,
                       rtol=1e-12, atol=1e-12):
        raise ValueError("Exported ensemble means disagree with their stored masses")
    signatures = []
    for replicate in REPLICATE_INDICES:
        subset = tiles.loc[tiles.replicate_index == replicate]
        if subset.empty or not np.isfinite(subset.g1_density).all():
            raise ValueError(f"Replicate {replicate} is unavailable")
        signatures.append(frozenset(subset.tile_geometry_wkt))
    if len(set(signatures)) != len(REPLICATE_INDICES):
        raise ValueError("Stack members must be genuinely different partitions")
    return run, provenance, frames


def map_canvas(title=None):
    """Match Panel A's map transform and optional heading position/style."""
    fig = plt.figure(figsize=CANVAS_SIZE, facecolor="none")
    ax = fig.add_axes(AXES_RECT, facecolor="none")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if title is not None:
        fig.text(0.5, 0.94, title, ha="center", va="center", fontsize=12)
    fig.canvas.draw()
    return fig, ax


def paint_units(ax, frame, geometry_col, value_col, *, prefix):
    """Flat polygon means, preserving holes and every exported geometry part."""
    cmap = expression_colormap()
    values = frame[value_col].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Unavailable values require an explicit support display")
    n_parts = 0
    for ordinal, (geometry_text, value) in enumerate(zip(frame[geometry_col], values)):
        geometry = wkt.loads(geometry_text)
        if not geometry.is_valid or geometry.is_empty:
            raise ValueError("Invalid exported unit geometry")
        for part_index, part in enumerate(iter_polygons(geometry)):
            patch = patch_from_polygon(
                part, facecolor=cmap(EXPRESSION_NORM(value)),
                edgecolor="#64686C", linewidth=0.36, antialiased=True,
            )
            patch.set_gid(f"{prefix}_unit_{ordinal}_part_{part_index}")
            ax.add_patch(patch)
            n_parts += 1
    return n_parts


def draw_roi_boundaries(ax, geometry):
    polygons = {row.component_id: wkt.loads(row.geometry_wkt) for row in geometry.itertuples()}
    draw_polygon_boundaries(ax, polygons, color="#3B4043", linewidth=0.80)




def render(frames, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "tilescope-figure1-B",
                         "pdf.fonttype": 42, "axes.linewidth": 0.6})
    output.mkdir(parents=True, exist_ok=True)
    files, descriptions = [], []
    geometry = frames["geometry"]

    fig, ax = map_canvas()
    source = frames["source"]
    spots = SpotSplatPainter(radius_scale=0.35).draw(
        ax, source[["x", "y"]].to_numpy(), source.gene1.to_numpy(),
        cmap=expression_colormap(), norm=EXPRESSION_NORM, canvas=None,
        median_nn_override=1.0,
    )
    if len(spots.get_offsets()) != len(source):
        raise ValueError("A source spot was omitted")
    draw_roi_boundaries(ax, geometry)
    files.extend(save_artwork(fig, output, "B0_spot_expression_gene_A"))
    descriptions.append({"stem": "B0_spot_expression_gene_A", "surface": "source",
                         "n_units": len(source), "value_column": "gene1"})

    for replicate in REPLICATE_INDICES:
        subset = frames["tiles"].loc[frames["tiles"].replicate_index == replicate]
        fig, ax = map_canvas("Tessellation")
        stem = f"B1_tessellation_{replicate + 1:02d}_gene_A"
        n_parts = paint_units(ax, subset, "tile_geometry_wkt", "g1_density", prefix=stem)
        draw_roi_boundaries(ax, geometry)
        files.extend(save_artwork(fig, output, stem))
        descriptions.append({"stem": stem, "surface": "tessellation",
                             "replicate_index": replicate, "n_units": len(subset),
                             "n_polygon_parts": n_parts, "value_column": "g1_density"})

    fig, ax = map_canvas("Consensus ensemble")
    hexes = frames["hexes"]
    n_parts = paint_units(ax, hexes, "hex_geometry_wkt", "gene1_density_ensemble",
                          prefix="ensemble")
    draw_roi_boundaries(ax, geometry)
    files.extend(save_artwork(fig, output, "B2_ensemble_consensus_gene_A"))
    descriptions.append({"stem": "B2_ensemble_consensus_gene_A", "surface": "consensus_ensemble",
                         "n_units": len(hexes), "n_polygon_parts": n_parts,
                         "value_column": "gene1_density_ensemble",
                         "n_contributing_tessellations": 10})

    # An independent colour key can be placed in the assembled legend.
    fig = plt.figure(figsize=(2.2, 0.44), facecolor="none")
    ax = fig.add_axes([0.055, 0.60, 0.89, 0.23])
    mappable = plt.cm.ScalarMappable(norm=EXPRESSION_NORM, cmap=expression_colormap())
    key = fig.colorbar(mappable, cax=ax, orientation="horizontal", ticks=[0, 0.5, 1])
    key.ax.tick_params(length=0, labelsize=8, pad=3)
    key.outline.set_visible(False)
    files.extend(save_artwork(fig, output, "B_expression_colour_key"))
    return files, descriptions
