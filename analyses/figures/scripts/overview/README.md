# Figure 1: overview illustration

**Figure 1 is a schematic.** It illustrates the method rather than
reporting a scientific result on biological data. The input is a
purpose-built synthetic field — not patient data, not a benchmark
case, not a scenario classifier. No real genes carry invented
expression patterns; "Gene A" and "Gene B" are explicitly illustrative
continuous fields in arbitrary units.

The published panels are rendered from the retained plotting
measurements under
`TileScope-data/supplementary_data/Supplementary_Data_1/overview/`.
A complete original TileScope run on this synthetic input is not
required to re-render the figure.

Inputs are in `TileScope-data/input_data/overview/`; retained plotting
measurements are in `TileScope-data/supplementary_data/Supplementary_Data_1/overview/`.
All artwork is in `analyses/figures/main/Figure1/panels/`; source data and
provenance are beside it. See the [figure workspace](../../README.md).

## Three separate panels

The files in `panels/` are independent, identically sized SVG, PDF and PNG files:

- `A1_roi_labels`: predefined ROIs; the two pieces labelled ROI 1 belong to the
  same biological region label but are disconnected spatial components.
- `A2_gene_A_expression`: broad expression with a second elevation in the
  detached ROI 1 component, attenuating further into ROI 2 away from ROI 1.
- `A3_gene_B_expression`: a moderate elevation throughout much of ROI 2, with
  a stronger localized patch partly overlapping Gene A and crossing the
  boundary between the main ROI 1 component and ROI 2.

The colours in the ROI map identify regions, not expression. Both expression
maps use the same light-blue through off-white to red palette and fixed 0–1
range. Low expression is blue without a dark navy background; high expression
is red. The pale midpoint is not a biological threshold or zero. Their dots represent the
same individual spots in exactly the same locations. Expression is not uniform
within an ROI, and ROI borders are not expression thresholds. The Gene B
background is deliberately aligned with the analytic ROI 2 geometry, with
smooth transitions across its borders and reduced expression within the
detached ROI 1 component. The ROI labels remain predefined: this is not a
clustering result or independent biological validation of those labels.

The same dataset also supplies Figure 1B: three saved tessellations (indices
0, 1 and 2), ensemble consensus from all ten tessellations, and a spot map,
exported by the common figure command into the same `panels/` directory.
Neither renderer assembles a combined multi-panel plate.

Figure 1C/D use the same saved outputs for five separate question-oriented
graphics: enrichment, internal association, cross-correlation over distance,
boundary response and sensitivity to aggregation. The common exporter records
the statistical units and display settings with the row-level CSV/Excel evidence and
provenance beside the panels. It does not rerun TileScope or modify core outputs.

## Final assembly

All artwork remains in separate SVG/PDF/PNG files; no assembled plate is imposed.
The C-prefixed filenames are retained to preserve existing layout links.

| Figure panel | Separate assets, in display order |
| --- | --- |
| A | `panels/A1_roi_labels`, `A2_gene_A_expression`, `A3_gene_B_expression` |
| B | `panels/B0_spot_expression_gene_A`, the three `B1_tessellation_*` maps, `B2_ensemble_consensus_gene_A`; separate colour key |
| C | `panels/C1_enrichment`, `C2_internal_association` |
| D, left to right | `panels/C4_boundary_response`, `C3_association_over_distance`, `C5_sensitivity_to_aggregation` |

The enrichment panel uses a **spot-level high-expression map and source ROI
enrichment matrix**. The scatterplot heading is **Expression correlation**;
its distinct Moran inset describes spatial association between neighbouring
spots across whole ROI 1. Boundary
response is illustrated on a zoom retaining every ROI 2 spot and all ROI 1/2
contacts. The sensitivity strip uses **ROI 3**, whose expression is unchanged.

## Reproduce

From the results repository, using the current core environment:

```sh
../TileScope/venv/bin/python -m analyses.figures.build overview
../TileScope/venv/bin/python -m pytest analyses/figures/scripts/overview -q
```

This command verifies the bundle indexes, source graph, component membership
and plotted values. It never launches TileScope. The original full run and
its configuration are missing; the retained evidence is sufficient to redraw
these panels, not to claim an exact rerun. No substitute configuration is supplied.

## Dataset and configuration

`input_data/overview/spots.csv` contains `x,y,gene1,gene2,roi_id`.
The substrate is a regular hexagonal spot lattice with a mildly irregular outer
shape. Three predefined ROIs produce four source graph components. The labels
are analytic spatial regions, not fitted from either expression field.
Coordinates and expression use arbitrary units. Generation seed: 20260916.

Gene B combines a regional background of at most 0.56 units with a
localized Gaussian elevation. Sigmoid transitions span the borders of ROI 2;
the background blends into the peak rather than creating a clipped plateau.

Within ROI 2, Gene A is attenuated at distance `d > 3` from the nearest ROI 1
spot (in median nearest-neighbour spacings): its base value becomes
`0.02 + (base - 0.02) * exp(-(d - 3)^2)`. Both ROI 1 components contribute to
that distance. The taper applies to 56 spots and leaves expression in ROIs 1
and 3 unchanged. Core defines boundary bands using graph hops, independently
of this geometric expression generator.

## Saved run

Core `ef8ab0df1332bda822a2cf3240fc856c466ed534`, 457 spots, 3 ROIs,
4 components. ROI 1 has 149- and 43-spot components; ROI 2 has 177 spots;
ROI 3 has 88. All 457 spots are retained and tessellation eligible. The ten
tessellations produce 779 tiles in total (at least 6 per component per
replicate), and the consensus has 120 hexagons. The ROI 1→2 source interface
is evaluable for both genes. The preserved provenance identifies the original computation; a recorded
configuration hash is not a replacement for the missing configuration file.

All three manuscript maps have been visually checked: no clipped tissue,
missing dots or misplaced labels; the two ROI 1 components share the same
identity colour and label. The expression fields differ visibly and cross
the region labels as intended. The neutral outlines are the actual core
polygons, including their spot-scale scalloping, not smoothed cartoon borders.

## Rendering and provenance

The manuscript panels use the core `SpotSplatPainter` and
`draw_polygon_boundaries` functions with the input expression values and the
exact exported `component_geometry.csv` polygons. This is a separate layout,
not a raster edit of the report. Cosmetic changes: three individual pages,
smaller clearly separated dots, neutral thinner outlines, short headings,
a light-blue to red expression palette with fixed shared colour limits, and no
correlation panel. SVG text stays editable. The standard core PDFs retain their
original palette; this colour change is confined to the manuscript renderer.
There is no boundary smoothing, displacement of points, expression editing,
interpolation of the displayed source values or removal of observations.

Bundle indexes check the retained input and measurement files. The generated
figure provenance records these inputs, the rendering code and the panels.
Artwork and panel source tables are under `analyses/figures/main/Figure1/`
and are gitignored. Captions belong in the manuscript.
