# Figures S7–S8: selected complete components

Figure S7 shows P28_post; Figure S8 shows the Janesick breast example.
These illustrations are not recurrence estimates: cohort contrasts use all
sixteen samples and complete parent ROIs, not the displayed components.

## Biological motivation

Meylan et al. (Cell, 2026; DOI 10.1016/j.cell.2025.12.055), Figure 4G,
compared stem-like and interferon marker expression in T cells in lymphoid
aggregates and tumour–T cell interfaces. We use the exact marker lists in
their Figure 4 notebook. Our arithmetic means and all-cell ROI statistics
are not the authors' deposited scores or a repeat of their T-cell differential
expression analysis. The pair is frozen in
`analyses/meylan_xenium_preparation/pair.json`.

Janesick et al. (Nature Communications, 2023;
DOI 10.1038/s41467-023-43458-x) used ACTA2 and FASN to illustrate myoepithelial
structures and invasive tumour. ACTA2 is not exclusive to myoepithelial cells.

## Spatial units

- GBM expression: complete component 784, the largest component of the tumour–T cell
  interface ROI: 6,892 cells, 6,554 eligible for tessellation (95.1%).
  The full parent ROI contains 21,865 cells, of which 15,416 (70.5%) are
  eligible. The whole sample retains all 97,488 cells as input.
- The second GBM expression component is the largest lymphoid aggregate,
  component 0: 233 cells, 217 eligible (93.1%). It directly neighbours component
  784. Its complete extent is shown in the broad maps and enlarged in separate
  consensus detail panels, together with the surrounding interface.
  This comparison follows the lymphoid/interface contrast in Meylan Figure 4G.
- Tumour component 501 (1,156 cells; 1,143 eligible) is shown in the local
  region map and outlined for context in the expression maps. It has the most
  contact edges with 784 among tumour components, but is
  not the biological comparator for these signatures. Its cells and expression
  never enter the expression colour scale or its limits.
- Breast: complete invasive component 7, 46,825 cells, of which 46,754 are
  eligible. Parent ROI: 46,862 cells.

The GBM expression figure shows every cell in components 784 and 0 and
all their actual saved hexagons. No coordinate crop or new region labels enter the analysis.
Whole-ROI enrichment in S7D and S8B must not be attributed to these components.

## Display and limits

S7A is cropped to the complete interface, aggregate and neighbouring tumour
components, with a 3% margin. Other measured cells in this window remain grey.
The full section is not displayed; full-section cell identities are retained
in the source table, and the local bounds are recorded in display settings.
S7B gives the two consensus maps most of the page, showing complete components
784 and 0. S7C has two standalone consensus detail panels, one per program.
The square detail window is centred on component 0 and has twice
its maximum extent, including surrounding interface. The detail window is
marked in the broad maps. Additional full-field cell maps are supplied as
individual panels E1 and E2, outside the page preview.

Each expression map uses one linear scale across both displayed components,
from their combined minimum to combined 99th percentile, with larger values
saturated. Cells and consensus intentionally have different colour limits,
as do the two programs. Each detail inherits its corresponding full-map limits;
it is never independently rescaled. Coordinates and saved hexagon geometry,
including holes, are unchanged. Expression colours should not be compared
quantitatively across representations. Cell maps retain cells excluded from
aggregation; those cells appear grey in consensus maps. S7D and S8B report
enrichment for whole parent ROIs, not the displayed components or detail windows.

Component numbers appear in the figure caption, not the GBM artwork.
ROI names are given in a colour legend to the right of the local ROI map
and directly labelled on the broad expression maps.
The categorical region colours are teal for the interface, orange for the
lymphoid aggregate and purple for tumour, separate from the expression scale.
All three component boundaries are shown on the expression maps, including
the enlarged consensus views where they intersect the window. Boundary colours
match the region map; the tumour is labelled as context and remains uncoloured.
These are outlines of the displayed components, not all components in each parent ROI.
They are derived from the saved core
polygons, simplified by one quarter of the sample median nearest-neighbour
distance for display only, preserving topology; no convex or alpha hull is
constructed. Full polygons and display outlines accompany the figure data.

Descriptive cell means (stem-like/interferon) are 0.056/0.400 in component 784,
0.280/0.286 in component 0, and 0.029/0.441 in component 501. Thus tumour 501
does not provide a lower-interferon comparator. These all-cell summaries are
not the paper's T-cell-only differential-expression analysis, and are not
whole-ROI enrichment ratios.

Selection used paper relevance and visual readability after exploratory
review. It is not blinded or a recurrence estimate. The exact values,
identities, counts and display settings accompany the individual panels.

The generated Figure S7/S8 folders contain individual SVG/PDF/PNG panels,
CSV/Excel source data and A4 layout previews.
