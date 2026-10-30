# Greenwald main and supplementary figures

Figure code and reproduction instructions are here. Individual artwork and source data are
generated in `analyses/figures/main/Figure3/` and
`analyses/figures/supplementary/FigureS4/` through `FigureS6/`.
See the [figure workspace](../../README.md).

From TileScope-results, using the environment with TileScope installed:

```sh
python -m analyses.figures.build greenwald
```

The renderer reads `TileScope-data/input_data/greenwald/` and
`TileScope-data/supplementary_data/Supplementary_Data_2/`. It verifies bundle
indexes, then renders without launching TileScope or fitting a new model.

Open `analyses/figures/index.html` to inspect individual assets.
SVG/PDF canvas and axes backgrounds are transparent for figure assembly.
PNG previews keep white backgrounds; small ROI-label backings are retained.

- Figure 3: 15 individual maps/plots, grouped by filename A–D.
- S4: four cohort checks; S5: six source maps; S6: two distance profiles and
  two reverse-boundary panels. No automatic combined plates.
- Each directory has `panels/`, CSV/Excel `source_data/`, a panel index
  and provenance. Optional manual layouts go in `assembly/`, never overwritten.
- [GBM findings](../../../gbm_spatial_repertoire/FINDINGS.md) records the
  retained spatial interpretations; captions are maintained in the manuscript.

The shared program display is checked against all three underlying pair runs.
The builder rederives displayed cohort Pearson summaries from the saved ROI
inventory, verifies current canonical export hashes, checks input/ledger ROI
identity, verifies source and ensemble correlations from plotted values, and
reconstructs boundary bands using the core source graph. The band counts and
both program responses must match the stored exports.

Keys remain at their original grains: source row order in `spot_ledger.csv`,
parent ROI identity in ROI tables, `(ROI, component, hex)` in consensus tables,
component IDs in radial tables, and ordered anchor/responder IDs plus semantic
feature identity in interface tables. The figure does not join them on a
generic `roi_id` without checking those roles.

Scientific runs use `ef8ab0d`, n=10. Their producer-code hashes remain unchanged;
the figure records its own producer separately. Numeric input changes
fail validation. Display settings and statistical units accompany the source data.

The complete 91-pair inventory is in
`TileScope-data/supplementary_data/Supplementary_Data_2/`, including availability. Exploratory
clustering and unused tissue examples remain in their original
`gbm_spatial_repertoire/` analyses, not in the manuscript figure package.
