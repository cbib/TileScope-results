# Mouse figure and source-data generation

Build Figure 4 and Supplementary Figures S9–S10 from Input Data and
Supplementary Data 4. No analyses or signature scores are recomputed.

```bash
python -m analyses.figures.build mouse
# Rebuild one figure:
python -m analyses.figures.scripts.mouse_publication.build --figure FigureS9
```

## Inputs

- `TileScope-data/input_data/mouse/`: score banks, spot/ROI identities,
  frozen gene definitions, scoring settings, pair inventory and run configuration.
- `TileScope-data/supplementary_data/Supplementary_Data_4/`: the complete
  222-pair numerical screen, canonical measurements, component/support ledgers
  and selected consensus geometry. `selected_pairs.csv` identifies illustrated pairs.

See the [repository data layout](../../../../docs/DATA_LAYOUT.md) for path settings.

## Outputs

Under the [figure workspace](../../README.md):

- `main/Figure4/`: ten separate panels.
- `supplementary/FigureS9/`: seven separate panels.
- `supplementary/FigureS10/`: seven separate panels.
- Each figure includes editable SVG/PDF artwork, PNG previews, CSV/Excel source
  data, display settings, a panel index and provenance.

Figure 4B contains consensus score maps; Figure 4C contains Pearson correlations.
S9A combines the two individual enrichment patterns using the core consensus
palette. It is not a map of the joint-high enrichment ratio. Original spot
scores and joint-high measurements remain in the numerical data.

Captions and manuscript prose belong in the paper, not this renderer. The
builder does not create manuscript drafts or touch manually assembled figures.

## Checks

`evidence.py` verifies all 2,064 ROI correlations against the score banks and
the saved numerical screen, along with the indexed shared configuration and feature identities.
Selected maps also verify spot/ledger alignment, consensus correlations,
component identities and boundary groups. Display limits and outlines are
recorded separately from the analysis measurements.

```bash
MPLBACKEND=Agg python -m pytest analyses/figures/scripts/mouse_publication/test_build.py analyses/figures/test_source_data.py -q
```
