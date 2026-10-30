# Xenium feasibility supplement

Scope: all sixteen Meylan GBM samples from eight paired patients, with
stem-like/interferon marker means, and Janesick breast Sample 1 Replicate 1,
ACTA2–FASN. Figures S7/S8 illustrate P28_post and breast; numerical results
include all 17 samples.

## Data and code

- Preparation/runners: `analyses/meylan_xenium_preparation/` and
  `analyses/janesick_xenium_feasibility/`.
- Downloadable inputs: `TileScope-data/input_data/xenium/`.
  `datasets.csv` identifies original cell tables, features, units,
  annotation records and settings for every sample.
- Downloadable results: `TileScope-data/supplementary_data/Supplementary_Data_3/`.
  All numerical measurements and per-cell support ledgers are included;
  full spatial geometry is supplied for the two illustrated samples.
- Figure scripts: `analyses/figures/scripts/xenium/`.
- Figure outputs: `analyses/figures/supplementary/FigureS7/` and `FigureS8/`.
- New-run working data: `TileScope-data/meylan_xenium/` and
  `TileScope-data/Janesick/`, separate from the frozen downloads.

Data/core roots default to sibling directories; override with
`TILESCOPE_DATA_DIR` and `TILESCOPE_CORE_DIR`.

## Redraw from the bundles

```sh
python -m analyses.figures.build xenium
```

For a staged replacement with the additional cohort/figure coherence check:

```sh
python -m analyses.xenium_feasibility.publish --replace-generated
```

Publication writes figures, not numerical data. It does not rerun TileScope
or refresh the deposited bundles from new analyses. Manually assembled
figures are preserved.

## Interpretation and checks

Cell identities, ROI labels and expression units remain those of the analysed
inputs. Numerical tables preserve unavailable results, component identities,
ROI contacts, coverage and pre/post cohort denominators. Illustrations retain
component identities; numerical cohort comparisons use complete ROIs.

Meylan ROIs are reconstructed from the authors' neighbourhood procedure.
Marker means describe all cells, whereas their paper's comparison concerned
T cells. Missing regions are not treated as absent effects.
