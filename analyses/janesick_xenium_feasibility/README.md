# Janesick Xenium feasibility

The current ACTA2–FASN analysis is in
`TileScope-data/Janesick/CurrentResults/sample1_rep1/ACTA2_FASN/`.
It uses ten tessellations for 87,770 cells in the three
published parent ROIs: DCIS_1, DCIS_2 and Invasive. The exact core
revision used is recorded in that run's `run_metadata.json`.

Prepared source data remain under `TileScope-data/Janesick/<sample>/tilescope_inputs/`.
One shared run recipe lives at `TileScope-data/Janesick/CurrentResults/tilescope_run_config.json`.
It uses ten tessellations, target 15 cells/tile, minimum 9 cells/tile,
minimum 50 cells/component, and a two-thirds consensus-coverage requirement.
Current core selects the locally pruned Xenium Delaunay graph.

`run.py` uses the configured core checkout and the three published parent ROIs
(DCIS_1, DCIS_2, Invasive) for Sample 1 Replicate 1. Validate without a run:

```sh
python -m analyses.janesick_xenium_feasibility.run --validate-only
```

Omit `--validate-only` to execute a new run in
`CurrentResults/sample1_rep1/ACTA2_FASN/`. Existing results are never overwritten.
Exact inputs, cell IDs, settings and numerical outputs are included in
Supplementary Data 3. See the current
[cross-study analysis](../xenium_feasibility/README.md).

## Rebuild inputs

Preparation code lives here, not inside TileScope-data. From TileScope-results:

```sh
Rscript analyses/janesick_xenium_feasibility/build_published_rois.R \
  ../TileScope-data/Janesick/sample1_rep1 ../TileScope-data/Janesick/sample1_rep1/roi
Rscript analyses/janesick_xenium_feasibility/build_tilescope_inputs.R \
  ../TileScope-data/Janesick/sample1_rep1 ../TileScope-data/Janesick/sample1_rep1/tilescope_inputs
```

The second command writes only the three-parent ACTA2–FASN input and matching
pair/ROI manifests. Original nine-lasso memberships remain annotation evidence,
not alternative analysis inputs. The current sample's original data are retained.
