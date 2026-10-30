# Greenwald inputs and TileScope runs

The distributed inputs cover **30 IDH-wildtype GBM samples × 91 program pairs**:
2,730 runs and 17,472 pair–ROI measurements. The original local preparation
also includes six IDH-mutant gliomas, which are outside this published cohort.

## Distributed inputs and results

The runner reads `pairs.csv`, `scores/<sample>.csv.gz` and
`tilescope_run_config.json` from the directory passed as `--input-dir`.
These are supplied in Input Data under `greenwald/`. The `pair` column records
the original feature order; the unordered `pair_id` is not used to reorder it.

`--output-dir` is required and can have any name. Results go to
`<output-dir>/<sample>/<pair>/`, with status in
`<output-dir>/run_accounting/run_status.csv`. No existing results are needed.
The input directory stays read-only. Temporary pair CSVs are removed after
each invocation of TileScope, rather than stored alongside the score banks.

## Source-data preparation

`TILESCOPE_DATA_DIR` defaults to sibling `TileScope-data`;
`TILESCOPE_CORE_DIR` defaults to sibling `TileScope`. Neither is a personal path.

Data-side `heiland_tyrosh/` contains:

- `Original/`: authors' archive and annotations. The ZIP is not redundant with
  the eight extracted files.
- `RDSs/`: 36 Seurat objects with original BayesSpace labels.
- `CSVs/`: 3,276 frozen inputs preserving physical gene1/gene2 order.
- `CurrentResults/{sample}/{pair}/`: the corresponding saved core outputs.
- `analysis/{scenario_cohort,two_level_evidence,gbm_spatial_repertoire}/`: generated
  numerical evidence and selected review material, not copies of core runs.

Scripts, tests, scoring definitions, authored reviews and figure preparation stay
here. Current biology/figures: `../gbm_spatial_repertoire`. Shared evidence
builders: `../greenwald_full_evaluation`.

`preparation.json` specifies the 14 program definitions and scoring settings:
Spatial/data, BayesSpace, AddModuleScore nbin=24, ctrl=50 and seed=1.
Program order is fixed for control sampling and physical
pair orientation. These are the project's selected marker programs, not the
authors' entire original metaprogram reconstruction.

```sh
Rscript analyses/greenwald_full/prepare_inputs.R \
  --rds-dir ../TileScope-data/heiland_tyrosh/RDSs \
  --output-dir /tmp/greenwald-input-check \
  --sample 242_T \
  --check-against ../TileScope-data/heiland_tyrosh/CSVs
```

Omit `--sample` for the full cohort. Existing files are never overwritten.
The optional baseline check requires identical ROI labels and numerical
agreement to absolute tolerance 1e-12. Requires R, Seurat and jsonlite.
Original CSVs remain authoritative until an entire proposed replacement
passes this check.

## Run TileScope

The supplied `tilescope_run_config.json` specifies n=10, coverage=2/3
(seven contributing tessellations), target five observations per tile, minimum
three, component minimum 15, hex minimum four, ROI/context minima 40,
99 permutations and seed 42. No per-sample exceptions are applied.

Use a Python environment containing the configured core and results dependencies:

```sh
python -m analyses.greenwald_full.run_all_pairs \
  --input-dir ../TileScope-data/input_data/greenwald \
  --output-dir ../TileScope-data/reruns/greenwald --dry-run
python -m analyses.greenwald_full.run_all_pairs \
  --input-dir ../TileScope-data/input_data/greenwald \
  --output-dir ../TileScope-data/reruns/greenwald \
  --sample 242_T --pair AC_OPC --workers 1
```

Without sample/pair filters this covers every entry in the supplied `pairs.csv`.
Use `--config PATH` to override the input directory's shared configuration.
`--dry-run` validates selected score banks and settings without writing results.

Resume requires matching score-bank bytes, resolved settings, core revision and all
current quantitative exports. Runs without these identities are
**not silently reused or overwritten**. Use a fresh output directory, or
explicit `--force` when a rerun is intended. Core Python changes must be
committed first. Existing saved runs remain readable without invoking the runner.

`run_metadata.json` supplies thresholds and feature identities to readers;
`effective_config.json` verifies settings. These are functional outputs,
not disposable logs. Each chosen output directory keeps its own run inventory.

## Analysis

Current analysis entry points are documented in
`../gbm_spatial_repertoire/README.md`. Analysis reads saved core outputs;
it does not rerun TileScope.
