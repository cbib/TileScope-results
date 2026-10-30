# Mouse GBM spatial programs

The analysis uses the fixed gene lists and scoring settings in
[inputs/signatures_harmonized.yaml](inputs/signatures_harmonized.yaml).
CTL and KO share one panel per topic; common panels are also used in V4
where measured, with one separately named V4 reactive-interface panel.

Every available unordered pair was analysed across all ROIs: 13 programs
and 78 pairs in each of CTL and KO, 12 programs and 66 pairs in V4
(222 runs, 2,064 pair–ROI rows). Low signal does not exclude a pair.
Pdpn is missing from the CTL/KO objects; Casp4, Il1b, Ripk2, Ddit4, Ddit3
and Atf4 are missing from V4. Panels requiring these genes are unavailable,
not replaced by shorter sample-specific definitions.

## Inputs and marker retrieval

[inputs/samples.csv](inputs/samples.csv) identifies the three Seurat objects,
their Spatial assays and BayesSpace columns.
[inputs/roi_roles.csv](inputs/roi_roles.csv) contains the exploratory ROI
annotations, with evidence in [ROI_ANNOTATIONS.md](ROI_ANNOTATIONS.md).

The primary definition contexts are the biologist-selected CTL ROIs 2/5 and
KO ROIs 4/9. V4 ROIs 10/2/1 are additional computationally annotated contexts.
CTL ROI 4 is uncertain and excluded from signature construction, not from
TileScope analysis.

`phase1_2_build_marker_views.R` retrieves two kinds of marker table:

- Every ROI versus the rest of its sample, plus the highlighted CTL/KO
  core–periphery comparisons: Spatial/data Wilcoxon tests, min.pct 0.1,
  logfc.threshold 0.25 and min.diff.pct 0.1. FindAllMarkers retains both
  directions without a p-value filter (`return.thresh = Inf`).
- Seven specified comparisons with min.pct, logfc.threshold and min.diff.pct
  all zero: CTL 2 versus 5; KO 4 versus 9; CTL 2 versus KO 4; CTL 5 versus
  KO 9; and V4 10 versus 1, 10 versus 2, and 1 versus 2.

Before merging CTL/KO objects, the script restricts both Spatial assays to
their 1,315 shared genes (1,981 CTL and 1,950 KO input genes), then joins the
layers. Missing measured features are not zero-filled. Ranking uses
`abs(avg_log2FC) * abs(pct.1 - pct.2)` separately by comparison and
fold-change direction; ties retain Seurat row order. These are exploratory
marker ranks, not formal differential-expression evidence from independent
biological replicates.

Core–periphery differences can reflect host-neural composition rather than
the periphery's own immune identity. Signature selection therefore uses both
these comparisons and the ROI-versus-rest markers, not one in place of the
other. [SIGNATURE_DECISIONS.md](SIGNATURE_DECISIONS.md) records the biological
topics, candidate comparisons, spatial checks and final choices.

## Scores and TileScope inputs

The maintained workflow proceeds through `phase4_harmonized_*` (candidates
and spatial diagnostics), `phase5_*` (panel and scoring comparisons), and
`phase6_harmonized_build_pair_csvs.R` (frozen scores and pair inputs).
R scripts take explicit file/root arguments; Python stages run as modules.

Each signature is scored once per sample with Seurat AddModuleScore on
Spatial/data after LogNormalize (10,000): 24 bins, 50 controls per member,
seed 1234 and the full measured gene pool. Scores are reused across pairs,
without clipping, imputation or post-score transformation. Absolute score
magnitudes are not calibrated between samples. Candidate YAMLs are review
inputs; only the frozen signature YAML governs the final runs.

## Storage

Code and authored definitions remain in this directory. Downloaded Input Data
under `mouse/` contains `pairs.csv`, `scores/<sample>.csv.gz`,
`signatures_harmonized.yaml` and `tilescope_run_config.json`. The runner selects
the two `__score` columns named in each pair, never the `__raw_mean` columns.
No AddModuleScore calculation is repeated during a TileScope rerun.

Both `--input-dir` and `--output-dir` are required; their directory names are
not fixed. Input files stay read-only. Per-pair CSVs are temporary and are
removed after each run. Results are written to
`<output-dir>/<sample>/<signature_a>__<signature_b>/`; status and provenance
are stored in `<output-dir>/run_accounting/`.

The source-data preparation and evidence builders use the local
`TileScope-data/mouse/` tree, which is not distributed as a `CurrentResults`
download:

```text
MCT1_CTL/RDSs/, MCT1_KO/RDSs/, V4_WT/RDSs/
CurrentResults/
  <sample>/<program_A>__<program_B>/     locally saved TileScope runs
mouse_gbm_spatial_programs/
  phase1/, phase2/, phase3/             marker and topic evidence
  phase4_harmonized/                    gene diagnostics and candidate tables
  phase5*/                             panel-comparison and scoring evidence
  phase6_harmonized/                    source-preparation score banks and pair inputs
    evidence/                          all-pair screen, periphery comparison, gene detection
```

The shared settings agree across all 222 published runs. Per-run metadata and
effective configurations record their execution environment. New reruns write
`run_status.csv` and `run_provenance.json` under their own `run_accounting/`.
Source Seurat objects, score banks and pair inputs are different products,
not competing datasets.

Preparation/evidence data default to the sibling TileScope-data directory.
`TILESCOPE_DATA_DIR` changes that root; `TILESCOPE_DATA_MOUSE_ROOT` overrides
its mouse subdirectory. The TileScope runner instead uses its explicit arguments.
See [data layout](../../docs/DATA_LAYOUT.md).

## Commands

From TileScope-results, with TileScope installed:

```bash
# Validate the distributed score banks without launching analyses.
python -m analyses.mouse_gbm_spatial_programs.phase6_harmonized_run_tilescope \
  --input-dir ../TileScope-data/input_data/mouse \
  --output-dir ../TileScope-data/reruns/mouse --all --dry-run

# The same inputs -> TileScope outputs (launches analysis).
python -m analyses.mouse_gbm_spatial_programs.phase6_harmonized_run_tilescope \
  --input-dir ../TileScope-data/input_data/mouse \
  --output-dir ../TileScope-data/reruns/mouse --all --workers 4

# Saved runs and Seurat objects -> numerical evidence (requires R/Seurat).
python -m analyses.mouse_gbm_spatial_programs.build_evidence

# Downloaded inputs and Supplementary Data 4 -> figures.
python -m analyses.figures.build mouse
```

`--config PATH` overrides the input directory's configuration. `--sample` and
`--pair signature_a__signature_b` restrict the selected runs. `--validate`
selects the five predefined smoke-test pairs; use a separate output directory
for that check. `--all` runs every selected pair, including existing runs.
`--remainder` skips successful runs with matching score banks, configuration,
signature definitions and core revision; changed inputs require a fresh output
directory or an explicit rerun with `--all`.

`build_evidence` accepts `--output <dir>` for an isolated extraction. It
does not change signatures or rerun TileScope. The figure builder uses
[inputs/selected_pairs.csv](inputs/selected_pairs.csv) for the illustrated
pairs and keeps the complete screen in Supplementary Data 4.
See the [figure guide](../figures/scripts/mouse_publication/README.md).
Manuscript text and Word reports are not generated by this workflow.
