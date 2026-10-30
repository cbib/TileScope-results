# Code and data access

[TileScope core](https://github.com/cbib/TileScope) implements the analysis.
TileScope-results contains preparation, evaluation and figure code.
TileScope-data is a separate directory, not part of either code repository.

## Downloads

**Zenodo record / DOI: [TO BE ADDED AFTER DEPOSIT].**

The release has two kinds of data, supplied as five archives:

- **Input Data** (`input_data.zip`): the values passed to TileScope, coordinates,
  spot/cell and ROI identities, program definitions, annotation records and settings.
- **Supplementary Data 1:** measurements for 30 benchmark constructions and
  27 controls, the statistics atlas and clustering results, comparator values
  for 18 positive cases, original correlation p/q-values for all 57 runs,
  and the retained Figure 1 illustration measurements.
- **Supplementary Data 2:** 30 IDH-wildtype human GBM samples, 192 ROIs and
  91 program pairs; all ROI measurements, component-distance measurements,
  ordered boundary measurements, individual tessellation correlation/enrichment
  estimates, original correlation p/q-values, the run inventory, cohort frequencies
  and geometry for the figures.
- **Supplementary Data 3:** all 16 Meylan GBM samples and the Janesick breast
  sample; all numerical tables, cell eligibility, cohort comparisons and full
  spatial geometry for the two illustrated samples.
- **Supplementary Data 4:** all 222 mouse signature-pair runs, 2,064 pair–ROI
  rows, boundary and component measurements, selected consensus maps and
  gene-selection evidence.

Inputs are not repeated in the supplementary archives. These five downloads
are sufficient to redraw the figures; there is no additional reproduction archive.
Each has a README and a checksum index. See [extraction instructions](../README.md#where-to-put-the-data).

**Figure 1 is a schematic.** Its synthetic input lives in
`input_data/overview/`; the published panels are rendered from the
retained plotting measurements under `Supplementary_Data_1/overview/`.
A complete original TileScope run on this synthetic input is not
required to re-render the figure.

## What Input Data includes

| Dataset | Prepared inputs and settings |
|---|---|
| Overview | All 457 illustrative spots and their two values |
| Benchmark | All 57 role datasets, truth ledgers, construction plan, corpus index and shared run settings |
| Human GBM | One 14-program spot table per sample, pair/sample inventories, program/scoring definitions, author-context sample annotations and shared settings |
| Mouse GBM | One score bank per sample, spot identities, coordinates, ROI labels, AddModuleScore values and member-gene means; frozen signatures and scoring/run settings |
| Xenium | All 17 analysed cell tables with original identifiers, coordinates, ROI labels and feature values; gene definitions, annotation/mapping records and study-specific user/resolved settings |

**Converting score banks to pairwise CSV inputs.** The human and
mouse `scores/<sample>.csv.gz` files carry one row per spot and one column
per scored signature. The CSV API (`run_tilescope_from_csv`) and CLI take
one pair per input file with
columns `x, y, gene1, gene2, roi_id`; to drive it from a score bank, select
the two columns of interest, rename them to `gene1` and `gene2`, and pass
the real signature names as `feature_a_id` / `feature_b_id` in Python or
`--feature-a-id` / `--feature-b-id` on the CLI. Per-pair CSVs are already materialised for Xenium and the
benchmark corpus — no column selection is needed there.

For Xenium, `xenium/datasets.csv` identifies each input, its two features,
expression units and configuration. Values must not be normalized again:
the GBM marker means and breast gene measurements retain their recorded units.
The full per-cell support ledgers are results in Supplementary Data 3, not
copies of the input tables. Full polygon geometry is supplied for P28_post
and the breast illustration; the other samples retain complete statistics,
component summaries and cell eligibility, with geometry regenerable by TileScope.

Human and mouse score banks avoid repeating the same coordinates and
single-program values in every pair CSV. Pair inventories identify the two
columns needed for each run. Prepared inputs allow analysis reruns without
the original Seurat objects; changing signatures or normalization requires
those objects or the underlying expression matrices.

## Reproducibility limits

Figure builders read Input Data plus Supplementary Data; they do not rerun
TileScope or replace measured values. Use the core revisions recorded with
the datasets when reproducing analyses, rather than assuming that a newer
version is numerically identical.

Figure 1 is a schematic. Its optional synthetic drawing inputs are supplied
in `input_data/overview/`; it does not require a scientific analysis rerun.

The downloads are not copies of every local `CurrentResults/` directory,
report PDF, raw matrix or tissue image. The 30 published benchmark cases can
be regenerated entirely from Input Data: `benchmark/source_samples.csv`
locates the 13 required source geometries. Nine reuse the bundled human or
mouse score tables; four geometry-only tables are included separately.
Rebuilding biological inputs from raw data, or searching for new benchmark
substrates with `materialize --audit`, requires the original source data
and preparation workflows.

## Original data

Obtain raw data through the study deposits and accessions cited in the paper:

- Greenwald et al. (2024) and Ravi et al. (2022): human Visium glioblastoma.
- Meylan et al. (2026): Xenium glioblastoma, DOI 10.1016/j.cell.2025.12.055.
- Janesick et al. (2023): Xenium breast, DOI 10.1038/s41467-023-43458-x.
- Mouse Visium generated for this study: MCT1_CTL, MCT1_KO and V4_WT.
  **GEO accession: pending.**

## Availability status

The archives have been prepared locally; the Zenodo deposit and mouse GEO
accession are pending. The Greenwald test fixture retains an unresolved
redistribution-license note in its
[manifest](../tests/e2e/fixtures/greenwald_242T/fixture_manifest.yaml).

The benchmark registry identifies liver tumour samples CHC20, TS18 and
TS20 but does not record their definitive source citation or accession.
Their supplied ROI labels are not attributed to an unverified clustering procedure.
