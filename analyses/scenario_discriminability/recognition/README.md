# Full catalogue clustering

The primary analysis has **all 30 positive constructions, all ten modes**, not a
selection of successfully separated modes. Controls remain separate contrasts.
Figure 2 uses a separate 18-case illustration of the six expression modes.

## Fixed analysis choices

- Read the saved all-statistics extraction and verify its underlying data and
  output hashes. Differences from the recorded extractor revision are reported separately;
  it must not excuse any changed numerical input.
- Include every extracted expression-effect quantity for both declared ROIs and
  both interface directions (160 quantities in ten families). Retain an explicit
  disposition for every other exported quantity. Statistical significance,
  duplicate quartile endpoints, individual replicate copies, IDs and sample
  labels are not clustering variables.
- Recover bounded tessellation enrichment from the core's saved cross-support
  effect summaries for every case and channel, not only missing ones. These
  summaries include positive-over-zero saturation. Zero-over-zero stays missing.
  Log each change; also repeat the analysis with the original extraction.
- Standardize each quantity using its observed values across all 30 cases.
  Preserve missing entries. Exclude only nonvarying or insufficiently observed
  quantities, with their disposition exported.
- For each pair of cases, compare only quantities observed in both. Average
  squared standardized differences within each family × representation block,
  then across represented spatial representations in each family, then across
  available families; take the square root. Export the quantities/families used
  for every pair. No unavailable interface is an observed zero response.
- Use average linkage. Pairwise available-data dissimilarities need not be
  Euclidean, so Ward is not appropriate for this primary analysis. Use Ward on
  the fixed subset of quantities complete across **all 30** as a sensitivity
  analysis; it cannot test missing boundary quantities.
- Scenario names annotate results only. Record the smallest branch containing
  the three constructions of each mode, including all other cases in that
  branch. Do not select a tree cut, predictor subset, or linkage to maximize
  agreement with labels. This is clustering, not classifier fitting.
- Predefined sensitivities: complete linkage; equal quantity weights; family
  weights without representation balancing; range rather than SD scaling;
  each family omitted; abundance families omitted together; individual spatial
  representations; target ROI only; original enrichment extraction; and
  clustering restricted to complete quantities.
- Compare identical whole-ROI Pearson/Moran sets across representations as well;
  otherwise a representation comparison also changes which statistics are included.
- Separate diagnostics: availability-only clustering, and an additional
  clustering including the core's component-support summaries. These summaries
  are geometry/support, not expression effects or proof of enclosure. Counts
  use log1p; fractions remain unchanged. No new morphology descriptor is added.
- Quantify statistical redundancy and inspect which families separate modes,
  but do not use those labels to tune the primary tree. Report disagreement
  across sensitivities and missingness confounding, not only successful branches.

## Outputs

Run `python -m analyses.scenario_discriminability.recognition.full_catalogue_clustering`
to produce the 30-case data tables, full hierarchy, pairwise coverage,
branch and sensitivity audits, plots and a numerical summary. No Word
document is generated. The default output directory is
`analyses/scenario_discriminability/outputs/recognition/full_catalogue/`.
The analysis reads saved TileScope results; it does not regenerate inputs
or run TileScope. Figure builders consume these saved analysis tables.

Use `--output-dir` to choose another destination. Rebuild the saved runs and
derived tables after changing the generator, plan or core calculations before
interpreting new numerical results. Keep modes and declared comparison types
separate, and retain unavailable values rather than dropping cases.
