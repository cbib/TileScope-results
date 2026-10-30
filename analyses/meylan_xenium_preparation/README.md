# Meylan Xenium: complete-cohort feasibility analysis

Scope: **all sixteen deposited samples**, 2,401,946 cells from eight paired
patients, stem-like/interferon marker means and ten tessellations.
The marker lists and regional question follow Figure 4G of Meylan et al.;
their P28 example is in Figures 4F and S4E. Definitions are frozen in
`pair.json`. Our P28_post illustration is Supplementary Figure S7;
complete numerical results are under
`TileScope-data/meylan_xenium/analysis/cohort/`.

Paper: Meylan et al., *Cell* (2026), DOI
<https://doi.org/10.1016/j.cell.2025.12.055>.
Data: [GSE296577](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE296577).
Author code: <https://github.com/maximemeylan/GBM_CAN3110>.

## Cohort question and interpretation

Compare stem-like enrichment in lymphoid aggregates versus tumour–T cell
interfaces, and interferon enrichment in the reverse comparison. Use the
complete parent ROIs, not only the components illustrated in S7, at cells and
ensemble consensus separately. Report ROI presence, size and tessellation
coverage alongside each comparison.

The cohort tables count each direction among samples evaluable for that
comparison, and whether both contrasts occur in the same sample. Missing ROIs
are distinct from unavailable measurements or an evaluable opposite result.
Pre- and post-treatment summaries remain separate; sixteen samples are not
sixteen independent patients. Scores include all cell types, unlike the
original paper's T-cell-only comparison. This is not a treatment-efficacy test.
Other statistics, ordered boundaries and tessellation variation remain in
the full numerical inventory.

### P28_post example

| Whole ROI | Stem-like enrichment, cells / ensemble | Interferon enrichment, cells / ensemble | Cells in tessellation / total |
|---|---:|---:|---:|
| Lymphoid aggregate | 3.583 / 4.770 | 0.845 / 0.868 | 965 / 2,002 |
| Tumour–T cell interface | 2.152 / 2.763 | 1.811 / 2.041 | 15,416 / 21,865 |
| Tumour | 0.338 / 0.293 | 0.743 / 0.635 | 57,162 / 61,638 |

These are high-expression prevalence ratios relative to the other labelled
ROIs. On eligible cells alone, stem-like enrichments are 4.753 and 2.754 in
the lymphoid/interface ROIs, close to ensemble values. Selective representation
therefore explains much of the magnitude change after aggregation.

Of 97,488 cells, 79,616 (81.67%) contribute to every tessellation; all cells
remain in the measured input. Coverage is 48.2% in lymphoid aggregates,
70.5% in interface and 92.7% in tumour; the 202 vasculature cells have no
eligible component. Illustrated interface component 784 contains 6,892 cells,
6,554 eligible (95.1%): this is not the entire interface ROI's coverage.
The parent interface Pearson correlation changes from 0.0058 at cells to
0.161 in ensemble consensus, rather than remaining invariant.

## Annotation recovery

The processed object lacks final neighbourhood labels but retains the T-cell
expression graph needed by the authors' correction procedure (`RNA_snn_T`).
Their released correction and neighbourhood recipes were inspected alongside
the STAR Methods and supplementary figures. Reconstruction used cohort-wide
cell-composition clustering, as in the original procedure.

1. `extract_sample.R`: inspect remaining Seurat slots and extract the complete
   selected sample without changing expression values.
2. `audit_correction.R`: identify cells affected by the published correction
   and preserve its literal indexing for inspection.
3. `reconstruct_neighbourhoods.R`: form 20-neighbour cell-type counts, then
   use the released 18-cluster, seed-35, 25-start recipe. P28 counts are checked
   exactly against the authors' table/melt implementation.
4. `validate_mapping.py`: identify cluster-number permutations using the
   cell compositions in Fig. S4D. All **360 rounded percentages** (18 clusters
   × 20 cell types) match; every match is the unique nearest profile. The
   maximum absolute difference from a displayed integer is 0.4996 percentage
   points. The reference table was visually transcribed and ambiguous digits
   rechecked at higher resolution against the PDF, not adjusted to fitted values.
5. Apply the authors' explicit mapping from those clusters to seven named
   neighbourhoods in every deposited sample. A sample need not contain all
   seven; no missing neighbourhood is invented.

The final winning k-means solution converged (`ifault=0`, 16 iterations).
Three non-winning starts reached the Quick-TRANSfer limit; their warnings are
not evidence that the retained fit failed. Published text/legend inconsistencies
in cluster count are resolved here by the actual 18-column Fig. S4D and the
released 18-cluster code.

These are **reconstructed, publication-matched regions**, not a newly located
deposit of final memberships. Rounded profile agreement and map review do not
prove cell-for-cell identity with an unavailable original assignment table.
No neighbourhood was named or changed based on the selected marker scores or TileScope results.

### Correction-code caveat

The literal correction affects two cells, both in P34_post; none in P28_post.
The released function indexes a column-compressed sparse matrix as though its
pointers enumerated rows. Its two chosen replacement cells have no edge to the
suspected cells in the deposited expression graph. This behaviour was retained
and recorded rather than silently repaired. The subsequent full profile match
supports this *published-workflow reconstruction*, not correctness of that
individual correction. Do not describe the correction as independently
validated or the recovered memberships as an exact deposited ground truth.

## Expression and configuration

`prepare_cohort.R` uses the shared `prepare.R` function to retain all cells,
their original coordinates and the deposited
normalized values of the listed genes, verified against
`log1p(100 * counts / cell_total)`. Each input feature is their arithmetic
mean per cell, including zeros. Individual gene values are also saved so
the means can be independently checked.
No crop, positive-cell filter, expression imputation or normalization after
selection is performed. Each ROI's enrichment reference is the other labelled
neighbourhoods in this complete sample.

`run.config.json` uses the existing Xenium 50/15/9 component/target/minimum tile
settings, hexagon sizing target 12 and coverage fraction 2/3. Ten tessellations,
99 permutations and seed 42 are used. No core settings were relaxed to make
this specimen appear more suitable. The core revision and worktree status are
recorded as provenance; they do not restrict which checkout can run.

`--resume` reuses a completed run of the same sample and program pair. It does
not establish that unchanged sample names imply unchanged inputs. Use a new
output directory when changing data or settings. Numerical validation checks
cell identities, ROI membership, enrichment and correlation against the inputs.

## Storage and commands

Code remains here; data, reconstruction intermediates and TileScope outputs
are under `TileScope-data/meylan_xenium/`. The data/core locations follow
`TILESCOPE_DATA_DIR` and `TILESCOPE_CORE_DIR`, or sibling checkouts by default.
The complete extracted P28 object is `preparation/sample/P28_post.rds`;
validated cell labels are in `preparation/validated/`.
Publication figures and source tables are generated by
`analyses/figures/scripts/xenium/`. The current complete numerical package
is Supplementary Data 3. Publication and delivery never modify DOCX/PPTX files.

```sh
Rscript analyses/meylan_xenium_preparation/extract_sample.R FULL_RDS P28_post EXTRACT_DIR
Rscript analyses/meylan_xenium_preparation/audit_correction.R FULL_RDS AUTHOR_TCELL_RMD AUDIT_DIR
Rscript analyses/meylan_xenium_preparation/reconstruct_neighbourhoods.R AUDIT_DIR/correction_inputs.rds AUDIT_DIR/literal_correction_audit.csv RECONSTRUCTION_DIR
python -m analyses.meylan_xenium_preparation.validate_mapping --reconstruction RECONSTRUCTION_DIR --output VALIDATED_DIR
Rscript analyses/meylan_xenium_preparation/prepare.R EXTRACT_DIR/P28_post.rds VALIDATED_DIR/P28_post_annotations.csv.gz STUDY_DIR/tilescope_inputs/Stemlike_Interferon analyses/meylan_xenium_preparation/pair.json
python -m analyses.meylan_xenium_preparation.run --study-dir STUDY_DIR --validate-only
python -m analyses.meylan_xenium_preparation.run --study-dir STUDY_DIR
python -m analyses.meylan_xenium_preparation.analyse --study-dir STUDY_DIR
```

After annotation reconstruction, the complete cohort workflow is:

```sh
python -m analyses.meylan_xenium_preparation.batch --study-dir STUDY_DIR --background
```

Use `--skip-prepare` only after input preparation has completed. Samples run
sequentially, smallest first. Completed runs of the same sample and pair are
reused and numerically checked against current inputs. Use a new output directory
when changing data or settings. Core revision does not gate reuse. The batch stops on an
error; logs and `analysis/cohort/batch_status.json` identify its stage. A resume
revalidates completed samples. Additional runs produce all core numerical
tables without PDF rendering. The batch validates all sixteen runs and aggregates their cohort results.
It does not overwrite the deposited Input Data or Supplementary Data 3.
Figure publication reads those frozen bundles; updating a deposit after new
analyses is a separate operation. See [the figure guide](../figures/README.md).

For reconstruction, use empty output directories. The runner refuses to
overwrite an incompatible or partial run. All paths are supplied through arguments or the
standard project-root configuration, not personal absolute paths.

## Numerical regression checks

`performance_check.py` runs the numerical pipeline in a specified core checkout
and compares exported CSV contents directly, without rounding. It also records
elapsed time and peak memory. Use separate output directories for each run:

```sh
python -m analyses.meylan_xenium_preparation.performance_check run --core CORE_CHECKOUT --output NEW_OUTPUT --sample P34_pre
python -m analyses.meylan_xenium_preparation.performance_check compare --before BASELINE_OUTPUT --after NEW_OUTPUT
```

These checks do not replace the cohort validation or overwrite saved runs.
