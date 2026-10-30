# Human GBM: local program organisation

Manuscript scope: **30 IDH-wildtype sections (12 Greenwald, 18 Ravi), 192
BayesSpace ROIs, 14 published programs and all 91 pairs**. Sections are not
automatically independent patients. Six other glioma sections remain in the
input/run corpus but are excluded from these GBM summaries.

Shared enrichment, within-ROI variation and boundary localisation need not
coincide. Figure 3 uses 259_T, 262_T and 260_T to show these distinctions.
Descriptive frequency cutoffs are not biological classifiers; consensus and
source measurements are not independent replications.

## One current numerical path

From the repository root, with saved runs available:

```sh
python -m analyses.gbm_spatial_repertoire.build_measurements
python analyses/gbm_spatial_repertoire/vet_shortlist.py
python analyses/gbm_spatial_repertoire/vet_shortlist.py --record
python -m analyses.figures.build greenwald
```

These commands read saved runs, not execute TileScope. The first rebuilds
ROI/component/boundary evidence, local frequencies and verified shared source
fields; `--output-root /tmp/gbm-check` permits an isolated parity build.
The shortlist prepares the declared seven-pair views. `--record` attaches
existing authored observations, not new visual judgments. The final command
writes individual Figure 3/S4–S6 panels and Supplementary Data 2.

The numerical builder reuses statistical
helpers in `../greenwald_full_evaluation/` without redefining estimates.
[Input preparation and core runs](../greenwald_full/README.md) are separate.

## Locations and identities

- `inputs/{sample_scope,pair_scope,roi_context}.csv`: frozen cohort, physical
  feature order and author-context annotation.
- `TileScope-data/heiland_tyrosh/CurrentResults/tilescope_run_config.json`:
  shared scientific config; current core runs remain beside it.
- `.../heiland_tyrosh/analysis/scenario_cohort/`: ROI/component/boundary
  evidence, without assigning scenario classes.
- `.../analysis/two_level_evidence/`: local property records and exact cutoffs.
- `.../analysis/gbm_spatial_repertoire/`: verified shared fields, frequencies
  and shortlist evidence.
- `../figures/`: manuscript panels, source data and full supplements.
  Manual `assembly/` contents are never regenerated.

Data 2 retains all 17,472 section–pair–ROI rows plus component-distance,
ordered-boundary and support-comparison tables. Units and unavailable results
stay explicit. Section-balanced frequencies differ from pooled-ROI frequencies.

## Selection record

[FINDINGS.md](FINDINGS.md) summarises the current examples. The shortlist plan,
`shortlist_vetting/observations.json` and `batch_01_observations.json`
retain the authored exploratory assessments;
`review_witnesses/` preserves the exact boundary image cited in that record.
Rebuilding numbers does not repeat the visual assessment.

Shared-field preparation checks all 2,730 input CSVs and all 13 partner copies
of each section/program. Marginal values are reused only after equality checks;
partner-dependent status messages remain explicit. Pairwise associations,
joint enrichment, boundary orientations and component identities stay separate.

## Frequency definitions

Each section contributes its fraction of evaluable local units, then these
fractions are averaged with equal section weights. ROI statistics use ROIs;
distance summaries use components, not individual shells. Unavailable values
are excluded from the relevant denominator, never counted as zero.
`mean` is section-balanced, `pooled` pools local units, and `coverage`
averages within-section evaluable fractions.

The reference descriptive bands are Pearson 0.20 and Moran/distance 0.10;
lower and higher settings remain in the exported tables. They are not
significance thresholds. Positive and negative fractions stay separate.
Boundary responses remain continuous and retain ordered interfaces; repeated
feature measurements in different pair exports are not independent.

See the [figure builder](../figures/scripts/greenwald/README.md) for Figure 3,
S4–S6 and Supplementary Data 2.
