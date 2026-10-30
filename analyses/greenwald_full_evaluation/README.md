# Shared GBM measurement helpers

Numerical readers, feature-order mapping, typed availability and
section-balanced frequencies for
[the current GBM analysis](../gbm_spatial_repertoire/README.md). Run
`python -m analyses.gbm_spatial_repertoire.build_measurements` from the
repository root; there is no separate entry point in this directory.

`context_matching.py` and `extract_context_annotations.R` route the
published author annotations to spot/cell identities for the ROI
context matching. The corrected
`greenwald_data_s1_classifications.csv` holds the audited author
label table. Author categories are context, not ground-truth spatial
classes.

`context_matching.py` validates barcodes, coordinates and ROI identity and
writes to `TileScope-data/heiland_tyrosh/analysis/context_matching/`.
The analysis uses the original unmasked ROIs: author annotations do not
change which spots enter an ROI's statistics.

`review_inputs/local_reviews.csv` and
`configuration_review_decisions.csv` record authored local assessments.
The builder checks their source/consensus PDF hashes. These records retain
ROI/component/interface identities, feature order, and distinctions between
unreviewed, candidate, geometry-only and compatible interpretations.
They are not an exhaustive detection or absence inventory.

Publication figures and their source data live under `../figures/`.
