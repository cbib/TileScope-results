# Benchmark figures

Build Figure 2 and Supplementary Figures S1–S3 from
`TileScope-data/input_data/benchmark/` and
`TileScope-data/supplementary_data/Supplementary_Data_1/`:

```bash
python -m analyses.figures.build benchmark
python -m pytest analyses/figures/scripts/benchmark -q
```

This does not generate datasets or rerun TileScope. See the
[data layout](../../../../docs/DATA_LAYOUT.md), [scenario contract](../../../../scenario_benchmark/CONTRACT.md)
and [clustering specification](../../../scenario_discriminability/recognition/README.md).
The [figure workspace](../../README.md) describes individual SVG/PDF/PNG panels
and their CSV/Excel source data. No assembled manuscript figure is imposed.

## Catalogue maps

Figure 2A contains separate A and B maps for three examples:

- Opposed gradients: 242_T ROI 3, framed on its largest component.
- Embedded domain: the 15-spot ROI 2 component inside 260_T ROI 5. The view
  retains all 782 ROI 2 spots, including its larger 412-spot component outside
  the host, rather than displaying the inclusion alone.
- Boundary response: CHC20 ROI 4 / ROI 3 contact, with additional B expression
  on the ROI 3 side.

S1 provides twenty maps covering all ten modes. Controls and biological modes
remain distinct: separated foci and opposed gradients are positives, interface
ablations are not nulls, and topology scenarios have no paired null.

Input values and labels are checked against the truth ledger, with source,
manifest and ledger hashes verified against the corpus index. Coordinates and
values are unchanged; crops alter only the viewport. Other ROIs appear grey.
There is no smoothing or downsampling. View widths are at least 50 median
spot spacings, not a shared physical scale across samples.

Colour limits use the full-input minimum and maximum across both genes and
contrasting modes within a family. Limits differ between families. Outlines are
display-only unions of source-spot Voronoi footprints, clipped within 0.65 median
nearest-neighbour spacings; they do not define statistical support. Source
tables record bounds, colour limits, original values and display membership.

## Quantitative panels

**Figure 2B0** clusters 18 positive cases from six expression modes using twelve
spot-level measurements: four enrichments (A, B, joint-high, joint-low),
whole-ROI and largest-component Pearson correlations, both Moran directions,
and four component-distance coefficients. All twelve are retained, grouped
into five measurement families. This illustration is not the full-catalogue
analysis, which includes 30 cases and 160 measurements (S2).

Enrichment is bounded as `E/(1+E)` for clustering and displayed as
`(E-1)/(E+1)`. Positive-over-zero enrichment reaches the upper limit;
zero-over-zero remains unavailable. Variables are standardized with population
SD; nonvarying variables are excluded with an explicit disposition. Family
weighting prevents a family with more variables from dominating. B0 uses the
full-catalogue implementation of average linkage and optimal leaf ordering;
mode labels annotate the result, not the clustering inputs. These are
descriptive clusters, not classifier accuracy estimates.

**Figure 2B1** compares nine positive cases and their controls across three
modes. It uses the same largest component for the positive and its control.
Same-spot Pearson correlation is distinct from the equal-anchor
cross-correlations across four distance bands: 0.5–1.5, 1.5–2.5, 2.5–4.5 and
4.5–8.5 times the component's median nearest-neighbour spacing.

**S3** shows boundary measurements and their matched controls. Near and far
groups are inside the responder ROI: source-graph hops 0–1 are near, hop 2 is
an excluded buffer, and hops ≥3 are far. Unreachable spots do not contribute.
The measurement is mean near minus mean far, not a difference between ROI
means. Figures read the saved eligibility and unavailable statuses.

The matched-representation diagnostic remains in `additional/Benchmark/`;
it is not a numbered manuscript figure. Clustering sensitivities, coverage and
all measurement dispositions are exported with the full catalogue.

## Comparator panel

Figure 2C reads
`comparator_methods/comparator_values.csv` in Supplementary Data 1 for the
same 18 positive cases, using B0's recomputed row order. The accompanying
`comparator_methods_schema.csv` defines each statistic. Figure generation
reads these files without rewriting the bundle.

The panel shows Pearson
and Spearman references, MERINGUE SCC, Voyager Lee's L, SpatialCorr derived
correlation and LLR, STopover J_comp, and SpaceBF Gaussian global coefficients
in both directions. Shared weights are used for MERINGUE/Voyager, and the
shared-edge arm for SpaceBF.

Each column is standardized over its finite values and displayed within ±2 SD;
columns are grouped by method, not clustered. Hatched STopover cells indicate
cases outside its one-sided scope. The external comparator input is labelled
provisional; the display is not a method-accuracy ranking or evidence that
coefficient magnitudes are interchangeable between methods.

## Saved outputs

Builders write separate panels and source tables under `main/Figure2/` and
`supplementary/FigureS1/` through `FigureS3/`. Supplementary Data 1 includes
the statistics atlas, control comparisons and clustering tables. The 57 role
datasets, truth ledgers, construction plan and run configuration are stored
once in Input Data, not copied into the supplementary results. Per-file
provenance records the inputs and generated outputs.
