# Mouse GBM — signature decisions

The final panel set has been run in the harmonized 222-pair corpus. This
record describes the candidate comparisons and reasons for selecting each
panel.

**Execution authority:** [inputs/signatures_harmonized.yaml](inputs/signatures_harmonized.yaml).
It contains the exact genes, intended samples, scoring settings and reviewed
workbook identity. Candidate YAMLs are comparison inputs, not approved defaults.
The [workflow README](README.md) describes marker retrieval, sample coverage
and the saved run-accounting tables.

## Shared definitions and sample-specific extensions

The rule is one common gene set per topic in CTL and KO, used unchanged in V4 when
measured, with a separately named V4 reactive-interface extension.

The biologist-selected CTL ROI 2/5 and KO ROI 4/9 contrasts remain the primary
definition contexts. V4 ROI 10/2/1 are additional, computationally annotated
contexts. A panel's discovery location does not restrict where it is evaluated.
All other ROIs remain evaluation contexts, not sources for tuning a desired
pairwise result.

## Evidence used to select panels

The Seurat marker generator supplies Phase-1/2 contrast tables. Enriched markers down to rank 1,000
in the focus views were combined with topic/process and GBM literature evidence.
The bounded review considered 58 gene-panel candidates and 49 score candidates,
including compact panels, justified extensions and reference panels.

Detection, target/comparison contrast, same-spot co-detection, within-ROI
Spearman coherence, component coverage, availability and leave-one-gene-out
robustness were separate evidence axes. Strong correlation was not required
when members repeatedly occurred in the same spots. Focal programs were not
discarded merely for being focal. There was no composite ranking score and no
selection to recover a preferred TileScope pairwise story.

## Decisions retained from the comparison

Exact final lists live in the YAML rather than a second editable gene table.

| Topic | Decision and reason |
|---|---|
| Cell-cycle/proliferation | Retain the established five genes; more leave-one-out robust than the compact Cdk1/Birc5/Top2a alternative. |
| Interferon-related | Retain the established five genes. Compact and Zbp1-extended alternatives were tested; novelty did not justify adding a gene without a necessary spatial contribution. This label does not identify a stimulus or an IFN subtype. |
| Inflammation | Retain complement plus chemokine genes rather than the compact chemokine-only panel. Ccl12 did not add enough support to justify the extension. |
| Myeloid/phagocytic | Use one four-gene common panel; the compact alternative improved some raw metrics but does not require a second sample-specific definition. |
| Antigen presentation | Retain a coherent four-gene MHC-II panel, distinct from generic myeloid identity; it describes an antigen-presentation state, not an exclusive cell label. |
| Cytotoxic T/NK | Retain the portable three-gene panel. Cd3d extension was compared but not selected. Prf1 is absent from the CTL object, so it cannot enter an unchanged CTL/KO panel. |
| ECM remodelling | Add Col11a1 to the four-gene candidate; co-detection and leave-one-out support improve. The same panel is used across samples. |
| Mesenchymal/stromal | Retain three genes; the compact alternative changes a focus contrast. Gfap would add host/reactive astroglial ambiguity rather than a clean common extension. |
| Innate immune | Keep focal Il1b with Casp4/Ripk2. The cleaner two-gene alternative narrows the biological question. The complete panel is not measured in V4. |
| Hypoxia/stress | Retain the four-gene MCT panel, interpreted as stress-related rather than a specific oxygen measurement. Slc16a3 is absent from KO. Trib3 alone in V4 is a comparator, not an equivalent multi-gene panel. |
| Endothelial/vascular | Use Cldn5/Vwf for endothelial identity. Adding focal Angpt2 is less stable and MCT-only; do not relabel the common pair as a measurement of angiogenesis. |
| Oligodendroglial/neural-like | Use the portable Olig1/Olig2 pair. Sox2 is absent from V4 and adds a different stem-like interpretation. |
| V4 reactive interface | Retain Pdpn/A2m/Serping1 as an additional V4 panel. Do not add S100a4, shared with the common mesenchymal panel, merely to strengthen correlation between the panels. Pdpn is absent from CTL/KO objects. |
| GBM tumour locator | Replace Cdkn2a/Tcim/Il11/Emp3 with Hmga2/Cd109/Cd44/Emp3. Minimum focus-context detection improved from 0.06 to 0.21 and mean pairwise co-detection from 0.44 to 0.60, with similar mean raw contrast (0.51). This remains a compartment locator, not a universal malignant-cell diagnosis. |

“Proliferation–immune interface” is a relationship to examine between programs,
not another panel. The locator's 35 pairs were run for completeness/reference;
they are not automatically biological program–program findings.

Single-gene comparators retained in the selection evidence are Mki67, Isg15,
Cxcl10, Tyrobp, Nkg7, Mmp2, Fn1, Casp4, Ddit4, Trib3, Cldn5, Olig2, Cd74, Pdpn
and Hmga2. These use direct LogNormalized expression, not AddModuleScore, and
are not extra frozen programs in the 222-run sweep.

## Scoring and interpretation checks

The final score is Seurat AddModuleScore on Spatial/data after LogNormalize
(10,000), nbin 24, ctrl 50 and seed 1234, using the full measured Spatial gene
pool for expression-matched controls. Score each panel once per sample and
reuse it; no clipping, post-score transformation, missing-gene imputation or
sample-specific shrinking. Required missing/invariant genes make a panel
unavailable. Do not compare absolute module-score magnitudes across sections.

Member-gene means and leave-one-gene-out results are diagnostic checks, not
replacement primary inputs. In V4 the common myeloid score agrees with the raw
member-gene mean within ROIs, but a between-ROI contrast can change direction
after control subtraction. Its ROI enrichment therefore needs the member-gene
check alongside the score. Sign agreement at near-zero correlation is not
evidence of a biological relationship.

The findings use the complete 222-run harmonized corpus. Source-score checks
alone do not establish equivalence at tessellation or consensus support.

## Where the evidence lives

The maintained generators are `phase4_harmonized_*`, `phase5_compare_signature_candidates.py`,
`phase5_candidate_panel_audit.R`, `phase5_score_candidate_panels.*` and
`phase5_select_harmonized_signatures.py`. The candidate YAMLs remain in `inputs/`.
Generated tables are external under
`TileScope-data/mouse/mouse_gbm_spatial_programs/`:

- `phase4_harmonized/`: retrieved genes and spatial diagnostics.
- `phase5_harmonized/`: first-pass panel/co-detection evidence (not final lists).
- `phase5_candidate_comparison/`: raw panel comparisons and gene-level evidence.
- `phase5_score_comparison/`: score/leave-one-out comparisons and `finding_traceability.csv`.
- `phase6_harmonized/`: frozen score banks, pair inputs and run accounting.

Gene-level evidence records the literature and orthology caveats. Mouse-human
mapping must not be assumed from capitalization alone; Ifi47, chemokines and
MHC-II families need particular care before human translation.
