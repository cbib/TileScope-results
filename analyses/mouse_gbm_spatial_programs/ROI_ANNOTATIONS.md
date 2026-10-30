# ROI annotations and supporting markers

Exploratory ROI annotations. The labels were accepted for
analysis, not independently validated by biologists. The observations below
do not establish tumour identity or invasion on their own.

This record explains the marker and literature evidence for five ROIs in
`inputs/roi_roles.csv`. Twenty-seven ROIs are marked `reviewed_for_analysis`;
CTL ROI 4 remains `reviewed_uncertain`.

## ROI assignments

- **`MCT1_CTL` / ROI 4 → `uncertain`**.
  Low-RNA, highly fragmented oligodendroglial/neuronal territory
  with no convincing tumour identity. Excluded from signature
  construction; retained as an ROI in every TileScope input.

- **`MCT1_KO` / ROI 9 → `highlighted_periphery`**. Tumour-bearing proliferative / IFN /
  ECM-rich rim with substantial cytotoxic and other immune
  admixture — not a single generic "immune" program.

- **`V4_WT` / ROI 1 → `intratumour_subregion`**. Proliferative malignant foci
  associated with ROI 10. Marker expression does not distinguish nested,
  juxtaposed or invasive geometry.

- **`V4_WT` / ROI 2 → `other_periphery`**. Reactive glial / immune tumour–brain
  interface containing substantial host white-matter signal.

- **`V4_WT` / ROI 10 → `tumour_core`**. Broad
  malignant ECM / IFN-active tumour territory.

`intratumour_subregion` is used
for a proliferative or otherwise distinguishable sub-region that
sits inside a broad tumour ROI and shares its identity but not its
scale.

## Per-ROI evidence

### `MCT1_CTL` / ROI 4 — uncertain identity

Nineteen disconnected components. Median UMI count per spot ≈ 1029,
against 31 271 in CTL 2 (`tumour_core`) and 5775 in CTL 5
(`highlighted_periphery`). All 164 nominal positive-logFC markers
are detected in a smaller fraction of spots than in the reference
ROIs. `Cdkn2a` occurs in 1.6% of CTL 4 spots and `Hmga2` in 0.8%,
whereas oligodendroglial markers `Mog` (56%) and `Opalin` (51%)
dominate detection. Calling this territory invasive tumour would be
manufactured from sparse high-intensity genes such as
`Psat1` / `Phgdh` rather than supported tumour identity. Retaining
it as an ROI is appropriate; using it as an invasion-signature
source is not.

### `MCT1_KO` / ROI 9 — tumour-associated and immune expression

Tumour-associated and proliferation markers include `Cdkn2a` 91.5%, `Birc5` 96.8%, and
`Cdk1` 97.6% of spots. Immune-associated genes include `Nkg7` 60.7%, `Cd3d` 46.4%,
`Gzmb` 51.5%, alongside several mouse granzymes, `Mcpt1` / `Mcpt2`,
`Igkc`, and `Tnfrsf13c`. The dominant interpretation is cytotoxic /
granulated lymphoid infiltration, potentially with smaller mast-
and B-cell components — not one generic "immune" program.

`Nkg7` / `Prf1` / `Cd3d` and `Cpa3` / `Kit` / `Mcpt` represent different
candidate cell populations. Murine `Gzmd` can be mast-cell-associated, and mast
cells have been observed within and around experimental gliomas.

- Granzyme D: https://pmc.ncbi.nlm.nih.gov/articles/PMC3676009/
- Mouse glioma mast cells: https://pmc.ncbi.nlm.nih.gov/articles/PMC3176317/

### `V4_WT` / ROI 1 — proliferative subregion, not "an invasive ROI"

V4/1 and V4/10 share 47 of their top-100 one-vs-rest markers,
including `Cdk1`, `Birc5`, `Mki67`, `Cdkn2a`, `Emp3`, and `Cd109`.
V4/1 is therefore a proliferative subregion of the broad V4 tumour,
not currently an "invasive ROI". `EMP3` and `CD109` are
independently associated with aggressive GBM states, but neither
tells us whether these foci are nested inside V4/10 or invading
away from it — that is a spatial question for TileScope, not a
marker-list question.

- EMP3 in GBM: https://pmc.ncbi.nlm.nih.gov/articles/PMC10792875/
- CD109 edge-to-core: https://pmc.ncbi.nlm.nih.gov/articles/PMC7764499/

### `V4_WT` / ROI 2 — reactive tumour–brain interface

Marker composition is strongly glial / reactive: `Aqp4` 87.8%, `Mog`
73.5%, `Opalin` 73.1%, `Ly86` 84.1%, with `Serping1`, `A2m`, `Cfh`,
`Ccl2`, `Pdpn`, and `S100a4`. Recent spatial work places
`AQP4` / `PDPN` reactive astrocytes preferentially at glioma
borders, while `S100A4` can mark immunosuppressive myeloid and
T-cell states. This supports "reactive tumour–brain interface",
not a pure tumour-cell periphery.

- Tumour-associated astrocytes: https://pmc.ncbi.nlm.nih.gov/articles/PMC12539703/
- `S100A4` single-cell study: https://www.nature.com/articles/s41467-022-28372-y

### `V4_WT` / ROI 10 — broad malignant territory

Broad ECM-remodelling, interferon-active tumour
identity supported by `Hmga2`, `Mcam`, `Cdkn2a`, `Cdk1`, `Top2a`,
`Col11a1`, `Tgfbi`, `Rsad2`, and `Gbp7`.

## Broader observation: CTL–KO axis

Top-200 one-vs-rest markers for the two cores (`MCT1_CTL` / 2 vs
`MCT1_KO` / 4) overlap by 130 genes — Jaccard ≈ 0.48. The two
peripheries (`MCT1_CTL` / 5 vs `MCT1_KO` / 9) share only 33 genes —
Jaccard ≈ 0.09.

Productive hypothesis (**descriptive, not causal**):

> The shared malignant core programme is comparatively conserved
> between MCT1 CTL and MCT1 KO, while MCT1 depletion is associated
> with a markedly different tumour–host boundary ecosystem.

This is a hypothesis about compartment ecosystems, not proof of an
MCT1 effect. It gives TileScope a well-scoped question: whether
frozen, shared malignant programmes change compartment enrichment,
within-ROI geometry, boundary response, and cross-grain behaviour
between the two conditions.

## Use in marker selection

CTL ROI 4 is excluded from signature construction but retained in TileScope
inputs. The seven marker contrasts use CTL 2/5, KO 4/9 and V4 10/1/2, including
CTL–KO core and periphery comparisons. See [the workflow README](README.md#inputs-and-marker-retrieval) for
their exact definitions. Spatial separation alone is not labelled invasion.

## Experimental provenance

`inputs/samples.csv` identifies the input objects and ROIs. The computational
annotations do not establish the manipulated cell type, experimental model or
perturbation protocol; those require the study's experimental records.

## Limits of these annotations

Everything under this file is reviewed for exploratory analysis
only. Independent biologist confirmation of the role calls above —
especially `MCT1_CTL` / 4 as `uncertain`, `MCT1_KO` / 9 as
immunologically composite, and `V4_WT` / 1 as an intratumour
subregion rather than invasion — is not established by this review.
