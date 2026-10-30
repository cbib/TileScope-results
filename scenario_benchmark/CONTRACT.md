# Scenario generator contract

## Authorities

1. `SampleScaffold` owns source spots, fixed ROI labels, components, and the
   observed adjacency graph.
2. Scenario modules own biological eligibility, fields, roles, and role
   feasibility.
3. `registry.py` owns `geometry_source`, per-field morphology, and ROI topology.
4. `benchmark_plan.json` selects cases and numerical parameters. It does not
   repeat live role contracts.
5. Generated case manifests record resolved contracts and provenance.

No scenario benchmark module may import `tilescope.*`.

## Field composition

Every role field is exactly:

```
role_field = frozen_background + role_overlay
```

Scenario overlays never redraw the background. Background draws are
deterministic and byte-identical across roles wherever the declared invariant
requires it. A control may change a signal marginal only when its scientific
meaning explicitly requires that change; role names and comparison metadata
must state this.

## Geometry and biology

- ROI labels are fixed upstream from broad transcriptomic clustering.
- Field scenarios alter expression only; they never relabel spots.
- Topology scenarios select observed B-labelled components. TileScope support
  classes are diagnostics and never biological eligibility criteria.
- A is elevated on A-labelled host spots in topology scenarios and is not
  forced onto B-labelled spots.
- Infeasibility is typed. A builder must not silently change support, roles, or
  estimand to make a case pass.

## Scenario definitions

These are specified arrangements, not mutually exclusive diagnoses of real
tissue. The seven families contain ten modes:

| Family | Positive arrangement | Comparison role |
|---|---|---|
| `broad_colocalization` | Broad A and B with shared local variation in one ROI | `b_independent_background` removes B signal; it does not preserve its marginal distribution |
| `focal_pattern` | Connected A and B foci, either colocated or separated | Both modes are positives; independent B background is a diagnostic baseline |
| `spatial_gradient` | Concordant or opposed ramps along one physical axis | `paired_randomized` permutes the B **overlay** within the ROI, preserving its values but not its ramp |
| `nested_foci` | Broad A, basal B and focal B increments within an A-labelled ROI | `b_foci_relocated_outside_a` moves the same focus sizes and values to the B support ROI; A and basal B stay fixed |
| `embedded_domains` | One to three observed B-labelled domains enclosed by A | Positive only; each participating domain has at least five spots |
| `dispersed_infiltration` | Many disconnected B-labelled components distributed through A | Positive only; geometry gates below |
| `contact_interface` | B increases near an A–B contact, or A extends across it | Mechanism-specific ablations; conditional matched relocation to another boundary |

Broad colocalization uses a shared random term independent of both nuisance
backgrounds; neither gene is generated from the other. Focal modes use matched
focus sizes and counts. Gradient randomization does not permute the frozen
background. In nested foci, the B-labelled ROI supplies relocation support,
not a second positive compartment.

Dispersed infiltration requires at least five enclosed B components and 12
enclosed B spots, with no component exceeding 40% of that population. Mean
component-centroid distance must reach 25% of the A-host bounding-box diagonal.
These are biological geometry gates, independent of TileScope eligibility.
All B-labelled spots receive the population baseline; A is elevated only on
the A-labelled host. The three selected cases have scattered TileScope support,
which limits corpus coverage but does not define the scenario.

Enclosure is conservative: external neighbours of a B component must all
belong to A. Contact with another label excludes enclosure; insufficient
observed neighbours can leave it indeterminate. These counts are not estimates
of biological prevalence.

For contact interfaces, all observed contact fragments contribute:

- `responder_activation`: B increases on the B side; `no_interface_increment`
  removes that increment. Where feasible, `unrelated_boundary` relocates matched
  increment mass to a third-ROI boundary.
- `anchor_fade` (displayed as **Anchor extension**): A extends into B through a
  two-hop decaying band; `no_anchor_extension` removes only that extension.

Adjacent broad territories without a boundary term already occur in these
ablations; there is no separate `territorial_exclusion` generator.

## Comparison types

- `primary_null`: absence of the defining B signal under frozen A/background.
- `diagnostic_baseline`: useful generic negative, not the main matched contrast.
- `mode_ablation`: the biological compartments remain; one mechanism is
  removed. It is a weakened positive, not a null.
- `matched_relocation` or `specificity_control`: signal mass is retained while
  its spatial relation changes.

Topology-only archetypes have no paired null. Their reference is the observed
labelled geometry, not an expression-based negative control.

## Determinism and reuse

- RNG streams are named and derived from stable case/sample identifiers.
- The materializer loads a sample scaffold and computes its source hash once
  per run, then reuses both across cases.
- Production materialization runs each generator once. The optional
  `--verify-determinism` audit repeats generation and compares outputs.
- Shared connected-patch and component conversion logic belongs in
  `spatial_primitives.py`; enclosure classification belongs in `enclosure.py`.

## Outputs

Each case emits one manifest, one combined truth ledger, and one TileScope
input per feasible role. Manifests include generator/source provenance,
resolved background policy, registry metadata, and the resolved live role
contract. Generated artefacts must be rebuilt after a plan or generator change.
