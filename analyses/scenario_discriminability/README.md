# Scenario statistic evaluation

Seven families, ten modes, 30 positive constructions and 27 matched controls.
Inputs/truth live in `TileScope-data/scenario_benchmark/corpus/`; saved core
runs and shared config live in `scenario_benchmark/CurrentResults/`.
Derived evaluation tables live here in `outputs/`.

```sh
python -m analyses.scenario_discriminability.recognition.all_statistics
python -m analyses.scenario_discriminability.recognition.full_catalogue_clustering
python -m analyses.figures.build benchmark
```

No command above runs TileScope. The extractor verifies all 57 saved roles
and extracts the 30 positive cases. Full-catalogue clustering covers ten modes
and 160 measurements, with missingness/linkage/ablation checks specified in the
[clustering specification](recognition/README.md). Main Figure 2 intentionally
uses a smaller 18-case/12-measurement display; that is not the supplementary scope.

The final command exports individual Figure 2/S1–S3 panels and source data,
plus **Supplementary Data 1**: all 57 simulated datasets, truth ledgers,
plan/index/config, atlas, distance/boundary controls and full-catalogue tables.
Observed geometries are not independent biological replications.

`build_atlas` is a separate core runner: changed inputs/settings/core can launch
substantial computation. Do not use it to inspect saved outputs. Dataset generation
is separately owned by `scenario_benchmark/materialize.py`.

Extraction, availability and clustering tests cover the current workflow.
See [data layout](../../docs/DATA_LAYOUT.md) and the
[figure-building guide](../figures/scripts/benchmark/README.md).
