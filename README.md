# TileScope-results

Analysis scripts, figure builders and tests for the [TileScope](https://github.com/cbib/TileScope)
paper. The scenario generator is independent of the core analysis engine.

## What's where, by paper figure

| Figures | Scripts under `analyses/figures/scripts/` | Numerical results |
|---|---|---|
| Figure 1: method overview | `overview/` | Supplementary Data 1 |
| Figure 2, S1–S3: benchmark and comparator methods | `benchmark/` | Supplementary Data 1 |
| Figure 3, S4–S6: human glioblastoma | `greenwald/` | Supplementary Data 2 |
| S7–S8: Xenium | `xenium/` | Supplementary Data 3 |
| Figure 4, S9–S10: mouse glioblastoma | `mouse_publication/` | Supplementary Data 4 |

Figure 2C comparator values are in Supplementary Data 1 at
`comparator_methods/comparator_values.csv`; the accompanying
`comparator_methods_schema.csv` defines the statistics and directions.

## Data download

**Zenodo: [10.5281/zenodo.23535235](https://doi.org/10.5281/zenodo.23535235)**

Data are not included in this Git repository. Five separate archives are
prepared for the deposit:

| Download | Contents | Extract into |
|---|---|---|
| `input_data.zip` | Prepared spot/cell values, coordinates, ROI labels, gene definitions and run settings | `TileScope-data/` |
| `Supplementary_Data_1.zip` | Benchmark, comparator and overview measurements | `TileScope-data/supplementary_data/` |
| `Supplementary_Data_2.zip` | Human GBM measurements and selected spatial geometry | `TileScope-data/supplementary_data/` |
| `Supplementary_Data_3.zip` | All 17 Xenium samples: measurements, coverage and illustrated geometry | `TileScope-data/supplementary_data/` |
| `Supplementary_Data_4.zip` | Mouse all-pairs measurements and signature-selection evidence | `TileScope-data/supplementary_data/` |

Each ZIP contains its named directory, a README and a relative file index with
checksums. Input data are stored once, not copied into each supplementary archive.

- **Read results:** download the relevant Supplementary Data archive; no code needed.
- **Redraw figures:** download Input Data and the relevant Supplementary Data archive,
  then use the matching code versions.
- **Rerun TileScope:** use the prepared inputs and recorded settings in Input Data.
  Reconstructing those inputs from raw expression requires the original-study data
  described in [Data access](docs/DATA_ACCESS.md). The CSV API (`run_tilescope_from_csv`)
  and CLI take one pair per input file, with columns `x, y, gene1, gene2, roi_id`.
  The human and mouse score banks in `input_data/{greenwald,mouse}/scores/<sample>.csv.gz`
  carry one row per spot and one column per scored signature; to drive core, select
  the two columns of interest, rename them to `gene1` and `gene2`, and pass the real
  signature names as `feature_a_id` / `feature_b_id` in Python or
  `--feature-a-id` / `--feature-b-id` on the CLI. Per-pair CSVs are already
  materialised for Xenium (`input_data/xenium/cells/<Study>/<sample>/<pair>.csv.gz`)
  and benchmark (`input_data/benchmark/corpus/<scenario>/<case>/<role>.csv`).
  The human and mouse [runners](docs/DATA_LAYOUT.md#analysis-reruns) perform
  this selection from `--input-dir` and create results under `--output-dir`;
  no pre-existing `CurrentResults/` directory is needed.

Figure 1 is a schematic method illustration. Its synthetic input and the
retained plotting measurements under `Supplementary_Data_1/overview/` are
sufficient to re-render the published panels; a complete original TileScope
run of this synthetic input is not required.

## Install

```bash
git clone https://github.com/cbib/TileScope-results.git
cd TileScope-results
python -m venv venv
source venv/bin/activate
python -m pip install -e '.[analysis,figures,test]'
python -m pip install -e '../TileScope/python[test]'
```

The last command assumes the [TileScope core](https://github.com/cbib/TileScope)
checkout is beside this repository.

## Where to put the data

```text
Projects/
  TileScope/
  TileScope-results/               # code; generated artwork in analyses/figures/
  TileScope-data/
    input_data/
    supplementary_data/
      Supplementary_Data_1/
      Supplementary_Data_2/
      Supplementary_Data_3/
      Supplementary_Data_4/
```

Set `TILESCOPE_DATA_DIR` or `TILESCOPE_CORE_DIR` for different locations.
No machine-specific absolute paths are required. Downloaded data never need
to be placed inside TileScope-results.

```bash
python -m tilescope_results.bundles all
python -m analyses.figures.build all
# Or one group: overview, benchmark, greenwald, mouse, xenium
```

Builders read the bundles without changing them. Individual SVG/PDF panels,
PNG previews and figure-specific CSV/Excel source data go to
`analyses/figures/`. Set `TILESCOPE_FIGURE_DIR` to use a separate output directory.

## Tests and documentation

```bash
python -m pytest tests/scenario_benchmark
python -m pytest analyses/figures -q
python -m pytest tests/e2e --run-real-data
```

Figure tests need the downloaded bundles. The optional integration test uses
the small 242_T fixture.

- [Data access](docs/DATA_ACCESS.md): scope, source studies and deposit status.
- [Data layout](docs/DATA_LAYOUT.md): paths and reproduction commands.
- [Figure generation](analyses/figures/README.md): individual panels and source data.
- [Scenario benchmark](scenario_benchmark/README.md) and [contract](scenario_benchmark/CONTRACT.md).
