# Data layout and run commands

Download instructions and the Zenodo DOI placeholder are in the
[root README](../README.md#data-download--zenodo-deposit-pending).
The deposit is pending.

## Directory layout

Extract `input_data.zip` into `TileScope-data/` and each supplementary ZIP
into `TileScope-data/supplementary_data/`. Each archive includes its own
named top-level folder.

```text
TileScope-data/
  input_data/
    overview/
    benchmark/
    greenwald/
    mouse/
    xenium/
    file_index.csv
  supplementary_data/
    Supplementary_Data_1/
    Supplementary_Data_2/
    Supplementary_Data_3/
    Supplementary_Data_4/
```

All downloaded numerical data stay here. No archive needs to populate or
overwrite the code checkout. Working directories for new analyses, such as
`mouse/CurrentResults/` or `scenario_benchmark/corpus/`, are separate from
the frozen downloadable bundles and are not needed to redraw the paper figures.

## Path configuration

Paths are resolved by `tilescope_results.paths`, relative to the checkouts
unless explicitly configured:

| Variable | Purpose | Default |
|---|---|---|
| `TILESCOPE_DATA_DIR` | Input Data and Supplementary Data root | sibling `TileScope-data/` |
| `TILESCOPE_CORE_DIR` | Core checkout | sibling `TileScope/` |
| `TILESCOPE_FIGURE_DIR` | Generated panels and figure source tables | `analyses/figures/` in this repo |

```sh
python -m tilescope_results.paths
```

Set overrides before starting the Python process. References in bundle indexes
are relative paths; no personal absolute path is required.

## Rebuild panels

Install the matching core and results code in one Python environment:

```sh
python -m pip install -e '.[analysis,figures,test]'
python -m pip install -e '../TileScope/python[test]'
python -m tilescope_results.bundles all
python -m analyses.figures.build all
```

Choose `overview`, `benchmark`, `greenwald`, `mouse` or `xenium` instead of
`all` for a single group. Builders verify the relevant indexes and read the
deposited inputs/results directly. They do not invoke upstream analyses,
create Word documents or modify Supplementary Data.

Panel CSVs and Excel workbooks are generated exports for figure assembly;
they are not an additional downloadable dataset needed for reproduction.
The verification command only checks downloaded files; it does not create
archives or change their contents.

## Analysis reruns

Input Data contains prepared values and settings. The benchmark has three
tessellations; human, mouse and Xenium analyses have ten. User configurations
and resolved scientific settings are stored with the relevant input dataset,
not rediscovered from a local code checkout. Original run identifiers remain
with the supplementary results.

Human and mouse runners require explicit input and output directories:

```sh
python -m analyses.greenwald_full.run_all_pairs \
  --input-dir ../TileScope-data/input_data/greenwald \
  --output-dir ../TileScope-data/reruns/greenwald --dry-run
python -m analyses.mouse_gbm_spatial_programs.phase6_harmonized_run_tilescope \
  --input-dir ../TileScope-data/input_data/mouse \
  --output-dir ../TileScope-data/reruns/mouse --all --dry-run
```

Remove `--dry-run` to execute. The runners select each pair from the supplied
score banks and read `tilescope_run_config.json` from the input directory;
`--config PATH` can override it. Inputs remain unchanged. Results and run
accounting are created under the chosen output directory, which must be separate
from the inputs. No downloaded `CurrentResults/` tree is required.

Rerunning the original preprocessing additionally needs the source data
listed in [Data access](DATA_ACCESS.md), R and Seurat where applicable.
The complete Figure 1 configuration is missing; only its preserved input
and plotting measurements are available.

Figure export uses XlsxWriter and verifies workbooks with openpyxl.
Some preview/test commands use `pdftoppm`. Tested benchmark dependency pins
are in [requirements/benchmark-tested.txt](../requirements/benchmark-tested.txt).
