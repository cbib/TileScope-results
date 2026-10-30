# Manuscript figures

Build individual panels from the downloadable Input Data and Supplementary
Data bundles. This command does not run TileScope, generate datasets or
change scientific settings.

```bash
python -m analyses.figures.build all
# Or: overview, benchmark, greenwald, mouse, xenium
```

See [data layout](../../docs/DATA_LAYOUT.md) for installation and extraction.
Configure `TILESCOPE_DATA_DIR` and `TILESCOPE_CORE_DIR` for non-sibling locations.
Set `TILESCOPE_FIGURE_DIR` for a separate artwork destination. Missing or changed
scientific inputs fail validation; they are not silently regenerated.

## Outputs

```text
analyses/figures/
  scripts/
  main/Figure1/ ... Figure4/
  supplementary/FigureS1/ ... FigureS10/
  additional/Benchmark/            across-support diagnostic
  index.html, index.csv
```

Open `index.html` after building. Each figure directory contains:

- `panels/`: individual SVG/PDF artwork and PNG previews. SVG text is editable.
  SVG/PDF backgrounds are transparent, apart from label backings needed for
  legibility; PNG previews have white backgrounds.
- `source_data/`: matching CSV tables and an Excel workbook, with a
  panel-to-table index, observation identities, units and display settings.
  Geometry exceeding Excel's cell limit remains complete in CSV.
- `panel_index.csv` and `provenance.json`.
- Optional `assembly/`: manually edited layouts, never written by builders.

No combined manuscript plate is imposed. A map or colour key is an individual
asset, not necessarily a separately lettered publication panel. Styles are
isolated between figure groups so that running `all` does not change a group's
appearance.

## Data and code

Input Data and all four Supplementary Data bundles live under TileScope-data.
Figure builders read them directly and never rewrite them. No supplementary
archive belongs under this code repository.

Generated artwork and figure-specific source tables are gitignored. Code,
tests and reproduction instructions are versioned. Captions and manuscript
prose belong in the paper; builders do not create Word reports.

| Group | Figures | Input dataset | Result bundle |
|---|---|---|---|
| `overview` | Figure 1 | `input_data/overview/` | Data 1, `overview/` |
| `benchmark` | Figure 2, S1–S3 | `input_data/benchmark/` | Data 1 |
| `greenwald` | Figure 3, S4–S6 | `input_data/greenwald/` | Data 2 |
| `mouse` | Figure 4, S9–S10 | `input_data/mouse/` | Data 4 |
| `xenium` | S7–S8 | `input_data/xenium/` | Data 3 |

The original full Figure 1 configuration is missing; its retained input and
plotting evidence reproduce the panels, not an exact new analysis run.

## Checks

```bash
python -m pytest analyses/figures -q
```

Checks cover identities, statistics, component support, boundary bands and
map values. Excel exports are opened with an independent reader and checked
for preserved identifiers and missing values. Manual assemblies are preserved.
