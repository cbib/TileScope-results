# Vendored fixture: greenwald_242T

Real-data fixture for the TileScope-results integration tests. **Immutable
test input.** Do not modify these files to change scientific behavior
— the scientific source authority is `TileScope-data`. If the source
changes, refresh the fixture in a single well-scoped commit that also
updates `fixture_manifest.yaml` (SHAs + per-ROI counts) and any tests
whose expectations depend on the change.

See `fixture_manifest.yaml` for provenance, SHAs, row/ROI counts,
selection rationale, and redistribution notes.

## Layout

- `242_T_expression_data_AC_Chromatin-Reg.csv` — pair 1
- `242_T_expression_data_MES_Reactive-Ast.csv` — pair 2
- `run.config.json` — TileScope run configuration for 242_T
- `fixture_manifest.yaml` — provenance + invariants

## How it is used

`tests/e2e/test_roi_surface_greenwald.py` invokes
`tilescope.api.run_tilescope_from_csv` on each pair, writes outputs
to pytest's `tmp_path`, then verifies TileScope's canonical ROI-grain
export surface. Never commit generated outputs.

Opt-in only:

```
pytest tests/e2e --run-real-data
```
