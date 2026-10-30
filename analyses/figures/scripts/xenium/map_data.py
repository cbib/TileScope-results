"""Load complete spatial units for Xenium figure maps."""
import numpy as np
from types import SimpleNamespace
from shapely import from_wkt
from analyses.xenium_feasibility.analyse import read_csv, align_ledger
from tilescope_results.paths import input_data_root, supplementary_data_root


def evidence_root():
    return supplementary_data_root(3)


def runs():
    root = input_data_root() / 'xenium'
    for row in read_csv(root / 'datasets.csv').to_dict('records'):
        identity = {k: row[k] for k in ('study', 'sample', 'patient', 'pair', 'feature_a', 'feature_b')}
        yield SimpleNamespace(**row, identity=identity, input=root / row['input_csv'],
            output=evidence_root() / 'spatial' / row['study'] / row['sample'],
            ledger=evidence_root() / 'cells' / row['study'] / row['sample'] / 'cell_aggregation_status.csv.gz',
            config=root / row['user_config'])


def complete_area(run, roi, component=None):
    """Return every cell and saved hexagon in the named ROI or component."""
    cells = read_csv(run.input)
    ledger = align_ledger(cells, read_csv(run.ledger))
    cells['input_row'] = np.arange(len(cells))
    cells['component_id'] = ledger.component_id
    cells['tessellation_eligible'] = ledger.tessellation_eligible.eq(True)
    cells = cells.loc[cells.roi_id.eq(roi)].copy()
    hexes = read_csv(run.output / 'consensus_hexes.csv.gz')
    hexes = hexes.loc[hexes.roi_id.eq(roi)].copy()
    geometry = read_csv(run.output / 'component_geometry.csv.gz')
    geometry = geometry.loc[geometry.roi_id.eq(roi)].copy()
    roi_count = len(cells)
    if component == 'largest':
        eligible = geometry.loc[geometry.n_spots_in_tessellation_eligible_regions.gt(0)]
        component = int(eligible.sort_values('n_component_spots', ascending=False).iloc[0].component_id)
    if component is not None:
        component = int(component)
        cells = cells.loc[cells.component_id.eq(component)].copy()
        hexes = hexes.loc[hexes.component_id.eq(component)].copy()
        geometry = geometry.loc[geometry.component_id.eq(component)].copy()
    if cells.empty or hexes.empty:
        raise ValueError(f'Unavailable selected area: {run.sample}/{roi}/{component}')
    hexes['geometry'] = from_wkt(hexes.hex_geometry_wkt)
    cell_lo, cell_hi = cells[['x','y']].min().to_numpy(), cells[['x','y']].max().to_numpy()
    all_bounds = np.asarray([g.bounds for g in hexes.geometry])
    low = np.minimum(cell_lo, all_bounds[:, :2].min(axis=0))
    high = np.maximum(cell_hi, all_bounds[:, 2:].max(axis=0))
    pad = .03 * max(high - low)
    bounds = (low[0]-pad, low[1]-pad, high[0]+pad, high[1]+pad)
    return cells, hexes, dict(**run.identity, roi_id=roi, component_id=component,
                             display_unit='ROI' if component is None else 'component',
                             n_cells=len(cells), n_roi_cells=roi_count,
                             n_eligible_cells=int(cells.tessellation_eligible.sum()),
                             n_hexagons=len(hexes), xmin=bounds[0], ymin=bounds[1],
                             xmax=bounds[2], ymax=bounds[3],
                             extent_rule='Complete selected ROI/component, including every cell and full hexagon geometry; 3% margin')
