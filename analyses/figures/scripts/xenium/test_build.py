from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, MultiPolygon
from matplotlib.path import Path as MplPath

from analyses.figures.scripts.xenium.build import polygon_patches, panel_settings, enrichment_contrasts, CASES
from analyses.figures.scripts.xenium.map_data import complete_area
from analyses.figures.scripts.xenium.paired_components import (
    COMPONENTS, REGION_COLOURS, prepare_comparison, expression_panel, location_panel,
    section_cells, detail_bounds, combined_bounds, polygon_rings,
)
from analyses.figures.scripts.xenium.map_data import evidence_root, runs
from analyses.xenium_feasibility.analyse import read_csv, align_ledger


def test_clipped_hexagons_preserve_holes_and_disconnected_pieces():
    polygon = Polygon([(0,0),(3,0),(3,3),(0,3)], [[(1,1),(1,2),(2,2),(2,1)]])
    patches = polygon_patches(MultiPolygon([polygon, Polygon([(5,0),(6,0),(6,1),(5,1)])]))
    assert len(patches) == 2
    assert np.count_nonzero(patches[0].get_path().codes == MplPath.MOVETO) == 2
    assert np.count_nonzero(patches[0].get_path().codes == MplPath.CLOSEPOLY) == 2


def test_colour_scales_are_local_to_each_representation_and_roi():
    cells = pd.DataFrame({'roi_id': ['IGN', 'IGN', 'IGN', 'other'], 'gene2': [0., 8., np.nan, 1000.]})
    hexes = pd.DataFrame({'roi_id': ['IGN', 'IGN', 'IGN', 'other'], 'gene2_density_ensemble': [1.9, 3.5, np.nan, 1000.]})
    settings = dict(roi_id='IGN', gene='GSN', feature_a='MAG')
    a = panel_settings(cells, hexes, settings, 'cells')
    b = panel_settings(cells, hexes, settings, 'ensemble')
    assert (a['vmin'], a['vmax']) == pytest.approx((0., 7.92))
    assert (b['vmin'], b['vmax']) == pytest.approx((1.9, 3.484))
    assert a['n_finite_coloured_units'] == b['n_finite_coloured_units'] == 2
    assert a['n_above_colour_limit'] == b['n_above_colour_limit'] == 1


@pytest.mark.parametrize('value, limits', [(0., (0., 1.)), (2., (0., 2.)), (-2., (-2., 0.))])
def test_constant_maps_have_a_valid_scale(value, limits):
    cells = pd.DataFrame({'roi_id': ['IGN', 'IGN'], 'gene1': [value, value]})
    settings = dict(roi_id='IGN', gene='GSN', feature_a='GSN')
    result = panel_settings(cells, pd.DataFrame(), settings, 'cells')
    assert (result['vmin'], result['vmax']) == limits
    assert result['displayed_min'] == result['displayed_max'] == value


def test_manuscript_scope_contains_only_retained_samples():
    from analyses.meylan_xenium_preparation.cohort import SAMPLE_CELLS
    selected = list(runs())
    assert [(r.study,r.sample,r.pair) for r in selected] == [
        *[('GBM',s,'Stemlike_Interferon') for s in SAMPLE_CELLS],
        ('Breast','sample1_rep1','ACTA2_FASN')]
    assert selected[0].genes_a.split(';') == ['IL7R','SELL','TCF7','CCR7','FCMR','CD27']


@pytest.mark.parametrize('case', CASES)
def test_complete_component_maps_do_not_crop_cells_or_hexagons(case):
    if not (evidence_root()/'correlations.csv').exists():
        pytest.skip('Two-sample Xenium extraction not installed')
    study, sample, pair, roi, component, _ = case
    run = next(r for r in runs() if (r.study,r.sample,r.pair)==(study,sample,pair))
    cells, hexes, setting = complete_area(run,roi,component)
    frame=read_csv(run.input); ledger=align_ledger(frame,read_csv(run.ledger))
    expected = ledger.loc[ledger.roi_id.eq(roi) & ledger.component_id.eq(component)]
    assert set(cells.input_row) == set(expected.spot_index)
    saved_hexes=read_csv(run.output/'consensus_hexes.csv.gz')
    assert len(hexes)==len(saved_hexes.loc[saved_hexes.roi_id.eq(roi) & saved_hexes.component_id.eq(component)])
    assert cells.x.between(setting['xmin'],setting['xmax']).all()
    assert cells.y.between(setting['ymin'],setting['ymax']).all()
    for geometry in hexes.geometry:
        xmin,ymin,xmax,ymax=geometry.bounds
        assert setting['xmin'] <= xmin <= xmax <= setting['xmax']
        assert setting['ymin'] <= ymin <= ymax <= setting['ymax']


def test_enrichment_contrasts_keep_whole_roi_identity():
    if not (evidence_root()/'correlations.csv').exists():
        pytest.skip('Two-sample Xenium extraction not installed')
    table=enrichment_contrasts()
    assert len(table)==24
    assert set(table.statistic_unit)=={'whole ROI'}
    assert set(table.status)=={'finite'}
    assert not table.duplicated(['study','sample','roi_id','gene','representation']).any()


def test_two_component_scale_is_shared_and_excludes_context_tumour():
    cells = pd.DataFrame({'roi_id': ['interface', 'aggregate', 'tumour'],
                          'gene1': [1., 9., 1000.]})
    hexes = pd.DataFrame({'roi_id': ['interface', 'aggregate', 'tumour'],
                          'gene1_density_ensemble': [2., 10., 2000.]})
    settings = dict(roi_id='interface;aggregate', gene='Stemlike', feature_a='Stemlike')
    a = panel_settings(cells, hexes, settings, 'cells')
    b = panel_settings(cells, hexes, settings, 'ensemble')
    assert (a['vmin'], a['vmax']) == pytest.approx((1., 8.92))
    assert (b['vmin'], b['vmax']) == pytest.approx((2., 9.92))
    assert a['n_finite_coloured_units'] == b['n_finite_coloured_units'] == 2


@pytest.fixture(scope='module')
def comparison():
    run = next(r for r in runs() if r.study == 'GBM' and r.sample == 'P28_post')
    if not run.input.exists():
        pytest.skip('Meylan input not installed')
    return run, prepare_comparison(run)


def test_paired_maps_keep_all_cells_and_hexagons_at_original_positions(comparison):
    run, (cells, hexes, context, context_hexes, geometry, summary, settings) = comparison
    original = read_csv(run.input)
    ledger = align_ledger(original, read_csv(run.ledger))
    expected = ledger.loc[ledger.component_id.isin([784, 0])]
    assert len(cells) == 7125
    assert set(cells.input_row) == set(expected.spot_index)
    assert not cells.input_row.duplicated().any()
    assert np.array_equal(cells[['x','y']].to_numpy(), original.iloc[cells.input_row][['x','y']].to_numpy())
    assert set(context.component_id) == {784, 0, 501}
    assert len(context) == 8281
    assert set(geometry.component_id) == {784, 0, 501}
    assert summary.set_index('component_id').loc[0, 'n_eligible_cells'] == 217
    saved = read_csv(run.output / 'consensus_hexes.csv.gz')
    expected_hexes = saved.loc[saved.component_id.isin([784, 0])]
    assert set(zip(hexes.component_id, hexes.hex_id)) == set(zip(expected_hexes.component_id, expected_hexes.hex_id))
    assert cells.x.between(settings['xmin'], settings['xmax']).all()
    assert cells.y.between(settings['ymin'], settings['ymax']).all()
    for g in hexes.geometry:
        xmin, ymin, xmax, ymax = g.bounds
        assert settings['xmin'] <= xmin <= xmax <= settings['xmax']
        assert settings['ymin'] <= ymin <= ymax <= settings['ymax']


def test_detail_keeps_aggregate_and_context_with_full_map_colour_limits(comparison):
    representation = 'ensemble'
    _, (cells, hexes, _, _, geometry, _, base) = comparison
    settings = panel_settings(cells, hexes, dict(base, gene='Stemlike'), representation)
    bounds = detail_bounds(cells, hexes)
    detail_settings = dict(settings, **bounds)
    main_fig = expression_panel(cells, hexes, geometry, settings, polygon_patches)
    detail_fig = expression_panel(cells, hexes, geometry, detail_settings, polygon_patches, detail=True)
    main, detail = main_fig.axes[0], detail_fig.axes[0]
    from shapely import from_wkt
    from matplotlib.colors import to_rgba
    for axis in [main, detail]:
        boundaries = {artist.get_gid(): artist for artist in axis.collections if artist.get_gid()}
        assert set(boundaries) == {f'roi-boundary-{component}' for component in REGION_COLOURS}
        for component, colour in REGION_COLOURS.items():
            artist = boundaries[f'roi-boundary-{component}']
            source = geometry.loc[geometry.component_id.eq(component), 'display_outline_wkt'].iloc[0]
            expected = polygon_rings(from_wkt(source))
            assert len(artist.get_segments()) == len(expected)
            for actual, ring in zip(artist.get_segments(), expected):
                np.testing.assert_array_equal(actual, ring)
            assert artist.get_colors()[0] == pytest.approx(to_rgba(colour, .9))
    from matplotlib.collections import PatchCollection
    if representation == 'cells':
        main_expression, detail_expression = main.collections[0], detail.collections[0]
    else:
        main_expression = next(c for c in main.collections if isinstance(c, PatchCollection))
        detail_expression = next(c for c in detail.collections if isinstance(c, PatchCollection))
    assert (main_expression.norm.vmin, main_expression.norm.vmax) == (detail_expression.norm.vmin, detail_expression.norm.vmax)
    np.testing.assert_array_equal(main_expression.get_array(), detail_expression.get_array())
    assert detail.get_xlim() == pytest.approx((bounds['xmin'], bounds['xmax']))
    assert detail.get_ylim() == pytest.approx((bounds['ymax'], bounds['ymin']))
    inside = cells.loc[cells.x.between(bounds['xmin'],bounds['xmax']) & cells.y.between(bounds['ymin'],bounds['ymax'])]
    assert inside.component_id.eq(0).sum() == 233
    assert inside.component_id.eq(784).sum() > 0
    for shape in hexes.loc[hexes.component_id.eq(0), 'geometry']:
        xmin,ymin,xmax,ymax = shape.bounds
        assert bounds['xmin'] <= xmin < xmax <= bounds['xmax']
        assert bounds['ymin'] <= ymin < ymax <= bounds['ymax']
    plt.close(main_fig)
    plt.close(detail_fig)


def test_locator_crops_to_local_context_without_losing_selected_components(comparison):
    run, (_, _, context, context_hexes, _, _, settings) = comparison
    cells = section_cells(run)
    settings = dict(settings, **combined_bounds(context, context_hexes))
    fig = location_panel(cells, settings)
    assert fig.axes[0].get_title() == 'ROIs'
    fig.canvas.draw()
    legend_bounds = fig.legends[0].get_window_extent().transformed(fig.transFigure.inverted())
    assert legend_bounds.x0 > fig.axes[0].get_position().x1
    assert legend_bounds.x1 < 1
    assert len(cells) == 97488
    visible = cells.loc[cells.x.between(settings['xmin'],settings['xmax']) &
                        cells.y.between(settings['ymin'],settings['ymax'])]
    assert 8281 <= len(visible) < len(cells) / 2
    assert fig.axes[0].get_xlim() == pytest.approx((settings['xmin'],settings['xmax']))
    assert fig.axes[0].get_ylim() == pytest.approx((settings['ymax'],settings['ymin']))
    assert set(visible.loc[visible.component_id.isin(REGION_COLOURS), 'input_row']) == set(context.input_row)
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [
        'Tumour–T cell interface', 'Lymphoid aggregate', 'Tumour', 'Other cells']
    from matplotlib.colors import to_rgba
    from matplotlib.collections import PathCollection
    points = [c for c in fig.axes[0].collections if isinstance(c, PathCollection)]
    assert sum(len(artist.get_offsets()) for artist in points) == len(visible)
    for artist, (_, component, _, _) in zip(points[1:], COMPONENTS):
        assert artist.get_facecolors()[0] == pytest.approx(to_rgba(REGION_COLOURS[component]))
        assert len(artist.get_offsets()) == cells.component_id.eq(component).sum()
    assert len(set(REGION_COLOURS.values())) == 3
    plt.close(fig)
