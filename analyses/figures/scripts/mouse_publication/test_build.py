"""Scientific handoff checks, not snapshots of formatting or directory trees."""
import numpy as np
import pandas as pd
import pytest

from .evidence import Evidence, PRIMARY, V4_PAIR
from analyses.mouse_gbm_spatial_programs.paths import locations


def test_feature_boundary_deduplication_rejects_pair_dependent_values():
    e = Evidence.__new__(Evidence)
    e.selection = pd.DataFrame({'sample': ['MCT1_CTL']*2, 'pair': ['a__b', 'a__c']})
    rows = [dict(sample='MCT1_CTL', pair=pair, g1_id='a', g2_id=pair[-1], anchor_roi_id=2,
        responder_roi_id=5, responder_feature_id='a', surface='source_native', availability_status='finite',
        interface_response=.3, source_near_mass=132, source_far_mass=36,
        represented_near_mass=132, represented_far_mass=36) for pair in ['a__b', 'a__c']]
    e.tables = {'roi_pair_interface_response': pd.DataFrame(rows)}
    assert len(e.boundaries()) == 1
    e.tables['roi_pair_interface_response'].loc[1, 'interface_response'] = -.3
    with pytest.raises(ValueError, match='Pair-independent'):
        e.boundaries()


@pytest.mark.skipif(not (locations()[0]/'CurrentResults/MCT1_CTL'/PRIMARY/'roi_correlations.csv').exists(),
                    reason='Current external mouse corpus is not installed')
def test_current_story_retains_support_identity_and_within_periphery_bands():
    e = Evidence()
    assert len(e.roi) == 2064 and len(e.run_inventory) == 222
    # S9A must match the production PDF colour at every retained hexagon.
    from .build import enrichment_pattern_map
    import matplotlib.pyplot as plt
    from shapely import wkt
    from tilescope.cols import (HEX_G1_ENRICHMENT, HEX_G1_ENRICHMENT_STATUS,
                                HEX_G2_ENRICHMENT, HEX_G2_ENRICHMENT_STATUS, HEX_GEOMETRY_WKT)
    from tilescope.viz.consensus import _pattern_style_by_status
    for sample, roi in [('MCT1_CTL', 5), ('MCT1_KO', 9)]:
        hexes = e.spatial(sample, PRIMARY)['hexes']
        expected = []
        for row in hexes.loc[hexes.roi_id.eq(roi)].itertuples():
            color = _pattern_style_by_status(
                getattr(row, HEX_G1_ENRICHMENT), getattr(row, HEX_G1_ENRICHMENT_STATUS),
                getattr(row, HEX_G2_ENRICHMENT), getattr(row, HEX_G2_ENRICHMENT_STATUS))[0]
            geom = wkt.loads(getattr(row, HEX_GEOMETRY_WKT))
            expected.extend([color] * (len(geom.geoms) if geom.geom_type == 'MultiPolygon' else 1))
        fig = enrichment_pattern_map(e, sample, 'test')
        try:
            artist, = [c for c in fig.axes[0].collections if c.get_gid() == 'consensus_enrichment_pattern']
            assert len(artist.get_paths()) == len(expected)
            np.testing.assert_allclose(artist.get_facecolors(), expected)
        finally:
            plt.close(fig)
    for sample, pair, near, far in [('MCT1_CTL', PRIMARY, 132, 36),
                                    ('MCT1_KO', PRIMARY, 209, 121),
                                    ('V4_WT', V4_PAIR, 80, 259)]:
        b = e.bands(sample, pair)
        assert b.band.eq('near').sum() == near and b.band.eq('far').sum() == far
        assert (b.loc[b.band.isin(['near', 'far']), 'roi_id'] == b.responder_roi_id.iloc[0]).all()
    focus = e.satellite().set_index('component_id')
    assert focus.n_spots.to_dict() == {6: 243, 7: 49}
    np.testing.assert_allclose(focus.source_r, [-.094706, -.450716], atol=1e-6)
