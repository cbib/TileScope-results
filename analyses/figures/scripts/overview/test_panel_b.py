"""Figure 1B rendering contracts; no new TileScope run in tests."""
from pathlib import Path
import importlib.util
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

SPEC = importlib.util.spec_from_file_location("figure1_panel_b", Path(__file__).with_name("plot_panel_b.py"))
panel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(panel)


def test_canvas_matches_panel_a_and_has_no_caption():
    fig, ax = panel.map_canvas()
    assert tuple(fig.get_size_inches()) == (4.2, 3.35)
    assert ax.get_xlim() == (-15.5, 15.5)
    assert ax.get_ylim() == (-10.5, 10.5)
    assert ax.get_title() == ""
    assert not fig.texts
    assert not ax.texts
    assert fig.get_facecolor()[-1] == 0
    plt.close(fig)


def test_polygon_colour_uses_fixed_scale_and_retains_multipart_geometry():
    fig, ax = panel.map_canvas()
    frame = pd.DataFrame({
        "geometry": ["MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)), ((2 0, 3 0, 3 1, 2 0)))"],
        "value": [0.25],
    })
    assert panel.paint_units(ax, frame, "geometry", "value", prefix="test") == 2
    expected = panel.expression_colormap()(panel.EXPRESSION_NORM(0.25))
    for patch in ax.patches:
        np.testing.assert_allclose(patch.get_facecolor(), expected)
    plt.close(fig)


def test_heading_matches_panel_a_position_and_style():
    fig, ax = panel.map_canvas("Consensus ensemble")
    assert len(fig.texts) == 1
    heading = fig.texts[0]
    assert heading.get_text() == "Consensus ensemble"
    assert heading.get_position() == (0.5, 0.94)
    assert heading.get_fontsize() == 12
    assert heading.get_ha() == heading.get_va() == "center"
    assert not ax.texts
    plt.close(fig)


def test_separate_maps_have_only_requested_headings(tmp_path):
    polygon = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
    frames = {
        "source": pd.DataFrame({"x": [0.2, 0.8], "y": [0.2, 0.8], "gene1": [0.2, 0.7]}),
        "geometry": pd.DataFrame({"component_id": ["0"], "roi_id": ["1"], "geometry_wkt": [polygon]}),
        "tiles": pd.DataFrame({"replicate_index": [0, 1, 2], "tile_geometry_wkt": [polygon] * 3,
                               "g1_density": [0.2, 0.4, 0.6]}),
        "hexes": pd.DataFrame({"hex_geometry_wkt": [polygon], "gene1_density_ensemble": [0.5]}),
    }
    files, records = panel.render(frames, tmp_path)
    assert len(files) == 18  # five maps + independent colour key, in three formats
    assert len(records) == 5
    boxes = set()
    for record in records:
        svg = ET.parse(tmp_path / (record["stem"] + ".svg")).getroot()
        boxes.add(svg.attrib["viewBox"])
        text = ["".join(e.itertext()).strip() for e in svg.iter("{http://www.w3.org/2000/svg}text")]
        expected = {"source": [], "tessellation": ["Tessellation"],
                    "consensus_ensemble": ["Consensus ensemble"]}
        assert text == expected[record["surface"]]
        with Image.open(tmp_path / (record["stem"] + ".png")) as image:
            assert image.mode == "RGBA"
            assert image.convert('RGB').getpixel((0, 0)) == (255, 255, 255)
    assert len(boxes) == 1
    assert records[-1]["value_column"] == "gene1_density_ensemble"
    assert records[-1]["n_contributing_tessellations"] == 10
