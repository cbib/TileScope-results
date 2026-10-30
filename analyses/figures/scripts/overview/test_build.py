"""Small fixture/design checks; the full core run is audited by build.py."""
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("figure1_build", Path(__file__).with_name("build.py"))
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


def test_reproducible_and_in_tilescope_format():
    df = build.make_input()
    pd.testing.assert_frame_equal(df, build.make_input())
    assert df.columns.tolist() == ["x", "y", "gene1", "gene2", "roi_id"]
    assert 300 <= len(df) <= 600
    assert np.isfinite(df.to_numpy()).all()
    assert not df[["x", "y"]].duplicated().any()
    assert ((df[["gene1", "gene2"]] >= 0) & (df[["gene1", "gene2"]] <= 1)).all().all()


def test_geometry_has_three_rois_and_four_components():
    audit = build.source_audit(build.make_input())
    assert audit["n_components"] == 4
    assert [c["roi_id"] for c in audit["components"]].count("1") == 2
    assert min(audit["roi_spot_counts"].values()) >= 40
    assert audit["cross_roi_edge_counts"]["1-2"] >= 20


def test_fields_differ_and_both_cross_roi_boundaries():
    df = build.make_input()
    assert not np.allclose(df.gene1, df.gene2)
    a_high, b_high = df.gene1 > 0.55, df.gene2 > 0.55
    assert (a_high & b_high).sum() >= 5
    assert (a_high & ~b_high).sum() >= 30
    assert set(df.loc[b_high, "roi_id"]) >= {1, 2}
    assert df.groupby("roi_id")[["gene1", "gene2"]].std().min().min() > 0.01


def test_random_seed_changes_expression_not_geometry():
    a, b = build.make_input(1), build.make_input(2)
    pd.testing.assert_frame_equal(a[["x", "y", "roi_id"]], b[["x", "y", "roi_id"]])
    assert not np.array_equal(a.gene1, b.gene1)


def test_gene_b_revision_preserves_gene_a_and_geometry():
    before = build.make_input(include_gene_b_background=False, include_gene_a_boundary_taper=False)
    after = build.make_input(include_gene_a_boundary_taper=False)
    pd.testing.assert_frame_equal(before.drop(columns="gene2"), after.drop(columns="gene2"))
    assert (after.gene2 >= before.gene2).all()
    # Pin the serialised previous input so this comparison is not circular.
    import hashlib
    previous_csv = before.to_csv(index=False, float_format="%.15g").encode()
    assert hashlib.sha256(previous_csv).hexdigest() == (
        "874f221f71f285d7f085bb29b53b10a3bd150853cc0cb20f543d1fb64158ef8d"
    )


def test_boundary_revision_changes_only_far_roi2_a_and_preserves_high_masks():
    from tilescope.spatial.source_spot_table import build_source_spot_table_from_frame
    from tilescope.spatial.source_spot_adjacency import build_source_spot_adjacency
    from tilescope.analyzers.parent_roi_interface_response import FAR_HOP_MIN
    import hashlib

    old = build.make_input(include_gene_a_boundary_taper=False)
    new = build.make_input()
    assert hashlib.sha256(old.to_csv(index=False, float_format="%.15g").encode()).hexdigest() == (
        "ce61c4233d9eed9f050546cfe109e3060427994d3c85506fffd08cd6e012f8ce"
    )
    pd.testing.assert_frame_equal(old.drop(columns="gene1"), new.drop(columns="gene1"))
    graph = build_source_spot_adjacency(build_source_spot_table_from_frame(old), technology="visium")
    _, endpoints = graph.interface_endpoints("1", "2")
    hops = graph.hop_distance_within_roi("2", endpoints)
    changed = old.gene1.ne(new.gene1)
    np.testing.assert_array_equal(changed, old.roi_id.eq(2) & (hops >= FAR_HOP_MIN))
    assert changed.sum() == 56
    assert (new.loc[changed, "gene1"] < old.loc[changed, "gene1"]).all()
    for gene in ("gene1", "gene2"):
        q_old, q_new = np.quantile(old[gene], .75), np.quantile(new[gene], .75)
        assert q_old == q_new
        np.testing.assert_array_equal(old[gene] > q_old, new[gene] > q_new)


def test_boundary_taper_is_invariant_to_global_coordinate_rescaling():
    old = build.make_input(include_gene_a_boundary_taper=False)
    scaled = old.copy()
    scaled[["x", "y"]] *= 10
    np.testing.assert_allclose(build.gene_a_boundary_taper(scaled).gene1,
                               build.gene_a_boundary_taper(old).gene1, rtol=1e-13, atol=1e-13)


def test_gene_b_has_broad_moderate_roi2_expression_and_a_stronger_focus():
    df = build.make_input()
    roi2 = df.loc[df.roi_id == 2, "gene2"]
    # Illustration-design checks, not biological or scenario decision bands.
    assert (roi2 > 0.4).mean() > 0.85
    assert 0.45 < roi2.median() < 0.65
    assert roi2.max() > roi2.median() + 0.30
    assert roi2.mean() > df.loc[df.roi_id != 2, "gene2"].mean() + 0.25
    assert (df.gene2 >= 1.0).sum() == 0


def test_gene_b_background_does_not_step_at_roi_boundary():
    # At y=0 the main ROI 1/2 analytic boundary lies at x=-4.
    values = build.gene_b_regional_background(
        np.array([-4.005, -3.995]), np.array([0.0, 0.0]),
    )
    assert values.min() > 0.20
    assert abs(values[1] - values[0]) < 0.005


def test_saved_input_matches_the_illustration_generator():
    from tilescope_results.paths import input_data_root
    saved = pd.read_csv(input_data_root() / 'overview/spots.csv', float_precision='round_trip')
    generated = build.make_input()
    pd.testing.assert_frame_equal(saved[generated.columns], generated, check_dtype=False,
                                  check_exact=False, rtol=1e-12, atol=1e-12)


def test_expression_palette_has_light_blue_low_values_and_red_high_values():
    cmap = build.expression_colormap()
    colours = np.asarray(cmap(np.linspace(0.0, 1.0, 256)))[:, :3]
    low, high = colours[0], colours[-1]
    assert low[2] > low[1] > low[0]  # blue, not grey or red
    assert low[2] > 1.35 * low[0]
    assert high[0] > 2.5 * max(high[1], high[2])
    # The low end is blue without becoming dark navy; the centre is pale.
    linear = np.where(colours <= 0.04045, colours / 12.92,
                      ((colours + 0.055) / 1.055) ** 2.4)
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
    assert luminance[0] > 0.35
    assert luminance[128] > max(luminance[0], luminance[-1])


def test_renderer_writes_three_separate_matching_canvases(tmp_path):
    import xml.etree.ElementTree as ET
    import re
    import shutil
    import subprocess
    from PIL import Image

    # Minimal rendering fixture; anatomy is tested independently above and
    # against the actual core exports by verify_run().
    geometry = pd.DataFrame({
        "roi_id": ["1"], "component_id": ["0"],
        "geometry_wkt": ["POLYGON ((-15 -10, 15 -10, 15 10, -15 10, -15 -10))"],
    })
    files = build.render_panels(build.make_input(), geometry, tmp_path)
    stems = {"A1_roi_labels", "A2_gene_A_expression", "A3_gene_B_expression"}
    assert {p.stem for p in files} == stems
    assert len(files) == 9
    svg_boxes, pdf_boxes, png_boxes = set(), set(), set()
    for stem in stems:
        root = ET.parse(tmp_path / f"{stem}.svg").getroot()
        svg_boxes.add(root.attrib["viewBox"])
        pdf = tmp_path / f"{stem}.pdf"
        assert pdf.read_bytes().startswith(b"%PDF-")
        if shutil.which("pdfinfo"):
            info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
            assert re.search(r"Pages:\s+1\b", info)
            pdf_boxes.add(re.search(r"Page size:\s+(.+)", info).group(1))
        with Image.open(tmp_path / f"{stem}.png") as img:
            png_boxes.add(img.size)
    assert len(svg_boxes) == len(png_boxes) == 1
    if shutil.which("pdfinfo"):
        assert len(pdf_boxes) == 1
