from __future__ import annotations

import json
import sys
from pathlib import Path
from tilescope_results.paths import data_root

import pandas as pd
import pytest


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_DIR))

from feasibility import (  # noqa: E402
    CANONICAL_COLUMNS,
    load_and_validate_input,
)


def _small_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "x": [0.0, 1.0, 2.0, 10.0],
        "y": [0.0, 0.0, 0.0, 0.0],
        "gene1": [0, 1, 2, 3],
        "gene2": [3, 2, 1, 0],
        "roi_id": ["A", "A", "B", "B"],
    })


def test_input_contract_accepts_canonical_synthetic_input(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    _small_frame().to_csv(path, index=False)
    observed = load_and_validate_input(
        path, expected_roi_ids=("A", "B"), expected_n_cells=4
    )
    assert tuple(observed.columns) == CANONICAL_COLUMNS


def test_input_contract_rejects_roi_drift(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    _small_frame().to_csv(path, index=False)
    with pytest.raises(ValueError, match="ROI identity mismatch"):
        load_and_validate_input(
            path, expected_roi_ids=("A", "C"), expected_n_cells=4
        )


def test_predeclared_config_is_explicit_about_technology_and_scale() -> None:
    config_path = (
        (data_root() / 'Janesick/CurrentResults/tilescope_run_config.json')
    )
    if not config_path.is_file():
        pytest.skip("Janesick data repository is not installed")
    config = json.loads(config_path.read_text())
    assert "_version" not in config
    assert config["technology"] == "xenium"
    assert config["tessellation"] == {
        "n": 10,
        "target_observations_per_tile": 15,
        "min_observations_per_tile": 9,
        "min_component_observations": 50,
    }
    assert config["consensus"]["min_spots_per_hex"] == 12
    assert config["consensus"]["min_tess_coverage_fraction"] == 2/3
    assert config["permutations"] == 99
    assert config["seed"] == 42
    assert "voronoi_fraction" not in config["tessellation"]
