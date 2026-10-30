"""Read paired TileScope inputs from the distributed per-sample score banks."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


def run_paths(input_dir, output_dir, config=None):
    inputs = Path(input_dir).expanduser().resolve()
    outputs = Path(output_dir).expanduser().resolve()
    settings = Path(config).expanduser().resolve() if config else inputs / "tilescope_run_config.json"
    if not inputs.is_dir():
        raise ValueError(f"Input directory does not exist: {inputs}")
    if inputs.is_relative_to(outputs) or outputs.is_relative_to(inputs):
        raise ValueError("Input and output directories must be separate, not nested")
    if not settings.is_file():
        raise ValueError(f"Run configuration does not exist: {settings}")
    return inputs, outputs, settings


def directory_name(value):
    if not isinstance(value, str) or not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"Invalid sample or pair identifier: {value!r}")
    return value


def read_scores(path, features):
    columns = ["x", "y", "roi_id", *sorted(set(features))]
    frame = pd.read_csv(path, usecols=columns, float_precision="round_trip")
    numeric = frame.drop(columns="roi_id").to_numpy(dtype=float)
    if frame.empty or frame.roi_id.isna().any() or not np.isfinite(numeric).all():
        raise ValueError(f"Missing or nonfinite coordinates, scores or ROI labels: {path}")
    return frame


@contextmanager
def pair_csv(path, feature_a, feature_b):
    """Keep pair inputs temporary; the supplied score bank is never modified."""
    if feature_a == feature_b:
        raise ValueError("A pair must contain two different features")
    frame = read_scores(path, [feature_a, feature_b])
    frame = frame.loc[:, ["x", "y", feature_a, feature_b, "roi_id"]]
    frame.columns = ["x", "y", "gene1", "gene2", "roi_id"]
    with TemporaryDirectory(prefix="tilescope-pair-") as temporary:
        csv = Path(temporary) / "input.csv"
        frame.to_csv(csv, index=False)
        yield csv
