"""E2E test-suite bootstrap: opt-in flag + shared markers.

E2E tests here invoke the full TileScope pipeline on vendored real
data and write several MB per pair to `tmp_path`. They are excluded
from the default pytest run and only executed when the caller passes
`--run-real-data`.

Markers registered here (also declared in the runtime `pytest_configure`
hook so pytest does not warn about unknown markers):

- `real_data` — depends on vendored real-data fixtures under
  `tests/e2e/fixtures/`.
- `slow` — a single test may take tens of seconds; the whole suite
  is only appropriate for explicit acceptance runs.
"""
from __future__ import annotations

import os

# Force a non-interactive Matplotlib backend before any TileScope /
# pyplot import. TileScope's visualization stack imports matplotlib
# eagerly; on macOS the default is the interactive Cocoa backend,
# which aborts under headless / non-GUI pytest sessions.
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-real-data", action="store_true", default=False,
        help=(
            "Run opt-in real-data E2E tests under tests/e2e/. Off by "
            "default; these invoke the full TileScope pipeline and are "
            "orders of magnitude slower than the synthetic suites."
        ),
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_data: depends on vendored real-data fixtures; opt-in only",
    )
    config.addinivalue_line(
        "markers",
        "slow: expensive test; opt-in only",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-real-data"):
        return
    skip_real = pytest.mark.skip(
        reason="opt-in only; pass --run-real-data to run"
    )
    for item in items:
        if "real_data" in item.keywords:
            item.add_marker(skip_real)
