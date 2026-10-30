"""TileScope-conformance suite bootstrap.

The tests under this directory verify that the independent benchmark
policies (`scenario_benchmark.support_class`) stay byte-compatible
with the authoritative TileScope core implementation. They REQUIRE
the ``tilescope`` package to be importable and MUST NOT be silently
skipped: a missing core is a genuine acceptance-suite failure.

Default pytest invocations (``pytest`` without arguments) exclude
this directory via pyproject ``testpaths``. Conformance runs are
opt-in:

    pytest tests/tilescope_conformance
"""
from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    try:
        import tilescope  # noqa: F401
    except ImportError as e:
        raise pytest.UsageError(
            "TileScope conformance suite requires the tilescope package "
            "in the current environment; install it with "
            "`python -m pip install -e ../TileScope/python` before running "
            f"`pytest tests/tilescope_conformance`. ({e})"
        )
