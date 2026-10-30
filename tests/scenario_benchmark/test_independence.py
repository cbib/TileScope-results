"""Package-level import-boundary contract: scenario_benchmark must
never import ``tilescope.*``. Enforced by AST-scanning every .py file
under the installed package. The mirrored spatial-support classifier
in ``scenario_benchmark.support_class`` is the correct place for
policy conformance; TileScope itself never enters the sister package.

Historical: CONTRACT.md § 1.1 is the frozen invariant. The prior
enforcement test was deleted with the retired v1 layout; this file
restores it under the flat package structure."""
from __future__ import annotations

import ast
from pathlib import Path

import scenario_benchmark


PKG_ROOT = Path(scenario_benchmark.__file__).resolve().parent


def _imported_top_levels(source: str) -> set[str]:
    """Extract the top-level module name from every ``import`` /
    ``from`` statement in ``source``. Uses AST, not string matching,
    so ``# tilescope`` in a comment doesn't produce a false positive."""
    tree = ast.parse(source)
    top_levels: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_levels.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                top_levels.add(node.module.split(".", 1)[0])
    return top_levels


def test_scenario_benchmark_package_never_imports_tilescope():
    offenders: list[str] = []
    for py in sorted(PKG_ROOT.rglob("*.py")):
        try:
            imports = _imported_top_levels(py.read_text())
        except SyntaxError as e:
            offenders.append(f"{py.relative_to(PKG_ROOT)}: {e}")
            continue
        if "tilescope" in imports:
            offenders.append(str(py.relative_to(PKG_ROOT)))
    assert not offenders, (
        "scenario_benchmark package must not import tilescope; "
        f"offenders: {offenders}"
    )
