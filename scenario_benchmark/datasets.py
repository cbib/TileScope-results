"""Dataset-source registry for scenario_benchmark.

Maps ``sample_id`` to a source CSV path across the three
on-disk dataset roots (Greenwald / HCC / mouse). Data roots
are configurable via environment variables so that no
machine-specific absolute path is baked into tracked artefacts.

The registry is deliberately small: one ``DatasetSource`` per
disk layout, one filename glob per source. Selection of a
specific per-pair CSV within a sample uses the
lexicographically first match; audit rows record which CSV was
loaded so downstream consumers can re-resolve deterministically.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tilescope_results.paths import data_root


@dataclass(frozen=True)
class DatasetSource:
    name: str
    default_root_env: str
    default_root: Path
    sample_root_pattern: str  # relative to root; "" means samples share root
    csv_pattern: str          # glob under sample_root

    def sample_dir(self, root: Path, sample_id: str) -> Path:
        rel = self.sample_root_pattern.format(sample_id=sample_id)
        return root / rel if rel else root

    def glob_pattern(self, sample_id: str) -> str:
        return self.csv_pattern.format(sample_id=sample_id)


DATASET_REGISTRY: Dict[str, DatasetSource] = {
    "greenwald": DatasetSource(
        name="greenwald",
        default_root_env="TILESCOPE_DATA_ROOT",
        default_root=Path("heiland_tyrosh/CSVs"),
        sample_root_pattern="",
        csv_pattern="{sample_id}_*expression_data_*.csv",
    ),
    "hcc": DatasetSource(
        name="hcc",
        default_root_env="TILESCOPE_HCC_ROOT",
        default_root=Path("HCC_human"),
        sample_root_pattern="{sample_id}/dchalopin_signatures/CSVs",
        csv_pattern="{sample_id}_*expression_data_*.csv",
    ),
    "mouse": DatasetSource(
        # Use the current score bank for coordinates and ROI membership.
        # No fallback to retired, pair-specific expression CSVs.
        name="mouse",
        default_root_env="TILESCOPE_MOUSE_ROOT",
        default_root=Path("mouse"),
        sample_root_pattern="mouse_gbm_spatial_programs/phase6_harmonized",
        csv_pattern="{sample_id}_signature_score_bank.csv",
    ),
}


def resolve_dataset_root(
    dataset: str, *, explicit: Optional[Path] = None,
) -> Path:
    src = DATASET_REGISTRY[dataset]
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(src.default_root_env)
    if env:
        return Path(env).expanduser().resolve()
    return (data_root() / src.default_root).resolve()


def resolve_sample_csv(
    sample_id: str, *, dataset: Optional[str] = None,
    data_root: Optional[Path] = None,
) -> Tuple[Path, str]:
    """Locate the source CSV for ``sample_id``.

    When ``dataset`` is given, resolves only within that dataset's
    root. Otherwise searches all registered datasets in registry
    order and returns the first match; raises FileNotFoundError if
    no dataset has a matching CSV, and ValueError if two datasets
    both claim the same sample_id.
    """
    if dataset is not None:
        src = DATASET_REGISTRY[dataset]
        root = resolve_dataset_root(dataset, explicit=data_root)
        matches = sorted(src.sample_dir(root, sample_id).glob(
            src.glob_pattern(sample_id)
        ))
        if not matches:
            raise FileNotFoundError(
                f"resolve_sample_csv: no CSV matching "
                f"{src.glob_pattern(sample_id)!r} under "
                f"{src.sample_dir(root, sample_id)}"
            )
        return matches[0], dataset

    found: List[Tuple[Path, str]] = []
    for name, src in DATASET_REGISTRY.items():
        root = resolve_dataset_root(name)
        matches = sorted(src.sample_dir(root, sample_id).glob(
            src.glob_pattern(sample_id)
        ))
        if matches:
            found.append((matches[0], name))
    if not found:
        raise FileNotFoundError(
            f"resolve_sample_csv: {sample_id!r} not found in any "
            f"registered dataset ({sorted(DATASET_REGISTRY)})"
        )
    if len(found) > 1:
        raise ValueError(
            f"resolve_sample_csv: {sample_id!r} ambiguous across "
            f"datasets {[d for _, d in found]}; pass dataset= explicitly"
        )
    return found[0]


def enumerate_samples(dataset: str) -> List[str]:
    """List sample_ids discoverable on disk for ``dataset``.

    Uses each dataset's naming convention: greenwald pulls the
    stem prefix before ``_expression_data_``; HCC enumerates sample
    folders; mouse enumerates current signature score banks. Sorted order.
    """
    src = DATASET_REGISTRY[dataset]
    root = resolve_dataset_root(dataset)
    if not root.exists():
        return []
    if dataset == "greenwald":
        seen = set()
        for p in root.glob("*_expression_data_*.csv"):
            stem = p.name.split("_expression_data_", 1)[0]
            seen.add(stem)
        return sorted(seen)
    if dataset == "mouse":
        bank_dir = src.sample_dir(root, "")
        return sorted(p.name.removesuffix("_signature_score_bank.csv")
                      for p in bank_dir.glob("*_signature_score_bank.csv"))
    # hcc: iterate sample subdirectories
    samples = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        candidate_dir = src.sample_dir(root, sub.name)
        if not candidate_dir.exists():
            continue
        matches = list(candidate_dir.glob(src.glob_pattern(sub.name)))
        if matches:
            samples.append(sub.name)
    return samples
