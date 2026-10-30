"""Portable code/data roots and resolution of saved provenance references.

Benchmark inputs and core runs live in TileScope-data. Figure artwork and
derived benchmark evaluations live beside their scripts in TileScope-results.
Historical references resolve to those canonical locations, without duplicates.
"""
from __future__ import annotations

import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def _configured(name: str, default: Path | None) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


def data_root() -> Path:
    """Entire TileScope-data root, not a particular dataset's CSV directory."""
    return _configured("TILESCOPE_DATA_DIR", REPOSITORY.parent / "TileScope-data")


def core_root() -> Path:
    return _configured("TILESCOPE_CORE_DIR", REPOSITORY.parent / "TileScope")


def input_data_root() -> Path:
    return data_root() / 'input_data'


def supplementary_data_root(number: int | None = None) -> Path:
    root = data_root() / 'supplementary_data'
    return root if number is None else root / f'Supplementary_Data_{number}'


def greenwald_evidence(name: str) -> Path:
    """Generated numerical evidence, distinct from scripts and authored reviews."""
    if name not in {
        'configuration_inputs',
        'context_matching',
        'gbm_spatial_repertoire',
        'local_frequencies',
        'manuscript_evidence_profiles',
        'scenario_cohort',
        'summaries',
        'two_level_evidence',
    }:
        raise ValueError(f"Unknown Greenwald evidence collection: {name}")
    return data_root() / "heiland_tyrosh" / "analysis" / name


def benchmark_root() -> Path:
    return data_root() / "scenario_benchmark"


def corpus_root() -> Path:
    """Read-only authoritative benchmark corpus (shipped in the input_data bundle).

    Holds the plan, index, per-scenario case directories and role CSVs. The
    materialize generator writes to :func:`generated_corpus_root` instead to
    keep the bundle immutable.
    """
    return input_data_root() / "benchmark" / "corpus"


def generated_corpus_root() -> Path:
    """Writable output location for scenario_benchmark.materialize.

    Separate from :func:`corpus_root` so a regeneration run never overwrites
    the read-only bundle shipped in input_data.
    """
    return benchmark_root() / "corpus"


def current_results_root() -> Path:
    return benchmark_root() / "CurrentResults"


def comparator_benchmark_root() -> Path:
    return benchmark_root() / "comparator_benchmark"


def atlas_root() -> Path:
    """Derived benchmark evaluation, not the TileScope core run tree."""
    return REPOSITORY / "analyses/scenario_discriminability/outputs"


def figure2_root() -> Path:
    """Figure 2 individual artwork and source-data package."""
    root = _configured('TILESCOPE_FIGURE_DIR', REPOSITORY / 'analyses/figures')
    return root / 'main/Figure2'


def feasibility_root() -> Path:
    return generated_corpus_root() / "feasibility"


def relocated_directories() -> dict[str, Path]:
    """Historical repository-relative identifiers, not fallback locations."""
    return {
        'analyses/greenwald_ac_opc/greenwald_organization/results': data_root() / 'heiland_tyrosh/analysis/greenwald_organization',
        'analyses/greenwald_full/checks': data_root() / 'heiland_tyrosh/analysis/workflow_checks',
        'analyses/statistic_candidates/contact_interface_audit': atlas_root() / 'contact_interface_audit',
        'analyses/statistic_candidates/interface_conditioned_periphery/retention_results.csv': atlas_root() / 'interface_conditioned_periphery/retention_results.csv',
        'analyses/statistic_candidates/interface_conditioned_periphery/corpus_diagnostic.csv': atlas_root() / 'interface_conditioned_periphery/corpus_diagnostic.csv',
        'analyses/gbm_spatial_repertoire/FINDINGS.md': REPOSITORY / 'analyses/gbm_spatial_repertoire/FINDINGS.md',
        'analyses/gbm_spatial_repertoire/README.md': REPOSITORY / 'analyses/gbm_spatial_repertoire/README.md',
        'analyses/gbm_spatial_repertoire/batch_01_observations.json': REPOSITORY / 'analyses/gbm_spatial_repertoire/batch_01_observations.json',
        'analyses/gbm_spatial_repertoire/batch_01_plan.json': REPOSITORY / 'analyses/gbm_spatial_repertoire/batch_01_plan.json',
        'analyses/gbm_spatial_repertoire/shortlist_vetting/observations.json': REPOSITORY / 'analyses/gbm_spatial_repertoire/shortlist_vetting/observations.json',
        "analyses/greenwald_full_evaluation/configuration_figure/panel_C.png": REPOSITORY / "analyses/gbm_spatial_repertoire/review_witnesses/panel_C.png",
        "analyses/greenwald_full_evaluation/configuration_figure/provenance.json": REPOSITORY / "analyses/gbm_spatial_repertoire/review_witnesses/provenance.json",
        "analyses/greenwald_full_evaluation/summaries": greenwald_evidence("summaries"),
        "analyses/greenwald_full_evaluation/configuration_inputs": greenwald_evidence("configuration_inputs"),
        "analyses/greenwald_full_evaluation/context_matching": greenwald_evidence("context_matching"),
        **{f"analyses/gbm_spatial_repertoire/{name}": greenwald_evidence("gbm_spatial_repertoire") / name
           for name in ("details", "pair_profiles", "shortlist_vetting", "statistical_frequencies",
                        "batch_01_maps.csv", "coverage.csv", "inspected_views.csv", "new_local_assessments.csv",
                        "provenance.json", "review.html", "review_provenance.json", "source_equivalence.csv.gz")},
        "analyses/greenwald_ac_opc/summaries": data_root() / "heiland_tyrosh/analysis/greenwald_ac_opc/summaries",
        "analyses/greenwald_full/production_config.json": data_root() / "heiland_tyrosh/CurrentResults/tilescope_run_config.json",
        "analyses/greenwald_full/summaries": data_root() / "heiland_tyrosh/CurrentResults/run_accounting",
        **{f"analyses/scenario_discriminability/recognition/{name}": atlas_root() / "recognition" / name
           for name in ("summaries", "positive_audit", "all_statistics", "stat_packs")},
        **{f"analyses/figure1_overview/{name}": data_root() / "figure1_overview" / name
           for name in ('inputs', 'runs', 'run.config.json', 'provenance.json', 'validation.json')},
        **{f"analyses/{old}": REPOSITORY / f"analyses/figures/scripts/{new}"
           for old,new in [('figure1_overview','overview'),('figure2_scenario_catalogue','benchmark'),('figure3_greenwald','greenwald')]},
        **{f"analyses/greenwald_full_evaluation/{name}": greenwald_evidence(name)
           for name in ("scenario_cohort", "two_level_evidence", "local_frequencies")},
        "benchmark_manifests/benchmark_plan.json": corpus_root() / "benchmark_plan.json",
        "benchmark_manifests/corpus_index.json": corpus_root() / "corpus_index.json",
        "benchmark_manifests/feasibility_availability.csv": generated_corpus_root() / "feasibility_availability.csv",
        "benchmark_manifests/tilescope_run_config.json": current_results_root() / "tilescope_run_config.json",
        "corpus": corpus_root(),
        "analyses/scenario_discriminability/outputs": atlas_root(),
        "analyses/scenario_feasibility/outputs": feasibility_root(),
    }


def _safe_join(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Reference must stay within its root: {relative!r}")
    return root / rel


def _relocate_data_path(path: Path) -> Path:
    """Resolve older layouts recorded in existing provenance."""
    if not path.is_relative_to(benchmark_root()):
        return path
    parts = path.relative_to(benchmark_root()).parts
    if parts[:1] == ("figure2_scenario_catalogue",):
        tail=parts[1:]
        if tail[:2] == ('panels','main'):
            return figure2_root().joinpath('panels',*tail[2:])
        return figure2_root().joinpath(*tail)
    if parts[:1] == ("scenario_discriminability",):
        return atlas_root().joinpath(*parts[1:])
    if len(parts) >= 4 and parts[0] == "corpus" and parts[3] == "inputs":
        return corpus_root().joinpath(*parts[1:3], *parts[4:])
    if len(parts) >= 4 and parts[0] == "corpus" and parts[3] == "tilescope":
        return current_results_root().joinpath(*parts[1:3], *parts[4:])
    if parts[:2] == ("analysis", "scenario_discriminability"):
        return atlas_root().joinpath(*parts[2:])
    if parts[:2] == ("analysis", "feasibility"):
        return feasibility_root().joinpath(*parts[2:])
    return path


def resolve_reference(value: str | Path, *, repository: Path = REPOSITORY) -> Path:
    """Resolve current paths and saved references without rewriting provenance.

    Absolute paths from an older machine are rebased only through named
    TileScope-results/TileScope-data roots. Arbitrary user paths stay explicit.
    """
    raw = str(value)
    if raw.startswith("data://"):
        return _relocate_data_path(_safe_join(data_root(), raw[7:]))
    if raw.startswith("core://"):
        return _safe_join(core_root(), raw[7:])
    path = Path(raw).expanduser()
    if ".." in path.parts:
        raise ValueError(f"Reference must stay within its root: {value!r}")
    if path.is_absolute():
        if path.is_relative_to(repository):
            path = path.relative_to(repository)
        elif path.is_relative_to(data_root()):
            return _relocate_data_path(path)
        else:
            parts = path.parts
            lowered = [p.lower() for p in parts]
            if "tilescope-results" in lowered:
                index = lowered.index("tilescope-results")
                path = Path(*parts[index+1:])
            elif "tilescope-data" in lowered:
                index = lowered.index("tilescope-data")
                return _relocate_data_path(_safe_join(data_root(), str(Path(*parts[index+1:]))))
            else:
                return path
    for name, target in relocated_directories().items():
        if path.is_relative_to(name):
            return _relocate_data_path(_safe_join(target, str(path.relative_to(name))))
    return _safe_join(repository, str(path))


def reference(path: Path, *, repository: Path = REPOSITORY) -> str:
    """Write current location-independent paths, not historical aliases."""
    path = Path(path).absolute()
    if path.is_relative_to(repository):
        return str(path.relative_to(repository))
    if path.is_relative_to(data_root()):
        return "data://" + path.relative_to(data_root()).as_posix()
    if path.is_relative_to(core_root()):
        return "core://" + path.relative_to(core_root()).as_posix()
    raise ValueError(f"Cannot publish a reference outside configured roots: {path}")


if __name__ == "__main__":
    import json
    print(json.dumps({"repository": str(REPOSITORY), "data": str(data_root()),
                      "input_data": str(input_data_root()),
                      "supplementary_data": str(supplementary_data_root()),
                      "core": str(core_root()),
                      "benchmark": str(benchmark_root())}, indent=2))
