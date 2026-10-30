"""Review checkout portability and code/data separation, without real datasets."""
from pathlib import Path
import ast

import pytest

from tilescope_results import paths
from scenario_benchmark.datasets import resolve_dataset_root


def test_data_root_not_a_personal_home(monkeypatch, tmp_path):
    monkeypatch.delenv("TILESCOPE_DATA_DIR", raising=False)
    monkeypatch.delenv('TILESCOPE_FIGURE_DIR', raising=False)
    monkeypatch.setattr(paths, "REPOSITORY", tmp_path / "review checkout")
    assert paths.data_root() == tmp_path / "TileScope-data"
    monkeypatch.setenv("TILESCOPE_DATA_DIR", str(tmp_path / "mounted data"))
    assert paths.corpus_root() == tmp_path / "mounted data/input_data/benchmark/corpus"
    assert paths.generated_corpus_root() == tmp_path / "mounted data/scenario_benchmark/corpus"
    assert paths.atlas_root() == tmp_path / "review checkout/analyses/scenario_discriminability/outputs"
    assert paths.figure2_root() == tmp_path / "review checkout/analyses/figures/main/Figure2"
    assert paths.input_data_root() == tmp_path / 'mounted data/input_data'
    assert paths.supplementary_data_root(3) == tmp_path / 'mounted data/supplementary_data/Supplementary_Data_3'
    monkeypatch.setenv('TILESCOPE_FIGURE_DIR', str(tmp_path / 'artwork'))
    assert paths.figure2_root() == tmp_path / 'artwork/main/Figure2'


def test_dataset_overrides_and_global_root(monkeypatch, tmp_path):
    monkeypatch.setenv("TILESCOPE_DATA_DIR", str(tmp_path))
    for var in ("TILESCOPE_DATA_ROOT", "TILESCOPE_HCC_ROOT", "TILESCOPE_MOUSE_ROOT"):
        monkeypatch.delenv(var, raising=False)
    assert resolve_dataset_root("greenwald") == tmp_path / "heiland_tyrosh/CSVs"
    assert resolve_dataset_root("mouse") == tmp_path / "mouse"
    monkeypatch.setenv("TILESCOPE_DATA_ROOT", str(tmp_path / "legacy override"))
    assert resolve_dataset_root("greenwald") == tmp_path / "legacy override"
    assert resolve_dataset_root("greenwald", explicit=tmp_path / "explicit") == tmp_path / "explicit"


@pytest.mark.parametrize("name", ["corpus/family/case/inputs/positive.csv",
                                  "corpus/family/case/tilescope/positive/roi_correlations.csv",
                                  "benchmark_manifests/tilescope_run_config.json",
                                  "benchmark_manifests/benchmark_plan.json",
                                  "analyses/scenario_discriminability/outputs/atlas.csv"])
def test_logical_references_survive_two_locations(monkeypatch, tmp_path, name):
    for location in (tmp_path / "first", tmp_path / "reviewer's data"):
        monkeypatch.setenv("TILESCOPE_DATA_DIR", str(location))
        path = paths.resolve_reference(name)
        if name.startswith("analyses/"):
            assert path.is_relative_to(paths.REPOSITORY)
        else:
            assert path.is_relative_to(location)
        assert paths.resolve_reference(paths.reference(path)) == path
        assert paths.resolve_reference(Path("/previous/machine/TileScope-results") / name) == path


def test_saved_external_paths_follow_layout_change(monkeypatch, tmp_path):
    monkeypatch.setenv("TILESCOPE_DATA_DIR", str(tmp_path / "data"))
    for old, expected in [
        ("corpus/family/case/inputs/positive.csv", paths.corpus_root() / "family/case/positive.csv"),
        ("corpus/family/case/tilescope/positive/roi_correlations.csv",
         paths.current_results_root() / "family/case/positive/roi_correlations.csv"),
        ("analysis/scenario_discriminability/atlas.csv", paths.atlas_root() / "atlas.csv"),
        ("scenario_discriminability/recognition/full_catalogue/values.csv",
         paths.atlas_root() / "recognition/full_catalogue/values.csv"),
        ("figure2_scenario_catalogue/panels/main/B0_clustered_measurement_profiles.pdf",
         paths.figure2_root() / "panels/B0_clustered_measurement_profiles.pdf"),
    ]:
        assert paths.resolve_reference("data://scenario_benchmark/" + old) == expected
        assert paths.resolve_reference(Path("/previous/TileScope-data/scenario_benchmark") / old) == expected
        assert paths.resolve_reference(paths.benchmark_root() / old) == expected
        assert paths.resolve_reference(paths.reference(expected)) == expected


def test_no_implicit_old_data_fallback(monkeypatch, tmp_path):
    repo = tmp_path / "checkout"
    (repo / "corpus").mkdir(parents=True)
    monkeypatch.setenv("TILESCOPE_DATA_DIR", str(tmp_path / "external"))
    assert paths.resolve_reference(repo / "corpus", repository=repo) == paths.corpus_root()


def test_figure_scripts_and_outputs_share_results_root():
    for name in ("panels/B0_clustered_measurement_profiles.pdf",
                 "source_data/clustering_matrix.csv", "provenance.json"):
        path = paths.figure2_root() / name
        assert paths.resolve_reference(paths.reference(path)) == path
        assert path.is_relative_to(paths.REPOSITORY)


def test_invalid_reference_cannot_escape_root(monkeypatch, tmp_path):
    monkeypatch.setenv("TILESCOPE_DATA_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        paths.resolve_reference("data://../outside")
    with pytest.raises(ValueError):
        paths.resolve_reference("corpus/../../outside")


def test_active_python_has_no_personal_path_literals():
    """Historical prose/provenance may name a machine; runtime code may not."""
    violations = []
    for folder in ("scenario_benchmark", "tilescope_results", "analyses"):
        for file in (paths.REPOSITORY / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(file.read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    value = node.value.strip()
                    if value.startswith(("/Users/", "/home/", "~/Projects/", "~/Notes/")):
                        violations.append(f"{file.relative_to(paths.REPOSITORY)}:{node.lineno}")
    assert not violations, violations


def test_generated_review_and_authored_inputs_have_distinct_roots(monkeypatch, tmp_path):
    monkeypatch.setenv("TILESCOPE_DATA_DIR", str(tmp_path / "data"))
    code = "analyses/gbm_spatial_repertoire/"
    output = paths.greenwald_evidence("gbm_spatial_repertoire")
    assert paths.resolve_reference(code + "review.html") == output / "review.html"
    assert paths.resolve_reference(code + "pair_profiles/features.csv") == output / "pair_profiles/features.csv"
    assert paths.resolve_reference(code + "review.py") == paths.REPOSITORY / code / "review.py"
    assert paths.resolve_reference(code + "shortlist_vetting/observations.json") == paths.REPOSITORY / code / "shortlist_vetting/observations.json"
    assert paths.resolve_reference(code + "pair_profiles/FINDINGS.md") == output / "pair_profiles/FINDINGS.md"


def test_mouse_uses_current_bank_without_legacy_fallback(monkeypatch, tmp_path):
    from scenario_benchmark.datasets import enumerate_samples, resolve_sample_csv

    monkeypatch.setenv("TILESCOPE_MOUSE_ROOT", str(tmp_path))
    old = tmp_path / "MCT1_CTL" / "CSVs" / "old_expression_data_A_B.csv"
    old.parent.mkdir(parents=True)
    old.write_text("x,y,roi_id\n1,2,1\n")
    bank = tmp_path / "mouse_gbm_spatial_programs" / "phase6_harmonized" / "MCT1_CTL_signature_score_bank.csv"
    bank.parent.mkdir(parents=True)
    bank.write_text("x,y,roi_id\n3,4,2\n")
    assert resolve_sample_csv("MCT1_CTL", dataset="mouse")[0] == bank
    assert enumerate_samples("mouse") == ["MCT1_CTL"]
    bank.unlink()
    assert enumerate_samples("mouse") == []
    with pytest.raises(FileNotFoundError):
        resolve_sample_csv("MCT1_CTL", dataset="mouse")
