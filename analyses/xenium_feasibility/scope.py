"""Complete Meylan cohort and the retained Janesick breast sample."""
from dataclasses import dataclass
from pathlib import Path

import json

from tilescope_results.paths import data_root
from analyses.meylan_xenium_preparation.cohort import SAMPLE_CELLS, input_path

TABLES = (
    "roi_enrichment", "roi_correlations", "roi_source_moran",
    "roi_replicate_correlations", "roi_replicate_enrichment",
    "roi_tessellation_correlations", "roi_tessellation_enrichment", "roi_tessellation_moran",
    "roi_consensus_correlations", "roi_consensus_enrichment", "roi_consensus_moran",
    "roi_pair_interface_response", "roi_cross_grain_sensitivity",
    "roi_support_status", "component_radial_shell_correlogram", "component_region_replicate_status",
    "component_geometry", "component_domain_regions",
)


def evidence_root():
    return data_root() / "xenium_feasibility/analysis"


@dataclass(frozen=True)
class Run:
    study: str
    sample: str
    patient: str
    pair: str
    feature_a: str
    feature_b: str
    input: Path
    output: Path
    status: Path
    config: Path
    genes_a: str
    genes_b: str
    expression_units: str

    @property
    def identity(self):
        return dict(study=self.study, sample=self.sample, patient=self.patient, pair=self.pair,
                    feature_a=self.feature_a, feature_b=self.feature_b)


def runs():
    root = data_root()
    preparation = Path(__file__).parents[1] / "meylan_xenium_preparation"
    definition = json.loads((preparation / "pair.json").read_text())
    gbm = root / "meylan_xenium"
    pair = definition["pair"]
    for sample in SAMPLE_CELLS:
        yield Run("GBM", sample, sample.split('_')[0], pair, definition["feature_a"], definition["feature_b"],
                  input_path(gbm,sample), gbm / f"CurrentResults/{sample}/{pair}",
                  gbm / f"CurrentResults/{sample}/{pair}/runner_status.json",
                  preparation / "run.config.json", ";".join(definition["genes_a"]),
                  ";".join(definition["genes_b"]), definition["calculation"])
    breast = root / "Janesick"
    yield Run("Breast", "sample1_rep1", "Janesick_sample1", "ACTA2_FASN", "ACTA2", "FASN",
              breast / "sample1_rep1/tilescope_inputs/CSVs/Janesick_S1R1_expression_data_ACTA2_FASN_parent_rois.csv",
              breast / "CurrentResults/sample1_rep1/ACTA2_FASN",
              breast / "CurrentResults/sample1_rep1/ACTA2_FASN/runner_status.json",
              breast / "CurrentResults/tilescope_run_config.json", "ACTA2", "FASN",
              "raw Xenium transcript count per segmented cell")
