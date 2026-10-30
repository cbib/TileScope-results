"""scenario_benchmark -- real-substrate scenario corpus generator.

Executable scenarios: three calibration (broad_colocalization,
focal_pattern, spatial_gradient) plus four challenge
(dispersed_infiltration, embedded_domains, nested_foci,
contact_interface). ``focal_pattern`` carries colocated + separated
modes; ``spatial_gradient`` carries concordant + opposed relationship
modes; ``contact_interface`` carries responder_activation + anchor_fade
modes.

Scientific contract: scenario_benchmark/CONTRACT.md.
Contract: scenario_benchmark/CONTRACT.md.

Import boundary: this package does NOT import TileScope
(tests/scenario_benchmark/test_independence.py enforces via AST scan).
"""

from scenario_benchmark.adapters.greenwald import (
    load_greenwald_sample_scaffold,
    stable_spot_id,
)
from scenario_benchmark.sample_scaffold import (
    SampleScaffold,
    load_sample_scaffold,
)
from scenario_benchmark.scenarios import (
    BROAD_COLOCALIZATION_NAME, BROAD_COLOCALIZATION_ROLES,
    NESTED_FOCI_NAME, NESTED_FOCI_ROLES,
    CONTACT_INTERFACE_ANCHOR_FADE_INVARIANTS,
    CONTACT_INTERFACE_MODE_ANCHOR_FADE,
    CONTACT_INTERFACE_MODE_RESPONDER_ACTIVATION,
    CONTACT_INTERFACE_MODES,
    CONTACT_INTERFACE_NAME,
    CONTACT_INTERFACE_RESPONDER_ACTIVATION_INVARIANTS,
    CONTACT_INTERFACE_ROLES,
    EMBEDDED_DOMAINS_NAME, EMBEDDED_DOMAINS_ROLES,
    FOCAL_PATTERN_INVARIANTS,
    FOCAL_PATTERN_MODE_COLOCATED,
    FOCAL_PATTERN_MODE_SEPARATED,
    FOCAL_PATTERN_MODES,
    FOCAL_PATTERN_NAME, FOCAL_PATTERN_ROLES,
    GRADIENT_RELATIONSHIP_CONCORDANT, GRADIENT_RELATIONSHIP_OPPOSED,
    DISPERSED_INFILTRATION_NAME, DISPERSED_INFILTRATION_ROLES,
    SPATIAL_GRADIENT_NAME, SPATIAL_GRADIENT_ROLES,
    ScenarioSet,
    build_broad_colocalization_set,
    build_nested_foci_set,
    build_contact_interface_set,
    build_embedded_domains_set,
    build_focal_pattern_set,
    build_dispersed_infiltration_set,
    build_spatial_gradient_set,
)
from scenario_benchmark.scenarios.serialization import (
    TILESCOPE_INPUT_COLUMNS,
    TRUTH_LEDGER_COLUMNS,
    rows_to_tilescope_dataframe,
    rows_to_truth_dataframe,
    write_tilescope_input_csv,
    write_truth_ledger_csv,
)
from scenario_benchmark.signal_model import SignalModel
from scenario_benchmark.truth_schemas import (
    CHALLENGE_SCENARIOS,
    EASY_CALIBRATION_SCENARIOS,
    EXECUTABLE_SCENARIO_NAMES,
    PHASE1_SCENARIO_NAMES,
    SPOT_TRUTH_PK,
    InstanceManifestRow,
    SpotTruthRow,
)


__all__ = [
    # Vocabulary
    "EXECUTABLE_SCENARIO_NAMES",
    "PHASE1_SCENARIO_NAMES",
    "EASY_CALIBRATION_SCENARIOS",
    "CHALLENGE_SCENARIOS",
    "SPOT_TRUTH_PK",
    "BROAD_COLOCALIZATION_NAME", "BROAD_COLOCALIZATION_ROLES",
    "FOCAL_PATTERN_NAME", "FOCAL_PATTERN_ROLES",
    "FOCAL_PATTERN_MODES",
    "FOCAL_PATTERN_MODE_COLOCATED",
    "FOCAL_PATTERN_MODE_SEPARATED",
    "FOCAL_PATTERN_INVARIANTS",
    "SPATIAL_GRADIENT_NAME", "SPATIAL_GRADIENT_ROLES",
    "GRADIENT_RELATIONSHIP_OPPOSED", "GRADIENT_RELATIONSHIP_CONCORDANT",
    "DISPERSED_INFILTRATION_NAME", "DISPERSED_INFILTRATION_ROLES",
    "EMBEDDED_DOMAINS_NAME", "EMBEDDED_DOMAINS_ROLES",
    "NESTED_FOCI_NAME", "NESTED_FOCI_ROLES",
    "CONTACT_INTERFACE_NAME",
    "CONTACT_INTERFACE_ROLES",
    "CONTACT_INTERFACE_MODES",
    "CONTACT_INTERFACE_MODE_RESPONDER_ACTIVATION",
    "CONTACT_INTERFACE_MODE_ANCHOR_FADE",
    "CONTACT_INTERFACE_RESPONDER_ACTIVATION_INVARIANTS",
    "CONTACT_INTERFACE_ANCHOR_FADE_INVARIANTS",
    # Primary records
    "InstanceManifestRow",
    "SpotTruthRow",
    # Sample scaffold
    "SampleScaffold",
    "load_sample_scaffold",
    # Signal model
    "SignalModel",
    # Scenario builders
    "ScenarioSet",
    "build_broad_colocalization_set",
    "build_focal_pattern_set",
    "build_spatial_gradient_set",
    "build_dispersed_infiltration_set",
    "build_embedded_domains_set",
    "build_nested_foci_set",
    "build_contact_interface_set",
    # Serialization
    "TILESCOPE_INPUT_COLUMNS",
    "TRUTH_LEDGER_COLUMNS",
    "rows_to_tilescope_dataframe",
    "rows_to_truth_dataframe",
    "write_tilescope_input_csv",
    "write_truth_ledger_csv",
    # Adapters
    "load_greenwald_sample_scaffold",
    "stable_spot_id",
]
