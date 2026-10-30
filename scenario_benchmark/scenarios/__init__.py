"""Scenario orchestrators.

Executable scenarios: three calibration (broad_colocalization,
focal_pattern, spatial_gradient) plus four challenge
(dispersed_infiltration, embedded_domains, nested_foci,
contact_interface). ``focal_pattern`` carries colocated + separated
modes; ``spatial_gradient`` carries concordant + opposed relationship
modes; ``contact_interface`` carries responder_activation + anchor_fade
modes.
"""

from scenario_benchmark.scenarios.broad_colocalization import (
    build_broad_colocalization_set,
    ROLES as BROAD_COLOCALIZATION_ROLES,
    SCENARIO_NAME as BROAD_COLOCALIZATION_NAME,
)
from scenario_benchmark.scenarios.nested_foci import (
    build_nested_foci_set,
    ROLES as NESTED_FOCI_ROLES,
    SCENARIO_NAME as NESTED_FOCI_NAME,
)
from scenario_benchmark.scenarios.contact_interface import (
    build_contact_interface_set,
    CONTACT_INTERFACE_ANCHOR_FADE_INVARIANTS,
    CONTACT_INTERFACE_RESPONDER_ACTIVATION_INVARIANTS,
    MODE_ANCHOR_FADE as CONTACT_INTERFACE_MODE_ANCHOR_FADE,
    MODE_RESPONDER_ACTIVATION as CONTACT_INTERFACE_MODE_RESPONDER_ACTIVATION,
    MODES as CONTACT_INTERFACE_MODES,
    ROLES as CONTACT_INTERFACE_ROLES,
    SCENARIO_NAME as CONTACT_INTERFACE_NAME,
)
from scenario_benchmark.scenarios.embedded_domains import (
    build_embedded_domains_set,
    ROLES as EMBEDDED_DOMAINS_ROLES,
    SCENARIO_NAME as EMBEDDED_DOMAINS_NAME,
)
from scenario_benchmark.scenarios.focal_pattern import (
    build_focal_pattern_set,
    FOCAL_PATTERN_INVARIANTS,
    MODE_COLOCATED as FOCAL_PATTERN_MODE_COLOCATED,
    MODE_SEPARATED as FOCAL_PATTERN_MODE_SEPARATED,
    MODES as FOCAL_PATTERN_MODES,
    ROLES as FOCAL_PATTERN_ROLES,
    SCENARIO_NAME as FOCAL_PATTERN_NAME,
)
from scenario_benchmark.scenarios.spatial_gradient import (
    build_spatial_gradient_set,
    ROLES as SPATIAL_GRADIENT_ROLES,
    SCENARIO_NAME as SPATIAL_GRADIENT_NAME,
    GRADIENT_RELATIONSHIP_OPPOSED,
    GRADIENT_RELATIONSHIP_CONCORDANT,
)
from scenario_benchmark.scenarios.result import ScenarioSet
from scenario_benchmark.scenarios.dispersed_infiltration import (
    build_dispersed_infiltration_set,
    ROLES as DISPERSED_INFILTRATION_ROLES,
    SCENARIO_NAME as DISPERSED_INFILTRATION_NAME,
)


__all__ = [
    "ScenarioSet",
    "BROAD_COLOCALIZATION_NAME",
    "BROAD_COLOCALIZATION_ROLES",
    "build_broad_colocalization_set",
    "FOCAL_PATTERN_NAME",
    "FOCAL_PATTERN_ROLES",
    "FOCAL_PATTERN_MODES",
    "FOCAL_PATTERN_MODE_COLOCATED",
    "FOCAL_PATTERN_MODE_SEPARATED",
    "FOCAL_PATTERN_INVARIANTS",
    "build_focal_pattern_set",
    "SPATIAL_GRADIENT_NAME",
    "SPATIAL_GRADIENT_ROLES",
    "GRADIENT_RELATIONSHIP_OPPOSED",
    "GRADIENT_RELATIONSHIP_CONCORDANT",
    "build_spatial_gradient_set",
    "EMBEDDED_DOMAINS_NAME",
    "EMBEDDED_DOMAINS_ROLES",
    "build_embedded_domains_set",
    "NESTED_FOCI_NAME",
    "NESTED_FOCI_ROLES",
    "build_nested_foci_set",
    "DISPERSED_INFILTRATION_NAME",
    "DISPERSED_INFILTRATION_ROLES",
    "build_dispersed_infiltration_set",
    "CONTACT_INTERFACE_NAME",
    "CONTACT_INTERFACE_ROLES",
    "CONTACT_INTERFACE_MODES",
    "CONTACT_INTERFACE_MODE_RESPONDER_ACTIVATION",
    "CONTACT_INTERFACE_MODE_ANCHOR_FADE",
    "CONTACT_INTERFACE_RESPONDER_ACTIVATION_INVARIANTS",
    "CONTACT_INTERFACE_ANCHOR_FADE_INVARIANTS",
    "build_contact_interface_set",
]
