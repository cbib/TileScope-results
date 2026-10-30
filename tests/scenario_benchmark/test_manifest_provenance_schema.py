"""Pin the intentional provenance-field change introduced by
STRUCT-SCENARIO-01: retired fields absent + new fields present on
:class:`InstanceManifestRow`. Combined with the existing scenario
invariants (test_scenario_invariants + per-scenario tests), this
freezes the "generator output byte-identical apart from intentional
provenance changes" acceptance contract."""
from __future__ import annotations

import dataclasses

from scenario_benchmark import InstanceManifestRow


RETIRED_FIELDS = frozenset({
    "generator_version",
    "config_sha256",
    "git_status",
})

NEW_PROVENANCE_FIELDS = frozenset({
    "manifest_schema_version",
    "generator_commit_sha",
    "generator_source_state",
    "source_data_sha256",
})


def _field_names() -> frozenset[str]:
    return frozenset(f.name for f in dataclasses.fields(InstanceManifestRow))


def test_retired_provenance_fields_are_gone():
    """The three v1/v2 provenance fields must be absent."""
    absent = RETIRED_FIELDS - _field_names()
    assert absent == RETIRED_FIELDS, (
        "InstanceManifestRow still carries retired provenance fields: "
        f"{sorted(RETIRED_FIELDS & _field_names())}"
    )


def test_new_provenance_fields_are_present():
    """The four post-cutover provenance fields must be present."""
    missing = NEW_PROVENANCE_FIELDS - _field_names()
    assert not missing, (
        f"InstanceManifestRow missing new provenance fields: {sorted(missing)}"
    )


def test_manifest_schema_version_default_is_one():
    """Schema version starts at 1 in the flat-package layout."""
    row = InstanceManifestRow(
        scenario_instance_id="test",
        counterfactual_set_id="test-set",
        counterfactual_role="positive",
        scenario_name="broad_colocalization",
        feature_a_id="gene_a",
        feature_b_id="gene_b",
        nuisance_id="none",
    )
    assert row.manifest_schema_version == 1
