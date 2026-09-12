from __future__ import annotations

import json

import pandas as pd

from geroprotector.screening_blend_external import (
    _prepare_records,
    _publication_geometry,
)


def test_external_preparation_excludes_d1_and_collapses_connectivity() -> None:
    mapping = pd.DataFrame(
        {
            "provider_record_id": ["one", "two", "three"],
            "compound_name": ["ethanol-a", "ethanol-b", "blocked-name"],
            "mapping_source": ["provider"] * 3,
            "provider_identifier": ["1", "2", "3"],
            "source_smiles": ["CCO", "OCC", "CCN"],
        }
    )
    identities, ledger = _prepare_records(
        cohort="synthetic",
        mapping=mapping,
        d1_identities=set(),
        d1_names={"blockedname"},
    )
    assert len(identities) == 1
    assert identities.iloc[0].collapsed_mapping_rows == 2
    assert json.loads(identities.iloc[0].source_names_json) == ["ethanol-a", "ethanol-b"]
    blocked = ledger[ledger.provider_record_id == "three"].iloc[0]
    assert blocked.exclusion_reason == "overlap_d1_normalized_name"


def test_publication_geometry_uses_transitive_compound_publication_union() -> None:
    frame = pd.DataFrame(
        {
            "external_id": ["a", "b", "c"],
            "publication_ids_json": [
                json.dumps(["p1"]),
                json.dumps(["p1", "p2"]),
                json.dumps(["p3"]),
            ],
        }
    )
    audit = _publication_geometry(frame)
    assert audit["publication_compound_union_units"] == 2
    assert audit["largest_union_compounds"] == 2
    assert audit["largest_union_fraction"] == 2 / 3
    assert audit["ordinary_cluster_inference_not_dominated"] is False
