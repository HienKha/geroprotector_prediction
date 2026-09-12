from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from geroprotector.config import (
    resolve_config,
    unresolved_placeholders,
    validate_protocol_lock,
)
from geroprotector.hashing import sha256_file
from geroprotector.validation.leakage_checks import assert_internal_config_safe
from geroprotector.validation.nested_cv import (
    _model_specs,
    _outer_axes,
    _split_strategy,
)
from geroprotector.validation.paper_split_registry import (
    INNER_REGISTRY_COLUMNS,
    PAPER_REGISTRY_COLUMNS,
    PaperSplitIntegrityError,
    _partition_components,
    validate_paper_registries,
)
from geroprotector.validation.split_registry import grouped_inner_folds

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_registries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    test_identities = tuple(
        line.strip()
        for line in (ROOT / "configs" / "paper80_v3_1_test_identities.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    train_identities = tuple(f"TRN{index:011d}" for index in range(305))
    identities = (*test_identities, *train_identities)
    labels = {
        **{value: int(index >= 29) for index, value in enumerate(test_identities)},
        **{value: int(index >= 151) for index, value in enumerate(train_identities)},
    }
    structures = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "C=C", "N=N", "O=O")
    curated = pd.DataFrame(
        {
            "compound_id": [f"cmp::{value}" for value in identities],
            "standardized_parent_smiles": [
                structures[index % len(structures)] for index in range(len(identities))
            ],
            "connectivity_inchikey": identities,
            "identity_group_id": identities,
            "label": [labels[value] for value in identities],
        }
    )
    test_compound_ids = {f"cmp::{value}" for value in test_identities}
    components = _partition_components(curated, test_compound_ids=test_compound_ids)
    outer = pd.DataFrame(
        [
            {
                "compound_id": f"cmp::{identity}",
                "label": labels[identity],
                "component_id": str(components.loc[f"cmp::{identity}"]),
                "repeat": 0,
                "outer_fold": 0 if identity in test_identities else 1,
                "role": "paper_test" if identity in test_identities else "paper_train",
            }
            for identity in identities
        ],
        columns=list(PAPER_REGISTRY_COLUMNS),
    )
    train_rows = outer.loc[outer["role"].eq("paper_train")].set_index("compound_id")
    folds = grouped_inner_folds(
        train_rows["label"], train_rows["component_id"], n_splits=3, seed=142
    )
    assignment = {
        compound: fold for fold, (_, validation) in enumerate(folds) for compound in validation
    }
    inner = pd.DataFrame(
        [
            {
                "compound_id": row.compound_id,
                "label": int(row.label),
                "component_id": row.component_id,
                "repeat": 0,
                "outer_fold": 0,
                "inner_fold": assignment[row.compound_id],
                "role": "inner_validation",
                "assignment_seed": 142,
            }
            for row in train_rows.reset_index().itertuples(index=False)
        ],
        columns=list(INNER_REGISTRY_COLUMNS),
    )
    return curated, outer, inner, test_identities


def test_bis_configs_are_locked_and_preserve_base_model_suites() -> None:
    v5 = resolve_config(ROOT / "configs" / "v5.yaml")
    v6 = resolve_config(ROOT / "configs" / "v6.yaml")
    v5bis = resolve_config(ROOT / "configs" / "v5bis.yaml")
    v6bis = resolve_config(ROOT / "configs" / "v6bis.yaml")
    for name, config in (("v5bis", v5bis), ("v6bis", v6bis)):
        assert_internal_config_safe(config)
        validate_protocol_lock(root=ROOT, contract_name=name, config=config)
        assert _outer_axes(config) == ((0, 0),)
        assert _split_strategy(config) == "paper_random_80_20"
    assert _model_specs(v5bis, suite="full") == _model_specs(v5, suite="full")
    assert _model_specs(v6bis, suite="full") == _model_specs(v6, suite="full")
    assert unresolved_placeholders(v5bis) == []
    assert len(unresolved_placeholders(v6bis)) == 24


def test_paper_identity_lock_is_exact_and_hash_pinned() -> None:
    config = resolve_config(ROOT / "configs" / "v5bis.yaml")
    path = ROOT / config["evaluation_design"]["test_identity_file"]
    assert sha256_file(path) == config["evaluation_design"]["test_identity_file_sha256"]
    values = path.read_text(encoding="utf-8").splitlines()
    assert len(values) == len(set(values)) == 77


def test_paper_registry_contract_accepts_only_locked_305_77_partition() -> None:
    curated, outer, inner, test_identities = _synthetic_registries()
    audit = validate_paper_registries(
        curated,
        outer,
        inner,
        expected_test_identities=test_identities,
        inner_folds=3,
    )
    assert audit["n_train"] == 305
    assert audit["n_test"] == 77
    assert audit["test_class_0"] == 29
    assert audit["test_class_1"] == 48
    assert audit["identity_overlap_count"] == 0

    corrupted = outer.copy()
    test_index = corrupted.index[corrupted["role"].eq("paper_test")][0]
    train_index = corrupted.index[corrupted["role"].eq("paper_train")][0]
    corrupted.loc[test_index, "role"] = "paper_train"
    corrupted.loc[test_index, "outer_fold"] = 1
    corrupted.loc[train_index, "role"] = "paper_test"
    corrupted.loc[train_index, "outer_fold"] = 0
    with pytest.raises(PaperSplitIntegrityError, match="membership differs"):
        validate_paper_registries(
            curated,
            corrupted,
            inner,
            expected_test_identities=test_identities,
            inner_folds=3,
        )

    fractional = outer.copy()
    fractional["outer_fold"] = fractional["outer_fold"].astype(object)
    fractional.loc[fractional.index[0], "outer_fold"] = 0.5
    with pytest.raises(PaperSplitIntegrityError, match="exact finite integers"):
        validate_paper_registries(
            curated,
            fractional,
            inner,
            expected_test_identities=test_identities,
            inner_folds=3,
        )

    wrong_inner_seed = inner.copy()
    wrong_inner_seed["assignment_seed"] = 143
    with pytest.raises(PaperSplitIntegrityError, match="assignment seed"):
        validate_paper_registries(
            curated,
            outer,
            wrong_inner_seed,
            expected_test_identities=test_identities,
            inner_folds=3,
        )
