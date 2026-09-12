from __future__ import annotations

import copy

import pandas as pd
import pytest

from geroprotector.validation.split_registry import (
    INNER_REGISTRY_COLUMNS,
    REGISTRY_COLUMNS,
    SplitIntegrityError,
    fold_ids,
    locked_inner_fold_ids,
    registry_canonical_sha256,
    validate_inner_registry,
    validate_registry,
    validate_selection_folds,
)


def _set_fractional_label(frame: pd.DataFrame) -> None:
    frame["label"] = frame["label"].astype(float)
    frame.loc[0, "label"] = 0.5


def _outer_registry(labels: pd.Series, groups: pd.Series) -> pd.DataFrame:
    assignments = (
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
    )
    rows = []
    for repeat, folds in enumerate(assignments):
        for compound, fold in zip(labels.index, folds, strict=True):
            rows.append(
                {
                    "compound_id": compound,
                    "label": int(labels.loc[compound]),
                    "component_id": str(groups.loc[compound]),
                    "repeat": repeat,
                    "outer_fold": fold,
                    "role": "outer_test",
                }
            )
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def _inner_registry(outer: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for repeat in (0, 1):
        repeated = outer.loc[outer["repeat"].eq(repeat)]
        for outer_fold in (0, 1):
            training = repeated.loc[~repeated["outer_fold"].eq(outer_fold)].copy()
            by_label = {
                label: sorted(training.loc[training["label"].eq(label), "compound_id"])
                for label in (0, 1)
            }
            assignment = {
                compound: index
                for label in (0, 1)
                for index, compound in enumerate(by_label[label])
            }
            for row in training.itertuples(index=False):
                rows.append(
                    {
                        "compound_id": row.compound_id,
                        "label": row.label,
                        "component_id": row.component_id,
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "inner_fold": assignment[row.compound_id],
                        "role": "inner_validation",
                        "assignment_seed": 1000 + repeat * 10 + outer_fold,
                    }
                )
    return pd.DataFrame(rows, columns=INNER_REGISTRY_COLUMNS)


def test_outer_and_inner_registry_happy_path(
    balanced_labels: pd.Series, unique_groups: pd.Series
) -> None:
    outer = _outer_registry(balanced_labels, unique_groups)
    audit = validate_registry(
        outer,
        expected_ids=balanced_labels.index,
        outer_repeats=2,
        outer_folds=2,
    )
    assert audit["passed"] is True
    assert audit["n_rows"] == 16
    assert audit["registry_sha256"] == registry_canonical_sha256(outer)
    assert len(audit["folds"]) == 4

    inner = _inner_registry(outer)
    validate_inner_registry(outer, inner, outer_repeats=2, outer_folds=2, inner_folds=2)
    folds = locked_inner_fold_ids(inner, repeat=0, outer_fold=0)
    train, test = fold_ids(outer, repeat=0, outer_fold=0)
    assert set().union(*(set(valid) for _, valid in folds)) == set(train)
    assert set(train).isdisjoint(test)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda frame: frame.__setitem__("role", "train"),
        _set_fractional_label,
        lambda frame: frame.loc.__setitem__((8, "label"), 1 - int(frame.loc[8, "label"])),
        lambda frame: frame.loc.__setitem__((8, "component_id"), "drifted"),
        lambda frame: frame.loc.__setitem__((0, "component_id"), frame.loc[1, "component_id"]),
    ],
)
def test_outer_registry_rejects_contract_violations(
    mutation, balanced_labels: pd.Series, unique_groups: pd.Series
) -> None:
    outer = _outer_registry(balanced_labels, unique_groups)
    mutation(outer)
    with pytest.raises(SplitIntegrityError):
        validate_registry(
            outer,
            expected_ids=balanced_labels.index,
            outer_repeats=2,
            outer_folds=2,
        )


def test_registry_hash_is_row_order_invariant(
    balanced_labels: pd.Series, unique_groups: pd.Series
) -> None:
    outer = _outer_registry(balanced_labels, unique_groups)
    shuffled = outer.sample(frac=1.0, random_state=73).reset_index(drop=True)
    assert registry_canonical_sha256(outer) == registry_canonical_sha256(shuffled)


def test_inner_registry_rejects_outer_test_membership_and_component_leakage(
    balanced_labels: pd.Series, unique_groups: pd.Series
) -> None:
    outer = _outer_registry(balanced_labels, unique_groups)
    inner = _inner_registry(outer)
    bad_membership = inner.copy()
    target = bad_membership.index[
        bad_membership["repeat"].eq(0) & bad_membership["outer_fold"].eq(0)
    ][0]
    bad_membership.loc[target, "compound_id"] = "outer-test-injected"
    with pytest.raises(SplitIntegrityError, match="outer-train membership"):
        validate_inner_registry(
            outer, bad_membership, outer_repeats=2, outer_folds=2, inner_folds=2
        )

    leaking = inner.copy()
    subset = leaking.loc[leaking["repeat"].eq(0) & leaking["outer_fold"].eq(0)]
    left, right = subset.index[:2]
    leaking.loc[right, "component_id"] = leaking.loc[left, "component_id"]
    with pytest.raises(SplitIntegrityError):
        validate_inner_registry(outer, leaking, outer_repeats=2, outer_folds=2, inner_folds=2)


def test_selection_folds_are_complete_grouped_oof(
    balanced_labels: pd.Series,
    unique_groups: pd.Series,
    two_fold_selection,
) -> None:
    validated = validate_selection_folds(
        fit_ids=tuple(balanced_labels.index),
        labels=balanced_labels,
        groups=unique_groups,
        folds=two_fold_selection,
    )
    assert len(validated) == 2

    repeated_validation = copy.deepcopy(two_fold_selection)
    repeated_validation[1] = repeated_validation[0]
    with pytest.raises(SplitIntegrityError, match="exactly once"):
        validate_selection_folds(
            fit_ids=tuple(balanced_labels.index),
            labels=balanced_labels,
            groups=unique_groups,
            folds=repeated_validation,
        )

    component_leak = unique_groups.copy()
    component_leak.iloc[2] = component_leak.iloc[0]
    with pytest.raises(SplitIntegrityError, match="crosses a component"):
        validate_selection_folds(
            fit_ids=tuple(balanced_labels.index),
            labels=balanced_labels,
            groups=component_leak,
            folds=two_fold_selection,
        )


def test_selection_folds_reject_fractional_labels_and_missing_groups(
    balanced_labels: pd.Series,
    unique_groups: pd.Series,
    two_fold_selection,
) -> None:
    fractional = balanced_labels.astype(float)
    fractional.iloc[0] = 0.5
    with pytest.raises(SplitIntegrityError, match="integer"):
        validate_selection_folds(
            fit_ids=tuple(fractional.index),
            labels=fractional,
            groups=unique_groups,
            folds=two_fold_selection,
        )

    missing_group = unique_groups.copy()
    missing_group.iloc[0] = pd.NA
    with pytest.raises(SplitIntegrityError, match="group"):
        validate_selection_folds(
            fit_ids=tuple(balanced_labels.index),
            labels=balanced_labels,
            groups=missing_group,
            folds=two_fold_selection,
        )
