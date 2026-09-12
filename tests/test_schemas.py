from __future__ import annotations

import copy
from pathlib import Path

import pytest

from geroprotector.data.schemas import load_schema, validate_records

ROOT = Path(__file__).resolve().parents[1]


def _prediction() -> dict:
    return {
        "run_id": "synthetic-run",
        "pipeline_id": "reference",
        "model_id": "R0_prevalence",
        "split_strategy": "similarity_components",
        "evaluation_role": "outer_test",
        "repeat": 0,
        "outer_fold": 0,
        "compound_id": "cmp::fixture",
        "identity_group_id": "FIXTUREIDENTITY",
        "primary_component_id": "component::fixture",
        "y_true": 0,
        "probability_raw": 0.4,
        "probability_calibrated": 0.45,
        "adaptation_used": False,
        "created_utc": "2026-08-15T00:00:00Z",
    }


def test_prediction_schema_supports_references_and_enforces_label_blindness() -> None:
    schema = load_schema(ROOT, "prediction.schema.json")
    validate_records([_prediction()], schema)

    external = _prediction()
    external.update(
        pipeline_id="V6_TABULAR_FOUNDATION_MODELS",
        evaluation_role="external_label_blind",
        y_true=None,
        repeat=None,
        outer_fold=None,
    )
    validate_records([external], schema)
    external["y_true"] = 1
    with pytest.raises(ValueError, match="Schema validation failed"):
        validate_records([external], schema)


def _metric_record() -> dict:
    return {
        "run_id": "synthetic-run",
        "pipeline_id": "V5_ELIXIRFP_REBUILT",
        "model_id": "v5",
        "split_strategy": "similarity_components",
        "evaluation_role": "outer_test",
        "aggregation_level": "one_final_prediction_per_compound",
        "n_compounds": 4,
        "n_positive": 2,
        "n_negative": 2,
        "prevalence_positive": 0.5,
        "metrics": {
            "ap_positive": 0.8,
            "ap_negative": 0.7,
            "auroc": 0.75,
            "brier": 0.2,
        },
        "paired_comparison": {
            "metric": "ap_positive",
            "delta_candidate_minus_reference": 0.1,
            "ci_low": -0.1,
            "ci_high": 0.2,
            "p_two_sided": 0.4,
            "superiority_probability": 0.7,
            "bootstrap_unit": "primary_component",
            "bootstrap_units": 4,
            "n_resamples": 1000,
            "n_valid_resamples": 998,
            "seed": 1,
            "multiple_testing_adjustment": "none",
        },
        "created_utc": "2026-08-15T00:00:00Z",
    }


def test_metrics_schema_requires_compound_aggregation_and_bootstrap_audit_fields() -> None:
    schema = load_schema(ROOT, "metrics.schema.json")
    record = _metric_record()
    validate_records([record], schema)

    missing_aggregation = copy.deepcopy(record)
    missing_aggregation.pop("aggregation_level")
    with pytest.raises(ValueError, match="aggregation_level"):
        validate_records([missing_aggregation], schema)

    missing_valid_resamples = copy.deepcopy(record)
    missing_valid_resamples["paired_comparison"].pop("n_valid_resamples")
    with pytest.raises(ValueError, match="n_valid_resamples"):
        validate_records([missing_valid_resamples], schema)


def _claims_record() -> dict:
    return {
        "pipeline_id": "V5_ELIXIRFP_REBUILT",
        "evidence_level": "internal_grouped_validation",
        "allowed_claims": ["predictive discrimination under grouped internal validation"],
        "forbidden_claims": ["causal geroprotection"],
        "required_qualifiers": ["source-defined weak reference endpoint"],
        "causal_language_allowed": False,
        "human_efficacy_language_allowed": False,
        "source_label_confounding_must_be_disclosed": True,
        "external_dataset_name": None,
        "locked_utc": "2026-08-15T00:00:00Z",
    }


def test_claims_schema_requires_safety_booleans_instead_of_defaulting_them() -> None:
    schema = load_schema(ROOT, "claims_allowed.schema.json")
    record = _claims_record()
    validate_records([record], schema)
    for field in (
        "causal_language_allowed",
        "human_efficacy_language_allowed",
        "source_label_confounding_must_be_disclosed",
    ):
        missing = copy.deepcopy(record)
        missing.pop(field)
        with pytest.raises(ValueError, match=field):
            validate_records([missing], schema)
