"""Command-line entrypoint for the leakage-audited V5/V6 workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .audit import validate_core_runtime
from .config import (
    resolve_config,
    resolved_config_sha256,
    unresolved_placeholders,
    validate_locked_shared_contract,
    validate_protocol_lock,
)
from .data.acquire import acquire_sources
from .data.curate import curate_cohort, load_curated_cohort
from .fixed_blend_paper405 import run as run_fixed_blend_paper405
from .hashing import sha256_file
from .hypermoltab_paper405 import run as run_hypermoltab_paper405
from .models.v6.checkpoints import stage_checkpoints
from .reporting import report_internal
from .screening_blend_paper405 import run as run_screening_blend_paper405
from .traditional_paper405 import run_benchmark as run_traditional_paper405
from .validation.leakage_checks import assert_internal_config_safe, assert_no_hagr_paths
from .validation.nested_cv import plan_nested_cv, run_nested_cv
from .validation.paper_split_registry import (
    build_paper_split_registry,
    load_verified_paper_registries,
)
from .validation.split_registry import build_split_registry, load_verified_registries
from .weighted_blend_paper405 import run as run_weighted_blend_paper405


def _root(value: str) -> Path:
    path = Path(value)
    if path.is_symlink() or not path.is_dir():
        raise argparse.ArgumentTypeError(
            f"Project root must be a non-symlink directory: {path}"
        )
    return path.resolve()


def _config(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def _project_path(root: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Configured artifact path is unsafe: {value}")
    candidate = root / path
    cursor = root
    for part in path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"Configured artifact path contains a symlink: {cursor}")
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Configured artifact escapes project root: {value}")
    assert_no_hagr_paths([resolved])
    return resolved


def _shared_config_guard(root: Path, config: dict[str, Any]) -> None:
    validate_locked_shared_contract(config)
    assert_internal_config_safe(config)
    contract_name = {
        "reference": "reference",
        "V5_ELIXIRFP_REBUILT": "v5",
        "V6_TABULAR_FOUNDATION_MODELS": "v6",
        "V5BIS_PAPER80": "v5bis",
        "V6BIS_PAPER80": "v6bis",
    }[str(config["pipeline"])]
    validate_protocol_lock(root=root, contract_name=contract_name, config=config)
    data = resolve_config(root / "configs" / "data.yaml")
    validation = resolve_config(root / "configs" / "validation.yaml")
    validate_protocol_lock(root=root, contract_name="data", config=data)
    validate_protocol_lock(root=root, contract_name="validation", config=validation)
    if str(config["pipeline"]) in {"V5BIS_PAPER80", "V6BIS_PAPER80"}:
        paper = resolve_config(root / "configs" / "paper80.yaml")
        validate_protocol_lock(root=root, contract_name="paper80", config=paper)
        for key in ("evaluation_design", "paper_registry_outputs"):
            if config.get(key) != paper.get(key):
                raise ValueError(f"Pipeline changes the shared paper80 contract: {key}")
    for key in ("project", "upstream", "roles", "curation", "outputs"):
        if config.get(key) != data.get(key):
            raise ValueError(f"Pipeline changes the shared data contract: {key}")
    for key in (
        "primary_split",
        "inner_cv",
        "secondary_splits",
        "selection",
        "calibration",
        "threshold",
        "metrics",
        "bootstrap",
        "applicability",
        "external_firewall",
        "registry_outputs",
    ):
        if config.get(key) != validation.get(key):
            raise ValueError(f"Pipeline changes the shared validation contract: {key}")


def _local_sources(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--local-source must be ROLE=/absolute/path")
        role, raw_path = value.split("=", maxsplit=1)
        path = Path(raw_path)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError(f"Local source must be absolute, regular and non-symlink: {path}")
        if role in result:
            raise ValueError(f"Local source role is duplicated: {role}")
        assert_no_hagr_paths([path])
        result[role] = path
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gero")
    parser.add_argument("--root", default=".", help="Project root (default: current directory)")
    subcommands = parser.add_subparsers(dest="command", required=True)

    config = subcommands.add_parser("config", help="Resolve and validate a pipeline config")
    config.add_argument("config_path")

    data = subcommands.add_parser("data", help="Pinned Stage-0 data operations")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    acquire = data_sub.add_parser("acquire")
    acquire.add_argument("--allow-network", action="store_true")
    acquire.add_argument("--local-source", action="append", default=[])
    data_sub.add_parser("curate")

    splits = subcommands.add_parser("splits", help="Build/audit the shared registry")
    split_sub = splits.add_subparsers(dest="split_command", required=True)
    split_sub.add_parser("build")
    split_sub.add_parser("audit")

    paper_splits = subcommands.add_parser(
        "paper-splits", help="Build/audit the locked identity-safe paper 80/20 registry"
    )
    paper_split_sub = paper_splits.add_subparsers(dest="paper_split_command", required=True)
    paper_split_sub.add_parser("build")
    paper_split_sub.add_parser("audit")

    checkpoint = subcommands.add_parser("v6-checkpoints", help="Stage V6 checkpoint ledger")
    checkpoint.add_argument("--config", default="configs/v6.yaml")

    run = subcommands.add_parser("run", help="Plan or execute shared nested CV")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    for name in ("plan", "nested-cv"):
        command = run_sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--suite", choices=("core", "full"), default="core")
        if name == "nested-cv":
            command.add_argument("--run-id", required=True)

    report = subcommands.add_parser("report", help="Report a sealed internal run")
    report.add_argument("run_directory")
    report.add_argument("--reference-model-id")

    traditional = subcommands.add_parser(
        "traditional-paper405",
        help="Run nine fixed traditional models on the publication's exact 405-row split",
    )
    traditional.add_argument("--config", required=True)
    traditional.add_argument("--positive", required=True)
    traditional.add_argument("--negative", required=True)
    traditional.add_argument("--run-id", required=True)
    hypermoltab = subcommands.add_parser(
        "hypermoltab-paper405",
        help="Run HyperMolTab variants on the publication's exact 405-row split",
    )
    hypermoltab.add_argument("--config", required=True)
    hypermoltab.add_argument("--positive", required=True)
    hypermoltab.add_argument("--negative", required=True)
    hypermoltab.add_argument("--run-id", required=True)
    fixed_blend = subcommands.add_parser(
        "fixed-blend-paper405",
        help="Run the fixed V3 blend and SVM variants on the exact 405-row split",
    )
    fixed_blend.add_argument("--config", required=True)
    fixed_blend.add_argument("--positive", required=True)
    fixed_blend.add_argument("--negative", required=True)
    fixed_blend.add_argument("--run-id", required=True)
    weighted_blend = subcommands.add_parser(
        "weighted-blend-paper405",
        help=(
            "Select SVM/Tanimoto/TabPFN weights by nested train-only CV and evaluate the "
            "locked 171-candidate exploratory sweep on the exact paper split"
        ),
    )
    weighted_blend.add_argument("--config", required=True)
    weighted_blend.add_argument("--positive", required=True)
    weighted_blend.add_argument("--negative", required=True)
    weighted_blend.add_argument("--run-id", required=True)
    screening_blend = subcommands.add_parser(
        "screening-blend-paper405",
        help=(
            "Package the locked 0.10/0.60/0.30 screening blend and run post-lock "
            "paired test analyses"
        ),
    )
    screening_blend.add_argument("--config", required=True)
    screening_blend.add_argument("--weighted-run", required=True)
    screening_blend.add_argument("--traditional-run", required=True)
    screening_blend.add_argument("--positive", required=True)
    screening_blend.add_argument("--negative", required=True)
    screening_blend.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = _root(args.root)
    validate_core_runtime(root / "requirements-lock.txt")
    if args.command == "config":
        config = resolve_config(_config(root, args.config_path))
        _shared_config_guard(root, config)
        print(
            json.dumps(
                {
                    "pipeline": config["pipeline"],
                    "resolved_config_sha256": resolved_config_sha256(config),
                    "runtime_placeholders": unresolved_placeholders(config),
                    "internal_firewall": "PASS",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "data":
        config = resolve_config(root / "configs" / "data.yaml")
        validate_protocol_lock(root=root, contract_name="data", config=config)
        # Rebind the standalone shared file to both implemented pipeline configs.
        for pipeline_path in (
            root / "configs" / "v5.yaml",
            root / "configs" / "v6.yaml",
            root / "configs" / "v5bis.yaml",
            root / "configs" / "v6bis.yaml",
        ):
            pipeline = resolve_config(pipeline_path)
            for key in ("project", "upstream", "roles", "curation", "outputs"):
                if pipeline.get(key) != config.get(key):
                    raise ValueError(
                        f"Shared data config differs in {pipeline_path.name}:{key}"
                    )
        if args.data_command == "acquire":
            result = acquire_sources(
                _project_path(root, config["outputs"]["raw_dir"]),
                allow_network=bool(args.allow_network),
                source_config=config["upstream"]["sources"],
                expected_upstream_commit=config["upstream"]["commit"],
                local_sources=_local_sources(args.local_source) or None,
            )
        else:
            outputs = config["outputs"]
            curation = config["curation"]
            curated_path = _project_path(root, outputs["curated_table"])
            provenance_path = _project_path(root, outputs["provenance_table"])
            manifest_path = _project_path(root, outputs["manifest"])
            override_path = _project_path(root, curation["curation_overrides"])
            metal_path = _project_path(root, curation["metal_adjudication"])
            if sha256_file(override_path) != curation["curation_overrides_sha256"]:
                raise ValueError("Curation override bytes differ from the locked config")
            if sha256_file(metal_path) != curation["metal_adjudication_sha256"]:
                raise ValueError("Metal adjudication bytes differ from the locked config")
            if manifest_path.exists():
                _, _, sealed = load_curated_cohort(
                    curated_table=curated_path,
                    provenance_table=provenance_path,
                    manifest_path=manifest_path,
                    resolved_config_sha256=resolved_config_sha256(config),
                )
                result = {**sealed, "stage0_resume_status": "existing_curated_bundle_verified"}
            else:
                result = curate_cohort(
                    raw_dir=_project_path(root, outputs["raw_dir"]),
                    curated_table=curated_path,
                    provenance_table=provenance_path,
                    identity_conflicts=_project_path(root, outputs["identity_conflicts"]),
                    standardization_ledger=_project_path(
                        root, outputs["standardization_ledger"]
                    ),
                    manifest_path=manifest_path,
                    curation_overrides=override_path,
                    expected_overrides_sha256=curation["curation_overrides_sha256"],
                    metal_adjudication=metal_path,
                    expected_metal_adjudication_sha256=(curation["metal_adjudication_sha256"]),
                    expected_counts=curation["expected_counts"],
                    expected_identity_label_sha256=(curation["expected_identity_label_sha256"]),
                    expected_rdkit_version=curation["rdkit_version_pin"],
                    resolved_config_sha256=resolved_config_sha256(config),
                )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "paper-splits":
        data_config = resolve_config(root / "configs" / "data.yaml")
        paper_config = resolve_config(root / "configs" / "v5bis.yaml")
        paper_contract = resolve_config(root / "configs" / "paper80.yaml")
        validate_protocol_lock(root=root, contract_name="data", config=data_config)
        validate_protocol_lock(root=root, contract_name="v5bis", config=paper_config)
        validate_protocol_lock(root=root, contract_name="paper80", config=paper_contract)
        outputs = data_config["outputs"]
        curated_path = _project_path(root, outputs["curated_table"])
        provenance_path = _project_path(root, outputs["provenance_table"])
        data_manifest_path = _project_path(root, outputs["manifest"])
        curated, _, _ = load_curated_cohort(
            curated_table=curated_path,
            provenance_table=provenance_path,
            manifest_path=data_manifest_path,
            resolved_config_sha256=resolved_config_sha256(data_config),
        )
        registry = paper_config["paper_registry_outputs"]
        design = paper_config["evaluation_design"]
        outer_path = _project_path(root, registry["outer"])
        inner_path = _project_path(root, registry["inner"])
        manifest_path = _project_path(root, registry["manifest"])
        identity_path = _project_path(root, design["test_identity_file"])
        if args.paper_split_command == "build" and not manifest_path.exists():
            result = build_paper_split_registry(
                curated,
                output_path=outer_path,
                inner_output_path=inner_path,
                manifest_path=manifest_path,
                curated_table_sha256=sha256_file(curated_path),
                data_manifest_path=data_manifest_path,
                resolved_config_sha256=resolved_config_sha256(paper_contract),
                test_identity_path=identity_path,
                test_identity_file_sha256=design["test_identity_file_sha256"],
                inner_folds=design["inner_selection"]["folds"],
                inner_seed=design["inner_selection"]["assignment_seed"],
            )
        else:
            outer, inner, manifest = load_verified_paper_registries(
                cohort=curated,
                outer_path=outer_path,
                inner_path=inner_path,
                manifest_path=manifest_path,
                curated_table_path=curated_path,
                data_manifest_path=data_manifest_path,
                resolved_config_sha256=resolved_config_sha256(paper_contract),
                test_identity_path=identity_path,
                test_identity_file_sha256=design["test_identity_file_sha256"],
                inner_folds=design["inner_selection"]["folds"],
            )
            result = {
                **manifest,
                "stage0_resume_status": "existing_paper80_bundle_verified",
                "outer_rows": len(outer),
                "inner_rows": len(inner),
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "splits":
        data_config = resolve_config(root / "configs" / "data.yaml")
        config = resolve_config(root / "configs" / "validation.yaml")
        validate_protocol_lock(root=root, contract_name="data", config=data_config)
        validate_protocol_lock(root=root, contract_name="validation", config=config)
        outputs = data_config["outputs"]
        curated_path = _project_path(root, outputs["curated_table"])
        data_manifest_path = _project_path(root, outputs["manifest"])
        curated, _, _ = load_curated_cohort(
            curated_table=curated_path,
            provenance_table=_project_path(root, outputs["provenance_table"]),
            manifest_path=data_manifest_path,
            resolved_config_sha256=resolved_config_sha256(data_config),
        )
        registry = config["registry_outputs"]
        if args.split_command == "build":
            outer_path = _project_path(root, registry["outer"])
            inner_path = _project_path(root, registry["inner"])
            split_manifest_path = _project_path(root, registry["manifest"])
            if split_manifest_path.exists():
                outer, inner, manifest = load_verified_registries(
                    outer_path=outer_path,
                    inner_path=inner_path,
                    manifest_path=split_manifest_path,
                    curated_table_path=curated_path,
                    data_manifest_path=data_manifest_path,
                    resolved_config_sha256=resolved_config_sha256(config),
                    outer_repeats=config["primary_split"]["outer_repeats"],
                    outer_folds=config["primary_split"]["outer_folds"],
                    inner_folds=config["inner_cv"]["folds"],
                )
                result = {
                    **manifest,
                    "stage0_resume_status": "existing_split_bundle_verified",
                    "outer_rows": len(outer),
                    "inner_rows": len(inner),
                }
            else:
                result = build_split_registry(
                    curated,
                    output_path=outer_path,
                    inner_output_path=inner_path,
                    manifest_path=split_manifest_path,
                    curated_table_sha256=sha256_file(curated_path),
                    data_manifest_path=data_manifest_path,
                    resolved_config_sha256=resolved_config_sha256(config),
                    fingerprint_config=config["primary_split"]["fingerprint"],
                    edge_threshold=config["primary_split"]["edge_threshold"],
                    outer_repeats=config["primary_split"]["outer_repeats"],
                    outer_folds=config["primary_split"]["outer_folds"],
                    outer_seeds=config["primary_split"]["outer_seeds"],
                    inner_folds=config["inner_cv"]["folds"],
                    inner_seeds=config["inner_cv"]["seeds"],
                )
        else:
            outer, inner, manifest = load_verified_registries(
                outer_path=_project_path(root, registry["outer"]),
                inner_path=_project_path(root, registry["inner"]),
                manifest_path=_project_path(root, registry["manifest"]),
                curated_table_path=curated_path,
                data_manifest_path=data_manifest_path,
                resolved_config_sha256=resolved_config_sha256(config),
                outer_repeats=config["primary_split"]["outer_repeats"],
                outer_folds=config["primary_split"]["outer_folds"],
                inner_folds=config["inner_cv"]["folds"],
            )
            result = {
                "status": "PASS",
                "outer_rows": len(outer),
                "inner_rows": len(inner),
                "registry_sha256": manifest["registry_sha256"],
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "v6-checkpoints":
        config = resolve_config(_config(root, args.config))
        _shared_config_guard(root, config)
        placeholders = unresolved_placeholders(config)
        if placeholders:
            raise ValueError("Resolve V6 config placeholders: " + ", ".join(placeholders))
        result = stage_checkpoints(
            config,
            root=root,
            output_path=_project_path(root, config["checkpoint_ledger"]),
            resolved_config_sha256=resolved_config_sha256(config),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        config_path = _config(root, args.config)
        if args.run_command == "plan":
            result = plan_nested_cv(root=root, config_path=config_path, suite=args.suite)
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            result = run_nested_cv(
                root=root,
                config_path=config_path,
                run_id=args.run_id,
                suite=args.suite,
            )
            print(result)
        return 0
    if args.command == "traditional-paper405":
        output = run_traditional_paper405(
            root=root,
            protocol_path=_config(root, args.config),
            positive_path=Path(args.positive),
            negative_path=Path(args.negative),
            run_id=args.run_id,
        )
        print(output)
        return 0
    if args.command == "hypermoltab-paper405":
        output = run_hypermoltab_paper405(
            root=root,
            config_path=_config(root, args.config),
            positive_path=Path(args.positive),
            negative_path=Path(args.negative),
            run_id=args.run_id,
        )
        print(output)
        return 0
    if args.command == "fixed-blend-paper405":
        output = run_fixed_blend_paper405(
            root=root,
            config_path=_config(root, args.config),
            positive_path=Path(args.positive),
            negative_path=Path(args.negative),
            run_id=args.run_id,
        )
        print(output)
        return 0
    if args.command == "weighted-blend-paper405":
        output = run_weighted_blend_paper405(
            root=root,
            config_path=_config(root, args.config),
            positive_path=Path(args.positive),
            negative_path=Path(args.negative),
            run_id=args.run_id,
        )
        print(output)
        return 0
    if args.command == "screening-blend-paper405":
        output = run_screening_blend_paper405(
            root=root,
            config_path=_config(root, args.config),
            weighted_run=_config(root, args.weighted_run),
            traditional_run=_config(root, args.traditional_run),
            positive_path=Path(args.positive),
            negative_path=Path(args.negative),
            run_id=args.run_id,
        )
        print(output)
        return 0
    report = report_internal(
        _config(root, args.run_directory),
        reference_model_id=args.reference_model_id,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
