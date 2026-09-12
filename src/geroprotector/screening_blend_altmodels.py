"""Equal-weight screening blends whose third slot is BiSHop, TabM or TabNet.

The locked 0.10/0.60/0.30 blend is not touched.  This module builds a *separate*
family of experiments in which the tabular-foundation-model slot of the three
component blend is replaced by one of three tabular deep-learning models, with
prespecified equal weights ``1/3`` for the paper SVM, the Morgan-Tanimoto SVC and
the alternative model.

Contract, in order:

1. Rebuild the 405-row D1 frame, the paper 324/81 split, Morgan bits and the RDKit2D
   descriptor panel, then *prove* that the rebuilt Morgan bits and the rebuilt
   median-imputed/variance-filtered descriptor context are identical to the sealed
   locked bundle.  Nothing continues if they differ.
2. Rebuild the paper-SVM and Tanimoto-SVC cross-fitted OOF and full-fit probability
   streams and prove they match the sealed weighted run.  These two components are
   therefore literally the same fitted chemistry components as the locked blend.
3. Fit the alternative model on the same fold-local feature panel, producing an OOF
   stream on the 324 training rows only.
4. Select each blend's decision threshold by MCC on the 324-row OOF, exactly as the
   locked blend did.  A fixed 0.5 rule is reported as the prespecified second
   operating point.
5. Only then score the 81 held-out D1 rows, the 446 DrugAge parent compounds and the
   69 AgeXtend Table 6 compounds.  External chemistry-component probabilities are
   *read* from the sealed external prediction files, so the external SVM/Tanimoto
   streams are byte-identical to the ones already published.

No test or external outcome touches a fit, a weight or a threshold anywhere.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer

from geroprotector.fixed_blend_paper405 import (
    _features,
    _selected_component_predictions,
    _tanimoto,
)
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, canonical_sha256, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.screening_blend_ablation import (
    _safe_metrics,
    _svc_probability_at_native_boundary,
)
from geroprotector.screening_blend_paper405 import load_locked_bundle
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices
from geroprotector.weighted_blend_paper405 import select_threshold


class ScreeningBlendAltModelsError(RuntimeError):
    """Raised when a sealed input, a parity proof or a leakage contract fails."""


ALT_MODELS = ("bishop", "tabm", "tabnet")
CHEMISTRY_COMPONENTS = ("paper_svm", "tanimoto_svc")
EQUAL_WEIGHT = 1.0 / 3.0
SCHEMA = "geroprotector.screening_blend_altmodels"

D1_ENDPOINT = "paper_binary"
DRUGAGE_ENDPOINT = "significant_positive_retrieval_background_not_certified_negative"
AGEXTEND_ENDPOINT = "published_independent_table6_binary"


# --------------------------------------------------------------------------- io


def _regular_file(path: Path, role: str, expected_sha256: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ScreeningBlendAltModelsError(f"{role} must be a regular file: {path}")
    resolved = path.resolve()
    if expected_sha256 is not None and sha256_file(resolved) != expected_sha256:
        raise ScreeningBlendAltModelsError(f"{role} SHA256 differs from the protocol lock")
    return resolved


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(
        _regular_file(path, "alt-model protocol").read_text(encoding="utf-8")
    )
    expected_schema = f"{SCHEMA}.protocol.v1"
    if not isinstance(protocol, dict) or protocol.get("schema_version") != expected_schema:
        raise ScreeningBlendAltModelsError("Unknown alt-model protocol schema")
    blend = protocol.get("blend", {})
    if (
        blend.get("components") != ["paper_svm", "tanimoto_svc", "alt_model"]
        or not np.allclose(blend.get("weights"), [EQUAL_WEIGHT] * 3, rtol=0.0, atol=1e-15)
        or tuple(blend.get("alt_models", ())) != ALT_MODELS
        or blend.get("weights_selected_from_data") is not False
        or blend.get("test_or_external_labels_used_for_fit_weight_or_threshold") is not False
        or blend.get("primary_threshold_source") != "full_324_d1_train_cross_fitted_oof_mcc"
        or float(blend.get("secondary_fixed_threshold", np.nan)) != 0.5
    ):
        raise ScreeningBlendAltModelsError("Equal-weight blend contract differs")
    parity = protocol.get("parity_contract", {})
    if (
        parity.get("rebuilt_paper_svm_and_tanimoto_streams_must_match_sealed_run") is not True
        or parity.get("rebuilt_rdkit2d_context_must_equal_sealed_tabpfn_context") is not True
        or parity.get("rebuilt_morgan_bits_must_equal_sealed_tanimoto_train_bits") is not True
    ):
        raise ScreeningBlendAltModelsError("Parity contract differs")
    immutability = protocol.get("immutability", {})
    if immutability.get("existing_run_directories_are_read_only") is not True:
        raise ScreeningBlendAltModelsError("Immutability contract differs")
    for record in protocol.get("sealed_inputs", {}).values():
        _regular_file(root / record["path"], f"sealed input {record['path']}", record["sha256"])
    return protocol, canonical_sha256(protocol)


# ------------------------------------------------------------------ feature panel


def _imputer_context(train_raw: np.ndarray) -> dict[str, np.ndarray]:
    """Reproduce the sealed fold-local RDKit2D imputation as a reusable context."""

    finite_any = np.isfinite(train_raw).any(axis=0)
    kept = train_raw[:, finite_any]
    medians = np.nanmedian(np.where(np.isfinite(kept), kept, np.nan), axis=0)
    filled = np.where(np.isfinite(kept), kept, medians)
    varying = np.ptp(filled, axis=0) > 0
    return {
        "finite_any_mask": finite_any,
        "medians_after_finite_any": medians,
        "varying_after_imputation_mask": varying,
    }


def _apply_context(context: dict[str, np.ndarray], raw: np.ndarray) -> np.ndarray:
    kept = raw[:, np.asarray(context["finite_any_mask"], dtype=bool)]
    medians = np.asarray(context["medians_after_finite_any"], dtype=float)
    filled = np.where(np.isfinite(kept), kept, medians)
    output = filled[:, np.asarray(context["varying_after_imputation_mask"], dtype=bool)]
    output = output.astype(np.float32)
    if not np.isfinite(output).all():
        raise ScreeningBlendAltModelsError(
            "Frozen descriptor transform produced non-finite values"
        )
    return output


def _rdkit2d_from_smiles(smiles: list[str], descriptor_names: list[str]) -> np.ndarray:
    names = [name for name, _function in Descriptors._descList]
    if names != list(descriptor_names):
        raise ScreeningBlendAltModelsError("RDKit descriptor registry differs from the bundle")
    float32_max = float(np.finfo(np.float32).max)
    values = np.full((len(smiles), len(names)), np.nan, dtype=np.float64)
    for row, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(text))
        if molecule is None:
            raise ScreeningBlendAltModelsError(f"Unparsable structure at row {row}")
        for column, (_name, function) in enumerate(Descriptors._descList):
            try:
                result = float(function(molecule))
            except Exception:
                result = np.nan
            values[row, column] = (
                result if np.isfinite(result) and abs(result) <= float32_max else np.nan
            )
    return values


def _morgan_from_smiles(smiles: list[str], contract: dict[str, Any]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(contract["morgan_radius"]),
        fpSize=int(contract["morgan_bits"]),
        includeChirality=bool(contract["morgan_include_chirality"]),
    )
    bits = np.zeros((len(smiles), int(contract["morgan_bits"])), dtype=np.uint8)
    for row, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(text))
        if molecule is None:
            raise ScreeningBlendAltModelsError(f"Unparsable structure at row {row}")
        bits[row] = np.asarray(generator.GetFingerprint(molecule), dtype=np.uint8)
    return bits


# ------------------------------------------------------------- alternative models


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _bishop_module(root: Path, settings: dict[str, Any]):
    vendored = root / settings["vendored_path"]
    for relative, expected in settings["vendored_sha256"].items():
        _regular_file(vendored / relative, f"vendored BiSHop {relative}", expected)
    if str(vendored) not in sys.path:
        sys.path.insert(0, str(vendored))
    from models.model import BiSHop

    return BiSHop


def _require_version(package: str, expected: str) -> str:
    observed = importlib.metadata.version(package)
    if observed != str(expected):
        raise ScreeningBlendAltModelsError(
            f"{package} version {observed} differs from the protocol lock {expected}"
        )
    return observed


def _torch_train_predict(
    build,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    targets: dict[str, np.ndarray],
    training: dict[str, Any],
    seed: int,
) -> dict[str, np.ndarray]:
    """One early-stopped training run; the inner split never leaves the fit rows."""

    import torch
    from torch import nn

    device = torch.device(str(training["device"]))
    inner_fit, inner_validation = train_test_split(
        np.arange(len(x_fit)),
        test_size=float(training["inner_validation_fraction"]),
        random_state=seed,
        stratify=y_fit if bool(training["inner_validation_stratified"]) else None,
    )
    model, forward = build(x_fit[inner_fit], y_fit[inner_fit], device, seed)
    optimiser = model["optimiser"]
    network = model["network"]
    criterion = nn.CrossEntropyLoss()

    xt = torch.tensor(x_fit[inner_fit], device=device)
    yt = torch.tensor(y_fit[inner_fit], device=device, dtype=torch.long)
    xv = torch.tensor(x_fit[inner_validation], device=device)
    yv = torch.tensor(y_fit[inner_validation], device=device, dtype=torch.long)

    def _loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:  # TabM emits (batch, k, classes)
            target = target[:, None].expand(-1, logits.shape[1]).reshape(-1)
            logits = logits.reshape(-1, logits.shape[-1])
        return criterion(logits, target)

    batch = int(training["batch_size"])
    patience = int(training["early_stopping_patience"])
    best_loss, best_state, waited, last_epoch = np.inf, None, 0, -1
    for epoch in range(int(training["max_epochs"])):
        last_epoch = epoch
        network.train()
        order = torch.randperm(len(xt), device=device)
        for start in range(0, len(xt), batch):
            index = order[start : start + batch]
            optimiser.zero_grad()
            _loss(forward(xt[index]), yt[index]).backward()
            optimiser.step()
        network.eval()
        with torch.no_grad():
            validation_loss = float(_loss(forward(xv), yv).item())
        if validation_loss < best_loss - 1e-6:
            best_loss, waited = validation_loss, 0
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is None:
        raise ScreeningBlendAltModelsError("Training produced no validated state")
    network.load_state_dict(best_state)
    network.eval()

    output: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, matrix in targets.items():
            logits = forward(torch.tensor(matrix, device=device))
            if logits.dim() == 3:
                probability = torch.softmax(logits, dim=-1)[..., 1].mean(dim=1)
            else:
                probability = torch.softmax(logits, dim=-1)[:, 1]
            output[name] = probability.detach().cpu().numpy().astype(np.float64)
    output["_state"] = best_state
    output["_epochs"] = np.asarray([last_epoch + 1, best_loss], dtype=float)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def _build_tabm(settings: dict[str, Any]):
    def build(x: np.ndarray, y: np.ndarray, device, seed: int):
        import tabm
        import torch

        _seed_everything(seed)
        hyper = settings["hyperparameters"]
        network = tabm.TabM.make(
            n_num_features=int(x.shape[1]),
            cat_cardinalities=[],
            d_out=2,
            n_blocks=int(hyper["n_blocks"]),
            d_block=int(hyper["d_block"]),
            dropout=float(hyper["dropout"]),
            k=int(hyper["k"]),
            arch_type=str(hyper["arch_type"]),
            start_scaling_init=str(hyper["start_scaling_init"]),
        ).to(device)
        optimiser = torch.optim.AdamW(
            network.parameters(),
            lr=float(hyper["learning_rate"]),
            weight_decay=float(hyper["weight_decay"]),
        )
        return {"network": network, "optimiser": optimiser}, network

    return build


def _build_bishop(root: Path, settings: dict[str, Any]):
    BiSHop = _bishop_module(root, settings)

    def build(x: np.ndarray, y: np.ndarray, device, seed: int):
        import torch
        from torch import nn

        _seed_everything(seed)
        hyper = settings["hyperparameters"]
        network = BiSHop(
            n_cat=0,
            n_num=int(x.shape[1]),
            n_out=2,
            emb_dim=int(hyper["emb_dim"]),
            out_dim=int(hyper["out_dim"]),
            patch_dim=int(hyper["patch_dim"]),
            factor=int(hyper["factor"]),
            flip=True,
            n_agg=int(hyper["n_agg"]),
            actv=str(hyper["actv"]),
            hopfield=bool(hyper["hopfield"]),
            d_model=int(hyper["d_model"]),
            d_ff=int(hyper["d_ff"]),
            n_heads=int(hyper["n_heads"]),
            e_layer=int(hyper["e_layer"]),
            d_layer=int(hyper["d_layer"]),
            dropout=float(hyper["dropout"]),
            share=True,
            share_div=8,
            share_add=False,
            full_dropout=False,
            emb_dropout=float(hyper["emb_dropout"]),
            mlp_actv=nn.ReLU(),
            mlp_bn=True,
            mlp_bn_final=False,
            mlp_dropout=float(hyper["mlp_dropout"]),
            mlp_hidden=tuple(int(value) for value in hyper["mlp_hidden"]),
            mlp_skip=bool(hyper["mlp_skip"]),
            mlp_softmax=False,
            device=device,
        ).to(device)
        # Quantile bins come from the current fit rows only.
        with torch.no_grad():
            network.get_bins(torch.tensor(x, device=device))
        network.NumEmb._to(device)
        optimiser = torch.optim.AdamW(
            network.parameters(),
            lr=float(hyper["learning_rate"]),
            weight_decay=float(hyper["weight_decay"]),
        )
        empty = torch.empty(0, dtype=torch.long, device=device)

        def forward(batch):
            return network(empty, batch)

        return {"network": network, "optimiser": optimiser}, forward

    return build


def _tabnet_train_predict(
    settings: dict[str, Any],
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    targets: dict[str, np.ndarray],
    training: dict[str, Any],
    seed: int,
) -> dict[str, np.ndarray]:
    from pytorch_tabnet.tab_model import TabNetClassifier

    _seed_everything(seed)
    hyper = settings["hyperparameters"]
    inner_fit, inner_validation = train_test_split(
        np.arange(len(x_fit)),
        test_size=float(training["inner_validation_fraction"]),
        random_state=seed,
        stratify=y_fit if bool(training["inner_validation_stratified"]) else None,
    )
    model = TabNetClassifier(
        n_d=int(hyper["n_d"]),
        n_a=int(hyper["n_a"]),
        n_steps=int(hyper["n_steps"]),
        gamma=float(hyper["gamma"]),
        n_independent=int(hyper["n_independent"]),
        n_shared=int(hyper["n_shared"]),
        lambda_sparse=float(hyper["lambda_sparse"]),
        optimizer_params={"lr": float(hyper["learning_rate"])},
        seed=int(seed),
        verbose=0,
        device_name=str(training["device"]),
    )
    model.fit(
        x_fit[inner_fit],
        y_fit[inner_fit],
        eval_set=[(x_fit[inner_validation], y_fit[inner_validation])],
        eval_metric=["logloss"],
        max_epochs=int(training["max_epochs"]),
        patience=int(training["early_stopping_patience"]),
        batch_size=int(training["batch_size"]),
        virtual_batch_size=int(hyper["virtual_batch_size"]),
    )
    output = {
        name: np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
        for name, matrix in targets.items()
    }
    output["_state"] = model
    output["_epochs"] = np.asarray(
        [float(model.best_epoch), float(model.best_cost)], dtype=float
    )
    return output


def _alt_probabilities(
    root: Path,
    name: str,
    protocol: dict[str, Any],
    x_fit_raw: np.ndarray,
    y_fit: np.ndarray,
    raw_targets: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fold-local imputation, fold-local scaling, seed-ensembled probabilities."""

    training = protocol["alt_model_training"]
    settings = protocol["alt_models"][name]
    context = _imputer_context(x_fit_raw)
    x_fit = _apply_context(context, x_fit_raw)
    scaler = QuantileTransformer(
        n_quantiles=int(min(256, len(x_fit))),
        output_distribution="normal",
        subsample=10**9,
        random_state=42,
    ).fit(x_fit)
    x_fit_scaled = scaler.transform(x_fit).astype(np.float32)
    targets = {
        key: scaler.transform(_apply_context(context, value)).astype(np.float32)
        for key, value in raw_targets.items()
    }

    accumulator = {
        key: np.zeros(len(value), dtype=np.float64) for key, value in targets.items()
    }
    states, epochs = [], []
    for seed in [int(value) for value in training["seed_ensemble"]]:
        if name == "tabnet":
            result = _tabnet_train_predict(
                settings, x_fit_scaled, y_fit, targets, training, seed
            )
        else:
            build = _build_bishop(root, settings) if name == "bishop" else _build_tabm(settings)
            result = _torch_train_predict(build, x_fit_scaled, y_fit, targets, training, seed)
        for key in accumulator:
            accumulator[key] += result[key]
        states.append(result["_state"])
        epochs.append([float(value) for value in result["_epochs"]])
        print(f"    {name} seed {seed} done (epochs/best-loss {epochs[-1]})", flush=True)

    divisor = float(len(training["seed_ensemble"]))
    probabilities = {
        key: np.clip(value / divisor, 1e-7, 1 - 1e-7) for key, value in accumulator.items()
    }
    audit = {
        "model": name,
        "fit_rows": len(x_fit_raw),
        "input_descriptor_columns": int(x_fit_raw.shape[1]),
        "retained_descriptor_columns": int(x_fit.shape[1]),
        "seed_ensemble": [int(value) for value in training["seed_ensemble"]],
        "per_seed_epochs_and_best_inner_loss": epochs,
        "scaler": "QuantileTransformer(output_distribution=normal)",
        "scaler_fitted_on_fit_rows_only": True,
        "target_or_external_labels_used": False,
    }
    return probabilities, {
        "audit": audit,
        "states": states,
        "context": context,
        "scaler": scaler,
    }


# ------------------------------------------------------------------------ metrics


def _metric_row(
    cohort: str,
    endpoint: str,
    model: str,
    operating_point: str,
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    source: str,
) -> dict[str, Any]:
    return {
        "cohort": cohort,
        "endpoint": endpoint,
        "model": model,
        "operating_point": operating_point,
        "result_source": source,
        **_safe_metrics(labels, probability, threshold),
    }


def _plain_markdown(frame: pd.DataFrame) -> str:
    """Small dependency-free markdown writer (the runtime has no tabulate)."""

    header = list(frame.columns)
    lines = ["| " + " | ".join(str(name) for name in header) + " |"]
    lines.append("|" + "---|" * len(header))
    for row in frame.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _summary_table(frame: pd.DataFrame, title: str) -> str:
    columns = [
        ("model", "Model"),
        ("n_test", "n"),
        ("auprc_average_precision_positive", "AP"),
        ("auroc", "AUROC"),
        ("brier", "Brier"),
        ("mcc", "MCC"),
        ("macro_f1", "Macro F1"),
        ("recall_sensitivity", "Recall"),
        ("specificity", "Specificity"),
    ]
    lines = [f"## {title}", ""]
    lines.append("| " + " | ".join(label for _key, label in columns) + " |")
    lines.append("|" + "---|" * len(columns))
    for row in frame.itertuples(index=False):
        values = []
        for key, _label in columns:
            value = getattr(row, key)
            if key == "model":
                values.append(str(value))
            elif key == "n_test":
                values.append(str(int(value)))
            else:
                blank = value is None or not np.isfinite(value)
                values.append("nan" if blank else f"{value:.4f}")
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------- run


def run(
    *,
    root: Path,
    config_path: Path,
    positive_path: Path,
    negative_path: Path,
    run_id: str,
) -> Path:
    if not re.fullmatch(r"screeningblend_altmodels_[a-z0-9_.-]+", run_id):
        raise ScreeningBlendAltModelsError("RUN_ID must start with screeningblend_altmodels_")
    root = root.resolve()
    protocol, protocol_sha256 = load_protocol(root, config_path)
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise ScreeningBlendAltModelsError(f"Run directory already exists: {destination}")

    sealed = protocol["sealed_inputs"]
    bundle = load_locked_bundle(
        root / sealed["screening_bundle"]["path"],
        expected_artifact_sha256=sealed["screening_bundle"]["sha256"],
    )

    # -- 1. D1 frame, split and feature panel ------------------------------------
    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, _source_audit = _read_sources(
        _regular_file(positive_path, "positive source", protocol["sources"]["positive_sha256"]),
        _regular_file(negative_path, "negative source", protocol["sources"]["negative_sha256"]),
        traditional,
    )
    if len(frame) != int(protocol["sources"]["expected_rows"]):
        raise ScreeningBlendAltModelsError("D1 row count differs from 405")
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    if split_sha256 != protocol["paper_split"]["expected_assignment_sha256"]:
        raise ScreeningBlendAltModelsError("Paper split differs from the sealed assignment")
    labels = frame["label"].to_numpy(dtype=int)
    smiles, _ = _validated_raw_smiles(frame)
    features = _features(frame, smiles, fixed_protocol)

    fit_indices = np.asarray(bundle["fit_paper_indices"], dtype=int)
    if not np.array_equal(np.sort(fit_indices), np.sort(train_indices)):
        raise ScreeningBlendAltModelsError(
            "Sealed fit indices differ from the paper train split"
        )
    if not np.array_equal(np.asarray(bundle["fit_labels"], dtype=int), labels[fit_indices]):
        raise ScreeningBlendAltModelsError("Sealed fit labels differ from D1")

    # -- 2. parity proofs against the sealed locked component state ---------------
    if not np.array_equal(
        features["morgan"][fit_indices],
        np.asarray(bundle["tanimoto_train_bits"], dtype=np.uint8),
    ):
        raise ScreeningBlendAltModelsError("Rebuilt Morgan bits differ from the sealed bundle")
    train_context = _imputer_context(features["rdkit2d"][fit_indices])
    rebuilt_context_features = _apply_context(train_context, features["rdkit2d"][fit_indices])
    if not np.array_equal(
        rebuilt_context_features,
        np.asarray(bundle["tabpfn_context"]["context_features"], dtype=np.float32),
    ):
        raise ScreeningBlendAltModelsError(
            "Rebuilt descriptor context differs from the sealed foundation-model context"
        )

    atol = float(protocol["parity_contract"]["parity_atol"])
    sealed_oof = pd.read_csv(root / sealed["weighted_train_oof"]["path"])
    sealed_test = pd.read_csv(root / sealed["weighted_test_components"]["path"])

    # -- 3. chemistry OOF and full-fit streams ------------------------------------
    settings = fixed_protocol["components"]
    folds = StratifiedKFold(
        n_splits=int(settings["cross_fitted_oof_folds"]),
        shuffle=True,
        random_state=int(settings["cross_fitted_oof_seed"]),
    )
    fold_split = list(folds.split(train_indices, labels[train_indices]))
    oof = {name: np.full(len(train_indices), np.nan) for name in CHEMISTRY_COMPONENTS}
    for fold, (relative_fit, relative_validation) in enumerate(fold_split):
        fit = train_indices[relative_fit]
        validation = train_indices[relative_validation]
        chemistry, _models, _audit = _selected_component_predictions(
            features,
            labels,
            fit,
            validation,
            settings,
            seed=42 + fold,
            requested=CHEMISTRY_COMPONENTS,
        )
        for name in CHEMISTRY_COMPONENTS:
            oof[name][relative_validation] = chemistry[name]
        print(f"chemistry OOF fold {fold + 1}/{len(fold_split)}", flush=True)
    if any(not np.isfinite(values).all() for values in oof.values()):
        raise ScreeningBlendAltModelsError("Chemistry OOF is incomplete")

    oof_frame = pd.DataFrame(
        {
            "paper_row_index": train_indices,
            "label": labels[train_indices],
            "probability_paper_svm": oof["paper_svm"],
            "probability_tanimoto_svc": oof["tanimoto_svc"],
        }
    )
    merged = sealed_oof.merge(oof_frame, on="paper_row_index", suffixes=("_sealed", "_rebuilt"))
    if len(merged) != len(train_indices):
        raise ScreeningBlendAltModelsError("Sealed OOF join is incomplete")
    oof_parity = {}
    for column in ("probability_paper_svm", "probability_tanimoto_svc"):
        difference = float(
            np.max(np.abs(merged[f"{column}_sealed"] - merged[f"{column}_rebuilt"]))
        )
        if difference > atol:
            raise ScreeningBlendAltModelsError(
                f"Rebuilt OOF {column} differs from the sealed run"
            )
        oof_parity[column] = difference

    full_chemistry, full_models, _audit = _selected_component_predictions(
        features,
        labels,
        train_indices,
        test_indices,
        settings,
        seed=42,
        requested=CHEMISTRY_COMPONENTS,
    )
    test_frame = pd.DataFrame(
        {
            "paper_row_index": test_indices,
            "label": labels[test_indices],
            "probability_paper_svm": full_chemistry["paper_svm"],
            "probability_tanimoto_svc": full_chemistry["tanimoto_svc"],
        }
    )
    merged_test = sealed_test.merge(
        test_frame.drop(columns=["label"]),
        on="paper_row_index",
        suffixes=("_sealed", "_rebuilt"),
    )
    if len(merged_test) != len(test_indices):
        raise ScreeningBlendAltModelsError("Sealed test join is incomplete")
    test_parity = {}
    for column in ("probability_paper_svm", "probability_tanimoto_svc"):
        difference = float(
            np.max(np.abs(merged_test[f"{column}_sealed"] - merged_test[f"{column}_rebuilt"]))
        )
        if difference > atol:
            raise ScreeningBlendAltModelsError(
                f"Rebuilt test {column} differs from the sealed run"
            )
        test_parity[column] = difference

    paper_svm_boundary = _svc_probability_at_native_boundary(full_models["paper_svm"])

    # -- 4. external cohorts (chemistry streams are read from the sealed run) -----
    cohorts: dict[str, dict[str, Any]] = {}
    for cohort, key, endpoint in (
        ("drugage", "drugage_scored", DRUGAGE_ENDPOINT),
        ("agextend", "agextend_scored", AGEXTEND_ENDPOINT),
    ):
        scored = pd.read_csv(root / sealed[key]["path"])
        if cohort == "drugage":
            scored = scored.assign(label=scored.has_significant_positive.astype(int))
        scored = scored.reset_index(drop=True)
        external_smiles = scored.source_smiles.astype(str).tolist()
        cohorts[cohort] = {
            "endpoint": endpoint,
            "frame": scored,
            "rdkit2d": _rdkit2d_from_smiles(external_smiles, bundle["descriptor_names"]),
            "morgan": _morgan_from_smiles(external_smiles, bundle["portable_contract"]),
        }
        print(f"external features rebuilt for {cohort}: {len(scored)} rows", flush=True)

    for cohort, payload in cohorts.items():
        similarity = _tanimoto(
            payload["morgan"], np.asarray(bundle["tanimoto_train_bits"], dtype=np.uint8)
        ).max(axis=1)
        stored = payload["frame"].maximum_tanimoto_to_fitted_train.to_numpy(dtype=float)
        if float(np.max(np.abs(similarity - stored))) > 1e-6:
            raise ScreeningBlendAltModelsError(
                f"Rebuilt {cohort} similarity differs from the sealed prediction file"
            )
        payload["max_tanimoto"] = similarity

    d1_test_similarity = _tanimoto(
        features["morgan"][test_indices],
        np.asarray(bundle["tanimoto_train_bits"], dtype=np.uint8),
    ).max(axis=1)

    # -- 5. alternative models ----------------------------------------------------
    versions = {
        "tabm": _require_version(
            "tabm", protocol["alt_models"]["tabm"]["required_package_version"]
        ),
        "pytorch_tabnet": _require_version(
            "pytorch-tabnet", protocol["alt_models"]["tabnet"]["required_package_version"]
        ),
    }
    raw_targets_full = {
        "d1_test": features["rdkit2d"][test_indices],
        "drugage": cohorts["drugage"]["rdkit2d"],
        "agextend": cohorts["agextend"]["rdkit2d"],
    }

    alt_oof: dict[str, np.ndarray] = {}
    alt_scored: dict[str, dict[str, np.ndarray]] = {}
    alt_audits: dict[str, Any] = {}
    alt_states: dict[str, Any] = {}
    for name in ALT_MODELS:
        print(f"[{name}] cross-fitted OOF on the 324 D1 training rows", flush=True)
        stream = np.full(len(train_indices), np.nan)
        fold_audits = []
        for fold, (relative_fit, relative_validation) in enumerate(fold_split):
            fit = train_indices[relative_fit]
            validation = train_indices[relative_validation]
            print(f"  fold {fold + 1}/{len(fold_split)}", flush=True)
            probabilities, extra = _alt_probabilities(
                root,
                name,
                protocol,
                features["rdkit2d"][fit],
                labels[fit],
                {"validation": features["rdkit2d"][validation]},
            )
            stream[relative_validation] = probabilities["validation"]
            fold_audits.append({"fold": fold, **extra["audit"]})
        if not np.isfinite(stream).all():
            raise ScreeningBlendAltModelsError(f"{name} OOF is incomplete")
        alt_oof[name] = stream

        print(f"[{name}] full 324-row fit and scoring", flush=True)
        probabilities, extra = _alt_probabilities(
            root,
            name,
            protocol,
            features["rdkit2d"][fit_indices],
            labels[fit_indices],
            raw_targets_full,
        )
        alt_scored[name] = probabilities
        alt_states[name] = extra
        alt_audits[name] = {"folds": fold_audits, "full_fit": extra["audit"]}

    # -- 6. thresholds from the 324-row OOF only ---------------------------------
    threshold_rows = []
    blend_oof: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    y_train = labels[train_indices]
    for name in ALT_MODELS:
        component_threshold, component_mcc = select_threshold(y_train, alt_oof[name])
        blend = (oof["paper_svm"] + oof["tanimoto_svc"] + alt_oof[name]) / 3.0
        blend_oof[name] = blend
        blend_threshold, blend_mcc = select_threshold(y_train, blend)
        thresholds[name] = float(component_threshold)
        thresholds[f"blend_{name}"] = float(blend_threshold)
        threshold_rows.append(
            {
                "model": name,
                "threshold": float(component_threshold),
                "threshold_source": "full_324_d1_train_cross_fitted_oof_mcc",
                "oof_mcc_at_selection": float(component_mcc),
                "external_labels_used": False,
            }
        )
        threshold_rows.append(
            {
                "model": f"blend_equal_thirds_{name}",
                "threshold": float(blend_threshold),
                "threshold_source": "full_324_d1_train_cross_fitted_oof_mcc",
                "oof_mcc_at_selection": float(blend_mcc),
                "external_labels_used": False,
            }
        )
    tanimoto_threshold, tanimoto_mcc = select_threshold(y_train, oof["tanimoto_svc"])
    threshold_rows.insert(
        0,
        {
            "model": "paper_svm",
            "threshold": float(paper_svm_boundary),
            "threshold_source": "published_svc_predict_decision_function_zero",
            "oof_mcc_at_selection": float(
                select_threshold(y_train, oof["paper_svm"])[1]
            ),
            "external_labels_used": False,
        },
    )
    threshold_rows.insert(
        1,
        {
            "model": "tanimoto_svc",
            "threshold": float(tanimoto_threshold),
            "threshold_source": "full_324_d1_train_cross_fitted_oof_mcc",
            "oof_mcc_at_selection": float(tanimoto_mcc),
            "external_labels_used": False,
        },
    )

    # -- 7. scored frames ---------------------------------------------------------
    for name in ALT_MODELS:
        test_frame[f"probability_{name}"] = alt_scored[name]["d1_test"]
        test_frame[f"blend_equal_thirds_{name}"] = (
            test_frame.probability_paper_svm
            + test_frame.probability_tanimoto_svc
            + test_frame[f"probability_{name}"]
        ) / 3.0
    test_frame["maximum_tanimoto_to_fitted_train"] = d1_test_similarity

    for cohort, payload in cohorts.items():
        scored = payload["frame"]
        for name in ALT_MODELS:
            scored[f"probability_{name}"] = alt_scored[name][cohort]
            scored[f"blend_equal_thirds_{name}"] = (
                scored.probability_paper_svm
                + scored.probability_tanimoto_svc
                + scored[f"probability_{name}"]
            ) / 3.0

    # -- 8. metrics ---------------------------------------------------------------
    def _model_streams(scored: pd.DataFrame) -> list[tuple[str, np.ndarray, float]]:
        rows: list[tuple[str, np.ndarray, float]] = [
            ("paper_svm", scored.probability_paper_svm.to_numpy(float), paper_svm_boundary),
            (
                "tanimoto_svc",
                scored.probability_tanimoto_svc.to_numpy(float),
                float(tanimoto_threshold),
            ),
        ]
        for name in ALT_MODELS:
            rows.append(
                (name, scored[f"probability_{name}"].to_numpy(float), thresholds[name])
            )
            rows.append(
                (
                    f"blend_equal_thirds_{name}",
                    scored[f"blend_equal_thirds_{name}"].to_numpy(float),
                    thresholds[f"blend_{name}"],
                )
            )
        return rows

    metric_rows: list[dict[str, Any]] = []
    for cohort, endpoint, scored in (
        ("d1_paper_test", D1_ENDPOINT, test_frame),
        ("drugage", DRUGAGE_ENDPOINT, cohorts["drugage"]["frame"]),
        ("agextend", AGEXTEND_ENDPOINT, cohorts["agextend"]["frame"]),
    ):
        y = scored.label.to_numpy(dtype=int)
        for model, probability, threshold in _model_streams(scored):
            metric_rows.append(
                _metric_row(
                    cohort,
                    endpoint,
                    model,
                    "oof_mcc" if model != "paper_svm" else "published_svc_native_predict",
                    y,
                    probability,
                    threshold,
                    "this_run",
                )
            )
            metric_rows.append(
                _metric_row(
                    cohort, endpoint, model, "fixed_0p5", y, probability, 0.5, "this_run"
                )
            )
    metrics = pd.DataFrame(metric_rows)

    # Applicability strata, reported exactly like the sealed ablation.
    applicability_rows: list[dict[str, Any]] = []
    for cohort, endpoint, scored, similarity in (
        ("d1_paper_test", D1_ENDPOINT, test_frame, d1_test_similarity),
        (
            "drugage",
            DRUGAGE_ENDPOINT,
            cohorts["drugage"]["frame"],
            cohorts["drugage"]["max_tanimoto"],
        ),
        (
            "agextend",
            AGEXTEND_ENDPOINT,
            cohorts["agextend"]["frame"],
            cohorts["agextend"]["max_tanimoto"],
        ),
    ):
        outside = np.asarray(similarity) < float(protocol["applicability"]["threshold"])
        subset = scored.loc[outside]
        if len(subset) == 0 or set(subset.label.astype(int)) != {0, 1}:
            continue
        y = subset.label.to_numpy(dtype=int)
        for model, probability, threshold in _model_streams(subset):
            applicability_rows.append(
                {
                    "stratum": "maximum_tanimoto_lt_0p40",
                    **_metric_row(
                        cohort,
                        endpoint,
                        model,
                        "oof_mcc",
                        y,
                        probability,
                        threshold,
                        "this_run",
                    ),
                }
            )
    applicability = pd.DataFrame(applicability_rows)

    # -- 9. sealed reference rows for the published SVM and the locked blend ------
    sealed_d1 = pd.read_csv(root / sealed["ablation_d1_metrics"]["path"])
    sealed_external = pd.read_csv(root / sealed["ablation_external_metrics"]["path"])
    reference = pd.concat([sealed_d1, sealed_external], ignore_index=True)
    reference = reference[reference.is_primary_operating_point.astype(bool)]
    reference = reference[
        reference.endpoint.isin(
            [D1_ENDPOINT, DRUGAGE_ENDPOINT, AGEXTEND_ENDPOINT]
        )
    ].copy()
    reference["result_source"] = "sealed_locked_run_20260818"

    primary = metrics[metrics.operating_point != "fixed_0p5"].copy()
    comparison_columns = [
        "cohort",
        "endpoint",
        "model",
        "result_source",
        "n_test",
        "auprc_average_precision_positive",
        "auroc",
        "brier",
        "mcc",
        "macro_f1",
        "recall_sensitivity",
        "specificity",
        "threshold",
    ]
    comparison = pd.concat(
        [
            reference[reference.model.isin(["paper_svm", "tabpfn_v2", "blend_010_060_030"])][
                comparison_columns
            ],
            primary[primary.model.str.startswith("blend_equal_thirds")][comparison_columns],
            primary[primary.model.isin(list(ALT_MODELS))][comparison_columns],
        ],
        ignore_index=True,
    )
    order = {"d1_paper_test": 0, "drugage": 1, "agextend": 2}
    comparison["_cohort_order"] = comparison.cohort.map(order)
    comparison = comparison.sort_values(
        ["_cohort_order", "model"], kind="stable"
    ).drop(columns=["_cohort_order"])

    # -- 10. write the run --------------------------------------------------------
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".altmodels.work-", dir=destination.parent))
    try:
        (temporary / "models").mkdir()
        for name in ALT_MODELS:
            payload = alt_states[name]
            joblib.dump(
                {
                    "model": name,
                    "descriptor_context": payload["context"],
                    "scaler": payload["scaler"],
                    "audit": payload["audit"],
                },
                temporary / "models" / f"{name}_preprocessing.joblib",
            )
            if name == "tabnet":
                for index, estimator in enumerate(payload["states"]):
                    estimator.save_model(
                        str(temporary / "models" / f"{name}_seed{index}")
                    )
            else:
                import torch

                torch.save(
                    [
                        {key: value.cpu() for key, value in state.items()}
                        for state in payload["states"]
                    ],
                    temporary / "models" / f"{name}_state_dicts.pt",
                )

        _write_csv(
            temporary / "d1_train_oof_predictions.csv",
            pd.DataFrame(
                {
                    "paper_row_index": train_indices,
                    "label": y_train,
                    "probability_paper_svm": oof["paper_svm"],
                    "probability_tanimoto_svc": oof["tanimoto_svc"],
                    **{f"probability_{name}": alt_oof[name] for name in ALT_MODELS},
                    **{
                        f"blend_equal_thirds_{name}": blend_oof[name] for name in ALT_MODELS
                    },
                }
            ),
        )
        _write_csv(temporary / "component_thresholds.csv", pd.DataFrame(threshold_rows))
        _write_csv(temporary / "d1_test_predictions.csv", test_frame)
        for cohort, payload in cohorts.items():
            _write_csv(
                temporary / f"external_predictions_{cohort}.csv", payload["frame"]
            )
        _write_csv(temporary / "metrics_all.csv", metrics)
        if len(applicability):
            _write_csv(temporary / "applicability_stratified_metrics.csv", applicability)
        _write_csv(temporary / "comparison_vs_paper_svm.csv", comparison)

        summary_parts = [
            "# Equal-weight screening blends with BiSHop / TabM / TabNet",
            "",
            "The paper SVM and the Morgan-Tanimoto SVC components are the same fitted state as",
            "the locked 0.10/0.60/0.30 blend; only the third slot and the weights change.",
            "Weights are a prespecified 1/3 each. Every threshold comes from the 324 D1-train",
            "cross-fitted OOF. The 81 D1 test rows, 446 DrugAge compounds and 69 AgeXtend",
            "compounds were scored once afterwards.",
            "",
            "Paper-SVM rows and the locked-blend/TabPFN rows are copied unchanged from the",
            f"sealed run `{sealed['ablation_completed']['path']}`.",
            "",
        ]
        for cohort, title in (
            ("d1_paper_test", "D1 held-out paper test (n=81)"),
            ("drugage", "DrugAge positive-retrieval endpoint (n=446)"),
            ("agextend", "AgeXtend Table 6 endpoint (n=69)"),
        ):
            summary_parts.append(
                _summary_table(comparison[comparison.cohort == cohort], title)
            )
        summary_parts.extend(
            [
                "## Operating points",
                "",
                _plain_markdown(pd.DataFrame(threshold_rows)),
                "",
                "## Interpretation boundary",
                "",
                "DrugAge background compounds are not certified experimental negatives and",
                "AgeXtend has only four negatives after the locked exclusions. These",
                "blends are",
                "additional prespecified-weight experiments; they do not replace the locked",
                "screening model and no external outcome selected any of them.",
                "",
            ]
        )
        (temporary / "summary.md").write_text("\n".join(summary_parts), encoding="utf-8")

        manifest = {
            "schema_version": f"{SCHEMA}.run_manifest.v1",
            "run_id": run_id,
            "protocol_sha256": protocol_sha256,
            "paper_split_sha256": split_sha256,
            "locked_component_state_sha256": bundle["component_state_sha256"],
            "blend_weights": [EQUAL_WEIGHT] * 3,
            "alt_models": list(ALT_MODELS),
            "thresholds": thresholds,
            "paper_svm_native_probability_boundary": float(paper_svm_boundary),
            "tanimoto_oof_mcc_threshold": float(tanimoto_threshold),
            "parity": {
                "morgan_bits_equal_sealed_bundle": True,
                "descriptor_context_equal_sealed_bundle": True,
                "oof_max_abs_difference": oof_parity,
                "d1_test_max_abs_difference": test_parity,
            },
            "alt_model_audit": alt_audits,
            "package_versions": versions,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "sealed_inputs_sha256": {
                name: sha256_file(root / record["path"])
                for name, record in sealed.items()
            },
            "test_or_external_labels_used_for_fit_weight_or_threshold": False,
            "existing_runs_modified": False,
        }
        atomic_write_json(temporary / "RUN_MANIFEST.json", manifest)
        atomic_write_json(
            temporary / "COMPLETED.json",
            {
                "schema_version": f"{SCHEMA}.completed.v1",
                "status": "COMPLETE",
                "run_id": run_id,
                "run_manifest_sha256": sha256_file(temporary / "RUN_MANIFEST.json"),
                "artifact_hashes": {
                    str(path.relative_to(temporary)): sha256_file(path)
                    for path in sorted(temporary.rglob("*"))
                    if path.is_file() and path.name != "COMPLETED.json"
                },
            },
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps({"run": str(destination), "status": "COMPLETE"}, indent=2))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args(argv)
    run(
        root=arguments.root,
        config_path=arguments.config,
        positive_path=arguments.positive,
        negative_path=arguments.negative,
        run_id=arguments.run_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
