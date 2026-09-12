"""Bayesian softmax-gated mixture-of-experts on D1, adapted from Bariletto et al. (2026).

REFERENCE.  "On Bayesian Softmax-Gated Mixture-of-Experts Models", Bariletto, Nguyen, Ho
and Rinaldo, arXiv:2604.20551, 23 April 2026.

WHAT THIS IS, AND WHAT IT IS NOT.  The reference is a theoretical paper: it establishes
posterior contraction rates, parameter-estimation guarantees under Voronoi-type losses, and
criteria for choosing the number of experts. Its own experiments are entirely synthetic
(uniform covariates, d in {2,4,6}, n in {100..2000}), it releases no code, and its developed
model uses GAUSSIAN experts for REGRESSION. This module is therefore an adaptation, not a
reproduction, and the distinction is worth stating precisely:

  FAITHFUL to the reference
    * softmax gating pi_k(x) proportional to exp(a_k . x + b_k), with the last expert's
      gating parameters fixed at zero for identifiability, as the paper's identifiability
      analysis requires;
    * a Bayesian treatment with Normal priors over all gating and expert parameters;
    * black-box variational inference with a mean-field Gaussian variational family and
      reparameterised gradients, which is the fitting procedure of their Appendix B.1;
    * selection of the number of experts K by maximising the variational ELBO, which is
      the practical criterion analysed in their Section 6.

  ADAPTED, AND NOT COVERED BY THEIR THEOREMS
    * the experts here are Bernoulli (logistic) rather than Gaussian, because D1 is a
      binary classification task. The reference only "briefly discusses" classification in
      its Section 7 and proves nothing about it. The conditional model implemented is
        p(y=1 | x) = sum_k pi_k(x) * sigmoid(beta_k . x + c_k)
      which is the natural classification analogue of their location model but carries none
      of their guarantees.
    * the covariate dimension is held to the seven published DataWarrior descriptors. Their
      theory and experiments cover d <= 6 and the contraction rates degrade with dimension;
      with n = 324 training compounds the 205-descriptor RDKit2D panel would give the gating
      network alone 205*(K-1) parameters and is not a defensible regime for this model.

LEAKAGE DISCIPLINE.  K is selected by ELBO on training data only. In the held-out
evaluation, K is chosen on all 324 training compounds and the resulting model is scored once
on the 81 test compounds. In cross-validation, K is re-selected inside every fold from that
fold's training rows, so the reported CV estimate contains no selection performed on the
rows being scored. Features are standardised with statistics fitted on training rows only.
The decision threshold is fixed at 0.5 and is never tuned.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from geroprotector.fixed_blend_paper405 import _features
from geroprotector.fixed_blend_paper405 import load_protocol as load_fixed_protocol
from geroprotector.hashing import atomic_write_json, sha256_file
from geroprotector.hypermoltab_paper405 import _paper_contract, _validated_raw_smiles
from geroprotector.traditional_paper405 import _read_sources, paper_split_indices


class BayesMoEError(RuntimeError):
    """Raised when a contract or a leakage guard fails."""


SCHEMA = "geroprotector.bayes_moe_paper405"
THRESHOLD = 0.5


# --------------------------------------------------------------------- model
class SoftmaxGatedMoE(torch.nn.Module):
    """Mean-field Gaussian variational posterior over a softmax-gated MoE.

    Parameters are the gating weights (a_k, b_k) for k < K, with the K-th expert's gating
    parameters fixed at zero to remove the softmax shift invariance, and the expert weights
    (beta_k, c_k) for every k. Each has an independent Normal prior and an independent
    Gaussian variational factor, so the KL term is available in closed form.
    """

    def __init__(self, d: int, k: int, prior_scale: float, seed: int):
        super().__init__()
        self.d, self.k, self.prior_scale = d, k, prior_scale
        generator = torch.Generator().manual_seed(seed)

        def parameter(*shape, scale=0.1):
            return torch.nn.Parameter(
                torch.randn(*shape, generator=generator) * scale)

        # gating: K-1 free experts (the last is the reference), weights and bias
        self.gate_mu = parameter(max(k - 1, 1), d)
        self.gate_rho = torch.nn.Parameter(torch.full((max(k - 1, 1), d), -3.0))
        self.gate_bias_mu = parameter(max(k - 1, 1))
        self.gate_bias_rho = torch.nn.Parameter(torch.full((max(k - 1, 1),), -3.0))
        # experts
        self.expert_mu = parameter(k, d)
        self.expert_rho = torch.nn.Parameter(torch.full((k, d), -3.0))
        self.expert_bias_mu = parameter(k)
        self.expert_bias_rho = torch.nn.Parameter(torch.full((k,), -3.0))

    @staticmethod
    def _sigma(rho):
        return torch.nn.functional.softplus(rho) + 1e-6

    def _sample(self, generator=None):
        out = []
        for mu, rho in ((self.gate_mu, self.gate_rho),
                        (self.gate_bias_mu, self.gate_bias_rho),
                        (self.expert_mu, self.expert_rho),
                        (self.expert_bias_mu, self.expert_bias_rho)):
            sigma = self._sigma(rho)
            noise = torch.randn(mu.shape, device=mu.device, generator=generator)
            out.append(mu + sigma * noise)
        return out

    def kl(self) -> torch.Tensor:
        """Closed-form KL(q || prior) summed over all parameters."""
        total = 0.0
        s0 = self.prior_scale
        for mu, rho in ((self.gate_mu, self.gate_rho),
                        (self.gate_bias_mu, self.gate_bias_rho),
                        (self.expert_mu, self.expert_rho),
                        (self.expert_bias_mu, self.expert_bias_rho)):
            sigma = self._sigma(rho)
            total = total + torch.sum(
                torch.log(s0 / sigma) + (sigma ** 2 + mu ** 2) / (2 * s0 ** 2) - 0.5)
        return total

    def probability(self, x: torch.Tensor, draws: int, generator=None) -> torch.Tensor:
        """Posterior-predictive P(y=1|x), averaged over `draws` variational samples."""
        acc = []
        for _ in range(draws):
            gw, gb, ew, eb = self._sample(generator)
            logits = x @ gw.T + gb                       # (n, K-1)
            if self.k > 1:
                logits = torch.cat(
                    [logits, torch.zeros(x.shape[0], 1, device=x.device)], dim=1)
                gate = torch.softmax(logits, dim=1)      # (n, K)
            else:
                gate = torch.ones(x.shape[0], 1, device=x.device)
            expert = torch.sigmoid(x @ ew.T + eb)        # (n, K)
            acc.append((gate * expert).sum(dim=1))
        return torch.stack(acc).mean(dim=0)

    def log_likelihood(self, x, y, generator=None) -> torch.Tensor:
        """Single-sample Monte Carlo estimate of E_q[log p(y|x, theta)]."""
        gw, gb, ew, eb = self._sample(generator)
        logits = x @ gw.T + gb
        if self.k > 1:
            logits = torch.cat(
                [logits, torch.zeros(x.shape[0], 1, device=x.device)], dim=1)
            gate = torch.softmax(logits, dim=1)
        else:
            gate = torch.ones(x.shape[0], 1, device=x.device)
        p = (gate * torch.sigmoid(x @ ew.T + eb)).sum(dim=1).clamp(1e-6, 1 - 1e-6)
        return torch.sum(y * torch.log(p) + (1 - y) * torch.log1p(-p))


def fit_bbvi(x, y, k, *, prior_scale, steps, learning_rate, restarts, elbo_draws,
             seed, device):
    """Fit by black-box VI; return the best restart by final ELBO, and that ELBO."""
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    best_model, best_elbo = None, -np.inf
    for restart in range(restarts):
        torch.manual_seed(seed + 1000 * restart)
        model = SoftmaxGatedMoE(x.shape[1], k, prior_scale,
                                seed + 1000 * restart).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for _ in range(steps):
            optimiser.zero_grad()
            loss = -(model.log_likelihood(xt, yt) - model.kl())
            loss.backward()
            optimiser.step()
        with torch.no_grad():                     # ELBO averaged over several draws
            elbo = float(np.mean([
                float(model.log_likelihood(xt, yt) - model.kl())
                for _ in range(elbo_draws)]))
        if elbo > best_elbo:
            best_elbo, best_model = elbo, model
    return best_model, best_elbo


def select_k(x, y, candidates, *, device, **kwargs):
    """Choose the number of experts by maximising the ELBO on these rows only."""
    table = []
    best = (None, -np.inf, None)
    for k in candidates:
        model, elbo = fit_bbvi(x, y, k, device=device, **kwargs)
        table.append({"K": int(k), "elbo": float(elbo)})
        if elbo > best[1]:
            best = (model, elbo, k)
    return best[0], best[2], pd.DataFrame(table)


def _metrics(y, p):
    d = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    return {"n": len(y), "n_positive": int(y.sum()),
            "accuracy": float(accuracy_score(y, d)),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
            "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
            "cohen_kappa": float(cohen_kappa_score(y, d)),
            "auroc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan"),
            "auprc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "mcc": float(matthews_corrcoef(y, d)),
            "macro_f1": float(f1_score(y, d, average="macro", zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "threshold": THRESHOLD}


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an immutable artifact: {path}")
    frame.to_csv(path, index=False, lineterminator="\n")


def run(*, root: Path, positive_path: Path, negative_path: Path, run_id: str,
        k_max: int, steps: int, restarts: int, prior_scale: float,
        learning_rate: float, predict_draws: int, elbo_draws: int, device: str) -> Path:
    if not re.fullmatch(r"bayes_moe_[a-z0-9_.-]+", run_id):
        raise BayesMoEError("RUN_ID must start with bayes_moe_")
    root = root.resolve()
    destination = root / "outputs" / run_id
    if destination.exists() or destination.is_symlink():
        raise BayesMoEError(f"Run directory already exists: {destination}")

    fixed_protocol, _ = load_fixed_protocol(root / "configs" / "fixed_blend_paper405.yaml")
    traditional = _paper_contract(root, fixed_protocol)
    frame, audit = _read_sources(positive_path.resolve(), negative_path.resolve(),
                                 traditional)
    train_indices, test_indices, split_sha256 = paper_split_indices(traditional)
    labels = frame["label"].to_numpy(int)
    smiles, _ = _validated_raw_smiles(frame)
    panel = _features(frame, smiles, fixed_protocol)["paper"].astype(np.float64)
    if panel.shape[1] != 7:
        raise BayesMoEError(f"Expected the 7 paper descriptors, got {panel.shape[1]}")
    y_train, y_test = labels[train_indices], labels[test_indices]
    candidates = list(range(1, k_max + 1))
    settings = dict(prior_scale=prior_scale, steps=steps, learning_rate=learning_rate,
                    restarts=restarts, elbo_draws=elbo_draws)
    print(f"panel {panel.shape}; train {len(train_indices)}, test {len(test_indices)}; "
          f"K candidates {candidates}; device {device}", flush=True)

    def standardise(fit_rows, apply_rows):
        mean, sd = fit_rows.mean(0), fit_rows.std(0)
        sd[sd == 0] = 1.0
        return (fit_rows - mean) / sd, [(a - mean) / sd for a in apply_rows]

    generator = torch.Generator(device=device).manual_seed(20260823)

    # ---- held-out evaluation: select K on train, fit, score the test once ----
    t0 = time.time()
    x_train, (x_test,) = standardise(panel[train_indices], [panel[test_indices]])
    model, k_star, elbo_table = select_k(x_train, y_train, candidates,
                                         device=device, seed=20260823, **settings)
    print(f"selected K={k_star} by training ELBO in {time.time()-t0:.0f}s", flush=True)
    with torch.no_grad():
        p_test = model.probability(
            torch.as_tensor(x_test, dtype=torch.float32, device=device),
            predict_draws, generator).cpu().numpy()
        p_train = model.probability(
            torch.as_tensor(x_train, dtype=torch.float32, device=device),
            predict_draws, generator).cpu().numpy()
    held_out = _metrics(y_test, p_test)
    print(f"  D1 test: AP+={held_out['auprc']:.4f} MCC={held_out['mcc']:.4f} "
          f"Acc={held_out['accuracy']:.4f}", flush=True)

    # ---- 5-fold CV on the training rows, K re-selected inside every fold ----
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_split = list(folds.split(train_indices, y_train))
    fold_id = np.full(len(train_indices), -1, dtype=int)
    oof = np.full(len(train_indices), np.nan)
    fold_rows = []
    for fold, (fit_rows, validation_rows) in enumerate(fold_split):
        fold_id[validation_rows] = fold
        xf, (xv,) = standardise(panel[train_indices][fit_rows],
                                [panel[train_indices][validation_rows]])
        fold_model, fold_k, _table = select_k(
            xf, y_train[fit_rows], candidates, device=device,
            seed=20260823 + 77 * fold, **settings)
        with torch.no_grad():
            oof[validation_rows] = fold_model.probability(
                torch.as_tensor(xv, dtype=torch.float32, device=device),
                predict_draws, generator).cpu().numpy()
        fold_rows.append({"fold": fold, "selected_K": int(fold_k),
                          "n_fit": int(len(fit_rows)),
                          "n_validation": int(len(validation_rows))})
        print(f"  CV fold {fold + 1}/5: K={fold_k}", flush=True)
    if not np.isfinite(oof).all():
        raise BayesMoEError("OOF stream is incomplete")
    cross_validated = _metrics(y_train, oof)
    print(f"  5-fold CV: AP+={cross_validated['auprc']:.4f} "
          f"MCC={cross_validated['mcc']:.4f} Acc={cross_validated['accuracy']:.4f}",
          flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".bayesmoe.work-", dir=destination.parent))
    try:
        _write_csv(tmp / "metrics.csv", pd.DataFrame([
            {"evaluation": "d1_test_heldout", "selected_K": int(k_star), **held_out},
            {"evaluation": "cv5_train_oof", "selected_K": -1, **cross_validated},
            {"evaluation": "train_in_sample", "selected_K": int(k_star),
             **_metrics(y_train, p_train)}]))
        _write_csv(tmp / "elbo_model_selection.csv", elbo_table)
        _write_csv(tmp / "cv5_fold_selection.csv", pd.DataFrame(fold_rows))
        _write_csv(tmp / "d1_test_predictions.csv", pd.DataFrame({
            "paper_row_index": test_indices, "label": y_test, "probability": p_test}))
        _write_csv(tmp / "cv5_train_oof_predictions.csv", pd.DataFrame({
            "paper_row_index": train_indices, "fold": fold_id, "label": y_train,
            "probability": oof}))
        atomic_write_json(tmp / "RUN_MANIFEST.json", {
            "schema_version": f"{SCHEMA}.run_manifest.v1", "run_id": run_id,
            "reference": ("Bariletto, Nguyen, Ho, Rinaldo, 'On Bayesian Softmax-Gated "
                          "Mixture-of-Experts Models', arXiv:2604.20551, 23 April 2026"),
            "relationship_to_reference": (
                "adaptation, not reproduction: the reference is a theoretical paper with "
                "synthetic experiments only, no released code, and Gaussian experts for "
                "regression. Softmax gating with a fixed reference expert, Normal priors, "
                "black-box VI and ELBO-based selection of K follow the reference; Bernoulli "
                "experts for binary classification are an adaptation its Section 7 sketches "
                "but does not analyse, and carry none of its guarantees."),
            "feature_panel": "paper 7 DataWarrior descriptors, standardised on fit rows",
            "dimension_rationale": (
                "the reference's theory and experiments cover d<=6; with n=324 the "
                "205-descriptor panel would give the gating network 205*(K-1) parameters "
                "and is not a defensible regime for this model"),
            "paper_split_sha256": split_sha256,
            "k_candidates": candidates, "selected_K_heldout": int(k_star),
            "selected_K_per_fold": [r["selected_K"] for r in fold_rows],
            "k_selected_on": "training rows only, by ELBO",
            "inference": {"method": "black-box variational inference",
                          "variational_family": "mean-field Gaussian",
                          "kl": "closed form", "steps": steps, "restarts": restarts,
                          "learning_rate": learning_rate, "prior_scale": prior_scale,
                          "predict_draws": predict_draws, "elbo_draws": elbo_draws},
            "fixed_threshold": THRESHOLD, "threshold_is_tuned": False,
            "test_rows_used_in_any_fit_or_selection": False,
            "folds": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "data_audit": audit,
            "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": device, "platform": platform.platform()},
            "existing_runs_modified": False})
        atomic_write_json(tmp / "COMPLETED.json", {
            "schema_version": f"{SCHEMA}.completed.v1", "status": "COMPLETE",
            "run_id": run_id,
            "run_manifest_sha256": sha256_file(tmp / "RUN_MANIFEST.json"),
            "artifact_hashes": {str(p.relative_to(tmp)): sha256_file(p)
                                for p in sorted(tmp.rglob("*"))
                                if p.is_file() and p.name != "COMPLETED.json"}})
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    print(json.dumps({"run": str(destination), "status": "COMPLETE"}, indent=2))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--prior-scale", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--predict-draws", type=int, default=512)
    parser.add_argument("--elbo-draws", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = parser.parse_args(argv)
    run(root=a.root, positive_path=a.positive, negative_path=a.negative,
        run_id=a.run_id, k_max=a.k_max, steps=a.steps, restarts=a.restarts,
        prior_scale=a.prior_scale, learning_rate=a.learning_rate,
        predict_draws=a.predict_draws, elbo_draws=a.elbo_draws, device=a.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
