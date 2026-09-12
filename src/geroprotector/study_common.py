"""Shared statistics, chemistry and reporting helpers for the insight-analysis suite.

Everything here is deterministic and dependency-light: the pinned `bishop`
environment has no `statsmodels` and no `tabulate`, and adding packages to a
locked environment mid-project is not acceptable, so the logistic regression,
its standard errors, the trend test and the markdown writer are implemented
directly.  Each routine is small enough to be checked against a closed form.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats

from geroprotector.hashing import atomic_write_json, canonical_sha256

RDLogger.DisableLog("rdApp.*")

Z95 = 1.959963984540054
QED_STRATA = ((0.0, 0.3, "[0,0.3)"), (0.3, 0.5, "[0.3,0.5)"),
              (0.5, 0.7, "[0.5,0.7)"), (0.7, 1.01, "[0.7,1.0]"))
BASIC_PROPERTIES = ("qed", "mol_weight", "clogp", "h_donors", "h_acceptors",
                    "rotatable_bonds")


def bind_checkpoint_directory(path: Path, contract: dict[str, Any], *, error_cls=RuntimeError) -> str:
    """Create or verify a canonical binding for resumable intermediate files."""
    path = Path(path)
    binding = canonical_sha256(contract)
    marker = path / "CHECKPOINT_BINDING.json"
    if path.exists():
        if not marker.is_file():
            raise error_cls(f"Unbound checkpoint directory cannot be resumed safely: {path}")
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("binding_sha256") != binding or recorded.get("contract") != contract:
            raise error_cls(
                "Checkpoint metadata does not match the current protocol, inputs, "
                "split, or source code")
    else:
        path.mkdir(parents=True, exist_ok=False)
        atomic_write_json(marker, {"binding_sha256": binding, "contract": contract})
    return binding


# --------------------------------------------------------------------- reporting

def md_table(frame: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """Plain GitHub-flavoured markdown; avoids a `tabulate` dependency."""
    def cell(value: Any) -> str:
        if isinstance(value, float):
            if not np.isfinite(value):
                return "n/a"
            return float_format.format(value)
        return "" if value is None else str(value)

    cols = [str(c) for c in frame.columns]
    rows = [[cell(v) for v in rec] for rec in frame.itertuples(index=False, name=None)]
    widths = [max([len(cols[i])] + [len(r[i]) for r in rows]) for i in range(len(cols))]
    line = lambda cs: "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cs)) + " |"
    return "\n".join([line(cols), "| " + " | ".join("-" * w for w in widths) + " |"]
                     + [line(r) for r in rows])


# ------------------------------------------------------------------- proportions

def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (float(centre - half), float(centre + half))


def cochran_armitage(successes: Sequence[int], totals: Sequence[int],
                     scores: Sequence[float] | None = None) -> dict[str, float]:
    """Two-sided Cochran-Armitage test for trend in proportions across ordered strata."""
    k = np.asarray(successes, dtype=float)
    n = np.asarray(totals, dtype=float)
    keep = n > 0
    k, n = k[keep], n[keep]
    if len(n) < 2:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "strata_used": int(len(n))}
    x = (np.asarray(scores, dtype=float)[keep] if scores is not None
         else np.arange(len(n), dtype=float))
    total, successes_total = n.sum(), k.sum()
    p = successes_total / total
    numerator = float(np.sum(x * (k - n * p)))
    variance = float(p * (1 - p) * (np.sum(n * x * x) - (np.sum(n * x) ** 2) / total))
    if variance <= 0:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "strata_used": int(len(n))}
    z = numerator / np.sqrt(variance)
    return {"statistic": float(z),
            "p_value": float(2.0 * stats.norm.sf(abs(z))),
            "strata_used": int(len(n))}


# ------------------------------------------------------------------ associations

def spearman_bootstrap(x: np.ndarray, y: np.ndarray, *, n_resamples: int,
                       seed: int) -> dict[str, float]:
    """Spearman rho with a compound-level percentile bootstrap interval."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = len(x)
    if n < 4:
        return {"n": int(n), "rho": float("nan"), "p_value": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "bootstrap_valid": 0}
    result = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_resamples, n))
    values = np.full(n_resamples, np.nan)
    for i in range(n_resamples):
        idx = draws[i]
        xs, ys = x[idx], y[idx]
        if len(np.unique(xs)) < 2 or len(np.unique(ys)) < 2:
            continue
        values[i] = stats.spearmanr(xs, ys).statistic
    valid = values[np.isfinite(values)]
    low, high = ((np.percentile(valid, 2.5), np.percentile(valid, 97.5))
                 if len(valid) >= 100 else (np.nan, np.nan))
    return {"n": int(n), "rho": float(result.statistic),
            "p_value": float(result.pvalue),
            "ci_low": float(low), "ci_high": float(high),
            "bootstrap_valid": int(len(valid))}


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta via the Mann-Whitney U statistic (exact, no sampling)."""
    a = np.asarray(a, float)[np.isfinite(a)]
    b = np.asarray(b, float)[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with a pooled standard deviation."""
    a = np.asarray(a, float)[np.isfinite(a)]
    b = np.asarray(b, float)[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


# ---------------------------------------------------------- logistic regression

def logistic_fit(x: np.ndarray, y: np.ndarray, *, max_iter: int = 200,
                 tol: float = 1e-10) -> dict[str, Any]:
    """Unpenalised logistic regression by Newton-Raphson, with Wald inference.

    Returns coefficients, standard errors from the inverse observed information,
    a convergence flag, and a separation flag.  Separation is detected from the
    fitted probabilities rather than from the coefficient magnitudes alone,
    because a large coefficient with a finite likelihood is not separation.
    Nothing is silently substituted when the fit degenerates; the caller is
    expected to report the failure and fall back to the prespecified trend test.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(y)), x]) if x.ndim > 0 and x.size else np.ones((len(y), 1))
    if x.ndim == 1:
        design = np.column_stack([np.ones(len(y)), x])
    p = design.shape[1]
    beta = np.zeros(p)
    converged, singular = False, False
    for _ in range(max_iter):
        eta = design @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        w = np.clip(mu * (1.0 - mu), 1e-12, None)
        gradient = design.T @ (y - mu)
        hessian = design.T @ (design * w[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            singular = True
            break
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    eta = design @ beta
    mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
    separation = bool(np.all((mu > 0.999) == (y > 0.5)) and len(np.unique(y)) == 2) \
        or bool(np.max(np.abs(beta[1:])) > 25.0 if p > 1 else False)
    w = np.clip(mu * (1.0 - mu), 1e-12, None)
    try:
        covariance = np.linalg.inv(design.T @ (design * w[:, None]))
        standard_error = np.sqrt(np.clip(np.diag(covariance), 0, None))
    except np.linalg.LinAlgError:
        standard_error = np.full(p, np.nan)
        singular = True
    z = np.divide(beta, standard_error, out=np.full(p, np.nan),
                  where=np.isfinite(standard_error) & (standard_error > 0))
    return {"coefficients": beta, "standard_errors": standard_error,
            "z": z, "p_values": 2.0 * stats.norm.sf(np.abs(z)),
            "converged": bool(converged and not singular),
            "separation": bool(separation), "singular": bool(singular),
            "n": int(len(y)), "n_positive": int(y.sum()),
            "fitted": mu}


def logistic_predict(fit: dict[str, Any], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    design = np.column_stack([np.ones(len(x)), x]) if x.ndim > 1 else \
        np.column_stack([np.ones(len(x)), x])
    return 1.0 / (1.0 + np.exp(-np.clip(design @ fit["coefficients"], -35, 35)))


# --------------------------------------------------------------- multiplicity

def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p values; NaNs pass through as NaN."""
    p = np.asarray(p_values, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    adjusted = np.full(len(p), np.nan)
    if len(finite) == 0:
        return adjusted
    order = finite[np.argsort(p[finite], kind="mergesort")]
    m = len(order)
    running = 0.0
    for rank, index in enumerate(order):
        value = (m - rank) * p[index]
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted


# ------------------------------------------------------------------- chemistry

def morgan_generator(radius: int = 2, bits: int = 2048, chirality: bool = False):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=bits, includeChirality=chirality)


def morgan_matrix(smiles: Sequence[str], generator=None) -> np.ndarray:
    generator = generator or morgan_generator()
    out = np.zeros((len(smiles), generator.GetOptions().fpSize), dtype=np.uint8)
    for row, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(text))
        if molecule is None:
            raise ValueError(f"Unparsable structure at row {row}: {text!r}")
        out[row] = np.asarray(generator.GetFingerprint(molecule), dtype=np.uint8)
    return out


def max_tanimoto(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Row-wise maximum Tanimoto of `query` bit vectors against `reference`."""
    if len(reference) == 0:
        return np.full(len(query), np.nan)
    q = query.astype(np.float32)
    r = reference.astype(np.float32)
    intersection = q @ r.T
    union = q.sum(1)[:, None] + r.sum(1)[None, :] - intersection
    similarity = np.divide(intersection, union, out=np.zeros_like(intersection),
                           where=union > 0)
    return similarity.max(axis=1).astype(np.float64)


def molecular_variables(smiles: Sequence[str]) -> pd.DataFrame:
    """QED and the basic physicochemical panel of section 5.4, plus the scaffold."""
    records = []
    for text in smiles:
        molecule = Chem.MolFromSmiles(str(text))
        if molecule is None:
            records.append({"parsable": False, "qed": np.nan, "mol_weight": np.nan,
                            "clogp": np.nan, "h_donors": np.nan, "h_acceptors": np.nan,
                            "rotatable_bonds": np.nan, "lipinski_violations": np.nan,
                            "scaffold": "", "is_acyclic": np.nan})
            continue
        weight = Descriptors.MolWt(molecule)
        logp = Crippen.MolLogP(molecule)
        donors = Lipinski.NumHDonors(molecule)
        acceptors = Lipinski.NumHAcceptors(molecule)
        violations = int(weight > 500) + int(logp > 5) + int(donors > 5) + int(acceptors > 10)
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
        except Exception:
            scaffold = ""
        records.append({
            "parsable": True, "qed": float(QED.qed(molecule)), "mol_weight": float(weight),
            "clogp": float(logp), "h_donors": int(donors), "h_acceptors": int(acceptors),
            "rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
            "lipinski_violations": int(violations), "scaffold": scaffold,
            "is_acyclic": bool(scaffold == "")})
    return pd.DataFrame.from_records(records)


def qed_stratum(values: np.ndarray) -> np.ndarray:
    """Locked QED stratum label for each value; empty string when QED is missing."""
    labels = np.full(len(values), "", dtype=object)
    for low, high, name in QED_STRATA:
        mask = np.isfinite(values) & (values >= low) & (values < high)
        labels[mask] = name
    return labels
