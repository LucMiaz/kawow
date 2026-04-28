"""
model.py
========
Ridge-regression model for group-additivity prediction of logKow and logKoa.

The additive structure (descriptor = sum of atom/group contributions + const)
is exactly a linear model with no feature interactions, so Ridge regression
with L2 regularisation is the correct statistical analogue of Naef's
Gauss-Seidel parameter fitting.

Training data
-------------
  logKow : S01.sdf  (Naef & Acree, Liquids 2024, ~3332 molecules)
  logKoa : S02.sdf  (same paper, ~1900 molecules)
  logKaw  = logKow - logKoa   (Naef Eq. 2, derived, not directly trained)

Fitted coefficients are saved as JSON in kawow/data/ so the package can
be imported and used without re-fitting.
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import compute_features
from .atom_types import FEATURE_LABELS
from .io import _read_sdf

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

_MODEL_LOGKOW = DATA_DIR / "logkow_model.json"
_MODEL_LOGKOA = DATA_DIR / "logkoa_model.json"

# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_Xy(sdf_path: str | Path, value_prop: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load SDF, compute features, return (X, y, names). Skips failures."""
    rows = _read_sdf(sdf_path, target_prop=value_prop)
    X, y, names = [], [], []
    for mol, name, val in rows:
        if val is None:
            continue
        feat = compute_features(mol)
        if feat is None:
            continue
        X.append(feat)
        y.append(val)
        names.append(name)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float64), names


def _fit_and_save(X: np.ndarray, y: np.ndarray, out_path: Path, target: str) -> dict:
    """Fit RidgeCV with iterative 3σ outlier removal, save coefficients to JSON."""
    alphas = np.logspace(-3, 4, 50)

    def _make_pipeline():
        return make_pipeline(
            StandardScaler(with_mean=False),
            RidgeCV(alphas=alphas, cv=5, fit_intercept=True),
        )

    # ── Iterative 3σ outlier removal (matches Naef & Acree paper) ────────────
    # Fit once, identify outliers (|residual| > 3 × RMSE), remove, refit once.
    mask = np.ones(len(y), dtype=bool)
    model = _make_pipeline()
    model.fit(X, y)
    residuals = y - model.predict(X)
    sigma = residuals.std()
    mask = np.abs(residuals) <= 3 * sigma
    n_removed = int((~mask).sum())
    if n_removed > 0:
        print(f"  [{target}] removing {n_removed} outliers (>3*sigma={sigma:.3f})")
        X, y = X[mask], y[mask]
        model = _make_pipeline()
        model.fit(X, y)

    # Cross-validated performance (on cleaned dataset)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_cv = cross_val_predict(model, X, y, cv=kf)
    r2   = r2_score(y, y_cv)
    rmse = float(mean_squared_error(y, y_cv) ** 0.5)

    # Extract linear coefficients for the JSON model
    scaler = model.named_steps["standardscaler"]
    ridge  = model.named_steps["ridgecv"]
    # Ridge coefs are in *scaled* space; convert back to original
    scale  = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    coefs  = ridge.coef_ / scale          # shape (N_FEATURES,)
    intercept = float(ridge.intercept_)

    # Correct intercept for mean-subtraction artefact
    # (with_mean=False so no mean subtraction — intercept is already on original scale)

    result = {
        "target": target,
        "n_train": len(y),
        "alpha": float(ridge.alpha_),
        "r2_cv": round(r2, 4),
        "rmse_cv": round(rmse, 4),
        "intercept": intercept,
        "weights": {label: float(coefs[i]) for i, label in enumerate(FEATURE_LABELS)},
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [{target}] n={len(y)}  R²={r2:.4f}  RMSE={rmse:.4f}  "
          f"alpha={ridge.alpha_:.4g}  → {out_path.name}")
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def fit(
    sdf_logkow: str | Path,
    sdf_logkoa: str | Path,
    logkow_prop: str = "logP",
    logkoa_prop: str = "logKoa",
) -> None:
    """
    Fit Ridge models on S01 (logKow) and S02 (logKoa) SDF files and save
    coefficients to kawow/data/*.json.

    Parameters
    ----------
    sdf_logkow : path to SDF file with logKow data (tag name in logkow_prop)
    sdf_logkoa : path to SDF file with logKoa data (tag name in logkoa_prop)
    """
    print("Fitting logKow model …")
    X_kow, y_kow, _ = _build_Xy(sdf_logkow, logkow_prop)
    _fit_and_save(X_kow, y_kow, _MODEL_LOGKOW, "logKow")

    print("Fitting logKoa model …")
    X_koa, y_koa, _ = _build_Xy(sdf_logkoa, logkoa_prop)
    _fit_and_save(X_koa, y_koa, _MODEL_LOGKOA, "logKoa")
    print("Done.")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run kawow.fit(sdf_logkow, sdf_logkoa) first."
        )
    with open(path) as f:
        return json.load(f)


def _predict_from_json(feat: np.ndarray, model_dict: dict) -> float:
    """Apply saved linear model to a feature vector."""
    weights = np.array([model_dict["weights"][label] for label in FEATURE_LABELS],
                       dtype=np.float64)
    return float(np.dot(feat.astype(np.float64), weights) + model_dict["intercept"])


class PartitionCalculator:
    """
    Predict logKow, logKoa, logKaw from molecular structure using the
    Naef group-additivity method (re-fitted on S01/S02 SDF training data).

    Usage
    -----
    >>> from kawow import PartitionCalculator
    >>> calc = PartitionCalculator()
    >>> calc.predict("CCCCO")            # 1-butanol
    {'logKow': 0.88, 'logKoa': 4.12, 'logKaw': -3.24, 'status': 'ok'}
    >>> calc.predict_batch(["CCCCO", "c1ccccc1"])
    [{'smiles': ..., 'logKow': ..., ...}, ...]
    """

    def __init__(self) -> None:
        self._kow = _load_json(_MODEL_LOGKOW)
        self._koa = _load_json(_MODEL_LOGKOA)

    def _predict_mol(self, mol) -> dict:
        feat = compute_features(mol)
        if feat is None:
            return {"status": "error", "error": "molecule has fewer than 2 heavy atoms"}
        logKow = _predict_from_json(feat, self._kow)
        logKoa = _predict_from_json(feat, self._koa)
        logKaw = logKow - logKoa
        return {
            "logKow": round(logKow, 3),
            "logKoa": round(logKoa, 3),
            "logKaw": round(logKaw, 3),
            "status": "ok",
        }

    def predict(self, inp, fmt: str = "auto") -> dict | list[dict]:
        """
        Predict partition coefficients for one or more molecules.

        Parameters
        ----------
        inp : SMILES str | InChI str | RDKit Mol | Path to SDF | list of any
        fmt : 'auto' | 'smiles' | 'inchi' | 'sdf' | 'mol'

        Returns
        -------
        dict  (single molecule)  or  list[dict]  (multiple molecules)
        Each dict has keys: logKow, logKoa, logKaw, status[, error, name].
        """
        from .io import parse_input
        pairs = parse_input(inp, fmt=fmt)
        results = []
        for mol, name in pairs:
            r = self._predict_mol(mol)
            r["name"] = name
            try:
                from rdkit.Chem import Descriptors
                r["smiles"] = mol.GetProp("_SMILES") if mol.HasProp("_SMILES") else ""
            except Exception:
                pass
            results.append(r)
        if len(results) == 1:
            return results[0]
        return results

    def predict_batch(self, smiles_list: list[str]) -> list[dict]:
        """Convenience wrapper: predict from a list of SMILES strings."""
        from rdkit import Chem
        results = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is None:
                results.append({"smiles": smi, "status": "error",
                                 "error": "invalid SMILES"})
                continue
            r = self._predict_mol(mol)
            r["smiles"] = smi
            results.append(r)
        return results

    @property
    def model_info(self) -> dict:
        """Return training statistics for both fitted models."""
        return {
            "logKow": {k: v for k, v in self._kow.items() if k != "weights"},
            "logKoa": {k: v for k, v in self._koa.items() if k != "weights"},
        }

    @staticmethod
    def fit(
        sdf_logkow: str | Path,
        sdf_logkoa: str | Path,
        logkow_prop: str = "logP",
        logkoa_prop: str = "logKoa",
    ) -> None:
        """Fit and save models (calls module-level :func:`fit`)."""
        fit(sdf_logkow, sdf_logkoa, logkow_prop, logkoa_prop)
