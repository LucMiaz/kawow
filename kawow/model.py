"""
model.py
========
Group-additivity models for prediction of logKow and logKoa.

The supported model variant is:

``"crippen"`` (default)
    Ridge regression (L2 regularisation) re-fitted on the Naef & Acree (2024)
    SDF training data. Stored in ``kawow/data/logk*_model.json``.

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
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import compute_features
from .atom_types import FEATURE_LABELS
from .io import _read_sdf

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None

try:
    from mqg.core import MolecularQuantumGraph
    _MQG_AVAILABLE = True
except Exception:  # pragma: no cover
    MolecularQuantumGraph = None
    _MQG_AVAILABLE = False

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

_MODEL_LOGKOW     = DATA_DIR / "logkow_model.json"
_MODEL_LOGKOA     = DATA_DIR / "logkoa_model.json"
_MODEL_MQG_LOGKOW = DATA_DIR / "logkow_mqg_model.pkl"
_MODEL_MQG_LOGKOA = DATA_DIR / "logkoa_mqg_model.pkl"

# Backward-compat aliases: old key → new canonical key
_KEY_ALIASES: dict[str, str] = {
    "naefacree": "naef",
    "naefacree_mixed": "naef_crippen",
}


def _normalize_model_key(key: str) -> str:
    """Return the canonical model key, emitting a DeprecationWarning for old names."""
    canonical = _KEY_ALIASES.get(key)
    if canonical is not None:
        warnings.warn(
            f"Model key {key!r} is deprecated; use {canonical!r} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return canonical
    return key


_MODEL_FILES = {
    "crippen": (_MODEL_LOGKOW, _MODEL_LOGKOA),
}

_MODEL_FILES_MQG = {
    "mqg": (_MODEL_MQG_LOGKOW, _MODEL_MQG_LOGKOA),
}

_AVAILABLE_MODEL_NAMES = (
    "crippen",
    "naef",
    "naef_crippen",
    "naef_mqg",
    "crippen_mqg",
    "mqg",
    "pfasgroups",
    "pfasgroups_crippen",
    "pfasgroups_naef",
    "pfasgroups_naef_crippen",
    "pfasgroups_naef_crippen_rf",
    "pfasgroups_naef_crippen_xgb",
    "pfasgroups_naef_crippen_nn",
)


def _classify_partition(
    logkow: float,
    logkoa: float,
    b_logkow_threshold: float = 2.0,
    vb_logkow_threshold: float = 5.0,
    b_logkoa_threshold: float = 6.0,
    logkoc_from_kow_offset: float = 0.4,
    m_logkoc_threshold: float = 4.5,
    vm_logkoc_threshold: float = 3.5,
) -> dict[str, Any]:
    """Classify B/vB and M/vM from predicted partition coefficients.

    Notes
    -----
    - B and vB follow: logKoa >= 6 plus logKow >= 2 (B) or >= 5 (vB)
      as used in Science 2006 (doi:10.1126/science.1138275).
    - Mobility classes use an estimated logKoc from logKow:
      logKoc_est = logKow - 0.4, then M if logKoc_est <= 4.5 and
      vM if logKoc_est <= 3.5 (UBA drinking-water source protection guidance).
    """
    logkoc_est = float(logkow - logkoc_from_kow_offset)
    is_B = bool((logkoa >= b_logkoa_threshold) and (logkow >= b_logkow_threshold))
    is_vB = bool((logkoa >= b_logkoa_threshold) and (logkow >= vb_logkow_threshold))
    is_M = bool(logkoc_est <= m_logkoc_threshold)
    is_vM = bool(logkoc_est <= vm_logkoc_threshold)

    b_class = "vB" if is_vB else ("B" if is_B else "non-B")
    m_class = "vM" if is_vM else ("M" if is_M else "non-M")

    # ── Regulatory gaps ────────────────────────────────────────────────────
    # Gap 1 (non vM, non vB): 3.5 < logKow < 5.0
    in_gap1 = bool((logkow > 3.5) and (logkow < vb_logkow_threshold))
    # Gap 2 (non M, non B, excl. aquatic): logKow > 4.9 AND logKoa < 6
    m_logkow_threshold = m_logkoc_threshold + logkoc_from_kow_offset
    in_gap2 = bool((logkow > m_logkow_threshold) and (logkoa < b_logkoa_threshold))
    # Gap 3 (non M, non B, incl. aquatic): 4.9 < logKow < 5.0 AND logKoa < 6
    in_gap3 = bool(
        (logkow > m_logkow_threshold)
        and (logkow < vb_logkow_threshold)
        and (logkoa < b_logkoa_threshold)
    )
    gap_labels = []
    if in_gap1:
        gap_labels.append("Gap 1")
    if in_gap2:
        gap_labels.append("Gap 2")
    if in_gap3:
        gap_labels.append("Gap 3")

    return {
        "is_B": is_B,
        "is_vB": is_vB,
        "is_M": is_M,
        "is_vM": is_vM,
        "logKoc_est": logkoc_est,
        "b_class": b_class,
        "m_class": m_class,
        "bm_class": b_class,
        "pb_class": m_class,
        "in_gap1": in_gap1,
        "in_gap2": in_gap2,
        "in_gap3": in_gap3,
        "gap_labels": gap_labels,
        "thresholds": {
            "B_logKoa_gte": float(b_logkoa_threshold),
            "B_logKow_gte": float(b_logkow_threshold),
            "vB_logKow_gte": float(vb_logkow_threshold),
            "logKoc_est_from_logKow_offset": float(logkoc_from_kow_offset),
            "M_logKoc_est_lte": float(m_logkoc_threshold),
            "vM_logKoc_est_lte": float(vm_logkoc_threshold),
            "gap1_logKow_gt": 3.5,
            "gap1_logKow_lt": float(vb_logkow_threshold),
            "gap2_logKow_gt": float(m_logkow_threshold),
            "gap2_logKoa_lt": float(b_logkoa_threshold),
            "gap3_logKow_gt": float(m_logkow_threshold),
            "gap3_logKow_lt": float(vb_logkow_threshold),
            "gap3_logKoa_lt": float(b_logkoa_threshold),
        },
    }


def _normalise_model_prediction(raw: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Normalize predictions from all calculator classes to one schema."""
    if not isinstance(raw, dict):
        return {
            "model": model_name,
            "status": "error",
            "error": "invalid prediction payload",
            "ok": False,
        }

    if "status" in raw:
        if raw.get("status") != "ok":
            return {
                "model": model_name,
                "status": "error",
                "error": str(raw.get("error", "prediction failed")),
                "ok": False,
            }
        out = {
            "model": model_name,
            "status": "ok",
            "ok": True,
            "logKow": float(raw.get("logKow")),
            "logKoa": float(raw.get("logKoa")),
            "logKaw": float(raw.get("logKaw")),
        }
    else:
        # NaefAcree* calculators return bare contribution dicts.
        out = {
            "model": model_name,
            "status": "ok",
            "ok": True,
            "logKow": float(raw.get("logKow")),
            "logKoa": float(raw.get("logKoa")),
            "logKaw": float(raw.get("logKaw")),
        }
        if "in_coverage" in raw:
            out["in_coverage"] = bool(raw.get("in_coverage"))

    out.update(_classify_partition(out["logKow"], out["logKoa"]))
    return out


def run_models(
    inp,
    models: list[str] | tuple[str, ...] | None = None,
    fmt: str = "auto",
) -> list[dict[str, Any]]:
    """Run one or more kawow models and return aligned results per molecule.

    Parameters
    ----------
    inp:
        Any input accepted by :func:`kawow.io.parse_input`.
    models:
        Sequence of model identifiers. Supported values are:
        ``crippen``, ``naef``, ``naef_crippen``,
        ``naef_mqg``, ``crippen_mqg``, ``mqg``,
        ``pfasgroups``, ``pfasgroups_naef_crippen``,
        ``pfasgroups_naef``, ``pfasgroups_naef_crippen``.
        If omitted, all available models are used.
    fmt:
        Input format forwarded to :func:`kawow.io.parse_input`.
    """
    from .io import parse_input
    from .smarts_model import (
        NaefAcreePartitionCalculator,
        NaefAcreeCrippenPartitionCalculator,
    )

    selected_raw = [m.lower() for m in (models or list(_AVAILABLE_MODEL_NAMES))]
    selected = []
    for m in selected_raw:
        canonical = _KEY_ALIASES.get(m)
        if canonical is not None:
            warnings.warn(
                f"Model key {m!r} is deprecated; use {canonical!r} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            m = canonical
        if m in _AVAILABLE_MODEL_NAMES:
            selected.append(m)
    if not selected:
        raise ValueError(
            "No valid model selected. Supported models: "
            f"{list(_AVAILABLE_MODEL_NAMES)}"
        )

    calculators: dict[str, Any] = {}
    init_errors: dict[str, str] = {}
    for model_name in selected:
        try:
            if model_name == "crippen":
                calculators[model_name] = PartitionCalculator()
            elif model_name == "naef":
                calculators[model_name] = NaefAcreePartitionCalculator()
            elif model_name == "naef_crippen":
                calculators[model_name] = NaefAcreeCrippenPartitionCalculator()
            elif model_name == "naef_mqg":
                calculators[model_name] = EnsemblePartitionCalculator("naef_mqg")
            elif model_name == "crippen_mqg":
                calculators[model_name] = EnsemblePartitionCalculator("crippen_mqg")
            elif model_name == "mqg":
                calculators[model_name] = MQGPartitionCalculator()
            elif model_name == "pfasgroups":
                calculators[model_name] = PFASGroupsPartitionCalculator("pfasgroups")
            elif model_name == "pfasgroups_crippen":
                calculators[model_name] = PFASGroupsPartitionCalculator("pfasgroups_crippen")
            elif model_name == "pfasgroups_naef_crippen":
                calculators[model_name] = PFASGroupsPartitionCalculator("pfasgroups_naef_crippen")
            elif model_name == "pfasgroups_naef":
                calculators[model_name] = PFASGroupsPartitionCalculator("pfasgroups_naef")
            elif model_name == "pfasgroups_naef_crippen_rf":
                calculators[model_name] = PFASGroupsRFPartitionCalculator()
            elif model_name == "pfasgroups_naef_crippen_xgb":
                calculators[model_name] = PFASGroupsXGBPartitionCalculator()
            elif model_name == "pfasgroups_naef_crippen_nn":
                calculators[model_name] = PFASGroupsNNPartitionCalculator()
        except Exception as exc:
            init_errors[model_name] = str(exc)

    pairs = parse_input(inp, fmt=fmt)
    out_rows: list[dict[str, Any]] = []

    for mol, name in pairs:
        smiles = ""
        if Chem is not None and mol is not None:
            try:
                smiles = Chem.MolToSmiles(mol)
            except Exception:
                smiles = ""

        row = {
            "name": name,
            "smiles": smiles,
            "models": {},
            "ok": False,
        }

        for model_name in selected:
            if model_name in init_errors:
                row["models"][model_name] = {
                    "model": model_name,
                    "status": "error",
                    "error": init_errors[model_name],
                    "ok": False,
                }
                continue

            calculator = calculators.get(model_name)
            if calculator is None:
                row["models"][model_name] = {
                    "model": model_name,
                    "status": "error",
                    "error": "calculator unavailable",
                    "ok": False,
                }
                continue

            try:
                try:
                    pred = calculator.predict(mol, fmt="mol")
                except TypeError:
                    pred = calculator.predict(mol)
                norm = _normalise_model_prediction(pred, model_name=model_name)
            except Exception as exc:
                norm = {
                    "model": model_name,
                    "status": "error",
                    "error": str(exc),
                    "ok": False,
                }
            row["models"][model_name] = norm

        row["ok"] = any(v.get("ok") for v in row["models"].values())
        out_rows.append(row)

    return out_rows

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

    # Additional metrics
    from kawow.metrics import lin_ccc as _lin_ccc, nrmse as _nrmse, jeffreys_bf_corr as _jbf
    _r_cv = float(np.corrcoef(y, y_cv)[0, 1]) if len(y) > 1 else float("nan")
    _ccc  = _lin_ccc(y, y_cv)
    _nm   = _nrmse(y, y_cv)
    _bf   = _jbf(_r_cv, len(y))

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
        "method": "ridge",
        "n_train": len(y),
        "alpha": float(ridge.alpha_),
        "r2_cv": round(r2, 4),
        "rmse_cv": round(rmse, 4),
        "ccc_cv": round(_ccc, 4),
        "nrmse_sd_cv": round(_nm["nrmse_sd"], 4),
        "nrmse_range_cv": round(_nm["nrmse_range"], 4),
        "bf10_log10_cv": round(_bf["log10_bf10"], 2),
        "r_ci95_cv": [round(_bf["ci95_lo"], 3), round(_bf["ci95_hi"], 3)],
        "intercept": intercept,
        "weights": {label: float(coefs[i]) for i, label in enumerate(FEATURE_LABELS)},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  [{target}] n={len(y)}  R\u00b2={r2:.4f}  RMSE={rmse:.4f}  "
          f"CCC={_ccc:.4f}  NRMSE\u03c3={_nm['nrmse_sd']:.4f}  "
          f"alpha={ridge.alpha_:.4g}  \u2192 {out_path.name}")
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def fit(
    sdf_logkow: str | Path,
    sdf_logkoa: str | Path,
    logkow_prop: str = "logP",
    logkoa_prop: str = "logKoa",
) -> None:
    """
    Fit models on S01 (logKow) and S02 (logKoa) SDF files and save
    coefficients to kawow/data/*.json.

    Parameters
    ----------
    sdf_logkow : path to SDF file with logKow data (tag name in logkow_prop)
    sdf_logkoa : path to SDF file with logKoa data (tag name in logkoa_prop)
    """
    _saver = _fit_and_save
    kow_path, koa_path = _MODEL_LOGKOW, _MODEL_LOGKOA

    print("Fitting logKow model (ridge) …")
    X_kow, y_kow, _ = _build_Xy(sdf_logkow, logkow_prop)
    _saver(X_kow, y_kow, kow_path, "logKow")

    print("Fitting logKoa model (ridge) …")
    X_koa, y_koa, _ = _build_Xy(sdf_logkoa, logkoa_prop)
    _saver(X_koa, y_koa, koa_path, "logKoa")
    print("Done.")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run kawow.fit(sdf_logkow, sdf_logkoa) first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _predict_from_json(feat: np.ndarray, model_dict: dict) -> float:
    """Apply saved linear model to a feature vector."""
    weights = np.array([model_dict["weights"][label] for label in FEATURE_LABELS],
                       dtype=np.float64)
    return float(np.dot(feat.astype(np.float64), weights) + model_dict["intercept"])


def _compute_mqg_features(mol, fp_size: int = 64) -> np.ndarray | None:
    """Compute zero-padded MQG eigenvalue fingerprint for one molecule."""
    if not _MQG_AVAILABLE or Chem is None:
        raise ImportError(
            "MQG model requires the molecular_quantum_graph package (import name: mqg)."
        )

    if mol is None or mol.GetNumBonds() == 0:
        return None

    try:
        ev = MolecularQuantumGraph(
            mol,
            weighting_scheme="bde",
            normalize_length=True,
        ).compute_spectrum(maxroots=fp_size)
    except Exception as exc:
        warnings.warn(f"MQG spectrum computation failed: {exc}")
        return None

    x = np.zeros(fp_size, dtype=np.float32)
    if ev is None:
        return x
    k = min(len(ev), fp_size)
    if k > 0:
        x[:k] = np.asarray(ev[:k], dtype=np.float32)
    return x


def _compute_mqg_features_with_ratios(mol, fp_size: int = 64) -> np.ndarray | None:
    """MQG eigenvalue vector + consecutive ratios ev[i+1]/ev[i].

    Returns shape ``(2*fp_size - 1,)``:
    - First ``fp_size`` values: zero-padded eigenvalues.
    - Next ``fp_size - 1`` values: consecutive ratios ev[i+1]/ev[i]
      (0 where denominator is ~0).
    """
    ev = _compute_mqg_features(mol, fp_size=fp_size)
    if ev is None:
        return None
    denom = ev[:-1]
    numer = ev[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = np.where(np.abs(denom) > 1e-9, numer / denom, 0.0).astype(np.float32)
    return np.concatenate([ev, ratios])


def _build_Xy_mqg(sdf_path: str | Path, value_prop: str, fp_size: int = 64) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load SDF, compute MQG eigenvalue + ratio features, return (X, y, names)."""
    rows = _read_sdf(sdf_path, target_prop=value_prop)
    X, y, names = [], [], []
    for mol, name, val in rows:
        if val is None:
            continue
        feat = _compute_mqg_features_with_ratios(mol, fp_size=fp_size)
        if feat is None:
            continue
        X.append(feat)
        y.append(val)
        names.append(name)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float64), names


def _fit_mqg_and_save(
    X: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    target: str,
    fp_size: int = 64,
    tuning_frac: float = 0.2,
) -> dict:
    """Fit MQG RandomForest with tuning-fold ratio feature selection.

    Pipeline
    --------
    1. Hold out ``tuning_frac`` of the data as the *tuning fold*.
    2. On the tuning fold, rank the ``fp_size - 1`` consecutive ratio features
       by |Spearman correlation| with ``y``.
    3. Grid-search k ∈ {0, 5, 10, 15, 20, 30, 40, 50, fp_size-1}: use 3-fold
       CV on the remaining training set to pick the k that maximises R².
    4. Evaluate the final model (raw eigenvalues + selected k ratio features)
       with 5-fold CV on the *full* dataset, then fit on all data.
    """
    n_raw    = fp_size           # eigenvalue columns (indices 0..n_raw-1)
    n_ratios = fp_size - 1       # consecutive ratio columns

    # ── Tuning fold ───────────────────────────────────────────────────────────
    idx_fit, idx_tune = train_test_split(
        np.arange(len(y)), test_size=tuning_frac, random_state=42, shuffle=True
    )
    X_tune, y_tune = X[idx_tune], y[idx_tune]
    X_fit,  y_fit  = X[idx_fit],  y[idx_fit]

    # Rank ratio features by |Spearman r| on tuning set
    ratio_block = X_tune[:, n_raw:]
    corrs = np.array(
        [abs(spearmanr(ratio_block[:, i], y_tune).statistic) for i in range(n_ratios)],
        dtype=np.float64,
    )
    sorted_by_corr = np.argsort(corrs)[::-1]   # best first

    # Grid-search k using 3-fold CV on training set
    k_candidates = [0, 5, 10, 15, 20, 30, 40, 50, n_ratios]
    best_k, best_cv_r2 = 0, -np.inf
    for k in k_candidates:
        cols = list(range(n_raw))
        if k > 0:
            cols += [int(n_raw + j) for j in sorted_by_corr[:k]]
        tmp_model = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        )
        scores = cross_val_score(tmp_model, X_fit[:, cols], y_fit, cv=3, scoring="r2")
        mean_r2 = float(scores.mean())
        if mean_r2 > best_cv_r2:
            best_cv_r2 = mean_r2
            best_k = k

    selected_ratio_idx = [int(j) for j in sorted_by_corr[:best_k]] if best_k > 0 else []
    feature_cols = list(range(n_raw)) + [n_raw + j for j in selected_ratio_idx]

    print(
        f"  [{target} MQG] tuning fold selected {best_k}/{n_ratios} ratio features "
        f"(3-fold CV R²={best_cv_r2:.3f})"
    )

    # ── Final 5-fold CV + full fit ────────────────────────────────────────────
    X_sel = X[:, feature_cols]
    model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_cv = cross_val_predict(model, X_sel, y, cv=kf)
    r2   = float(r2_score(y, y_cv))
    rmse = float(mean_squared_error(y, y_cv) ** 0.5)
    model.fit(X_sel, y)

    # Additional metrics for MQG model
    from kawow.metrics import lin_ccc as _lin_ccc, nrmse as _nrmse, jeffreys_bf_corr as _jbf
    _r_cv_mqg = float(np.corrcoef(y, y_cv)[0, 1]) if len(y) > 1 else float("nan")
    _ccc_mqg  = _lin_ccc(y, y_cv)
    _nm_mqg   = _nrmse(y, y_cv)
    _bf_mqg   = _jbf(_r_cv_mqg, len(y))

    payload = {
        "target": target,
        "method": "mqg_random_forest",
        "n_train": int(len(y)),
        "fp_size": int(fp_size),
        "weighting_scheme": "bde",
        "normalize_length": True,
        "n_ratio_features": int(best_k),
        "selected_ratio_indices": selected_ratio_idx,
        "feature_cols": feature_cols,
        "r2_cv": round(r2, 4),
        "rmse_cv": round(rmse, 4),
        "ccc_cv": round(_ccc_mqg, 4),
        "nrmse_sd_cv": round(_nm_mqg["nrmse_sd"], 4),
        "nrmse_range_cv": round(_nm_mqg["nrmse_range"], 4),
        "bf10_log10_cv": round(_bf_mqg["log10_bf10"], 2),
        "r_ci95_cv": [round(_bf_mqg["ci95_lo"], 3), round(_bf_mqg["ci95_hi"], 3)],
        "model": model,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(
        f"  [{target} MQG] n={len(y)}  R\u00b2={r2:.4f}  RMSE={rmse:.4f}  "
        f"CCC={_ccc_mqg:.4f}  NRMSE\u03c3={_nm_mqg['nrmse_sd']:.4f}  "
        f"({len(feature_cols)} features total)  -> {out_path.name}"
    )
    return payload


def fit_mqg(
    sdf_logkow: str | Path,
    sdf_logkoa: str | Path,
    logkow_prop: str = "logP",
    logkoa_prop: str = "logKoa",
    fp_size: int = 64,
) -> None:
    """Fit MQG-based models on S01/S02 and save to kawow/data/*.pkl."""
    if not _MQG_AVAILABLE:
        raise ImportError(
            "MQG model requires the molecular_quantum_graph package (import name: mqg)."
        )

    print("Fitting MQG logKow model (RandomForest) ...")
    X_kow, y_kow, _ = _build_Xy_mqg(sdf_logkow, logkow_prop, fp_size=fp_size)
    _fit_mqg_and_save(X_kow, y_kow, _MODEL_MQG_LOGKOW, "logKow", fp_size=fp_size)

    print("Fitting MQG logKoa model (RandomForest) ...")
    X_koa, y_koa, _ = _build_Xy_mqg(sdf_logkoa, logkoa_prop, fp_size=fp_size)
    _fit_mqg_and_save(X_koa, y_koa, _MODEL_MQG_LOGKOA, "logKoa", fp_size=fp_size)
    print("Done.")


def _load_pickle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run kawow.fit_mqg(sdf_logkow, sdf_logkoa) first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Naef group-count feature helpers (used by ensemble models) ────────────────

def _compile_naef_patterns(param_csv_path: str | Path) -> list:
    """Compile a Naef parameter CSV into a list of (smarts_mol, pi, fnc) tuples.

    Skips the 'Const' row.  The returned list has one entry per non-Const row.
    SMARTS patterns are compiled once; the list can then be reused across many
    molecules without re-parsing.
    """
    import pandas as pd
    from . import smarts_model as _sm

    df = pd.read_csv(param_csv_path)
    patterns: list = []
    for _, row in df.iterrows():
        if row["Atom Type"] == "Const":
            continue
        smarts_str = row["SMARTS"]
        pi = int(row["pi"]) if not pd.isna(row.get("pi", float("nan"))) else None
        fnc_name = row["fnc"] if not pd.isna(row.get("fnc", float("nan"))) else None

        smarts_mol = None
        if not pd.isna(smarts_str):
            smarts_mol = Chem.MolFromSmarts(smarts_str)

        fnc = None
        if fnc_name:
            fnc = getattr(_sm, fnc_name, None)

        patterns.append((smarts_mol, pi, fnc))
    return patterns


def _compute_naef_group_counts(mol, compiled_patterns: list) -> np.ndarray:
    """Return a float32 count vector for all Naef SMARTS rows (non-Const).

    Parameters
    ----------
    mol : RDKit Mol
    compiled_patterns : list of (smarts_mol | None, pi | None, fnc | None)
        Pre-compiled pattern list from :func:`_compile_naef_patterns`.

    Returns
    -------
    np.ndarray of shape ``(len(compiled_patterns),)``, dtype float32.
    """
    from .smarts_model import count_conjugated_neighbor_moieties

    counts = np.zeros(len(compiled_patterns), dtype=np.float32)
    for i, (smarts_mol, pi, fnc) in enumerate(compiled_patterns):
        if smarts_mol is not None:
            matches = mol.GetSubstructMatches(smarts_mol)
            if matches:
                if pi is not None:
                    c = 0
                    for match in matches:
                        n_moi, _ = count_conjugated_neighbor_moieties(mol, match[0])
                        if n_moi == pi:
                            c += 1
                    counts[i] = float(c)
                else:
                    counts[i] = float(len(matches))
        elif fnc is not None:
            counts[i] = float(fnc(mol))
    return counts


def _build_ensemble_feature_vector(
    mol,
    pkl: dict,
    naef_patterns=None,
) -> np.ndarray | None:
    """Build the combined feature vector for an ensemble Ridge model.

    The ``pkl`` dict must contain keys ``ensemble_type``, ``mqg_feature_cols``,
    and optionally ``fp_size``.  ``naef_patterns`` must be supplied when
    ``ensemble_type`` contains 'naef'.

    Returns ``None`` if any required feature computation fails.
    """
    ensemble_type: str = pkl["ensemble_type"]
    fp_size: int = int(pkl.get("fp_size", 64))
    mqg_feature_cols: list = pkl["mqg_feature_cols"]

    # MQG eigenvalue + ratio features
    mqg_full = _compute_mqg_features_with_ratios(mol, fp_size=fp_size)
    if mqg_full is None:
        return None
    x_mqg = mqg_full[mqg_feature_cols]

    parts: list[np.ndarray] = []
    if "naef" in ensemble_type:
        if naef_patterns is None:
            return None
        parts.append(_compute_naef_group_counts(mol, naef_patterns))
    if "crippen" in ensemble_type:
        x_crippen = compute_features(mol)
        if x_crippen is None:
            return None
        parts.append(x_crippen.astype(np.float32))
    parts.append(x_mqg)
    return np.concatenate(parts)


class PartitionCalculator:
    """
    Predict logKow, logKoa, logKaw from molecular structure using the
    Naef group-additivity method (re-fitted on S01/S02 SDF training data).

    Parameters
    ----------
    model : str, optional
        Must be ``'crippen'`` (default). ``'kawow'`` is accepted as a
        deprecated alias.

    Usage
    -----
    >>> from kawow import PartitionCalculator
    >>> calc = PartitionCalculator()                    # Ridge (default)
    >>> calc.predict("CCCCO")            # 1-butanol
    {'logKow': 0.88, 'logKoa': 4.12, 'logKaw': -3.24, 'status': 'ok'}
    >>> calc.predict_batch(["CCCCO", "c1ccccc1"])
    [{'smiles': ..., 'logKow': ..., ...}, ...]
    """

    def __init__(self, model: str = "crippen") -> None:
        resolved = _normalize_model_key(model)
        if resolved not in ("crippen",):
            raise ValueError(
                f"Unknown model {model!r}. Valid option: ['crippen'] (deprecated alias: 'kawow')"
            )
        self._model_name = "crippen"
        kow_path, koa_path = _MODEL_FILES["crippen"]
        self._kow = _load_json(kow_path)
        self._koa = _load_json(koa_path)

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
            "model": self._model_name,
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


class MQGPartitionCalculator:
    """Predict logKow/logKoa/logKaw from MQG eigenvalue fingerprints.

    Transparently handles both the original RandomForest pkl and any
    ensemble Ridge pkl placed at the same paths (detected via the
    ``method`` field in the pkl dict).
    """

    def __init__(self) -> None:
        if not _MQG_AVAILABLE:
            raise ImportError(
                "MQGPartitionCalculator requires the molecular_quantum_graph package (import name: mqg)."
            )
        kow_path, koa_path = _MODEL_FILES_MQG["mqg"]
        self._kow = _load_pickle(kow_path)
        self._koa = _load_pickle(koa_path)

        self._is_ensemble = self._kow.get("method") == "ridge_ensemble"
        if self._is_ensemble:
            naef_kow = self._kow.get("naef_param_file")
            naef_koa = self._koa.get("naef_param_file")
            self._naef_patterns_kow = _compile_naef_patterns(naef_kow) if naef_kow else None
            self._naef_patterns_koa = _compile_naef_patterns(naef_koa) if naef_koa else None
        else:
            self._naef_patterns_kow = None
            self._naef_patterns_koa = None

    def _predict_mol(self, mol) -> dict:
        if self._is_ensemble:
            x_kow = _build_ensemble_feature_vector(mol, self._kow, self._naef_patterns_kow)
            x_koa = _build_ensemble_feature_vector(mol, self._koa, self._naef_patterns_koa)
            if x_kow is None or x_koa is None:
                return {"status": "error", "error": "ensemble feature computation failed"}
            logKow = float(self._kow["model"].predict(x_kow.reshape(1, -1))[0])
            logKoa = float(self._koa["model"].predict(x_koa.reshape(1, -1))[0])
        else:
            fp_size = int(self._kow.get("fp_size", 64))
            feat = _compute_mqg_features_with_ratios(mol, fp_size=fp_size)
            if feat is None:
                return {"status": "error", "error": "molecule has no valid MQG spectrum"}
            feature_cols_kow = self._kow.get("feature_cols")
            feature_cols_koa = self._koa.get("feature_cols")
            x_kow = feat[feature_cols_kow].reshape(1, -1) if feature_cols_kow is not None else feat.reshape(1, -1)
            x_koa = feat[feature_cols_koa].reshape(1, -1) if feature_cols_koa is not None else feat.reshape(1, -1)
            logKow = float(self._kow["model"].predict(x_kow)[0])
            logKoa = float(self._koa["model"].predict(x_koa)[0])

        logKaw = logKow - logKoa
        return {
            "logKow": round(logKow, 3),
            "logKoa": round(logKoa, 3),
            "logKaw": round(logKaw, 3),
            "status": "ok",
        }

    def predict(self, inp, fmt: str = "auto") -> dict | list[dict]:
        from .io import parse_input
        pairs = parse_input(inp, fmt=fmt)
        results = []
        for mol, name in pairs:
            r = self._predict_mol(mol)
            r["name"] = name
            results.append(r)
        if len(results) == 1:
            return results[0]
        return results

    def predict_batch(self, smiles_list: list[str]) -> list[dict]:
        if Chem is None:
            raise ImportError("RDKit is required for MQGPartitionCalculator.")
        results = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is None:
                results.append({"smiles": smi, "status": "error", "error": "invalid SMILES"})
                continue
            r = self._predict_mol(mol)
            r["smiles"] = smi
            results.append(r)
        return results

    @property
    def model_info(self) -> dict:
        return {
            "model": "mqg",
            "logKow": {k: v for k, v in self._kow.items() if k != "model"},
            "logKoa": {k: v for k, v in self._koa.items() if k != "model"},
        }

    @staticmethod
    def fit(
        sdf_logkow: str | Path,
        sdf_logkoa: str | Path,
        logkow_prop: str = "logP",
        logkoa_prop: str = "logKoa",
        fp_size: int = 64,
    ) -> None:
        fit_mqg(
            sdf_logkow=sdf_logkow,
            sdf_logkoa=sdf_logkoa,
            logkow_prop=logkow_prop,
            logkoa_prop=logkoa_prop,
            fp_size=fp_size,
        )


class EnsemblePartitionCalculator:
    """Predict logKow/logKoa/logKaw using an ensemble Ridge model.

    Combines MQG eigenvalue features with Naef group counts and/or Crippen
    (kawow) atom-type counts.  The ``ensemble_type`` must match one of the
    fitted pkl files in ``kawow/data/``.

    Parameters
    ----------
    ensemble_type : str
        One of ``"naef_mqg"``, ``"crippen_mqg"``, ``"naef_crippen_mqg"``.

    Usage
    -----
    >>> calc = EnsemblePartitionCalculator("crippen_mqg")
    >>> calc.predict("CCCCO")
    {'logKow': ..., 'logKoa': ..., 'logKaw': ..., 'status': 'ok'}
    """

    _VALID_TYPES = ("naef_mqg", "crippen_mqg", "naef_crippen_mqg")

    def __init__(self, ensemble_type: str) -> None:
        if ensemble_type not in self._VALID_TYPES:
            raise ValueError(
                f"Unknown ensemble_type {ensemble_type!r}. "
                f"Valid options: {self._VALID_TYPES}"
            )
        if not _MQG_AVAILABLE:
            raise ImportError(
                "EnsemblePartitionCalculator requires the molecular_quantum_graph package (import name: mqg)."
            )
        self._ensemble_type = ensemble_type
        kow_path = DATA_DIR / f"logkow_{ensemble_type}_model.pkl"
        koa_path = DATA_DIR / f"logkoa_{ensemble_type}_model.pkl"
        self._kow = _load_pickle(kow_path)
        self._koa = _load_pickle(koa_path)

        naef_kow = self._kow.get("naef_param_file")
        naef_koa = self._koa.get("naef_param_file")
        self._naef_patterns_kow = _compile_naef_patterns(naef_kow) if naef_kow else None
        self._naef_patterns_koa = _compile_naef_patterns(naef_koa) if naef_koa else None

    def _predict_mol(self, mol) -> dict:
        x_kow = _build_ensemble_feature_vector(mol, self._kow, self._naef_patterns_kow)
        x_koa = _build_ensemble_feature_vector(mol, self._koa, self._naef_patterns_koa)
        if x_kow is None or x_koa is None:
            return {"status": "error", "error": "feature computation failed"}
        logKow = float(self._kow["model"].predict(x_kow.reshape(1, -1))[0])
        logKoa = float(self._koa["model"].predict(x_koa.reshape(1, -1))[0])
        logKaw = logKow - logKoa
        return {
            "logKow": round(logKow, 3),
            "logKoa": round(logKoa, 3),
            "logKaw": round(logKaw, 3),
            "status": "ok",
        }

    def predict(self, inp, fmt: str = "auto") -> dict | list[dict]:
        from .io import parse_input
        pairs = parse_input(inp, fmt=fmt)
        results = []
        for mol, name in pairs:
            r = self._predict_mol(mol)
            r["name"] = name
            results.append(r)
        if len(results) == 1:
            return results[0]
        return results

    def predict_batch(self, smiles_list: list[str]) -> list[dict]:
        if Chem is None:
            raise ImportError("RDKit is required for EnsemblePartitionCalculator.")
        results = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is None:
                results.append({"smiles": smi, "status": "error", "error": "invalid SMILES"})
                continue
            r = self._predict_mol(mol)
            r["smiles"] = smi
            results.append(r)
        return results

    @property
    def model_info(self) -> dict:
        return {
            "model": f"ensemble_{self._ensemble_type}",
            "ensemble_type": self._ensemble_type,
            "logKow": {k: v for k, v in self._kow.items() if k != "model"},
            "logKoa": {k: v for k, v in self._koa.items() if k != "model"},
        }


class PFASGroupsPartitionCalculator:
    """Predict logKow/logKoa/logKaw using the PFASGroups 77-dim descriptor.

    Uses a Ridge regression pipeline (StandardScaler + RidgeCV) fitted on
    the S01/S02 Naef & Acree (2024) experimental datasets.  Feature
    extraction relies on :mod:`kawow.pfasgroups_features`.

    Parameters
    ----------
    variant : str
        ``"pfasgroups"`` — Ridge on 77-dim PFASGroups feature vector.
        ``"pfasgroups_crippen"`` — Ridge on PFASGroups (77) + Crippen (77)
        concatenated features (154-dim).
        ``"pfasgroups_naef"`` — Ridge on PFASGroups (77) + Naef group counts.
        ``"pfasgroups_naef_crippen"`` — Ridge on PFASGroups + Naef group counts + Crippen.

    Usage
    -----
    >>> calc = PFASGroupsPartitionCalculator("pfasgroups")
    >>> calc.predict("FC(F)(F)C(F)(F)F")
    {'logKow': ..., 'logKoa': ..., 'logKaw': ..., 'status': 'ok'}
    """

    _VALID_VARIANTS = (
        "pfasgroups",
        "pfasgroups_crippen",
        "pfasgroups_naef",
        "pfasgroups_naef_crippen",
    )

    def __init__(self, variant: str = "pfasgroups") -> None:
        if variant not in self._VALID_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. Valid options: {self._VALID_VARIANTS}"
            )
        from .pfasgroups_features import compute_pfasgroups_features as _cpf
        self._compute_pfasgroups_features = _cpf
        self._variant = variant
        self._use_crippen = variant in {"pfasgroups_crippen", "pfasgroups_naef_crippen"}
        self._use_naef = variant in {"pfasgroups_naef", "pfasgroups_naef_crippen"}

        self._naef_patterns_kow = None
        self._naef_patterns_koa = None
        if self._use_naef:
            self._naef_patterns_kow = _compile_naef_patterns(DATA_DIR / "naef2024_logkow_parameters.csv")
            self._naef_patterns_koa = _compile_naef_patterns(DATA_DIR / "naef2024_logkoa_parameters.csv")

        kow_path = DATA_DIR / f"logkow_{variant}_model.pkl"
        koa_path = DATA_DIR / f"logkoa_{variant}_model.pkl"
        self._kow = _load_pickle(kow_path)
        self._koa = _load_pickle(koa_path)

    def _feature_vector(self, mol) -> tuple:
        """Return (x_kow, x_koa) feature arrays or (None, None) on failure."""
        x_pg = self._compute_pfasgroups_features(mol)
        if x_pg is None:
            return None, None

        x_cr = None
        if self._use_crippen:
            x_cr = compute_features(mol)
            if x_cr is None:
                return None, None
            x_cr = x_cr.astype(np.float32)

        x_naef_kow = None
        x_naef_koa = None
        if self._use_naef:
            x_naef_kow = _compute_naef_group_counts(mol, self._naef_patterns_kow)
            x_naef_koa = _compute_naef_group_counts(mol, self._naef_patterns_koa)

        parts_kow = [x_pg]
        parts_koa = [x_pg]
        if self._use_naef:
            parts_kow.append(x_naef_kow)
            parts_koa.append(x_naef_koa)
        if self._use_crippen:
            parts_kow.append(x_cr)
            parts_koa.append(x_cr)

        x_kow = np.hstack(parts_kow).astype(np.float32)
        x_koa = np.hstack(parts_koa).astype(np.float32)
        return x_kow, x_koa

    def _predict_mol(self, mol) -> dict:
        x_kow, x_koa = self._feature_vector(mol)
        if x_kow is None:
            return {"status": "error", "error": "PFASGroups feature computation failed"}
        logKow = float(self._kow["model"].predict(x_kow.reshape(1, -1))[0])
        logKoa = float(self._koa["model"].predict(x_koa.reshape(1, -1))[0])
        logKaw = logKow - logKoa
        return {
            "logKow": round(logKow, 3),
            "logKoa": round(logKoa, 3),
            "logKaw": round(logKaw, 3),
            "status": "ok",
        }

    def predict(self, inp, fmt: str = "auto") -> dict | list[dict]:
        from .io import parse_input
        pairs = parse_input(inp, fmt=fmt)
        results = []
        for mol, name in pairs:
            r = self._predict_mol(mol)
            r["name"] = name
            results.append(r)
        if len(results) == 1:
            return results[0]
        return results

    def predict_batch(self, smiles_list: list[str]) -> list[dict]:
        if Chem is None:
            raise ImportError("RDKit is required for PFASGroupsPartitionCalculator.")
        results = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is None:
                results.append({"smiles": smi, "status": "error", "error": "invalid SMILES"})
                continue
            r = self._predict_mol(mol)
            r["smiles"] = smi
            results.append(r)
        return results

    @property
    def model_info(self) -> dict:
        return {
            "model": self._variant,
            "logKow": {k: v for k, v in self._kow.items() if k != "model"},
            "logKoa": {k: v for k, v in self._koa.items() if k != "model"},
        }


# ── Helper shared by the three advanced PFASGroups calculators ────────────────

def _make_pfasgroups_naef_crippen_features(
    mol,
    compute_pfasgroups_features,
    naef_patterns_kow,
    naef_patterns_koa,
) -> "tuple[np.ndarray | None, np.ndarray | None]":
    """Return (x_kow, x_koa) using the pfasgroups+naef+crippen feature set."""
    x_pg = compute_pfasgroups_features(mol)
    if x_pg is None:
        return None, None
    x_cr = compute_features(mol)
    if x_cr is None:
        return None, None
    x_cr = x_cr.astype(np.float32)
    x_naef_kow = _compute_naef_group_counts(mol, naef_patterns_kow)
    x_naef_koa = _compute_naef_group_counts(mol, naef_patterns_koa)
    return (
        np.hstack([x_pg, x_naef_kow, x_cr]).astype(np.float32),
        np.hstack([x_pg, x_naef_koa, x_cr]).astype(np.float32),
    )


# ── Random-Forest calculator ──────────────────────────────────────────────────

class PFASGroupsRFPartitionCalculator:
    """Predict logKow/logKoa/logKaw via a Random Forest on PFASGroups+Naef+Crippen features.

    Uses a RandomForestRegressor (300 trees, ``max_features=0.33``) inside a
    ``StandardScaler`` pipeline, fitted with 5-fold cross-validation on the
    S01/S02 Naef & Acree (2024) benchmark datasets.  Feature extraction is
    identical to the ``pfasgroups_naef_crippen`` Ridge model:
    PFASGroups (77-dim) + Naef group counts + Crippen atom types (91-dim).

    No additional dependencies beyond ``scikit-learn`` (already required by
    the base package) are needed.

    Usage
    -----
    >>> calc = PFASGroupsRFPartitionCalculator()
    >>> calc.predict("FC(F)(F)C(F)(F)F")
    {'logKow': ..., 'logKoa': ..., 'logKaw': ..., 'status': 'ok'}
    """

    def __init__(self) -> None:
        from .pfasgroups_features import compute_pfasgroups_features as _cpf
        self._cpf = _cpf
        self._naef_kow = _compile_naef_patterns(DATA_DIR / "naef2024_logkow_parameters.csv")
        self._naef_koa = _compile_naef_patterns(DATA_DIR / "naef2024_logkoa_parameters.csv")
        self._kow = _load_pickle(DATA_DIR / "logkow_pfasgroups_naef_crippen_rf_model.pkl")
        self._koa = _load_pickle(DATA_DIR / "logkoa_pfasgroups_naef_crippen_rf_model.pkl")

    def _feature_vector(self, mol):
        return _make_pfasgroups_naef_crippen_features(mol, self._cpf, self._naef_kow, self._naef_koa)

    def _predict_mol(self, mol) -> dict:
        x_kow, x_koa = self._feature_vector(mol)
        if x_kow is None:
            return {"status": "error", "error": "feature computation failed"}
        logKow = float(self._kow["model"].predict(x_kow.reshape(1, -1))[0])
        logKoa = float(self._koa["model"].predict(x_koa.reshape(1, -1))[0])
        return {
            "logKow": round(logKow, 3),
            "logKoa": round(logKoa, 3),
            "logKaw": round(logKow - logKoa, 3),
            "status": "ok",
        }

    def predict(self, inp, fmt: str = "auto") -> "dict | list[dict]":
        from .io import parse_input
        pairs = parse_input(inp, fmt=fmt)
        results = []
        for mol, name in pairs:
            r = self._predict_mol(mol)
            r["name"] = name
            results.append(r)
        return results[0] if len(results) == 1 else results

    def predict_batch(self, smiles_list: "list[str]") -> "list[dict]":
        if Chem is None:
            raise ImportError("RDKit is required.")
        results = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is None:
                results.append({"smiles": smi, "status": "error", "error": "invalid SMILES"})
                continue
            r = self._predict_mol(mol)
            r["smiles"] = smi
            results.append(r)
        return results

    @property
    def model_info(self) -> dict:
        return {
            "model": "pfasgroups_naef_crippen_rf",
            "logKow": {k: v for k, v in self._kow.items() if k != "model"},
            "logKoa": {k: v for k, v in self._koa.items() if k != "model"},
        }


# ── XGBoost calculator ────────────────────────────────────────────────────────

class PFASGroupsXGBPartitionCalculator:
    """Predict logKow/logKoa/logKaw via XGBoost on PFASGroups+Naef+Crippen features.

    Uses an ``XGBRegressor`` (gradient-boosted trees) with early stopping on a
    held-out validation split, fitted with 5-fold cross-validation on the
    S01/S02 Naef & Acree (2024) datasets.  The same PFASGroups+Naef+Crippen
    feature matrix as ``pfasgroups_naef_crippen`` is used.

    Requires ``xgboost>=2.0``.  Install with ``pip install kawow[ml]``.

    Usage
    -----
    >>> calc = PFASGroupsXGBPartitionCalculator()
    >>> calc.predict("FC(F)(F)C(F)(F)F")
    {'logKow': ..., 'logKoa': ..., 'logKaw': ..., 'status': 'ok'}
    """

    def __init__(self) -> None:
        try:
            import xgboost  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for PFASGroupsXGBPartitionCalculator. "
                "Install with: pip install kawow[ml]"
            ) from exc
        from .pfasgroups_features import compute_pfasgroups_features as _cpf
        self._cpf = _cpf
        self._naef_kow = _compile_naef_patterns(DATA_DIR / "naef2024_logkow_parameters.csv")
        self._naef_koa = _compile_naef_patterns(DATA_DIR / "naef2024_logkoa_parameters.csv")
        self._kow = _load_pickle(DATA_DIR / "logkow_pfasgroups_naef_crippen_xgb_model.pkl")
        self._koa = _load_pickle(DATA_DIR / "logkoa_pfasgroups_naef_crippen_xgb_model.pkl")

    def _feature_vector(self, mol):
        return _make_pfasgroups_naef_crippen_features(mol, self._cpf, self._naef_kow, self._naef_koa)

    def _predict_mol(self, mol) -> dict:
        x_kow, x_koa = self._feature_vector(mol)
        if x_kow is None:
            return {"status": "error", "error": "feature computation failed"}
        logKow = float(self._kow["model"].predict(x_kow.reshape(1, -1))[0])
        logKoa = float(self._koa["model"].predict(x_koa.reshape(1, -1))[0])
        return {
            "logKow": round(logKow, 3),
            "logKoa": round(logKoa, 3),
            "logKaw": round(logKow - logKoa, 3),
            "status": "ok",
        }

    def predict(self, inp, fmt: str = "auto") -> "dict | list[dict]":
        from .io import parse_input
        pairs = parse_input(inp, fmt=fmt)
        results = []
        for mol, name in pairs:
            r = self._predict_mol(mol)
            r["name"] = name
            results.append(r)
        return results[0] if len(results) == 1 else results

    def predict_batch(self, smiles_list: "list[str]") -> "list[dict]":
        if Chem is None:
            raise ImportError("RDKit is required.")
        results = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is None:
                results.append({"smiles": smi, "status": "error", "error": "invalid SMILES"})
                continue
            r = self._predict_mol(mol)
            r["smiles"] = smi
            results.append(r)
        return results

    @property
    def model_info(self) -> dict:
        return {
            "model": "pfasgroups_naef_crippen_xgb",
            "logKow": {k: v for k, v in self._kow.items() if k != "model"},
            "logKoa": {k: v for k, v in self._koa.items() if k != "model"},
        }


# ── Keras neural-network calculator ──────────────────────────────────────────

class PFASGroupsNNPartitionCalculator:
    """Predict logKow/logKoa/logKaw via a Keras MLP on PFASGroups+Naef+Crippen features.

    Uses a three-hidden-layer MLP ([256, 128, 64] units, BatchNormalization +
    Dropout(0.2) after each layer, linear output) trained with Adam and early
    stopping, fitted with 5-fold cross-validation on the S01/S02 Naef & Acree
    (2024) datasets.  The same PFASGroups+Naef+Crippen feature matrix as
    ``pfasgroups_naef_crippen`` is used.

    Requires ``keras>=3.0``.  Install with ``pip install kawow[ml]``.

    Usage
    -----
    >>> calc = PFASGroupsNNPartitionCalculator()
    >>> calc.predict("FC(F)(F)C(F)(F)F")
    {'logKow': ..., 'logKoa': ..., 'logKaw': ..., 'status': 'ok'}
    """

    def __init__(self) -> None:
        try:
            import keras as _keras
            self._keras = _keras
        except ImportError as exc:
            raise ImportError(
                "keras is required for PFASGroupsNNPartitionCalculator. "
                "Install with: pip install kawow[ml]"
            ) from exc
        from .pfasgroups_features import compute_pfasgroups_features as _cpf
        self._cpf = _cpf
        self._naef_kow = _compile_naef_patterns(DATA_DIR / "naef2024_logkow_parameters.csv")
        self._naef_koa = _compile_naef_patterns(DATA_DIR / "naef2024_logkoa_parameters.csv")
        self._kow_meta = _load_pickle(DATA_DIR / "logkow_pfasgroups_naef_crippen_nn_meta.pkl")
        self._koa_meta = _load_pickle(DATA_DIR / "logkoa_pfasgroups_naef_crippen_nn_meta.pkl")
        self._kow_nn = _keras.models.load_model(
            DATA_DIR / "logkow_pfasgroups_naef_crippen_nn_model.keras"
        )
        self._koa_nn = _keras.models.load_model(
            DATA_DIR / "logkoa_pfasgroups_naef_crippen_nn_model.keras"
        )

    def _feature_vector(self, mol):
        return _make_pfasgroups_naef_crippen_features(mol, self._cpf, self._naef_kow, self._naef_koa)

    def _predict_mol(self, mol) -> dict:
        x_kow, x_koa = self._feature_vector(mol)
        if x_kow is None:
            return {"status": "error", "error": "feature computation failed"}
        x_kow_s = self._kow_meta["scaler"].transform(x_kow.reshape(1, -1))
        x_koa_s = self._koa_meta["scaler"].transform(x_koa.reshape(1, -1))
        logKow = float(self._kow_nn.predict(x_kow_s, verbose=0)[0, 0])
        logKoa = float(self._koa_nn.predict(x_koa_s, verbose=0)[0, 0])
        return {
            "logKow": round(logKow, 3),
            "logKoa": round(logKoa, 3),
            "logKaw": round(logKow - logKoa, 3),
            "status": "ok",
        }

    def predict(self, inp, fmt: str = "auto") -> "dict | list[dict]":
        from .io import parse_input
        pairs = parse_input(inp, fmt=fmt)
        results = []
        for mol, name in pairs:
            r = self._predict_mol(mol)
            r["name"] = name
            results.append(r)
        return results[0] if len(results) == 1 else results

    def predict_batch(self, smiles_list: "list[str]") -> "list[dict]":
        if Chem is None:
            raise ImportError("RDKit is required.")
        results = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is None:
                results.append({"smiles": smi, "status": "error", "error": "invalid SMILES"})
                continue
            r = self._predict_mol(mol)
            r["smiles"] = smi
            results.append(r)
        return results

    @property
    def model_info(self) -> dict:
        return {
            "model": "pfasgroups_naef_crippen_nn",
            "logKow": {k: v for k, v in self._kow_meta.items() if k != "scaler"},
            "logKoa": {k: v for k, v in self._koa_meta.items() if k != "scaler"},
        }
