from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDMolSupplier
from rdkit import RDLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kawow.features import compute_features  # noqa: E402
from kawow.model import (  # noqa: E402
    _compile_naef_patterns,
    _compute_mqg_features_with_ratios,
    _compute_naef_group_counts,
    _load_pickle,
)
from kawow.smarts_model import NaefAcreePartitionCalculator  # noqa: E402
from scripts.fit_mixed_naef_crippen import (  # noqa: E402
    _append_crippen_rows,
    _compile_specs,
    _feature_vector,
)

RDLogger.DisableLog("rdApp.*")

DATA_DIR = REPO_ROOT / "kawow" / "data"
TEST_DIR = REPO_ROOT / "tests" / "test_data"
OUT_DIR = REPO_ROOT / "tests" / "out"

S01_SDF = TEST_DIR / "S01. Compounds List for logPow-Parameters Calculations.sdf"
S02_SDF = TEST_DIR / "S02. Compounds List for logKoa-Parameters Calculations.sdf"

KOW_BASE = DATA_DIR / "naef2024_logkow_parameters.csv"
KOA_BASE = DATA_DIR / "naef2024_logkoa_parameters.csv"

PURE_MQG_KOW = DATA_DIR / "logkow_mqg_model_pure_backup.pkl"
PURE_MQG_KOA = DATA_DIR / "logkoa_mqg_model_pure_backup.pkl"
WINNER_KOW = DATA_DIR / "logkow_naef_crippen_mqg_model.pkl"
WINNER_KOA = DATA_DIR / "logkoa_naef_crippen_mqg_model.pkl"


@dataclass
class EndpointData:
    endpoint: str
    threshold: float
    raw_n: int
    common_n: int
    y: np.ndarray
    X_kawow: np.ndarray
    X_mixed: np.ndarray
    X_mqg: np.ndarray
    X_winner: np.ndarray
    pred_smarts: np.ndarray


def _load_rows(sdf_path: Path, value_prop: str) -> list[tuple[Chem.Mol, str, float]]:
    rows: list[tuple[Chem.Mol, str, float]] = []
    suppl = SDMolSupplier(str(sdf_path), removeHs=True)
    for mol in suppl:
        if mol is None or not mol.HasProp(value_prop):
            continue
        try:
            value = float(mol.GetProp(value_prop))
        except Exception:
            continue
        name = mol.GetProp("Alias name") if mol.HasProp("Alias name") else (mol.GetProp("_Name") if mol.HasProp("_Name") else "")
        rows.append((mol, name, value))
    return rows


def _safe_smarts_predict(calc: NaefAcreePartitionCalculator, mol: Chem.Mol, endpoint: str) -> float:
    try:
        out = calc.parse(mol)
        if isinstance(out, dict):
            hit = out.get(mol)
            if isinstance(hit, dict) and endpoint in hit:
                return float(hit[endpoint])
            if endpoint in out:
                return float(out[endpoint])
            for value in out.values():
                if isinstance(value, dict) and endpoint in value:
                    return float(value[endpoint])
        return float("nan")
    except Exception:
        return float("nan")


def _build_endpoint_data(
    sdf_path: Path,
    value_prop: str,
    endpoint: str,
    threshold: float,
    naef_csv: Path,
    pure_mqg_pkl_path: Path,
    winner_pkl_path: Path,
    mixed_base_csv: Path,
) -> EndpointData:
    rows = _load_rows(sdf_path, value_prop)
    raw_n = len(rows)

    pure_mqg_pkl = _load_pickle(pure_mqg_pkl_path)
    winner_pkl = _load_pickle(winner_pkl_path)

    naef_patterns = _compile_naef_patterns(naef_csv)
    mixed_df = pd.read_csv(mixed_base_csv)
    mixed_specs, _ = _compile_specs(_append_crippen_rows(mixed_df))

    smarts_calc = NaefAcreePartitionCalculator()

    ys = []
    X_kawow = []
    X_mixed = []
    X_mqg = []
    X_winner = []
    pred_smarts = []

    for mol, _name, value in rows:
        x_crippen = compute_features(mol)
        x_mqg_full = _compute_mqg_features_with_ratios(mol, fp_size=int(pure_mqg_pkl.get("fp_size", 64)))
        x_smarts = _safe_smarts_predict(smarts_calc, mol, endpoint)
        x_mixed = _feature_vector(mol, mixed_specs)
        x_naef = _compute_naef_group_counts(mol, naef_patterns)

        if x_crippen is None or x_mqg_full is None or not np.isfinite(x_smarts) or x_mixed is None:
            continue

        ys.append(value)
        X_kawow.append(x_crippen.astype(np.float32))
        X_mixed.append(x_mixed.astype(np.float32))
        X_mqg.append(x_mqg_full[pure_mqg_pkl["feature_cols"]].astype(np.float32))
        X_winner.append(np.concatenate([x_naef, x_crippen.astype(np.float32), x_mqg_full[winner_pkl["mqg_feature_cols"]].astype(np.float32)]))
        pred_smarts.append(x_smarts)

    return EndpointData(
        endpoint=endpoint,
        threshold=threshold,
        raw_n=raw_n,
        common_n=len(ys),
        y=np.array(ys, dtype=np.float64),
        X_kawow=np.array(X_kawow, dtype=np.float32),
        X_mixed=np.array(X_mixed, dtype=np.float32),
        X_mqg=np.array(X_mqg, dtype=np.float32),
        X_winner=np.array(X_winner, dtype=np.float32),
        pred_smarts=np.array(pred_smarts, dtype=np.float64),
    )


def _fit_predict_kawow(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    alphas = np.logspace(-3, 4, 50)

    def _make_pipeline():
        return make_pipeline(
            StandardScaler(with_mean=False),
            RidgeCV(alphas=alphas, cv=5, fit_intercept=True),
        )

    model = _make_pipeline()
    model.fit(X_train, y_train)
    residuals = y_train - model.predict(X_train)
    sigma = residuals.std()
    mask = np.abs(residuals) <= 3 * sigma
    if int((~mask).sum()) > 0:
        model = _make_pipeline()
        model.fit(X_train[mask], y_train[mask])
    return model.predict(X_test)


def _fit_predict_mixed(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    alphas = np.logspace(-3, 4, 30)
    best_alpha = None
    best_rmse = np.inf

    for alpha in alphas:
        fold_pred = np.zeros_like(y_train, dtype=np.float64)
        for inner_train, inner_test in inner_kf.split(X_train):
            model = make_pipeline(
                StandardScaler(with_mean=False),
                Ridge(alpha=float(alpha), fit_intercept=True, solver="sag", max_iter=8000, tol=1e-4, random_state=42),
            )
            model.fit(X_train[inner_train], y_train[inner_train])
            fold_pred[inner_test] = model.predict(X_train[inner_test])
        rmse = float(math.sqrt(mean_squared_error(y_train, fold_pred)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)

    model = make_pipeline(
        StandardScaler(with_mean=False),
        Ridge(alpha=float(best_alpha), fit_intercept=True, solver="sag", max_iter=8000, tol=1e-4, random_state=42),
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _fit_predict_mqg(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, fp_size: int = 64) -> np.ndarray:
    n_raw = fp_size
    n_ratios = fp_size - 1
    idx_fit, idx_tune = train_test_split(
        np.arange(len(y_train)), test_size=0.2, random_state=42, shuffle=True
    )
    X_tune = X_train[idx_tune]
    y_tune = y_train[idx_tune]
    X_fit = X_train[idx_fit]
    y_fit = y_train[idx_fit]

    ratio_block = X_tune[:, n_raw:]
    corrs = np.array([
        abs(spearmanr(ratio_block[:, i], y_tune).statistic) if len(y_tune) > 1 else 0.0
        for i in range(n_ratios)
    ], dtype=np.float64)
    corrs = np.nan_to_num(corrs, nan=0.0)
    sorted_by_corr = np.argsort(corrs)[::-1]

    k_candidates = [0, 5, 10, 15, 20, 30, 40, 50, n_ratios]
    best_k = 0
    best_cv_r2 = -np.inf
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

    feature_cols = list(range(n_raw)) + ([n_raw + int(j) for j in sorted_by_corr[:best_k]] if best_k > 0 else [])
    model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    )
    model.fit(X_train[:, feature_cols], y_train)
    return model.predict(X_test[:, feature_cols])


def _fit_predict_ensemble(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 4, 50), cv=5, fit_intercept=True),
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    y_bin = (y_true >= threshold).astype(int)
    auc = float(roc_auc_score(y_bin, y_pred)) if len(np.unique(y_bin)) > 1 else float("nan")
    return {"r2": r2, "rmse": rmse, "auc": auc}


def _benchmark_endpoint(data: EndpointData) -> list[dict[str, float | str | int]]:
    outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = {
        "kawow": np.full(len(data.y), np.nan, dtype=np.float64),
        "smarts": data.pred_smarts.copy(),
        "smarts_mixed": np.full(len(data.y), np.nan, dtype=np.float64),
        "mqg": np.full(len(data.y), np.nan, dtype=np.float64),
        "naef_crippen_mqg": np.full(len(data.y), np.nan, dtype=np.float64),
    }

    for train_idx, test_idx in outer_kf.split(data.y):
        preds["kawow"][test_idx] = _fit_predict_kawow(data.X_kawow[train_idx], data.y[train_idx], data.X_kawow[test_idx])
        preds["smarts_mixed"][test_idx] = _fit_predict_mixed(data.X_mixed[train_idx], data.y[train_idx], data.X_mixed[test_idx])
        preds["mqg"][test_idx] = _fit_predict_mqg(data.X_mqg[train_idx], data.y[train_idx], data.X_mqg[test_idx])
        preds["naef_crippen_mqg"][test_idx] = _fit_predict_ensemble(data.X_winner[train_idx], data.y[train_idx], data.X_winner[test_idx])

    rows: list[dict[str, float | str | int]] = []
    for model_name, y_pred in preds.items():
        metrics = _compute_metrics(data.y, y_pred, data.threshold)
        rows.append({
            "endpoint": data.endpoint,
            "model": model_name,
            "n": int(len(data.y)),
            "r2_cv": round(metrics["r2"], 4),
            "rmse_cv": round(metrics["rmse"], 4),
            "auc_cv": round(metrics["auc"], 4) if np.isfinite(metrics["auc"]) else np.nan,
            "raw_n": int(data.raw_n),
            "common_n": int(data.common_n),
            "threshold": data.threshold,
        })
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kow = _build_endpoint_data(
        sdf_path=S01_SDF,
        value_prop="logP",
        endpoint="logKow",
        threshold=5.0,
        naef_csv=KOW_BASE,
        pure_mqg_pkl_path=PURE_MQG_KOW,
        winner_pkl_path=WINNER_KOW,
        mixed_base_csv=KOW_BASE,
    )
    koa = _build_endpoint_data(
        sdf_path=S02_SDF,
        value_prop="logKoa",
        endpoint="logKoa",
        threshold=6.0,
        naef_csv=KOA_BASE,
        pure_mqg_pkl_path=PURE_MQG_KOA,
        winner_pkl_path=WINNER_KOA,
        mixed_base_csv=KOA_BASE,
    )

    rows = _benchmark_endpoint(kow) + _benchmark_endpoint(koa)
    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "shared_fold_benchmark.csv"
    df.to_csv(out_path, index=False)

    print("Shared-fold benchmark on common valid molecules")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()