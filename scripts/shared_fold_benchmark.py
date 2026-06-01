from __future__ import annotations

import argparse
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kawow.features import compute_features  # noqa: E402
from kawow.metrics import (  # noqa: E402
    compare_correlations_bf,
    jeffreys_bf_corr,
    lin_ccc,
    nrmse,
    y_randomization_test,
)
from kawow.model import (  # noqa: E402
    _compile_naef_patterns,
    _compute_mqg_features_with_ratios,
    _compute_naef_group_counts,
    _load_pickle,
)
from kawow.smarts_model import NaefAcreePartitionCalculator  # noqa: E402
from kawow.pfasgroups_features import compute_pfasgroups_features  # noqa: E402
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
    X_pfasgroups: np.ndarray
    X_naef: np.ndarray


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
    max_samples: int | None = None,
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
    X_pfasgroups = []
    X_naef = []

    for i, (mol, _name, value) in enumerate(rows):
        x_crippen = compute_features(mol)
        x_mqg_full = _compute_mqg_features_with_ratios(mol, fp_size=int(pure_mqg_pkl.get("fp_size", 64)))
        x_smarts = _safe_smarts_predict(smarts_calc, mol, endpoint)
        x_mixed = _feature_vector(mol, mixed_specs)
        x_naef = _compute_naef_group_counts(mol, naef_patterns)
        x_pg = compute_pfasgroups_features(mol)

        if x_crippen is None or x_mqg_full is None or not np.isfinite(x_smarts) or x_mixed is None or x_pg is None:
            continue

        if (i + 1) % 500 == 0:
            print(f"  [{endpoint}] feature extraction: {i + 1}/{len(rows)}", flush=True)

        ys.append(value)
        X_kawow.append(x_crippen.astype(np.float32))
        X_mixed.append(x_mixed.astype(np.float32))
        X_mqg.append(x_mqg_full[pure_mqg_pkl["feature_cols"]].astype(np.float32))
        X_winner.append(np.concatenate([x_naef, x_crippen.astype(np.float32), x_mqg_full[winner_pkl["mqg_feature_cols"]].astype(np.float32)]))
        pred_smarts.append(x_smarts)
        X_pfasgroups.append(x_pg)
        X_naef.append(x_naef)

        if max_samples is not None and len(ys) >= int(max_samples):
            break

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
        X_pfasgroups=np.array(X_pfasgroups, dtype=np.float32),
        X_naef=np.array(X_naef, dtype=np.float32),
    )


def _fit_predict_kawow(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
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
    residuals = y_train - model.predict(X_train)
    sigma = residuals.std()
    mask = np.abs(residuals) <= 3 * sigma
    if int((~mask).sum()) > 0:
        model = make_pipeline(
            StandardScaler(with_mean=False),
            Ridge(alpha=float(best_alpha), fit_intercept=True, solver="sag", max_iter=8000, tol=1e-4, random_state=42),
        )
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


def _fit_predict_rf(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Random-forest regressor in a StandardScaler pipeline."""
    pipe = make_pipeline(
        StandardScaler(with_mean=False),
        RandomForestRegressor(
            n_estimators=300,
            max_features=0.33,
            random_state=42,
            n_jobs=-1,
        ),
    )
    pipe.fit(X_train, y_train)
    return pipe.predict(X_test)


def _fit_predict_nn(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
) -> np.ndarray:
    """Keras MLP [256,128,64] with BatchNorm + Dropout, early stopping."""
    try:
        import keras
    except ImportError:
        return np.full(len(X_test), np.nan, dtype=np.float64)

    scaler = StandardScaler(with_mean=False)
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    idx_fit, idx_val = train_test_split(
        np.arange(len(y_train)), test_size=0.15, random_state=42
    )
    inputs = keras.Input(shape=(X_tr_s.shape[1],))
    x = inputs
    for units in (256, 128, 64):
        x = keras.layers.Dense(units, activation="relu", kernel_initializer="he_normal")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1, activation="linear")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    model.fit(
        X_tr_s[idx_fit], y_train[idx_fit],
        validation_data=(X_tr_s[idx_val], y_train[idx_val]),
        epochs=500, batch_size=64, callbacks=[cb], verbose=0,
    )
    return model.predict(X_te_s, verbose=0).ravel()


def _fit_predict_xgb(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """XGBoost regressor with early stopping on a held-out validation split."""
    scaler = StandardScaler(with_mean=False)
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    # 20% of training data used for early stopping
    idx_fit, idx_val = train_test_split(
        np.arange(len(y_train)), test_size=0.2, random_state=42, shuffle=True
    )
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )
    model.fit(
        X_tr_s[idx_fit], y_train[idx_fit],
        eval_set=[(X_tr_s[idx_val], y_train[idx_val])],
        verbose=False,
    )
    return model.predict(X_te_s)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    r2 = float(r2_score(y_true, y_pred))
    rmse_val = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    y_bin = (y_true >= threshold).astype(int)
    auc = float(roc_auc_score(y_bin, y_pred)) if len(np.unique(y_bin)) > 1 else float("nan")
    r_corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    _bf = jeffreys_bf_corr(r_corr, len(y_true))
    _nm = nrmse(y_true, y_pred)
    return {
        "r2":          r2,
        "rmse":        rmse_val,
        "auc":         auc,
        "ccc":         float(lin_ccc(y_true, y_pred)),
        "nrmse_sd":    _nm["nrmse_sd"],
        "nrmse_range": _nm["nrmse_range"],
        "bf10_log10":  _bf["log10_bf10"],
    }


def _benchmark_endpoint(
    data: EndpointData,
    do_y_rand: bool = False,
    y_rand_permutations: int = 1000,
) -> tuple[list[dict[str, float | str | int]], list[dict], list[dict]]:
    """Returns (rows, pairwise_bf_rows, y_rand_rows)."""
    outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_pg_mixed = np.hstack([data.X_pfasgroups, data.X_kawow])
    X_pg_naef = np.hstack([data.X_pfasgroups, data.X_naef])
    X_pg_naef_mixed = np.hstack([data.X_pfasgroups, data.X_naef, data.X_kawow])
    X_pg_naef_mqg = np.hstack([data.X_pfasgroups, data.X_naef, data.X_kawow, data.X_mqg])
    preds = {
        "kawow": np.full(len(data.y), np.nan, dtype=np.float64),
        "smarts": data.pred_smarts.copy(),
        "smarts_mixed": np.full(len(data.y), np.nan, dtype=np.float64),
        "mqg": np.full(len(data.y), np.nan, dtype=np.float64),
        "naef_crippen_mqg": np.full(len(data.y), np.nan, dtype=np.float64),
        "pfasgroups": np.full(len(data.y), np.nan, dtype=np.float64),
        "pfasgroups_mixed": np.full(len(data.y), np.nan, dtype=np.float64),
        "pfasgroups_naef": np.full(len(data.y), np.nan, dtype=np.float64),
        "pfasgroups_naef_mixed": np.full(len(data.y), np.nan, dtype=np.float64),
        "xgb_pfasgroups": np.full(len(data.y), np.nan, dtype=np.float64),
        "pfasgroups_naef_mixed_xgb": np.full(len(data.y), np.nan, dtype=np.float64),
        "xgb_pfasgroups_naef_mqg": np.full(len(data.y), np.nan, dtype=np.float64),
        "pfasgroups_naef_mixed_rf": np.full(len(data.y), np.nan, dtype=np.float64),
        "pfasgroups_naef_mixed_nn": np.full(len(data.y), np.nan, dtype=np.float64),
    }

    # Store per-fold X matrices for y-randomisation (only models with X)
    X_by_model = {
        "kawow": data.X_kawow,
        "smarts_mixed": data.X_mixed,
        "mqg": data.X_mqg,
        "naef_crippen_mqg": data.X_winner,
        "pfasgroups": data.X_pfasgroups,
        "pfasgroups_mixed": X_pg_mixed,
        "pfasgroups_naef": X_pg_naef,
        "pfasgroups_naef_mixed": X_pg_naef_mixed,
        "xgb_pfasgroups": data.X_pfasgroups,
        "pfasgroups_naef_mixed_xgb": X_pg_naef_mixed,
        "xgb_pfasgroups_naef_mqg": X_pg_naef_mqg,
        "pfasgroups_naef_mixed_rf": X_pg_naef_mixed,
        "pfasgroups_naef_mixed_nn": X_pg_naef_mixed,
    }

    for train_idx, test_idx in outer_kf.split(data.y):
        preds["kawow"][test_idx] = _fit_predict_kawow(data.X_kawow[train_idx], data.y[train_idx], data.X_kawow[test_idx])
        preds["smarts_mixed"][test_idx] = _fit_predict_mixed(data.X_mixed[train_idx], data.y[train_idx], data.X_mixed[test_idx])
        preds["mqg"][test_idx] = _fit_predict_mqg(data.X_mqg[train_idx], data.y[train_idx], data.X_mqg[test_idx])
        preds["naef_crippen_mqg"][test_idx] = _fit_predict_ensemble(data.X_winner[train_idx], data.y[train_idx], data.X_winner[test_idx])
        preds["pfasgroups"][test_idx] = _fit_predict_ensemble(data.X_pfasgroups[train_idx], data.y[train_idx], data.X_pfasgroups[test_idx])
        preds["pfasgroups_mixed"][test_idx] = _fit_predict_ensemble(X_pg_mixed[train_idx], data.y[train_idx], X_pg_mixed[test_idx])
        preds["pfasgroups_naef"][test_idx] = _fit_predict_ensemble(X_pg_naef[train_idx], data.y[train_idx], X_pg_naef[test_idx])
        preds["pfasgroups_naef_mixed"][test_idx] = _fit_predict_ensemble(X_pg_naef_mixed[train_idx], data.y[train_idx], X_pg_naef_mixed[test_idx])
        preds["xgb_pfasgroups"][test_idx] = _fit_predict_xgb(data.X_pfasgroups[train_idx], data.y[train_idx], data.X_pfasgroups[test_idx])
        preds["pfasgroups_naef_mixed_xgb"][test_idx] = _fit_predict_xgb(X_pg_naef_mixed[train_idx], data.y[train_idx], X_pg_naef_mixed[test_idx])
        preds["xgb_pfasgroups_naef_mqg"][test_idx] = _fit_predict_xgb(X_pg_naef_mqg[train_idx], data.y[train_idx], X_pg_naef_mqg[test_idx])
        preds["pfasgroups_naef_mixed_rf"][test_idx] = _fit_predict_rf(X_pg_naef_mixed[train_idx], data.y[train_idx], X_pg_naef_mixed[test_idx])
        preds["pfasgroups_naef_mixed_nn"][test_idx] = _fit_predict_nn(X_pg_naef_mixed[train_idx], data.y[train_idx], X_pg_naef_mixed[test_idx])

    # Per-model metrics
    rows: list[dict[str, float | str | int]] = []
    for model_name, y_pred in preds.items():
        metrics = _compute_metrics(data.y, y_pred, data.threshold)
        rows.append({
            "endpoint":      data.endpoint,
            "model":         model_name,
            "n":             int(len(data.y)),
            "r2_cv":         round(metrics["r2"],          4),
            "rmse_cv":       round(metrics["rmse"],        4),
            "ccc_cv":        round(metrics["ccc"],         4),
            "nrmse_sd_cv":   round(metrics["nrmse_sd"],    4),
            "nrmse_range_cv":round(metrics["nrmse_range"], 4),
            "bf10_log10_cv": round(metrics["bf10_log10"],  2) if np.isfinite(metrics["bf10_log10"]) else np.nan,
            "auc_cv":        round(metrics["auc"],         4) if np.isfinite(metrics["auc"]) else np.nan,
            "raw_n":         int(data.raw_n),
            "common_n":      int(data.common_n),
            "threshold":     data.threshold,
        })

    # Pairwise Steiger BF rows
    model_names = list(preds.keys())
    corrs = {}
    for mn, yp in preds.items():
        fin = np.isfinite(yp) & np.isfinite(data.y)
        if fin.sum() > 1:
            corrs[mn] = float(np.corrcoef(data.y[fin], yp[fin])[0, 1])
        else:
            corrs[mn] = float("nan")

    pairwise_rows: list[dict] = []
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i + 1:]:
            r12 = corrs.get(m1, float("nan"))
            r13 = corrs.get(m2, float("nan"))
            # r23: correlation between the two model predictions
            p1, p2 = preds[m1], preds[m2]
            fin23 = np.isfinite(p1) & np.isfinite(p2)
            r23 = float(np.corrcoef(p1[fin23], p2[fin23])[0, 1]) if fin23.sum() > 1 else float("nan")
            bf_res = compare_correlations_bf(r12, r13, r23, int(len(data.y)))
            pairwise_rows.append({
                "endpoint": data.endpoint,
                "model_1":  m1,
                "model_2":  m2,
                "r_model1": round(r12, 4),
                "r_model2": round(r13, 4),
                "r_inter":  round(r23, 4),
                "t":        round(bf_res["t"], 3) if np.isfinite(bf_res["t"]) else np.nan,
                "log10_bf": round(bf_res["log10_bf"], 2) if np.isfinite(bf_res["log10_bf"]) else np.nan,
                "interpretation": bf_res["interpretation"],
            })

    # Y-randomisation rows (only for models with X matrices)
    y_rand_rows: list[dict] = []
    if do_y_rand:
        for mn, X_mat in X_by_model.items():
            print(f"  Y-randomisation: {data.endpoint} × {mn} ({y_rand_permutations} permutations) …", flush=True)
            res = y_randomization_test(X_mat, data.y, n_permutations=y_rand_permutations, n_splits=5)
            y_rand_rows.append({
                "endpoint":      data.endpoint,
                "model":         mn,
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()},
            })

    return rows, pairwise_rows, y_rand_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared-fold benchmark for kawow models.")
    parser.add_argument(
        "--y-randomization", action="store_true",
        help="Run Y-randomisation test (1000 permutations) for all trainable models. "
             "Saves y_randomization.csv to the output directory.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on valid molecules per endpoint for quick smoke tests.",
    )
    parser.add_argument(
        "--y-rand-permutations",
        type=int,
        default=1000,
        help="Permutation count for y-randomisation; lower values are useful for smoke tests.",
    )
    args = parser.parse_args()

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
        max_samples=args.max_samples,
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
        max_samples=args.max_samples,
    )

    kow_rows, kow_pairs, kow_yrand = _benchmark_endpoint(
        kow,
        do_y_rand=args.y_randomization,
        y_rand_permutations=args.y_rand_permutations,
    )
    koa_rows, koa_pairs, koa_yrand = _benchmark_endpoint(
        koa,
        do_y_rand=args.y_randomization,
        y_rand_permutations=args.y_rand_permutations,
    )

    all_rows = kow_rows + koa_rows
    df = pd.DataFrame(all_rows)
    out_path = OUT_DIR / "shared_fold_benchmark.csv"
    df.to_csv(out_path, index=False)
    print("Shared-fold benchmark on common valid molecules")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_path}")

    # Pairwise Steiger BF table
    all_pairs = kow_pairs + koa_pairs
    if all_pairs:
        df_pairs = pd.DataFrame(all_pairs)
        out_pairs = OUT_DIR / "pairwise_bfs.csv"
        df_pairs.to_csv(out_pairs, index=False)
        print(f"\nPairwise Bayes factors (Steiger 1980):")
        print(df_pairs[["endpoint", "model_1", "model_2", "log10_bf", "interpretation"]].to_string(index=False))
        print(f"\nSaved: {out_pairs}")

    # Y-randomisation results
    if args.y_randomization:
        all_yrand = kow_yrand + koa_yrand
        if all_yrand:
            df_yrand = pd.DataFrame(all_yrand)
            out_yrand = OUT_DIR / "y_randomization.csv"
            df_yrand.to_csv(out_yrand, index=False)
            print(f"\nY-randomisation test results:")
            print(df_yrand[["endpoint", "model", "observed_r2", "perm_r2_mean", "perm_r2_std", "p_value"]].to_string(index=False))
            print(f"\nSaved: {out_yrand}")


if __name__ == "__main__":
    main()