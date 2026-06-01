"""fit_pfasgroups_advanced_models.py — Fit RF, XGB, and Keras-NN models.

All three models use the same feature matrix as ``pfasgroups_naef_mixed``:
  PFASGroups (77-dim) + Naef group counts + Crippen atom types (91-dim).

Produces 6 files in kawow/data/:
  logkow/logkoa_pfasgroups_naef_mixed_rf_model.pkl     — RandomForest pipeline
  logkow/logkoa_pfasgroups_naef_mixed_xgb_model.pkl    — XGBoost pipeline
  logkow/logkoa_pfasgroups_naef_mixed_nn_meta.pkl      — NN StandardScaler + metadata
  logkow/logkoa_pfasgroups_naef_mixed_nn_model.keras   — Keras MLP (native format)

Run from the kawow repo root:
    python scripts/fit_pfasgroups_advanced_models.py [--skip-nn]
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import SDMolSupplier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kawow.features import compute_features  # noqa: E402
from kawow.pfasgroups_features import compute_pfasgroups_features  # noqa: E402
from kawow.model import _compile_naef_patterns, _compute_naef_group_counts  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DATA_DIR = REPO_ROOT / "kawow" / "data"
TEST_DIR = REPO_ROOT / "tests" / "test_data"

S01_SDF = TEST_DIR / "S01. Compounds List for logPow-Parameters Calculations.sdf"
S02_SDF = TEST_DIR / "S02. Compounds List for logKoa-Parameters Calculations.sdf"

N_CV_FOLDS = 5
RF_N_ESTIMATORS = 300
RF_MAX_FEATURES = 0.33
XGB_N_ESTIMATORS = 1000
XGB_LR = 0.05
XGB_MAX_DEPTH = 6
NN_HIDDEN = (256, 128, 64)
NN_DROPOUT = 0.2
NN_EPOCHS = 500
NN_PATIENCE = 30
NN_BATCH = 64


# ── Data loading (re-uses pattern from fit_pfasgroups_model.py) ──────────────

def _load_data(
    sdf_path: Path,
    value_prop: str,
    naef_csv: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load full (PFASGroups + Naef + Crippen) feature matrix and targets.

    Returns
    -------
    X : ndarray, shape (n, d)
    y : ndarray, shape (n,)
    """
    patterns = _compile_naef_patterns(naef_csv)
    suppl = SDMolSupplier(str(sdf_path), removeHs=True)
    X_rows, y_vals = [], []
    raw_n = 0
    for mol in suppl:
        if mol is None or not mol.HasProp(value_prop):
            continue
        raw_n += 1
        try:
            value = float(mol.GetProp(value_prop))
        except Exception:
            continue
        x_pg = compute_pfasgroups_features(mol)
        x_cr = compute_features(mol)
        x_naef = _compute_naef_group_counts(mol, patterns)
        if x_pg is None or x_cr is None:
            continue
        X_rows.append(np.hstack([x_pg, x_naef, x_cr.astype(np.float32)]).astype(np.float32))
        y_vals.append(value)

    print(f"  {sdf_path.name}: {len(y_vals)}/{raw_n} molecules retained")
    return np.array(X_rows, dtype=np.float32), np.array(y_vals, dtype=np.float64)


# ── CV helpers ────────────────────────────────────────────────────────────────

def _cv_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    return {"r2_cv": round(r2, 4), "rmse_cv": round(rmse, 4)}


def _fit_cv_rf(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros_like(y, dtype=np.float64)
    for train_idx, test_idx in kf.split(X):
        pipe = make_pipeline(
            StandardScaler(with_mean=False),
            RandomForestRegressor(
                n_estimators=RF_N_ESTIMATORS,
                max_features=RF_MAX_FEATURES,
                random_state=42,
                n_jobs=-1,
            ),
        )
        pipe.fit(X[train_idx], y[train_idx])
        oof[test_idx] = pipe.predict(X[test_idx])
    return oof


def _fit_cv_xgb(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)
    scaler = StandardScaler(with_mean=False)
    X_s = scaler.fit_transform(X)
    oof = np.zeros_like(y, dtype=np.float64)
    for train_idx, test_idx in kf.split(X_s):
        idx_fit, idx_val = train_test_split(train_idx, test_size=0.2, random_state=42)
        model = XGBRegressor(
            n_estimators=XGB_N_ESTIMATORS,
            learning_rate=XGB_LR,
            max_depth=XGB_MAX_DEPTH,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=50,
            eval_metric="rmse",
        )
        model.fit(
            X_s[idx_fit], y[idx_fit],
            eval_set=[(X_s[idx_val], y[idx_val])],
            verbose=False,
        )
        oof[test_idx] = model.predict(X_s[test_idx])
    return oof


def _fit_cv_nn(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """5-fold CV for the Keras MLP; returns OOF predictions."""
    import keras
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros_like(y, dtype=np.float64)
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"    NN fold {fold_i + 1}/{N_CV_FOLDS} …", flush=True)
        scaler = StandardScaler(with_mean=False)
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        y_tr = y[train_idx]

        idx_fit, idx_val = train_test_split(
            np.arange(len(y_tr)), test_size=0.15, random_state=42
        )

        model = _build_keras_model(X_tr.shape[1])
        cb = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=NN_PATIENCE, restore_best_weights=True
        )
        model.fit(
            X_tr[idx_fit], y_tr[idx_fit],
            validation_data=(X_tr[idx_val], y_tr[idx_val]),
            epochs=NN_EPOCHS,
            batch_size=NN_BATCH,
            callbacks=[cb],
            verbose=0,
        )
        oof[test_idx] = model.predict(X_te, verbose=0).ravel()
    return oof


def _build_keras_model(n_features: int):
    """Return a compiled Keras MLP with 3 hidden layers."""
    import keras
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for units in NN_HIDDEN:
        x = keras.layers.Dense(units, activation="relu", kernel_initializer="he_normal")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(NN_DROPOUT)(x)
    outputs = keras.layers.Dense(1, activation="linear")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


# ── Full-data fitting and saving ──────────────────────────────────────────────

def _save_pkl(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    print(f"  Saved: {path.name}")


def _fit_and_save_rf(X: np.ndarray, y: np.ndarray, endpoint: str, path: Path) -> dict:
    """Fit RF on full data; return metrics dict."""
    oof = _fit_cv_rf(X, y)
    metrics = _cv_metrics(y, oof)
    print(f"  [{endpoint}] RF  CV: R²={metrics['r2_cv']:.4f}  RMSE={metrics['rmse_cv']:.4f}")
    pipe = make_pipeline(
        StandardScaler(with_mean=False),
        RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_features=RF_MAX_FEATURES,
            random_state=42,
            n_jobs=-1,
        ),
    )
    pipe.fit(X, y)
    _save_pkl(
        {"model": pipe, "endpoint": endpoint, "n_train": len(y),
         "feature_dim": X.shape[1], **metrics},
        path,
    )
    return metrics


def _fit_and_save_xgb(X: np.ndarray, y: np.ndarray, endpoint: str, path: Path) -> dict:
    """Fit XGB on full data; return metrics dict."""
    oof = _fit_cv_xgb(X, y)
    metrics = _cv_metrics(y, oof)
    print(f"  [{endpoint}] XGB CV: R²={metrics['r2_cv']:.4f}  RMSE={metrics['rmse_cv']:.4f}")
    scaler = StandardScaler(with_mean=False)
    X_s = scaler.fit_transform(X)
    idx_fit, idx_val = train_test_split(np.arange(len(y)), test_size=0.1, random_state=42)
    model = XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        learning_rate=XGB_LR,
        max_depth=XGB_MAX_DEPTH,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )
    model.fit(
        X_s[idx_fit], y[idx_fit],
        eval_set=[(X_s[idx_val], y[idx_val])],
        verbose=False,
    )
    # Wrap scaler + model in a sklearn Pipeline for consistent .predict(X_raw) API
    final_pipe = make_pipeline(scaler, model)
    _save_pkl(
        {"model": final_pipe, "endpoint": endpoint, "n_train": len(y),
         "feature_dim": X.shape[1],
         "best_iteration": int(model.best_iteration), **metrics},
        path,
    )
    return metrics


def _fit_and_save_nn(
    X: np.ndarray, y: np.ndarray, endpoint: str,
    meta_path: Path, model_path: Path,
) -> dict:
    """Fit Keras NN on full data; saves scaler pkl + .keras model."""
    import keras
    oof = _fit_cv_nn(X, y)
    metrics = _cv_metrics(y, oof)
    print(f"  [{endpoint}] NN  CV: R²={metrics['r2_cv']:.4f}  RMSE={metrics['rmse_cv']:.4f}")

    scaler = StandardScaler(with_mean=False)
    X_s = scaler.fit_transform(X)
    idx_fit, idx_val = train_test_split(np.arange(len(y)), test_size=0.1, random_state=42)
    model = _build_keras_model(X_s.shape[1])
    cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=NN_PATIENCE, restore_best_weights=True
    )
    model.fit(
        X_s[idx_fit], y[idx_fit],
        validation_data=(X_s[idx_val], y[idx_val]),
        epochs=NN_EPOCHS,
        batch_size=NN_BATCH,
        callbacks=[cb],
        verbose=1,
    )
    # Save scaler + metadata as pkl
    _save_pkl(
        {"scaler": scaler, "endpoint": endpoint, "n_train": len(y),
         "feature_dim": X.shape[1], "architecture": list(NN_HIDDEN),
         "dropout": NN_DROPOUT, **metrics},
        meta_path,
    )
    # Save Keras model in native .keras format
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"  Saved: {model_path.name}")
    return metrics


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit RF, XGB, and Keras-NN models on PFASGroups+Naef+Crippen features."
    )
    parser.add_argument("--skip-nn", action="store_true",
                        help="Skip the Keras neural-network models (no keras/tensorflow required).")
    args = parser.parse_args()

    print("=== Fitting advanced PFASGroups models (RF, XGB, NN) ===\n")

    # ── logKow (S01) ─────────────────────────────────────────────────────────
    print("[logKow] Loading S01 …")
    X_kow, y_kow = _load_data(S01_SDF, "logP", DATA_DIR / "naef2024_logkow_parameters.csv")

    print("[logKow] Fitting Random Forest …")
    m_rf_kow = _fit_and_save_rf(X_kow, y_kow, "logKow",
                                 DATA_DIR / "logkow_pfasgroups_naef_mixed_rf_model.pkl")

    print("[logKow] Fitting XGBoost …")
    m_xgb_kow = _fit_and_save_xgb(X_kow, y_kow, "logKow",
                                    DATA_DIR / "logkow_pfasgroups_naef_mixed_xgb_model.pkl")

    if not args.skip_nn:
        print("[logKow] Fitting Keras NN …")
        m_nn_kow = _fit_and_save_nn(
            X_kow, y_kow, "logKow",
            DATA_DIR / "logkow_pfasgroups_naef_mixed_nn_meta.pkl",
            DATA_DIR / "logkow_pfasgroups_naef_mixed_nn_model.keras",
        )
    else:
        m_nn_kow = {}

    # ── logKoa (S02) ─────────────────────────────────────────────────────────
    print("\n[logKoa] Loading S02 …")
    X_koa, y_koa = _load_data(S02_SDF, "logKoa", DATA_DIR / "naef2024_logkoa_parameters.csv")

    print("[logKoa] Fitting Random Forest …")
    m_rf_koa = _fit_and_save_rf(X_koa, y_koa, "logKoa",
                                 DATA_DIR / "logkoa_pfasgroups_naef_mixed_rf_model.pkl")

    print("[logKoa] Fitting XGBoost …")
    m_xgb_koa = _fit_and_save_xgb(X_koa, y_koa, "logKoa",
                                    DATA_DIR / "logkoa_pfasgroups_naef_mixed_xgb_model.pkl")

    if not args.skip_nn:
        print("[logKoa] Fitting Keras NN …")
        m_nn_koa = _fit_and_save_nn(
            X_koa, y_koa, "logKoa",
            DATA_DIR / "logkoa_pfasgroups_naef_mixed_nn_meta.pkl",
            DATA_DIR / "logkoa_pfasgroups_naef_mixed_nn_model.keras",
        )
    else:
        m_nn_koa = {}

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== 5-fold CV summary (PFASGroups+Naef+Crippen feature set) ===")
    print(f"{'Model':<30} {'logKow R²':>10} {'logKow RMSE':>12} {'logKoa R²':>10} {'logKoa RMSE':>12}")
    print("-" * 76)
    for label, mk, ma in [("RF",  m_rf_kow,  m_rf_koa),
                           ("XGB", m_xgb_kow, m_xgb_koa),
                           ("NN",  m_nn_kow,  m_nn_koa)]:
        r2k  = f"{mk.get('r2_cv',  float('nan')):.4f}" if mk else "skipped"
        rmk  = f"{mk.get('rmse_cv', float('nan')):.4f}" if mk else "skipped"
        r2a  = f"{ma.get('r2_cv',  float('nan')):.4f}" if ma else "skipped"
        rma  = f"{ma.get('rmse_cv', float('nan')):.4f}" if ma else "skipped"
        print(f"{'pfasgroups_naef_mixed_' + label.lower():<30} {r2k:>10} {rmk:>12} {r2a:>10} {rma:>12}")
    print()


if __name__ == "__main__":
    main()
