"""fit_pfasgroups_model.py — Fit PFASGroups Ridge models on full S01/S02 data.

Trains two models on all available molecules (no CV) and saves sklearn
pipeline objects to kawow/data/ so PFASGroupsPartitionCalculator can load them
at import time:

    logkow_pfasgroups_model.pkl
    logkoa_pfasgroups_model.pkl
    logkow_pfasgroups_mixed_model.pkl        (PFASGroups + Crippen features)
    logkoa_pfasgroups_mixed_model.pkl
    logkow_pfasgroups_naef_model.pkl         (PFASGroups + Naef group counts)
    logkoa_pfasgroups_naef_model.pkl
    logkow_pfasgroups_naef_mixed_model.pkl   (PFASGroups + Naef + Crippen)
    logkoa_pfasgroups_naef_mixed_model.pkl

Run from the kawow repo root:
    python scripts/fit_pfasgroups_model.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import SDMolSupplier
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kawow.features import compute_features  # noqa: E402
from kawow.pfasgroups_features import compute_pfasgroups_features  # noqa: E402
from kawow.model import _compile_naef_patterns, _compute_naef_group_counts  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DATA_DIR   = REPO_ROOT / "kawow" / "data"
TEST_DIR   = REPO_ROOT / "tests" / "test_data"

S01_SDF = TEST_DIR / "S01. Compounds List for logPow-Parameters Calculations.sdf"
S02_SDF = TEST_DIR / "S02. Compounds List for logKoa-Parameters Calculations.sdf"

ALPHA = 1.0


def _load_data(sdf_path: Path, value_prop: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Load molecules from an SDF, compute features, return (X_pg, X_crippen, y)."""
    suppl = SDMolSupplier(str(sdf_path), removeHs=True)
    X_pg_rows = []
    X_cr_rows = []
    y_vals = []
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

        if x_pg is None or x_cr is None:
            continue

        X_pg_rows.append(x_pg)
        X_cr_rows.append(x_cr.astype(np.float32))
        y_vals.append(value)

    print(f"  {sdf_path.name}: {len(y_vals)}/{raw_n} molecules retained after feature extraction")

    X_pg = np.array(X_pg_rows, dtype=np.float32)
    X_cr = np.array(X_cr_rows, dtype=np.float32)
    y    = np.array(y_vals,   dtype=np.float64)
    return X_pg, X_cr, y


def _load_naef_counts(sdf_path: Path, value_prop: str, naef_csv: Path) -> np.ndarray:
    """Load Naef group-count features for molecules retained by _load_data order."""
    patterns = _compile_naef_patterns(naef_csv)
    suppl = SDMolSupplier(str(sdf_path), removeHs=True)
    rows = []
    for mol in suppl:
        if mol is None or not mol.HasProp(value_prop):
            continue
        try:
            float(mol.GetProp(value_prop))
        except Exception:
            continue
        x_pg = compute_pfasgroups_features(mol)
        x_cr = compute_features(mol)
        if x_pg is None or x_cr is None:
            continue
        rows.append(_compute_naef_group_counts(mol, patterns))
    return np.array(rows, dtype=np.float32)


def _fit_ridge(X: np.ndarray, y: np.ndarray) -> object:
    pipe = make_pipeline(
        StandardScaler(with_mean=False),
        Ridge(alpha=ALPHA, fit_intercept=True, solver="sag", max_iter=8000, tol=1e-4, random_state=42),
    )
    pipe.fit(X, y)
    return pipe


def _save_model(pipeline, filepath: Path, extra: dict | None = None) -> None:
    payload = {"model": pipeline}
    if extra:
        payload.update(extra)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as fh:
        pickle.dump(payload, fh)
    print(f"  Saved: {filepath.name}")


def main() -> None:
    print("=== Fitting PFASGroups Ridge models on full S01/S02 data ===\n")

    # ── logKow ──────────────────────────────────────────────────────────────
    print("[logKow] Loading S01 …")
    X_pg_kow, X_cr_kow, y_kow = _load_data(S01_SDF, "logP")
    X_naef_kow = _load_naef_counts(S01_SDF, "logP", DATA_DIR / "naef2024_logkow_parameters.csv")
    X_mix_kow = np.hstack([X_pg_kow, X_cr_kow])
    X_pg_naef_kow = np.hstack([X_pg_kow, X_naef_kow])
    X_pg_naef_mix_kow = np.hstack([X_pg_kow, X_naef_kow, X_cr_kow])

    print("[logKow] Fitting pfasgroups …")
    pipe_pg_kow = _fit_ridge(X_pg_kow, y_kow)
    _save_model(pipe_pg_kow, DATA_DIR / "logkow_pfasgroups_model.pkl",
                extra={"endpoint": "logKow", "n_train": len(y_kow),
                       "feature_dim": X_pg_kow.shape[1]})

    print("[logKow] Fitting pfasgroups_mixed …")
    pipe_mix_kow = _fit_ridge(X_mix_kow, y_kow)
    _save_model(pipe_mix_kow, DATA_DIR / "logkow_pfasgroups_mixed_model.pkl",
                extra={"endpoint": "logKow", "n_train": len(y_kow),
                       "feature_dim": X_mix_kow.shape[1]})

    print("[logKow] Fitting pfasgroups_naef …")
    pipe_pg_naef_kow = _fit_ridge(X_pg_naef_kow, y_kow)
    _save_model(pipe_pg_naef_kow, DATA_DIR / "logkow_pfasgroups_naef_model.pkl",
                extra={"endpoint": "logKow", "n_train": len(y_kow),
                       "feature_dim": X_pg_naef_kow.shape[1]})

    print("[logKow] Fitting pfasgroups_naef_mixed …")
    pipe_pg_naef_mix_kow = _fit_ridge(X_pg_naef_mix_kow, y_kow)
    _save_model(pipe_pg_naef_mix_kow, DATA_DIR / "logkow_pfasgroups_naef_mixed_model.pkl",
                extra={"endpoint": "logKow", "n_train": len(y_kow),
                       "feature_dim": X_pg_naef_mix_kow.shape[1]})

    # ── logKoa ──────────────────────────────────────────────────────────────
    print("\n[logKoa] Loading S02 …")
    X_pg_koa, X_cr_koa, y_koa = _load_data(S02_SDF, "logKoa")
    X_naef_koa = _load_naef_counts(S02_SDF, "logKoa", DATA_DIR / "naef2024_logkoa_parameters.csv")
    X_mix_koa = np.hstack([X_pg_koa, X_cr_koa])
    X_pg_naef_koa = np.hstack([X_pg_koa, X_naef_koa])
    X_pg_naef_mix_koa = np.hstack([X_pg_koa, X_naef_koa, X_cr_koa])

    print("[logKoa] Fitting pfasgroups …")
    pipe_pg_koa = _fit_ridge(X_pg_koa, y_koa)
    _save_model(pipe_pg_koa, DATA_DIR / "logkoa_pfasgroups_model.pkl",
                extra={"endpoint": "logKoa", "n_train": len(y_koa),
                       "feature_dim": X_pg_koa.shape[1]})

    print("[logKoa] Fitting pfasgroups_mixed …")
    pipe_mix_koa = _fit_ridge(X_mix_koa, y_koa)
    _save_model(pipe_mix_koa, DATA_DIR / "logkoa_pfasgroups_mixed_model.pkl",
                extra={"endpoint": "logKoa", "n_train": len(y_koa),
                       "feature_dim": X_mix_koa.shape[1]})

    print("[logKoa] Fitting pfasgroups_naef …")
    pipe_pg_naef_koa = _fit_ridge(X_pg_naef_koa, y_koa)
    _save_model(pipe_pg_naef_koa, DATA_DIR / "logkoa_pfasgroups_naef_model.pkl",
                extra={"endpoint": "logKoa", "n_train": len(y_koa),
                       "feature_dim": X_pg_naef_koa.shape[1]})

    print("[logKoa] Fitting pfasgroups_naef_mixed …")
    pipe_pg_naef_mix_koa = _fit_ridge(X_pg_naef_mix_koa, y_koa)
    _save_model(pipe_pg_naef_mix_koa, DATA_DIR / "logkoa_pfasgroups_naef_mixed_model.pkl",
                extra={"endpoint": "logKoa", "n_train": len(y_koa),
                       "feature_dim": X_pg_naef_mix_koa.shape[1]})

    print("\nDone. Eight model pkl files written to kawow/data/.")


if __name__ == "__main__":
    main()
