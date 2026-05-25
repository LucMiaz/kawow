"""Quick 5-fold CV for pfasgroups and pfasgroups_mixed models only.

Much faster than shared_fold_benchmark.py because it skips all other models
and their expensive feature computations (MQG, naef SMARTS, etc.).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import SDMolSupplier
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kawow.pfasgroups_features import compute_pfasgroups_features  # noqa: E402
from kawow.features import compute_features  # noqa: E402

TEST_DIR = REPO_ROOT / "tests" / "test_data"

S01_SDF = TEST_DIR / "S01. Compounds List for logPow-Parameters Calculations.sdf"
S02_SDF = TEST_DIR / "S02. Compounds List for logKoa-Parameters Calculations.sdf"


def load_endpoint(sdf_path: Path, prop: str, threshold: float):
    """Load molecules and compute pfasgroups features."""
    supplier = SDMolSupplier(str(sdf_path), removeHs=True)
    ys = []
    X_pg = []
    X_kawow = []
    total = 0
    for mol in supplier:
        if mol is None:
            continue
        if mol.GetNumAtoms() == 0:
            continue
        if not mol.HasProp(prop):
            continue
        try:
            y = float(mol.GetProp(prop))
        except (ValueError, KeyError):
            continue
        if not np.isfinite(y):
            continue

        x_pg = compute_pfasgroups_features(mol)
        if x_pg is None:
            continue

        x_cr = compute_features(mol)
        if x_cr is None:
            continue

        ys.append(y)
        X_pg.append(x_pg.astype(np.float32))
        X_kawow.append(x_cr.astype(np.float32))
        total += 1
        if total % 500 == 0:
            print(f"  [{prop}] {total} molecules processed", flush=True)

    print(f"  [{prop}] {total} molecules total", flush=True)
    return np.array(ys), np.array(X_pg), np.array(X_kawow)


def cv_metrics(y, X):
    """5-fold CV R² and RMSE."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in kf.split(y):
        pipe = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 4, 30), cv=5),
        )
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx] = pipe.predict(X[test_idx])
    fin = np.isfinite(preds)
    r2 = r2_score(y[fin], preds[fin])
    rmse = np.sqrt(mean_squared_error(y[fin], preds[fin]))
    return r2, rmse, int(fin.sum())


def main():
    # Find property names by checking first molecule
    prop_s01 = "logPow"
    prop_s02 = "logKoa"

    # Try to determine property name from S01 SDF
    supp = SDMolSupplier(str(S01_SDF), removeHs=True)
    for mol in supp:
        if mol is not None:
            props = list(mol.GetPropsAsDict().keys())
            for p in ["logPow", "logKow", "LogPow", "LogKow", "log_pow"]:
                if p in props:
                    prop_s01 = p
                    break
            break
    print(f"S01 property: {prop_s01}")

    supp2 = SDMolSupplier(str(S02_SDF), removeHs=True)
    for mol in supp2:
        if mol is not None:
            props = list(mol.GetPropsAsDict().keys())
            for p in ["logKoa", "LogKoa", "log_koa"]:
                if p in props:
                    prop_s02 = p
                    break
            break
    print(f"S02 property: {prop_s02}")

    print("\n--- logKow (S01) ---")
    y_kow, X_pg_kow, X_cr_kow = load_endpoint(S01_SDF, prop_s01, 5.0)
    X_pgm_kow = np.hstack([X_pg_kow, X_cr_kow])

    print("  CV pfasgroups...")
    r2_pg_kow, rmse_pg_kow, n_pg_kow = cv_metrics(y_kow, X_pg_kow)
    print("  CV pfasgroups_mixed...")
    r2_pgm_kow, rmse_pgm_kow, n_pgm_kow = cv_metrics(y_kow, X_pgm_kow)

    print("\n--- logKoa (S02) ---")
    y_koa, X_pg_koa, X_cr_koa = load_endpoint(S02_SDF, prop_s02, 6.0)
    X_pgm_koa = np.hstack([X_pg_koa, X_cr_koa])

    print("  CV pfasgroups...")
    r2_pg_koa, rmse_pg_koa, n_pg_koa = cv_metrics(y_koa, X_pg_koa)
    print("  CV pfasgroups_mixed...")
    r2_pgm_koa, rmse_pgm_koa, n_pgm_koa = cv_metrics(y_koa, X_pgm_koa)

    print("\n=== RESULTS ===")
    print(f"pfasgroups    logKow  R²={r2_pg_kow:.4f}  RMSE={rmse_pg_kow:.4f}  n={n_pg_kow}")
    print(f"pfasgroups    logKoa  R²={r2_pg_koa:.4f}  RMSE={rmse_pg_koa:.4f}  n={n_pg_koa}")
    print(f"pfasgroups_mixed logKow  R²={r2_pgm_kow:.4f}  RMSE={rmse_pgm_kow:.4f}  n={n_pgm_kow}")
    print(f"pfasgroups_mixed logKoa  R²={r2_pgm_koa:.4f}  RMSE={rmse_pgm_koa:.4f}  n={n_pgm_koa}")

    print("\n=== kawow.js MODEL_PERFORMANCE entries ===")
    print(f"{{ model: 'pfasgroups', metricType: 'R2', property: 'logKow', value: {r2_pg_kow:.4f} }},")
    print(f"{{ model: 'pfasgroups', metricType: 'R2', property: 'logKoa', value: {r2_pg_koa:.4f} }},")
    print(f"{{ model: 'pfasgroups_mixed', metricType: 'R2', property: 'logKow', value: {r2_pgm_kow:.4f} }},")
    print(f"{{ model: 'pfasgroups_mixed', metricType: 'R2', property: 'logKoa', value: {r2_pgm_koa:.4f} }},")
    print(f"{{ model: 'pfasgroups', metricType: 'RMSE', property: 'logKow', value: {rmse_pg_kow:.4f} }},")
    print(f"{{ model: 'pfasgroups', metricType: 'RMSE', property: 'logKoa', value: {rmse_pg_koa:.4f} }},")
    print(f"{{ model: 'pfasgroups_mixed', metricType: 'RMSE', property: 'logKow', value: {rmse_pgm_kow:.4f} }},")
    print(f"{{ model: 'pfasgroups_mixed', metricType: 'RMSE', property: 'logKoa', value: {rmse_pgm_koa:.4f} }},")


if __name__ == "__main__":
    main()
