"""benchmark_5model.py
=======================
Shared 5-fold cross-validated benchmark for the 5 models reported in
"A Vanishing Partitioning Gap" (Crippen ridge, Naef & Acree fixed
parameters, the combined Crippen+Naef & Acree ridge model, a Random
Forest and an XGBoost model -- both trained on the same combined
Naef+Crippen feature matrix as the combined ridge model).

This reuses the exact same fold assignments, feature construction and
per-model fitting routines as scripts/shared_fold_benchmark.py (imported
directly, not re-implemented), restricted to just the 5 models used in
the simplified article -- MQG and PFASGroups features are not computed
at all here.

Outputs (written to tests/out/, both git-tracked):
  benchmark_predictions_5model.csv - out-of-fold prediction for every
      molecule in the common evaluation set, for all 5 models and both
      endpoints (used by generate_figures_simple.py's residual figure
      and by this script's own metric computation).
  benchmark_metrics_5model.csv - one row per (endpoint, model) with
      R2, RMSE, and the four threshold-dependent classification metrics
      (AUC, average precision, balanced accuracy, MCC) -- this is the
      source of Table S4 in the Supporting Information.

Usage:
    python scripts/benchmark_5model.py
    python scripts/benchmark_5model.py --kow-threshold 4.6
    python scripts/benchmark_5model.py --kow-threshold 4.6 --koa-threshold 6.0

The thresholds classify a molecule as "positive" at logKow>=kow_threshold
(vB/B proxy) or logKoa>=koa_threshold (B proxy, Kelly et al.) -- see
Supporting Information S1.2 for the regulatory rationale. Re-run this
script whenever that threshold changes, rather than hand-editing
Table S4.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kawow.features import compute_features  # noqa: E402
from kawow.model import _compile_naef_patterns, _compute_naef_group_counts  # noqa: E402
from kawow.smarts_model import NaefAcreePartitionCalculator  # noqa: E402
from scripts.fit_mixed_naef_crippen import _append_crippen_rows, _compile_specs, _feature_vector  # noqa: E402
from scripts.shared_fold_benchmark import (  # noqa: E402
    _fit_predict_kawow,
    _fit_predict_mixed,
    _fit_predict_rf,
    _fit_predict_xgb,
    _load_rows,
    _safe_smarts_predict,
)

RDLogger.DisableLog("rdApp.*")

DATA_DIR = REPO_ROOT / "kawow" / "data"
TEST_DIR = REPO_ROOT / "tests" / "test_data"
OUT_DIR = REPO_ROOT / "tests" / "out"

S01_SDF = TEST_DIR / "S01. Compounds List for logPow-Parameters Calculations.sdf"
S02_SDF = TEST_DIR / "S02. Compounds List for logKoa-Parameters Calculations.sdf"
KOW_BASE = DATA_DIR / "naef2024_logkow_parameters.csv"
KOA_BASE = DATA_DIR / "naef2024_logkoa_parameters.csv"

MODEL_ORDER = ["crippen", "naefacree", "naefacree_mixed", "naef_crippen_rf", "naef_crippen_xgb"]
MODEL_LABELS = {
    "crippen": "Crippen",
    "naefacree": "Naef & Acree",
    "naefacree_mixed": "Crippen+Naef & Acree",
    "naef_crippen_rf": "Random Forest",
    "naef_crippen_xgb": "XGBoost",
}


def _build_common_set(sdf_path: Path, value_prop: str, endpoint: str, naef_csv: Path, mixed_base_csv: Path):
    """Load an endpoint's SDF and compute the 3 feature representations
    needed by the 5 models (Crippen, Naef group counts, and the
    concatenated Crippen+Naef "mixed" matrix used by the combined ridge/
    RF/XGBoost models), keeping only molecules valid for all three --
    this is the "common evaluation set" reported in Table S1 (backbone
    <2 / RDKit-parse-failure exclusions), independent of MQG/PFASGroups
    (not used by any of these 5 models, so not required here)."""
    rows = _load_rows(sdf_path, value_prop)
    naef_patterns = _compile_naef_patterns(naef_csv)
    mixed_df = pd.read_csv(mixed_base_csv)
    mixed_specs, _ = _compile_specs(_append_crippen_rows(mixed_df))
    smarts_calc = NaefAcreePartitionCalculator()

    inchikeys, names, ys = [], [], []
    X_crippen, X_mixed, X_naef, pred_smarts = [], [], [], []
    from rdkit import Chem

    for mol, name, value in rows:
        x_crippen = compute_features(mol)
        x_mixed = _feature_vector(mol, mixed_specs)
        x_naef = _compute_naef_group_counts(mol, naef_patterns)
        x_smarts = _safe_smarts_predict(smarts_calc, mol, endpoint)
        if x_crippen is None or x_mixed is None or x_naef is None or not np.isfinite(x_smarts):
            continue
        try:
            inchikey = Chem.MolToInchiKey(mol)
        except Exception:
            inchikey = ""
        inchikeys.append(inchikey)
        names.append(name)
        ys.append(value)
        X_crippen.append(x_crippen.astype(np.float32))
        X_mixed.append(x_mixed.astype(np.float32))
        X_naef.append(x_naef.astype(np.float32))
        pred_smarts.append(x_smarts)

    return {
        "inchikey": np.array(inchikeys),
        "name": np.array(names),
        "y": np.array(ys, dtype=np.float64),
        "X_crippen": np.array(X_crippen, dtype=np.float32),
        "X_mixed": np.array(X_mixed, dtype=np.float32),
        "X_naef": np.array(X_naef, dtype=np.float32),
        "pred_smarts": np.array(pred_smarts, dtype=np.float64),
        "raw_n": len(rows),
    }


def _run_outer_cv(data: dict) -> dict[str, np.ndarray]:
    y = data["y"]
    outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = {mkey: np.full(len(y), np.nan, dtype=np.float64) for mkey in MODEL_ORDER}
    preds["naefacree"] = data["pred_smarts"].copy()

    for train_idx, test_idx in outer_kf.split(y):
        preds["crippen"][test_idx] = _fit_predict_kawow(
            data["X_crippen"][train_idx], y[train_idx], data["X_crippen"][test_idx])
        preds["naefacree_mixed"][test_idx] = _fit_predict_mixed(
            data["X_mixed"][train_idx], y[train_idx], data["X_mixed"][test_idx])
        preds["naef_crippen_rf"][test_idx] = _fit_predict_rf(
            data["X_mixed"][train_idx], y[train_idx], data["X_mixed"][test_idx])
        preds["naef_crippen_xgb"][test_idx] = _fit_predict_xgb(
            data["X_mixed"][train_idx], y[train_idx], data["X_mixed"][test_idx])
    return preds


def _compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    y_bin_true = (y_true >= threshold).astype(int)
    y_bin_pred = (y_pred >= threshold).astype(int)
    if len(np.unique(y_bin_true)) > 1:
        auc = float(roc_auc_score(y_bin_true, y_pred))
        ap = float(average_precision_score(y_bin_true, y_pred))
    else:
        auc = float("nan")
        ap = float("nan")
    bal_acc = float(balanced_accuracy_score(y_bin_true, y_bin_pred))
    mcc = float(matthews_corrcoef(y_bin_true, y_bin_pred))
    return {"r2": r2, "rmse": rmse, "auc": auc, "ap": ap, "bal_acc": bal_acc, "mcc": mcc}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kow-threshold", type=float, default=4.6,
                         help="logKow classification threshold (default: 4.6, the aquatic-only B "
                              "proxy from BCF~=Lc*Kow, Lc~=0.05; see SI S1.2). Was 5.0 (the vB proxy) "
                              "in an earlier draft of Table S4.")
    parser.add_argument("--koa-threshold", type=float, default=6.0,
                         help="logKoa classification threshold (default: 6.0, Kelly et al.'s B proxy).")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building common evaluation sets (logKow, logKoa) ...", flush=True)
    kow_data = _build_common_set(S01_SDF, "logP", "logKow", KOW_BASE, KOW_BASE)
    koa_data = _build_common_set(S02_SDF, "logKoa", "logKoa", KOA_BASE, KOA_BASE)
    print(f"  logKow common set: n={len(kow_data['y'])} (raw n={kow_data['raw_n']})")
    print(f"  logKoa common set: n={len(koa_data['y'])} (raw n={koa_data['raw_n']})")

    print("Running outer 5-fold CV for logKow ...", flush=True)
    kow_preds = _run_outer_cv(kow_data)
    print("Running outer 5-fold CV for logKoa ...", flush=True)
    koa_preds = _run_outer_cv(koa_data)

    pred_rows = []
    for endpoint, data, preds in [("logKow", kow_data, kow_preds), ("logKoa", koa_data, koa_preds)]:
        for i in range(len(data["y"])):
            row = {
                "dataset": endpoint,
                "inchikey": data["inchikey"][i],
                "name": data["name"][i],
                "exp_value": data["y"][i],
            }
            for mkey in MODEL_ORDER:
                suffix = "kow" if endpoint == "logKow" else "koa"
                row[f"pred_{mkey}_{suffix}"] = preds[mkey][i]
            pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)
    pred_path = OUT_DIR / "benchmark_predictions_5model.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved {pred_path}")

    metric_rows = []
    for endpoint, data, preds, threshold in [
        ("logKow", kow_data, kow_preds, args.kow_threshold),
        ("logKoa", koa_data, koa_preds, args.koa_threshold),
    ]:
        for mkey in MODEL_ORDER:
            m = _compute_all_metrics(data["y"], preds[mkey], threshold)
            metric_rows.append({
                "endpoint": endpoint,
                "model": mkey,
                "model_label": MODEL_LABELS[mkey],
                "n": int(len(data["y"])),
                "threshold": threshold,
                "r2": round(m["r2"], 4),
                "rmse": round(m["rmse"], 3),
                "auc": round(m["auc"], 4),
                "ap": round(m["ap"], 4),
                "bal_acc": round(m["bal_acc"], 4),
                "mcc": round(m["mcc"], 4),
            })
    metric_df = pd.DataFrame(metric_rows)
    metric_path = OUT_DIR / "benchmark_metrics_5model.csv"
    metric_df.to_csv(metric_path, index=False)
    print(f"Saved {metric_path}")

    pd.set_option("display.width", 160)
    print("\n" + metric_df.to_string(index=False))


if __name__ == "__main__":
    main()
