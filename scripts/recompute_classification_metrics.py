"""recompute_classification_metrics.py
=======================================
Recompute the threshold-dependent classification metrics (AUC, average
precision, balanced accuracy, MCC) from the already-saved out-of-fold
predictions in tests/out/benchmark_predictions_5model.csv, without
re-running the (slow) 5-fold CV fitting in benchmark_5model.py.

Use this whenever only the classification threshold changes -- R2/RMSE
are threshold-independent and unaffected, and the predictions themselves
don't need to be regenerated.

Usage:
    python scripts/recompute_classification_metrics.py --kow-threshold 4.6
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "tests" / "out"
PRED_CSV = OUT_DIR / "benchmark_predictions_5model.csv"

MODEL_ORDER = ["crippen", "naefacree", "naefacree_mixed", "naef_crippen_rf", "naef_crippen_xgb"]
MODEL_LABELS = {
    "crippen": "Crippen", "naefacree": "Naef & Acree", "naefacree_mixed": "Crippen+Naef & Acree",
    "naef_crippen_rf": "Random Forest", "naef_crippen_xgb": "XGBoost",
}


def compute(y_true, y_pred, threshold):
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    y_bin_true = (y_true >= threshold).astype(int)
    y_bin_pred = (y_pred >= threshold).astype(int)
    auc = float(roc_auc_score(y_bin_true, y_pred))
    ap = float(average_precision_score(y_bin_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_bin_true, y_bin_pred))
    mcc = float(matthews_corrcoef(y_bin_true, y_bin_pred))
    return {"r2": r2, "rmse": rmse, "auc": auc, "ap": ap, "bal_acc": bal_acc, "mcc": mcc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kow-threshold", type=float, default=4.6)
    parser.add_argument("--koa-threshold", type=float, default=6.0)
    args = parser.parse_args()

    df = pd.read_csv(PRED_CSV)
    rows = []
    for endpoint, threshold, suffix in [("logKow", args.kow_threshold, "kow"), ("logKoa", args.koa_threshold, "koa")]:
        sub = df[df["dataset"] == endpoint]
        y_true = sub["exp_value"].values
        for mkey in MODEL_ORDER:
            y_pred = sub[f"pred_{mkey}_{suffix}"].values
            m = compute(y_true, y_pred, threshold)
            rows.append({
                "endpoint": endpoint, "model": mkey, "model_label": MODEL_LABELS[mkey],
                "n": int(len(sub)), "threshold": threshold,
                "r2": round(m["r2"], 4), "rmse": round(m["rmse"], 3),
                "auc": round(m["auc"], 4), "ap": round(m["ap"], 4),
                "bal_acc": round(m["bal_acc"], 4), "mcc": round(m["mcc"], 4),
            })
    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "benchmark_metrics_5model.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    pd.set_option("display.width", 160)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
