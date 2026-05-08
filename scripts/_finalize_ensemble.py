"""Post-processing: select winner from already-saved ensemble pkls.

Run this AFTER fit_ensemble_models.py has saved the 6 ensemble pkls but
before it completed (e.g. if it crashed at the CSV comparison step).

Usage:
    python scripts/_finalize_ensemble.py
"""
from __future__ import annotations

import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from kawow.model import DATA_DIR  # noqa: E402

OUT_DIR = REPO_ROOT / "tests" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENSEMBLE_TYPES = ["naef_mqg", "crippen_mqg", "naef_crippen_mqg"]
SIMPLICITY_THRESHOLD = 0.005


def _load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    print("=" * 60)
    print("Ensemble winner selection from saved pkls")
    print("=" * 60)

    payloads: dict[str, dict] = {}
    for etype in ENSEMBLE_TYPES:
        kow_path = DATA_DIR / f"logkow_{etype}_model.pkl"
        koa_path = DATA_DIR / f"logkoa_{etype}_model.pkl"
        if not kow_path.exists() or not koa_path.exists():
            print(f"  MISSING pkls for {etype} — skipping")
            continue
        pkl_kow = _load_pkl(kow_path)
        pkl_koa = _load_pkl(koa_path)
        payloads[etype] = {"kow": pkl_kow, "koa": pkl_koa}
        print(
            f"  {etype}: logKow R²={pkl_kow['r2_cv']:.4f} RMSE={pkl_kow['rmse_cv']:.4f} "
            f"| logKoa R²={pkl_koa['r2_cv']:.4f} RMSE={pkl_koa['rmse_cv']:.4f}"
        )

    if not payloads:
        print("No ensemble pkls found. Run fit_ensemble_models.py first.")
        return

    # Select winner
    best_model, best_combined_r2 = None, -np.inf
    for etype, pl in payloads.items():
        combined = pl["kow"]["r2_cv"] + pl["koa"]["r2_cv"]
        if combined > best_combined_r2:
            best_combined_r2 = combined
            best_model = etype

    preference_order = ["crippen_mqg", "naef_mqg", "naef_crippen_mqg"]
    for simpler in preference_order:
        if simpler == best_model or simpler not in payloads:
            break
        r2s = payloads[simpler]["kow"]["r2_cv"] + payloads[simpler]["koa"]["r2_cv"]
        if best_combined_r2 - r2s <= SIMPLICITY_THRESHOLD:
            print(
                f"  Preferring simpler '{simpler}' over '{best_model}' "
                f"(gap={best_combined_r2 - r2s:.4f} ≤ {SIMPLICITY_THRESHOLD})"
            )
            best_model = simpler
            break

    print(f"\n✓ Winner: {best_model}")
    w = payloads[best_model]
    print(f"  logKow: R²={w['kow']['r2_cv']:.4f}  RMSE={w['kow']['rmse_cv']:.4f}")
    print(f"  logKoa: R²={w['koa']['r2_cv']:.4f}  RMSE={w['koa']['rmse_cv']:.4f}")

    # Backup and replace
    print("\nBacking up original MQG pkls ...")
    for stem in ["logkow_mqg_model", "logkoa_mqg_model"]:
        src = DATA_DIR / f"{stem}.pkl"
        dst = DATA_DIR / f"{stem}_pure_backup.pkl"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"  {src.name} → {dst.name}")
        elif dst.exists():
            print(f"  {dst.name} already exists — skipping backup")

    print(f"Replacing mqg pkls with winner ({best_model}) ...")
    shutil.copy2(DATA_DIR / f"logkow_{best_model}_model.pkl", DATA_DIR / "logkow_mqg_model.pkl")
    shutil.copy2(DATA_DIR / f"logkoa_{best_model}_model.pkl", DATA_DIR / "logkoa_mqg_model.pkl")
    print("  Done.")

    # Save summary CSV
    rows = []
    for etype, pl in payloads.items():
        for key, endpoint in [("kow", "logKow"), ("koa", "logKoa")]:
            p = pl[key]
            rows.append({
                "model": etype,
                "endpoint": endpoint,
                "ion_class": "overall",
                "n": p["n_train"],
                "r2_cv": p["r2_cv"],
                "rmse_cv": p["rmse_cv"],
                "is_winner": etype == best_model,
            })
    df = pd.DataFrame(rows)
    out = OUT_DIR / "ensemble_performance_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(df.to_string(index=False))

    print("\n" + "=" * 60)
    print(f"WINNER: {best_model}")
    print("logkow/logkoa_mqg_model.pkl replaced with winner.")
    print("=" * 60)
    return best_model, payloads


if __name__ == "__main__":
    main()
