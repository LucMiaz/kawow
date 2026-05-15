"""Augment ionization_map_s01_s02.csv with naef_mqg and crippen_mqg predictions.

Uses shared MQG feature computation (the slow step) across both ensemble variants.
Targets only the S01∩S02 intersection molecules (~1102), which are the only ones
used by the web app confusion matrix.

Usage:
    python scripts/augment_benchmark_csv.py
"""
import csv
import pathlib
import time
import warnings
import sys
from concurrent.futures import as_completed

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

import numpy as np

CSV_PATH = pathlib.Path(__file__).parent.parent / "tests" / "out" / "ionization_map_s01_s02.csv"
NEW_COLS = [
    "pred_naef_mqg_logKow", "pred_naef_mqg_logKoa",
    "pred_crippen_mqg_logKow", "pred_crippen_mqg_logKoa",
]


def main():
    t_start = time.perf_counter()
    print(f"Reading CSV: {CSV_PATH}")

    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_fields = list(reader.fieldnames)
        for row in reader:
            rows.append(dict(row))

    print(f"  {len(rows)} rows, {len(original_fields)} columns")

    # Check if columns already exist and filled
    if all(c in original_fields for c in NEW_COLS):
        n = sum(1 for r in rows if r.get("pred_naef_mqg_logKow"))
        print(f"New columns already present ({n} rows filled). Nothing to do.")
        return

    # Load models
    print("Loading ensemble models...")
    from kawow.model import (
        _load_pickle, _compile_naef_patterns, _compute_mqg_features_with_ratios,
        _compute_naef_group_counts, compute_features, DATA_DIR,
    )

    kow_naef_mqg = _load_pickle(DATA_DIR / "logkow_naef_mqg_model.pkl")
    koa_naef_mqg = _load_pickle(DATA_DIR / "logkoa_naef_mqg_model.pkl")
    kow_crippen_mqg = _load_pickle(DATA_DIR / "logkow_crippen_mqg_model.pkl")
    koa_crippen_mqg = _load_pickle(DATA_DIR / "logkoa_crippen_mqg_model.pkl")

    naef_patterns_kow = _compile_naef_patterns(kow_naef_mqg["naef_param_file"])
    naef_patterns_koa = _compile_naef_patterns(koa_naef_mqg["naef_param_file"])
    print("  Models loaded.")

    def predict_smiles(smiles: str) -> dict:
        """Compute all variant predictions for one SMILES, sharing MQG features."""
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return {"smiles": smiles}
        try:
            # MQG features are molecular (not property-specific) — compute ONCE for all 4 predictions
            mqg_full = _compute_mqg_features_with_ratios(mol, fp_size=64)
            if mqg_full is None:
                return {"smiles": smiles}

            naef_kow = _compute_naef_group_counts(mol, naef_patterns_kow)
            naef_koa = _compute_naef_group_counts(mol, naef_patterns_koa)
            crippen = compute_features(mol)
            if crippen is None:
                return {"smiles": smiles}
            crippen = crippen.astype(np.float32)

            # naef_mqg: [naef | mqg_cols]
            x_nm_kow = np.concatenate([naef_kow, mqg_full[kow_naef_mqg["mqg_feature_cols"]]])
            x_nm_koa = np.concatenate([naef_koa, mqg_full[koa_naef_mqg["mqg_feature_cols"]]])
            kow_nm = round(float(kow_naef_mqg["model"].predict(x_nm_kow.reshape(1, -1))[0]), 3)
            koa_nm = round(float(koa_naef_mqg["model"].predict(x_nm_koa.reshape(1, -1))[0]), 3)

            # crippen_mqg: [crippen | mqg_cols]
            x_cm_kow = np.concatenate([crippen, mqg_full[kow_crippen_mqg["mqg_feature_cols"]]])
            x_cm_koa = np.concatenate([crippen, mqg_full[koa_crippen_mqg["mqg_feature_cols"]]])
            kow_cm = round(float(kow_crippen_mqg["model"].predict(x_cm_kow.reshape(1, -1))[0]), 3)
            koa_cm = round(float(koa_crippen_mqg["model"].predict(x_cm_koa.reshape(1, -1))[0]), 3)

            return {
                "smiles": smiles,
                "pred_naef_mqg_logKow": kow_nm, "pred_naef_mqg_logKoa": koa_nm,
                "pred_crippen_mqg_logKow": kow_cm, "pred_crippen_mqg_logKoa": koa_cm,
            }
        except Exception as e:
            return {"smiles": smiles, "_err": str(e)}

    # Identify molecules in S01∩S02 (server only uses these)
    s01_smiles = set()
    s02_smiles = set()
    for row in rows:
        ds = row.get("dataset", "").strip().upper()
        smi = row.get("smiles", "").strip()
        if not smi:
            continue
        if ds == "S01":
            s01_smiles.add(smi)
        elif ds == "S02":
            s02_smiles.add(smi)

    target_smiles = sorted(s01_smiles & s02_smiles)  # intersection only: server only uses these
    print(f"  Predicting S01+S02 intersection: {len(target_smiles)} unique SMILES")

    # Test speed on first molecule
    t0 = time.perf_counter()
    test_result = predict_smiles(target_smiles[0])
    t_one = time.perf_counter() - t0
    print(f"  Test prediction ({target_smiles[0][:40]}...): {t_one*1000:.0f}ms")
    print(f"  Estimated time (sequential): {t_one * len(target_smiles) / 60:.1f} min")

    # Run predictions sequentially (ThreadPoolExecutor deadlocks with MQG on Windows)
    print(f"  Running sequentially (~{t_one * len(target_smiles) / 60:.0f} min total)...")

    predictions = {}
    done = 0
    errors = 0
    t0 = time.perf_counter()
    for smi in target_smiles:
        try:
            result = predict_smiles(smi)
            if "_err" in result:
                errors += 1
            predictions[smi] = result
        except Exception as e:
            predictions[smi] = {"smiles": smi}
            errors += 1
        done += 1
        if done % 50 == 0 or done == len(target_smiles):
            elapsed = time.perf_counter() - t0
            rate = done / elapsed
            remaining = (len(target_smiles) - done) / rate if rate > 0 else 0
            print(f"  {done}/{len(target_smiles)}  rate={rate:.2f}/s  ~{remaining/60:.1f} min left  errors={errors}")
            sys.stdout.flush()

    # Map predictions back to CSV rows
    for row in rows:
        smi = row.get("smiles", "").strip()
        pred = predictions.get(smi, {})
        for col in NEW_COLS:
            row[col] = pred.get(col, "")

    # Build output fieldnames
    out_fields = list(original_fields)
    for col in NEW_COLS:
        if col not in out_fields:
            out_fields.append(col)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - t_start
    n_filled = sum(1 for r in rows if r.get("pred_naef_mqg_logKow"))
    print(f"\nDone in {elapsed/60:.1f} min. Written to {CSV_PATH}")
    print(f"Rows with naef_mqg prediction: {n_filled}/{len(rows)}")


if __name__ == "__main__":
    main()

