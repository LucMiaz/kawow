"""Augment ionization_map_s01_s02.csv with pfasgroups and pfasgroups_mixed predictions.

Uses pre-fitted Ridge pipeline pkl files from kawow/data/logkow_pfasgroups_model.pkl
and logkoa_pfasgroups_model.pkl (and the *_mixed variants).

Run AFTER fit_pfasgroups_model.py:
    python scripts/fit_pfasgroups_model.py
    python scripts/augment_benchmark_csv_pfasgroups.py
"""
import csv
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")

import numpy as np  # noqa: E402

CSV_PATH = REPO_ROOT / "tests" / "out" / "ionization_map_s01_s02.csv"
NEW_COLS = [
    "pred_pfasgroups_logKow", "pred_pfasgroups_logKoa",
    "pred_pfasgroups_mixed_logKow", "pred_pfasgroups_mixed_logKoa",
]


def main() -> None:
    t_start = time.perf_counter()
    print(f"Reading CSV: {CSV_PATH}")

    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_fields = list(reader.fieldnames)
        for row in reader:
            rows.append(dict(row))

    print(f"  {len(rows)} rows, {len(original_fields)} columns")

    if all(c in original_fields for c in NEW_COLS):
        n = sum(1 for r in rows if r.get("pred_pfasgroups_logKow"))
        print(f"PFASGroups columns already present ({n} rows filled). Nothing to do.")
        return

    # Load fitted models
    print("Loading PFASGroups models...")
    from kawow.model import _load_pickle, DATA_DIR  # noqa: E402
    from kawow.pfasgroups_features import compute_pfasgroups_features  # noqa: E402
    from kawow.features import compute_features  # noqa: E402

    kow_pg   = _load_pickle(DATA_DIR / "logkow_pfasgroups_model.pkl")
    koa_pg   = _load_pickle(DATA_DIR / "logkoa_pfasgroups_model.pkl")
    kow_pgm  = _load_pickle(DATA_DIR / "logkow_pfasgroups_mixed_model.pkl")
    koa_pgm  = _load_pickle(DATA_DIR / "logkoa_pfasgroups_mixed_model.pkl")
    print("  Models loaded.")

    def predict_smiles(smiles: str) -> dict:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return {}
        try:
            x_pg = compute_pfasgroups_features(mol)
            if x_pg is None:
                return {}
            x_cr = compute_features(mol)
            if x_cr is None:
                return {}
            x_cr = x_cr.astype(np.float32)
            x_pgm = np.hstack([x_pg, x_cr])

            kow_pg_pred   = round(float(kow_pg["model"].predict(x_pg.reshape(1, -1))[0]),   3)
            koa_pg_pred   = round(float(koa_pg["model"].predict(x_pg.reshape(1, -1))[0]),   3)
            kow_pgm_pred  = round(float(kow_pgm["model"].predict(x_pgm.reshape(1, -1))[0]), 3)
            koa_pgm_pred  = round(float(koa_pgm["model"].predict(x_pgm.reshape(1, -1))[0]), 3)

            return {
                "pred_pfasgroups_logKow": kow_pg_pred,
                "pred_pfasgroups_logKoa": koa_pg_pred,
                "pred_pfasgroups_mixed_logKow": kow_pgm_pred,
                "pred_pfasgroups_mixed_logKoa": koa_pgm_pred,
            }
        except Exception as exc:
            return {"_err": str(exc)}

    # Identify S01∩S02 molecules
    s01_smiles: set[str] = set()
    s02_smiles: set[str] = set()
    for row in rows:
        ds  = row.get("dataset", "").strip().upper()
        smi = row.get("smiles",  "").strip()
        if not smi:
            continue
        if ds == "S01":
            s01_smiles.add(smi)
        elif ds == "S02":
            s02_smiles.add(smi)

    target = sorted(s01_smiles & s02_smiles)
    print(f"  S01∩S02 intersection: {len(target)} unique SMILES to predict")

    # Warm-up timing estimate
    t0 = time.perf_counter()
    _ = predict_smiles(target[0])
    t_one = time.perf_counter() - t0
    print(f"  First prediction: {t_one*1000:.0f} ms  →  estimated total: {t_one * len(target) / 60:.1f} min")

    predictions: dict[str, dict] = {}
    errors = 0
    t0 = time.perf_counter()
    for i, smi in enumerate(target, 1):
        result = predict_smiles(smi)
        if "_err" in result or not result:
            errors += 1
        predictions[smi] = result
        if i % 50 == 0 or i == len(target):
            elapsed = time.perf_counter() - t0
            rate = i / elapsed
            remaining = (len(target) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(target)}  {rate:.2f}/s  ~{remaining/60:.1f} min left  errors={errors}")
            sys.stdout.flush()

    # Write predictions back to rows
    for row in rows:
        smi  = row.get("smiles", "").strip()
        pred = predictions.get(smi, {})
        for col in NEW_COLS:
            row[col] = pred.get(col, "")

    out_fields = list(original_fields)
    for col in NEW_COLS:
        if col not in out_fields:
            out_fields.append(col)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - t_start
    n_filled = sum(1 for r in rows if r.get("pred_pfasgroups_logKow"))
    print(f"\nDone in {elapsed/60:.1f} min. Written {n_filled} pfasgroups predictions to {CSV_PATH}")


if __name__ == "__main__":
    main()
