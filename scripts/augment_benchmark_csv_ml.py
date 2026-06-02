"""Augment ionization_map_s01_s02.csv with RF, XGB, and NN predictions.

Run from the repo root:
    python scripts/augment_benchmark_csv_ml.py [--models rf xgb nn]

NN requires keras/tensorflow; RF and XGB require scikit-learn / xgboost.
"""
import argparse
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

CSV_PATH = REPO_ROOT / "tests" / "out" / "ionization_map_s01_s02.csv"

MODEL_NEW_COLS = {
    "rf":  ["pred_pfasgroups_naef_mixed_rf_logKow",  "pred_pfasgroups_naef_mixed_rf_logKoa"],
    "xgb": ["pred_pfasgroups_naef_mixed_xgb_logKow", "pred_pfasgroups_naef_mixed_xgb_logKoa"],
    "nn":  ["pred_pfasgroups_naef_mixed_nn_logKow",  "pred_pfasgroups_naef_mixed_nn_logKoa"],
}


def load_calculator(name: str):
    import kawow
    if name == "rf":
        return kawow.PFASGroupsRFPartitionCalculator()
    if name == "xgb":
        return kawow.PFASGroupsXGBPartitionCalculator()
    if name == "nn":
        return kawow.PFASGroupsNNPartitionCalculator()
    raise ValueError(f"Unknown model: {name}")


def predict_smiles(calc, smi: str) -> dict:
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None:
        return {}
    try:
        r = calc._predict_mol(mol)
        if r.get("status") != "ok":
            return {}
        return {"logKow": r["logKow"], "logKoa": r["logKoa"]}
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["rf", "xgb", "nn"],
                        default=["rf", "xgb", "nn"],
                        help="Which models to add (default: rf xgb nn)")
    args = parser.parse_args()

    print(f"Reading CSV: {CSV_PATH}")
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_fields = list(reader.fieldnames)
        for row in reader:
            rows.append(dict(row))
    print(f"  {len(rows)} rows, {len(original_fields)} columns")

    # Collect S01 ∩ S02 SMILES
    s01_smiles, s02_smiles = set(), set()
    for row in rows:
        ds = row.get("dataset", "").strip().upper()
        smi = row.get("smiles", "").strip()
        if not smi:
            continue
        if ds == "S01":
            s01_smiles.add(smi)
        elif ds == "S02":
            s02_smiles.add(smi)
    target = sorted(s01_smiles & s02_smiles)
    print(f"  S01∩S02: {len(target)} unique SMILES")

    all_new_cols = []
    all_predictions: dict[str, dict] = {smi: {} for smi in target}

    for mname in args.models:
        new_cols = MODEL_NEW_COLS[mname]
        kow_col, koa_col = new_cols

        # Skip if already filled
        if all(c in original_fields for c in new_cols):
            n_filled = sum(1 for r in rows if r.get(kow_col))
            print(f"  [{mname}] columns already present ({n_filled} filled) — skipping")
            continue

        print(f"\n  Loading {mname.upper()} calculator...")
        try:
            calc = load_calculator(mname)
        except Exception as exc:
            print(f"  [{mname}] FAILED to load: {exc}")
            continue

        # Timing estimate
        t0 = time.perf_counter()
        _ = predict_smiles(calc, target[0])
        t_one = time.perf_counter() - t0
        print(f"  [{mname}] first prediction: {t_one*1000:.0f} ms → "
              f"estimated: {t_one * len(target) / 60:.1f} min")

        errors = 0
        t0 = time.perf_counter()
        for i, smi in enumerate(target, 1):
            r = predict_smiles(calc, smi)
            if r:
                all_predictions[smi][kow_col] = r["logKow"]
                all_predictions[smi][koa_col] = r["logKoa"]
            else:
                errors += 1
            if i % 100 == 0 or i == len(target):
                elapsed = time.perf_counter() - t0
                rate = i / elapsed
                remaining = (len(target) - i) / rate if rate > 0 else 0
                print(f"  [{mname}] {i}/{len(target)}  {rate:.1f}/s  "
                      f"~{remaining/60:.1f} min left  errors={errors}")
                sys.stdout.flush()

        all_new_cols.extend(new_cols)

    if not all_new_cols:
        print("\nNothing to write — all models already present.")
        return

    # Write predictions back
    for row in rows:
        smi = row.get("smiles", "").strip()
        pred = all_predictions.get(smi, {})
        for col in all_new_cols:
            row[col] = pred.get(col, "")

    out_fields = list(original_fields)
    for col in all_new_cols:
        if col not in out_fields:
            out_fields.append(col)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Written to {CSV_PATH}")


if __name__ == "__main__":
    main()
