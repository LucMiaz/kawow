"""build_deployment_predictions.py
==================================
Builds the final-model (non-CV, external-inference) prediction tables that
generate_figures.py and generate_tables.py read for main-text Figures 2 to 4,
SI Figures S4 to S6 and SI Table S6. Written to cache/.

Every model is fit ONCE on the complete Naef & Acree common evaluation set
for an endpoint (n=3,319 logKow, n=1,956 logKoa), with exactly the same
features and fit routines as kawow's scripts/benchmark_5model.py (imported,
not re-implemented), and then applied to:

  cross_endpoint_predictions.csv  - Naef & Acree molecules with only one
      endpoint measured: the missing endpoint, predicted by each model.
  arphale_predictions.csv         - the Arp & Hale (2022) REACH dataset.
  arphale_naef_experimental.csv   - Arp & Hale molecules that also have an
      experimental logKow (S01) AND logKoa (S02), matched by RDKit InChIKey.
  clinventory_predictions_5model.csv - the C&L Inventory extract
      (data/clinventory_molecules.csv, not in git, see README.md), molecules
      valid for all models.

Usage:
    python build_deployment_predictions.py            # all four tables (~15 min)
    python build_deployment_predictions.py --skip-clinventory
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

import paths  # noqa: E402  (also puts the kawow repository on sys.path)

from kawow.features import compute_features  # noqa: E402
from kawow.model import _compile_naef_patterns, _compute_naef_group_counts  # noqa: E402
from kawow.smarts_model import NaefAcreePartitionCalculator  # noqa: E402
from scripts.benchmark_5model import (  # noqa: E402
    KOA_BASE, KOW_BASE, MODEL_ORDER, S01_SDF, S02_SDF, _build_common_set,
)
from scripts.fit_mixed_naef_crippen import _append_crippen_rows, _compile_specs, _feature_vector  # noqa: E402
from scripts.shared_fold_benchmark import (  # noqa: E402
    _fit_predict_kawow, _fit_predict_mixed, _fit_predict_rf, _fit_predict_xgb, _load_rows,
)

RDLogger.DisableLog("rdApp.*")

OUT = paths.CACHE

ENDPOINTS = {"logKow": KOW_BASE, "logKoa": KOA_BASE}


# ── Ridge models, inner CV in parallel ─────────────────────────────────────────
# Same procedure as kawow's _fit_predict_kawow (crippen, with the 3-sigma
# outlier refit) and _fit_predict_mixed (combined ridge): 5-fold inner CV
# over 30 alphas, sag solver, fixed seeds. Only the 150 independent inner
# fits are run in parallel; each fit is deterministic, so the selected alpha
# and the final model are identical to the serial routine (--serial-ridge).
def _ridge(alpha):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(with_mean=False),
        Ridge(alpha=float(alpha), fit_intercept=True, solver="sag", max_iter=8000, tol=1e-4, random_state=42),
    )


def _inner_fit(alpha, X, y, tr, te):
    return _ridge(alpha).fit(X[tr], y[tr]).predict(X[te])


def _fit_predict_ridge_parallel(X_train, y_train, X_test, outlier_refit, n_jobs):
    from joblib import Parallel, delayed
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold
    alphas = np.logspace(-3, 4, 30)
    splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train))
    out = Parallel(n_jobs=n_jobs)(
        delayed(_inner_fit)(a, X_train, y_train, tr, te) for a in alphas for tr, te in splits)
    rmses = []
    for k in range(len(alphas)):
        fold_pred = np.zeros_like(y_train, dtype=np.float64)
        for (tr, te), p in zip(splits, out[k * len(splits):(k + 1) * len(splits)]):
            fold_pred[te] = p
        rmses.append(np.sqrt(mean_squared_error(y_train, fold_pred)))
    best_alpha = alphas[int(np.argmin(rmses))]  # first minimum, as the serial strict "<"
    model = _ridge(best_alpha).fit(X_train, y_train)
    if outlier_refit:
        residuals = y_train - model.predict(X_train)
        mask = np.abs(residuals) <= 3 * residuals.std()
        if int((~mask).sum()) > 0:
            model = _ridge(best_alpha).fit(X_train[mask], y_train[mask])
    return model.predict(X_test)


def _fit_table(serial_ridge, n_jobs):
    if serial_ridge:
        crippen, mixed = _fit_predict_kawow, _fit_predict_mixed
    else:
        def crippen(X, y, Xt):
            return _fit_predict_ridge_parallel(X, y, Xt, True, n_jobs)

        def mixed(X, y, Xt):
            return _fit_predict_ridge_parallel(X, y, Xt, False, n_jobs)
    return {
        "crippen":          ("X_crippen", crippen),
        "naefacree_mixed":  ("X_mixed", mixed),
        "naef_crippen_rf":  ("X_mixed", _fit_predict_rf),
        "naef_crippen_xgb": ("X_mixed", _fit_predict_xgb),
    }


# ── Feature extraction (same validity checks as _build_common_set) ────────────
_W = {}


def _init_worker():
    RDLogger.DisableLog("rdApp.*")
    _W["specs"] = {ep: _compile_specs(_append_crippen_rows(pd.read_csv(p)))[0] for ep, p in ENDPOINTS.items()}
    _W["naef"] = {ep: _compile_naef_patterns(p) for ep, p in ENDPOINTS.items()}
    _W["calc"] = NaefAcreePartitionCalculator()


def _smarts_both(mol):
    try:
        out = _W["calc"].parse(mol)
        hit = next(v for v in out.values() if isinstance(v, dict))
        return float(hit["logKow"]), float(hit["logKoa"])
    except Exception:
        return np.nan, np.nan


def _featurise_mol(mol):
    """Return {endpoint: (x_crippen, x_mixed, pred_naefacree)} for every
    endpoint the molecule is valid for (an endpoint is absent if any of the
    feature constructions fail, exactly as in _build_common_set)."""
    out = {}
    x_crippen = compute_features(mol)
    smarts = dict(zip(ENDPOINTS, _smarts_both(mol)))
    for ep in ENDPOINTS:
        x_mixed = _feature_vector(mol, _W["specs"][ep])
        x_naef = _compute_naef_group_counts(mol, _W["naef"][ep])
        if x_crippen is None or x_mixed is None or x_naef is None or not np.isfinite(smarts[ep]):
            continue
        out[ep] = (x_crippen.astype(np.float32), x_mixed.astype(np.float32), smarts[ep])
    return out


def _featurise_chunk(smiles_list):
    return [(_featurise_mol(m) if (m := Chem.MolFromSmiles(s)) is not None else {}) if isinstance(s, str) else {}
            for s in smiles_list]


def featurise(smiles, n_jobs):
    """Featurise a list of SMILES in parallel; returns one dict per SMILES."""
    chunks = [smiles[i:i + 1000] for i in range(0, len(smiles), 1000)]
    results = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker) as ex:
        for k, res in enumerate(ex.map(_featurise_chunk, chunks)):
            results.extend(res)
            if (k + 1) % 20 == 0:
                print(f"    featurised {len(results):,}/{len(smiles):,} ({time.perf_counter() - t0:.0f} s)", flush=True)
    return results


# ── Final models ──────────────────────────────────────────────────────────────
def predict_final(train, feats, ep, fit_table):
    """Fit each model on the full common set `train` for endpoint `ep` and
    predict every molecule in `feats` (list of _featurise_mol outputs).
    Returns {model: array}, NaN where the molecule is invalid for `ep`."""
    idx = np.array([i for i, f in enumerate(feats) if ep in f], dtype=int)
    preds = {m: np.full(len(feats), np.nan) for m in MODEL_ORDER}
    if not len(idx):
        return preds
    X = {
        "X_crippen": np.stack([feats[i][ep][0] for i in idx]),
        "X_mixed": np.stack([feats[i][ep][1] for i in idx]),
    }
    preds["naefacree"][idx] = [feats[i][ep][2] for i in idx]
    for mkey, (xkey, fit) in fit_table.items():
        t0 = time.perf_counter()
        preds[mkey][idx] = fit(train[xkey], train["y"], X[xkey])
        print(f"    {ep} {mkey}: fit + predict {len(idx):,} in {time.perf_counter() - t0:.0f} s", flush=True)
    return preds


def _inchikey(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToInchiKey(m) if m is not None else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-clinventory", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--serial-ridge", action="store_true",
                    help="use kawow's serial ridge routines directly (identical result, ~10x slower)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out_dir = args.out
    out_dir.mkdir(exist_ok=True)
    if not args.skip_clinventory:
        paths.require_clinventory()

    print("Building Naef & Acree common evaluation sets ...", flush=True)
    train = {
        "logKow": _build_common_set(S01_SDF, "logP", "logKow", KOW_BASE, KOW_BASE),
        "logKoa": _build_common_set(S02_SDF, "logKoa", "logKoa", KOA_BASE, KOA_BASE),
    }
    for ep, d in train.items():
        print(f"  {ep}: n={len(d['y'])} (raw n={d['raw_n']})")

    # Pools of molecules to predict, concatenated so each model is fit once
    # per endpoint. Naef & Acree molecules are passed as SMILES re-written
    # from their SDF structure.
    pools = {}
    keys = {ep: set(train[ep]["inchikey"]) for ep in train}
    sdf_mols = {}
    for ep, sdf, prop in [("logKow", S01_SDF, "logP"), ("logKoa", S02_SDF, "logKoa")]:
        for mol, _name, value in _load_rows(sdf, prop):
            ik = Chem.MolToInchiKey(mol)
            sdf_mols.setdefault(ik, Chem.MolToSmiles(mol))
    kow_only = sorted(keys["logKow"] - keys["logKoa"])
    koa_only = sorted(keys["logKoa"] - keys["logKow"])
    pools["naef"] = pd.DataFrame({
        "inchikey": kow_only + koa_only,
        "category": ["logKow_only"] * len(kow_only) + ["logKoa_only"] * len(koa_only),
    })
    pools["naef"]["smiles"] = pools["naef"]["inchikey"].map(sdf_mols)

    arphale = pd.read_csv(paths.ARPHALE_CSV)
    pools["arphale"] = arphale

    if not args.skip_clinventory:
        pools["clinventory"] = pd.read_csv(paths.CLINVENTORY_CSV)

    smiles_col = {"naef": "smiles", "arphale": "Smiles", "clinventory": "smiles"}
    all_smiles, owner = [], []
    for name, df in pools.items():
        all_smiles.extend(df[smiles_col[name]].tolist())
        owner.extend([name] * len(df))
    owner = np.array(owner)
    print(f"Featurising {len(all_smiles):,} molecules with {args.jobs} workers ...", flush=True)
    feats = featurise(all_smiles, args.jobs)

    print("Fitting final models and predicting ...", flush=True)
    fit_table = _fit_table(args.serial_ridge, args.jobs)
    preds = {ep: predict_final(train[ep], feats, ep, fit_table) for ep in ENDPOINTS}

    def _slice(name):
        sel = owner == name
        return {ep: {m: preds[ep][m][sel] for m in MODEL_ORDER} for ep in ENDPOINTS}

    # Cross-endpoint fill-ins for single-observation Naef & Acree molecules.
    p = _slice("naef")
    cross = pools["naef"][["inchikey", "category"]].copy()
    for m in MODEL_ORDER:
        cross[f"cross_pred_{m}_logKoa"] = np.where(cross["category"] == "logKow_only", p["logKoa"][m], np.nan)
        cross[f"cross_pred_{m}_logKow"] = np.where(cross["category"] == "logKoa_only", p["logKow"][m], np.nan)
    cross.to_csv(out_dir / "cross_endpoint_predictions.csv", index=False)
    print(f"  Saved cross_endpoint_predictions.csv (n={len(cross)})")

    # Arp & Hale predictions (rows valid for at least one endpoint).
    p = _slice("arphale")
    ah = arphale.copy()
    for ep in ENDPOINTS:
        for m in MODEL_ORDER:
            ah[f"pred_{m}_{ep}"] = p[ep][m]
    ok = np.isfinite(ah["pred_naefacree_mixed_logKow"]) & np.isfinite(ah["pred_naefacree_mixed_logKoa"])
    ah[ok].to_csv(out_dir / "arphale_predictions.csv", index=False)
    print(f"  Saved arphale_predictions.csv (n={int(ok.sum())} of {len(ah)})")

    # Arp & Hale x Naef & Acree experimental pairs, by RDKit InChIKey.
    exp = {}
    for ep, sdf, prop in [("logKow", S01_SDF, "logP"), ("logKoa", S02_SDF, "logKoa")]:
        rows = [(Chem.MolToInchiKey(mol), v) for mol, _n, v in _load_rows(sdf, prop)]
        exp[ep] = pd.DataFrame(rows, columns=["inchikey", ep]).groupby("inchikey")[ep].mean()
    ah_exp = arphale.copy()
    ah_exp["inchikey"] = ah_exp["Smiles"].map(_inchikey)
    ah_exp["naef_logKow"] = ah_exp["inchikey"].map(exp["logKow"])
    ah_exp["naef_logKoa"] = ah_exp["inchikey"].map(exp["logKoa"])
    # One row per Arp & Hale substance (not per structure), like the LSER and
    # predicted sets: n=703 substances, 583 distinct InChIKeys.
    ah_exp = ah_exp.dropna(subset=["naef_logKow", "naef_logKoa"])
    ah_exp.to_csv(out_dir / "arphale_naef_experimental.csv", index=False)
    print(f"  Saved arphale_naef_experimental.csv (n={len(ah_exp)})")

    if "clinventory" in pools:
        p = _slice("clinventory")
        cl = pools["clinventory"][["molecule_id", "inchikey", "smiles", "ec_primary"]].copy()
        for m in MODEL_ORDER:
            cl[f"{m}_logKow"] = p["logKow"][m]
            cl[f"{m}_logKoa"] = p["logKoa"][m]
        ok = np.all([np.isfinite(cl[f"{m}_{ep}"]) for m in MODEL_ORDER for ep in ENDPOINTS], axis=0)
        cl[ok].to_csv(out_dir / "clinventory_predictions_5model.csv", index=False)
        print(f"  Saved clinventory_predictions_5model.csv (n={int(ok.sum()):,} of {len(cl):,})")


if __name__ == "__main__":
    main()
