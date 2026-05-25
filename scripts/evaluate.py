"""
evaluate.py
===========
Evaluate kawow predictions against experimental data in S01-S03.

Tests:
  1. logKow: 5-fold CV R² and RMSE on S01 (held-out from training)
  2. logKoa: 5-fold CV R² and RMSE on S02 (held-out from training)
  3. logKaw: R² and RMSE on S03 (using logKow-logKoa, no direct training)
  4. Baseline comparison: RDKit MolLogP for logKow

Usage
-----
    cd ~/kawow
    python scripts/evaluate.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from rdkit.Chem import Descriptors

from kawow.io import _read_sdf
from kawow.features import compute_features
from kawow.metrics import jeffreys_bf_corr, lin_ccc, nrmse, format_bf10
from kawow.model import PartitionCalculator, _build_Xy

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDF_DIR = os.path.join(MAIN_DIR, "tests", "test_data")
S01 = os.path.join(SDF_DIR, "S01. Compounds List for logPow-Parameters Calculations.sdf")
S02 = os.path.join(SDF_DIR, "S02. Compounds List for logKoa-Parameters Calculations.sdf")
S03 = os.path.join(SDF_DIR, "S03. Compounds List with exp logKaw Data.sdf")

SEP = "=" * 60


def cv_stats(X, y, n_splits=5):
    pipe = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeCV(alphas=np.logspace(-3, 4, 50), cv=5),
    )
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_cv = cross_val_predict(pipe, X, y, cv=kf)
    r2   = r2_score(y, y_cv)
    rmse = float(mean_squared_error(y, y_cv) ** 0.5)
    ccc_val  = lin_ccc(y, y_cv)
    nm       = nrmse(y, y_cv)
    r_corr   = float(np.corrcoef(y, y_cv)[0, 1]) if len(y) > 1 else float("nan")
    bf_res   = jeffreys_bf_corr(r_corr, len(y))
    return y_cv, r2, rmse, ccc_val, nm["nrmse_sd"], nm["nrmse_range"], bf_res["log10_bf10"], bf_res["ci95_lo"], bf_res["ci95_hi"]


# ── 1. logKow on S01 ─────────────────────────────────────────────────────────
print(SEP)
print("1. logKow — 5-fold CV on S01 training set")
print(SEP)
X_kow, y_kow, names_kow = _build_Xy(S01, "logP")
print(f"   Molecules loaded: {len(y_kow)}")
y_cv_kow, r2_kow, rmse_kow, ccc_kow, nrmse_sd_kow, nrmse_range_kow, bf10_kow, ci_lo_kow, ci_hi_kow = cv_stats(X_kow, y_kow)
print(f"   R\u00b2 (5-fold CV)  : {r2_kow:.4f}")
print(f"   RMSE           : {rmse_kow:.4f}")
print(f"   CCC            : {ccc_kow:.4f}")
print(f"   NRMSE (\u03c3)      : {nrmse_sd_kow:.4f}")
print(f"   NRMSE (range)  : {nrmse_range_kow:.4f}")
print(f"   log\u2081\u2080(BF\u2081\u2080)     : {format_bf10(bf10_kow)}")
print(f"   95 % CI on \u03c1   : [{ci_lo_kow:.3f}, {ci_hi_kow:.3f}]")

# RDKit MolLogP baseline
rows_kow = _read_sdf(S01, "logP")
rdkit_kow = []
y_base    = []
for mol, name, val in rows_kow:
    if mol is None or val is None:
        continue
    try:
        rdkit_kow.append(Descriptors.MolLogP(mol))
        y_base.append(val)
    except Exception:
        pass
r2_base  = r2_score(y_base, rdkit_kow)
rmse_base = float(mean_squared_error(y_base, rdkit_kow) ** 0.5)
print(f"\n   Baseline (RDKit MolLogP):")
print(f"   R²   : {r2_base:.4f}")
print(f"   RMSE : {rmse_base:.4f}")

# ── 2. logKoa on S02 ─────────────────────────────────────────────────────────
print()
print(SEP)
print("2. logKoa — 5-fold CV on S02 training set")
print(SEP)
X_koa, y_koa, names_koa = _build_Xy(S02, "logKoa")
print(f"   Molecules loaded: {len(y_koa)}")
y_cv_koa, r2_koa, rmse_koa, ccc_koa, nrmse_sd_koa, nrmse_range_koa, bf10_koa, ci_lo_koa, ci_hi_koa = cv_stats(X_koa, y_koa)
print(f"   R\u00b2 (5-fold CV)  : {r2_koa:.4f}")
print(f"   RMSE           : {rmse_koa:.4f}")
print(f"   CCC            : {ccc_koa:.4f}")
print(f"   NRMSE (\u03c3)      : {nrmse_sd_koa:.4f}")
print(f"   NRMSE (range)  : {nrmse_range_koa:.4f}")
print(f"   log\u2081\u2080(BF\u2081\u2080)     : {format_bf10(bf10_koa)}")
print(f"   95 % CI on \u03c1   : [{ci_lo_koa:.3f}, {ci_hi_koa:.3f}]")

# ── 3. logKaw on S03 (derived: logKow - logKoa) ───────────────────────────────
print()
print(SEP)
print("3. logKaw — S03 validation (logKow_pred - logKoa_pred vs exp logKaw)")
print(SEP)

try:
    calc = PartitionCalculator()
except FileNotFoundError:
    print("   Model files not found — run scripts/fit_model.py first, then re-run evaluate.py")
    sys.exit(0)

rows_s03 = _read_sdf(S03, "logKaw")
y_kaw_exp, y_kaw_pred = [], []
for mol, name, val in rows_s03:
    if mol is None or val is None:
        continue
    feat = compute_features(mol)
    if feat is None:
        continue
    from kawow.model import _predict_from_json
    logKow_p = _predict_from_json(feat, calc._kow)
    logKoa_p = _predict_from_json(feat, calc._koa)
    logKaw_p = logKow_p - logKoa_p
    y_kaw_exp.append(val)
    y_kaw_pred.append(logKaw_p)

y_kaw_exp  = np.array(y_kaw_exp)
y_kaw_pred = np.array(y_kaw_pred)
print(f"   Molecules in S03: {len(y_kaw_exp)}")
if len(y_kaw_exp) > 1:
    r2_kaw   = r2_score(y_kaw_exp, y_kaw_pred)
    rmse_kaw = float(mean_squared_error(y_kaw_exp, y_kaw_pred) ** 0.5)
    from kawow.metrics import lin_ccc as _lc, nrmse as _nm, jeffreys_bf_corr as _jbf
    ccc_kaw = _lc(y_kaw_exp, y_kaw_pred)
    nm_kaw  = _nm(y_kaw_exp, y_kaw_pred)
    r_kaw   = float(np.corrcoef(y_kaw_exp, y_kaw_pred)[0, 1])
    bf_kaw  = _jbf(r_kaw, len(y_kaw_exp))
    print(f"   R\u00b2           : {r2_kaw:.4f}")
    print(f"   RMSE         : {rmse_kaw:.4f}")
    print(f"   CCC          : {ccc_kaw:.4f}")
    print(f"   NRMSE (\u03c3)    : {nm_kaw['nrmse_sd']:.4f}")
    print(f"   NRMSE (range): {nm_kaw['nrmse_range']:.4f}")
    print(f"   log\u2081\u2080(BF\u2081\u2080)   : {format_bf10(bf_kaw['log10_bf10'])}")
    print(f"   (Paper reports RMSE \u2248 0.67 with full 37K-molecule training set)")
else:
    r2_kaw, rmse_kaw = float("nan"), float("nan")
    print("   Not enough molecules to compute statistics.")

# ── 4. Summary ────────────────────────────────────────────────────────────────
print()
print(SEP)
print("Summary")
print(SEP)
print(f"  logKow  R\u00b2={r2_kow:.3f}  RMSE={rmse_kow:.3f}  CCC={ccc_kow:.3f}  NRMSE\u03c3={nrmse_sd_kow:.3f}  log\u2081\u2080BF\u2081\u2080={format_bf10(bf10_kow)}  (trained on {len(y_kow)} mols)")
print(f"  logKoa  R\u00b2={r2_koa:.3f}  RMSE={rmse_koa:.3f}  CCC={ccc_koa:.3f}  NRMSE\u03c3={nrmse_sd_koa:.3f}  log\u2081\u2080BF\u2081\u2080={format_bf10(bf10_koa)}  (trained on {len(y_koa)} mols)")
if len(y_kaw_exp) > 1:
    print(f"  logKaw  R\u00b2={r2_kaw:.3f}  RMSE={rmse_kaw:.3f}  (S03, {len(y_kaw_exp)} mols, derived)")
print(f"\n  RDKit MolLogP baseline:  R² = {r2_base:.3f}  RMSE = {rmse_base:.3f}")

# ── 5. Quick sanity check ─────────────────────────────────────────────────────
print()
print(SEP)
print("Quick checks on known molecules (literature values in brackets)")
print(SEP)
tests = [
    ("CCCCO",     "1-butanol",       0.88,  4.12, -3.24),
    ("c1ccccc1",  "benzene",         2.13,  4.15, -2.02),
    ("ClC(Cl)(Cl)Cl", "CCl4",        2.83,  4.44, -1.61),
    ("CC(=O)O",   "acetic acid",    -0.17,  6.58, -6.75),
    ("CCOCCO",    "ethylene glycol monoethyl ether", -0.32, 5.70, -6.02),
]
for smi, name, lit_kow, lit_koa, lit_kaw in tests:
    r = calc.predict(smi)
    if r["status"] == "ok":
        print(f"  {name:<35} "
              f"logKow={r['logKow']:+.2f} (lit {lit_kow:+.2f})  "
              f"logKoa={r['logKoa']:+.2f} (lit {lit_koa:+.2f})  "
              f"logKaw={r['logKaw']:+.2f} (lit {lit_kaw:+.2f})")
    else:
        print(f"  {name}: ERROR — {r.get('error')}")
