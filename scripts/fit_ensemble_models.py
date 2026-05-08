"""Fit and benchmark ensemble models that concatenate Naef, Crippen and MQG features.

Three ensemble types are evaluated:
  naef_mqg         -- Naef group counts  +  MQG selected features
  crippen_mqg      -- Crippen atom-type counts (85-dim)  +  MQG selected features
  naef_crippen_mqg -- Naef  +  Crippen  +  MQG selected features

Each ensemble uses a StandardScaler + Ridge (with RidgeCV alpha selection via
5-fold CV).  The MQG feature column selection inherited from the existing pure
MQG pkl is kept fixed (not re-tuned).

Usage:
    python scripts/fit_ensemble_models.py

Output artefacts (in kawow/data/):
    logkow_naef_mqg_model.pkl
    logkoa_naef_mqg_model.pkl
    logkow_crippen_mqg_model.pkl
    logkoa_crippen_mqg_model.pkl
    logkow_naef_crippen_mqg_model.pkl
    logkoa_naef_crippen_mqg_model.pkl
    logkow_mqg_model_pure_backup.pkl   (backup of original pure-MQG pkl)
    logkoa_mqg_model_pure_backup.pkl

Output report (in kawow/tests/out/):
    ensemble_performance_summary.csv

Output figures (in kawow/docs/imgs/):
    ensemble_overall_comparison.png
    ensemble_ionization_comparison.png

After running, the WINNER (highest combined R² across logKow + logKoa) is
copied over logkow/logkoa_mqg_model.pkl so that MQGPartitionCalculator
automatically uses the best ensemble.
"""

from __future__ import annotations

import math
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDMolSupplier
from rdkit import RDLogger
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kawow.model import (                                         # noqa: E402
    _build_Xy,
    _build_Xy_mqg,
    _compile_naef_patterns,
    _compute_naef_group_counts,
    _compute_mqg_features_with_ratios,
    _load_pickle,
    DATA_DIR,
)
from kawow.features import compute_features                       # noqa: E402

RDLogger.DisableLog("rdApp.*")

SDF_DIR   = REPO_ROOT / "tests" / "test_data"
S01_SDF   = SDF_DIR / "S01. Compounds List for logPow-Parameters Calculations.sdf"
S02_SDF   = SDF_DIR / "S02. Compounds List for logKoa-Parameters Calculations.sdf"
OUT_DIR   = REPO_ROOT / "tests" / "out"
IMGS_DIR  = REPO_ROOT / "docs" / "imgs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMGS_DIR.mkdir(parents=True, exist_ok=True)

NAEF_KOW_CSV = DATA_DIR / "naef2024_logkow_parameters.csv"
NAEF_KOA_CSV = DATA_DIR / "naef2024_logkoa_parameters.csv"

ENSEMBLE_TYPES = ["naef_mqg", "crippen_mqg", "naef_crippen_mqg"]
EXISTING_MODELS = ["kawow", "smarts", "smarts_mixed", "mqg"]

# ── ionisation helpers (reused from evaluate_by_ionization.py) ────────────────

from dataclasses import dataclass

@dataclass(frozen=True)
class GroupRule:
    name: str
    smarts: str
    pka: float


_ACID_RULES = [
    GroupRule("carboxylic_acid", "[CX3](=O)[OX2H1]", 4.5),
    GroupRule("sulfonic_acid", "[SX4](=O)(=O)[OX2H1]", -1.0),
    GroupRule("phosphonic_acid", "[PX4](=O)([OX2H1])[OX2H1]", 2.0),
    GroupRule("phosphate_monoester", "[PX4](=O)([OX2H1])[OX2][#6]", 2.2),
    GroupRule("tetrazole", "[n]1[n][n][n][c]1", 4.8),
    GroupRule("phenol", "[c][OX2H1]", 10.0),
    GroupRule("thiol", "[#16X2H1]", 10.5),
    GroupRule("imide", "[NX3H][CX3](=O)[#6][CX3](=O)", 8.5),
]
_BASE_RULES = [
    GroupRule("guanidine", "NC(=N)N", 13.5),
    GroupRule("amidine", "NC(=N)[#6]", 12.0),
    GroupRule("aliphatic_amine", "[NX3;H2,H1,H0;!$(N-C=O);!$(N-S(=O)=O);!$(N-c)]", 10.0),
    GroupRule("aniline_like", "[NX3;H2,H1,H0]-c", 5.0),
    GroupRule("pyridine_like", "[nH0;r6]", 5.2),
    GroupRule("imidazole_like", "[nH0;r5]", 7.0),
]

def _compile_ion_rules(rules):
    return [(r, Chem.MolFromSmarts(r.smarts)) for r in rules]

_COMPILED_ACIDS = _compile_ion_rules(_ACID_RULES)
_COMPILED_BASES = _compile_ion_rules(_BASE_RULES)


def _estimate_ion_class(mol, ph: float = 7.0) -> str:
    def _acid_frac(pka): return 1.0 / (1.0 + 10.0 ** (pka - ph))
    def _base_frac(pka): return 1.0 / (1.0 + 10.0 ** (ph - pka))

    net = 0.0
    found = False
    for rule, patt in _COMPILED_ACIDS:
        n = len(mol.GetSubstructMatches(patt, uniquify=True))
        if n:
            net -= n * _acid_frac(rule.pka)
            found = True
    for rule, patt in _COMPILED_BASES:
        n = len(mol.GetSubstructMatches(patt, uniquify=True))
        if n:
            net += n * _base_frac(rule.pka)
            found = True

    if net > 0.25:
        return "basic"
    if net < -0.25:
        return "acid"
    return "neutral"


# ── feature builders ──────────────────────────────────────────────────────────

def _load_sdf_mols(sdf_path: Path, value_prop: str):
    """Yield (mol, name, value) tuples from SDF, skipping invalid entries."""
    suppl = SDMolSupplier(str(sdf_path), removeHs=True)
    for mol in suppl:
        if mol is None:
            continue
        name = ""
        if mol.HasProp("Alias name"):
            name = mol.GetProp("Alias name")
        elif mol.HasProp("_Name"):
            name = mol.GetProp("_Name")
        val = None
        if mol.HasProp(value_prop):
            try:
                val = float(mol.GetProp(value_prop))
            except (ValueError, TypeError):
                val = None
        if val is not None:
            yield mol, name, val


def _build_all_features(
    sdf_path: Path,
    value_prop: str,
    naef_patterns: list,
    mqg_feature_cols: list,
    fp_size: int = 64,
) -> dict:
    """Build Crippen, Naef, and MQG feature vectors for all valid molecules.

    Returns a dict with keys:
        names, y, ion_class,
        X_crippen, X_naef, X_mqg_sel  (all aligned to same molecules)
    """
    names, ys, ion_classes = [], [], []
    X_crippen, X_naef, X_mqg = [], [], []

    for mol, name, val in _load_sdf_mols(sdf_path, value_prop):
        # Crippen features
        f_crippen = compute_features(mol)
        if f_crippen is None:
            continue
        # Naef group counts
        f_naef = _compute_naef_group_counts(mol, naef_patterns)
        # MQG features
        f_mqg_full = _compute_mqg_features_with_ratios(mol, fp_size=fp_size)
        if f_mqg_full is None:
            continue
        f_mqg_sel = f_mqg_full[mqg_feature_cols]

        ion = _estimate_ion_class(mol)

        names.append(name)
        ys.append(val)
        ion_classes.append(ion)
        X_crippen.append(f_crippen.astype(np.float32))
        X_naef.append(f_naef)
        X_mqg.append(f_mqg_sel)

    return {
        "names": names,
        "y": np.array(ys, dtype=np.float64),
        "ion_class": ion_classes,
        "X_crippen": np.array(X_crippen, dtype=np.float32),
        "X_naef": np.array(X_naef, dtype=np.float32),
        "X_mqg_sel": np.array(X_mqg, dtype=np.float32),
    }


# ── fitting helpers ───────────────────────────────────────────────────────────

def _fit_ridge_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    target: str,
    ensemble_type: str,
    naef_param_file: str | None,
    mqg_feature_cols: list,
    fp_size: int = 64,
) -> dict:
    """Fit StandardScaler + RidgeCV, 5-fold CV, save pkl, return payload."""
    alphas = np.logspace(-3, 4, 50)
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, cv=5, fit_intercept=True),
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_cv = cross_val_predict(model, X, y, cv=kf)
    r2   = float(r2_score(y, y_cv))
    rmse = float(math.sqrt(mean_squared_error(y, y_cv)))
    model.fit(X, y)

    payload = {
        "target": target,
        "method": "ridge_ensemble",
        "ensemble_type": ensemble_type,
        "naef_param_file": naef_param_file,
        "mqg_feature_cols": mqg_feature_cols,
        "fp_size": fp_size,
        "n_train": int(len(y)),
        "n_features": int(X.shape[1]),
        "r2_cv": round(r2, 4),
        "rmse_cv": round(rmse, 4),
        "model": model,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(
        f"  [{target} {ensemble_type}] n={len(y)}  R²={r2:.4f}  RMSE={rmse:.4f}"
        f"  ({X.shape[1]} features)  → {out_path.name}"
    )
    return payload


def _cv_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(mask.sum())
    if n < 2:
        return {"n": n, "r2": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "n": n,
        "r2": float(r2_score(yt, yp)),
        "rmse": float(math.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
    }


# ── load existing model predictions for comparison ───────────────────────────

def _load_existing_preds(endpoint_key: str) -> dict[str, np.ndarray] | None:
    """Try to load previously computed benchmark CSVs for existing models."""
    # endpoint_key: "logKow" → looks for s01_*_vs_experimental.csv
    dataset = "s01" if "Kow" in endpoint_key else "s02"
    model_map = {
        "kawow": f"{dataset}_kawow_vs_experimental.csv",
        "smarts": f"{dataset}_smarts_vs_experimental.csv",
        "smarts_mixed": f"{dataset}_smarts_m_vs_experimental.csv",
        "mqg": f"{dataset}_mqg_vs_experimental.csv",
    }
    preds = {}
    for model, fname in model_map.items():
        fpath = OUT_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            col = "predicted" if "predicted" in df.columns else df.columns[-1]
            preds[model] = df[col].to_numpy(dtype=float)
    return preds if preds else None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Ensemble model fitting for kawow")
    print("=" * 70)

    # ── 1. Load existing MQG pkls to extract feature_cols ─────────────────
    print("\n[1] Loading existing MQG pkls ...")
    mqg_kow_pkl = _load_pickle(DATA_DIR / "logkow_mqg_model.pkl")
    mqg_koa_pkl = _load_pickle(DATA_DIR / "logkoa_mqg_model.pkl")
    kow_feature_cols: list = mqg_kow_pkl["feature_cols"]
    koa_feature_cols: list = mqg_koa_pkl["feature_cols"]
    fp_size: int = int(mqg_kow_pkl.get("fp_size", 64))
    print(f"  logKow MQG: {len(kow_feature_cols)} selected features, fp_size={fp_size}")
    print(f"  logKoa MQG: {len(koa_feature_cols)} selected features")

    # ── 2. Compile Naef patterns once ─────────────────────────────────────
    print("\n[2] Compiling Naef SMARTS patterns ...")
    naef_kow_patterns = _compile_naef_patterns(NAEF_KOW_CSV)
    naef_koa_patterns = _compile_naef_patterns(NAEF_KOA_CSV)
    print(f"  logKow: {len(naef_kow_patterns)} patterns")
    print(f"  logKoa: {len(naef_koa_patterns)} patterns")

    # ── 3. Build feature matrices ─────────────────────────────────────────
    print("\n[3] Computing features for S01 (logKow) ...")
    kow_data = _build_all_features(
        S01_SDF, "logP", naef_kow_patterns, kow_feature_cols, fp_size=fp_size
    )
    print(f"  S01: {len(kow_data['y'])} molecules retained")

    print("[3] Computing features for S02 (logKoa) ...")
    koa_data = _build_all_features(
        S02_SDF, "logKoa", naef_koa_patterns, koa_feature_cols, fp_size=fp_size
    )
    print(f"  S02: {len(koa_data['y'])} molecules retained")

    # ── 4. Build combined X matrices for each ensemble type ───────────────
    def _build_X(data: dict, etype: str) -> np.ndarray:
        parts = []
        if "naef" in etype:
            parts.append(data["X_naef"])
        if "crippen" in etype:
            parts.append(data["X_crippen"])
        parts.append(data["X_mqg_sel"])
        return np.hstack(parts).astype(np.float32)

    # ── 5. Fit ensemble models ─────────────────────────────────────────────
    print("\n[4] Fitting ensemble models ...")
    ensemble_payloads: dict[str, dict] = {}

    for etype in ENSEMBLE_TYPES:
        print(f"\n  -- {etype} --")
        X_kow = _build_X(kow_data, etype)
        X_koa = _build_X(koa_data, etype)

        naef_kow_path = str(NAEF_KOW_CSV) if "naef" in etype else None
        naef_koa_path = str(NAEF_KOA_CSV) if "naef" in etype else None

        pkl_kow = _fit_ridge_ensemble(
            X_kow, kow_data["y"],
            DATA_DIR / f"logkow_{etype}_model.pkl",
            "logKow", etype,
            naef_kow_path, kow_feature_cols, fp_size=fp_size,
        )
        pkl_koa = _fit_ridge_ensemble(
            X_koa, koa_data["y"],
            DATA_DIR / f"logkoa_{etype}_model.pkl",
            "logKoa", etype,
            naef_koa_path, koa_feature_cols, fp_size=fp_size,
        )
        ensemble_payloads[etype] = {"kow": pkl_kow, "koa": pkl_koa}

    # ── 6. Ionisation-stratified CV evaluation ─────────────────────────────
    print("\n[5] Ionisation-stratified evaluation ...")
    rows: list[dict] = []

    def _eval_stratified(data: dict, model: str, endpoint: str, dataset: str, y_cv: np.ndarray):
        y_true = data["y"]
        for ion_class in ["acid", "neutral", "basic"]:
            idx = [i for i, c in enumerate(data["ion_class"]) if c == ion_class]
            if not idx:
                continue
            m = _cv_metrics(y_true[idx], y_cv[idx])
            rows.append({
                "dataset": dataset,
                "endpoint": endpoint,
                "model": model,
                "ion_class": ion_class,
                **m,
            })

    for etype in ENSEMBLE_TYPES:
        pkl_kow = ensemble_payloads[etype]["kow"]
        pkl_koa = ensemble_payloads[etype]["koa"]

        X_kow = _build_X(kow_data, etype)
        X_koa = _build_X(koa_data, etype)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        y_cv_kow = cross_val_predict(pkl_kow["model"], X_kow, kow_data["y"], cv=kf)
        y_cv_koa = cross_val_predict(pkl_koa["model"], X_koa, koa_data["y"], cv=kf)

        _eval_stratified(kow_data, etype, "logKow", "S01", y_cv_kow)
        _eval_stratified(koa_data, etype, "logKoa", "S02", y_cv_koa)

    ion_df = pd.DataFrame(rows)

    # ── 7. Overall performance table ─────────────────────────────────────
    print("\n[6] Overall performance summary ...")
    overall_rows: list[dict] = []

    # Ensemble models
    for etype in ENSEMBLE_TYPES:
        for endpoint, data, pkl_key in [
            ("logKow", kow_data, "kow"),
            ("logKoa", koa_data, "koa"),
        ]:
            pl = ensemble_payloads[etype][pkl_key]
            overall_rows.append({
                "model": etype,
                "endpoint": endpoint,
                "n": pl["n_train"],
                "r2_cv": pl["r2_cv"],
                "rmse_cv": pl["rmse_cv"],
            })

    # Existing models (from saved CSV benchmarks if available)
    for model in EXISTING_MODELS:
        for endpoint, dataset in [("logKow", "s01"), ("logKoa", "s02")]:
            fpath = OUT_DIR / f"{dataset}_{model}_vs_experimental.csv"
            if not fpath.exists():
                # Try alternative naming
                fpath = OUT_DIR / f"{dataset}_smarts_m_vs_experimental.csv" if model == "smarts_mixed" else fpath
            if fpath.exists():
                df = pd.read_csv(fpath)
                # All existing benchmark CSVs have: name,name_norm,inchikey,smiles,logX_exp_sNN,logX_model
                col_pred = df.columns[-1]
                col_exp = df.columns[-2]
                y_true = df[col_exp].to_numpy(dtype=float)
                y_pred = df[col_pred].to_numpy(dtype=float)
                mask = np.isfinite(y_true) & np.isfinite(y_pred)
                yt, yp = y_true[mask], y_pred[mask]
                r2 = float(r2_score(yt, yp)) if mask.sum() >= 2 else float("nan")
                rmse = float(math.sqrt(mean_squared_error(yt, yp))) if mask.sum() >= 2 else float("nan")
                overall_rows.append({
                    "model": model,
                    "endpoint": endpoint,
                    "n": int(mask.sum()),
                    "r2_cv": round(r2, 4),
                    "rmse_cv": round(rmse, 4),
                })

    overall_df = pd.DataFrame(overall_rows)
    print("\nOverall model performance (5-fold CV R²):")
    print(overall_df.pivot(index="model", columns="endpoint", values="r2_cv").to_string())

    # ── 8. Choose winner ──────────────────────────────────────────────────
    print("\n[7] Selecting winner ...")
    best_model, best_combined_r2 = None, -np.inf
    for etype in ENSEMBLE_TYPES:
        r2_kow = ensemble_payloads[etype]["kow"]["r2_cv"]
        r2_koa = ensemble_payloads[etype]["koa"]["r2_cv"]
        combined = r2_kow + r2_koa
        print(f"  {etype}: R²_kow={r2_kow:.4f}  R²_koa={r2_koa:.4f}  sum={combined:.4f}")
        if combined > best_combined_r2:
            best_combined_r2 = combined
            best_model = etype

    # Penalise complexity: prefer simpler model if within 0.005 of best
    SIMPLICITY_THRESHOLD = 0.005
    preference_order = ["crippen_mqg", "naef_mqg", "naef_crippen_mqg"]
    for simpler in preference_order:
        if simpler == best_model:
            break
        r2s = (
            ensemble_payloads[simpler]["kow"]["r2_cv"]
            + ensemble_payloads[simpler]["koa"]["r2_cv"]
        )
        if best_combined_r2 - r2s <= SIMPLICITY_THRESHOLD:
            print(f"  Preferring simpler '{simpler}' over '{best_model}' (gap ≤ {SIMPLICITY_THRESHOLD})")
            best_model = simpler
            break

    print(f"\n  ✓ Winner: {best_model}")

    # ── 9. Backup existing MQG pkls and replace with winner ───────────────
    print("\n[8] Backing up existing MQG pkls ...")
    for stem in ["logkow_mqg_model", "logkoa_mqg_model"]:
        src = DATA_DIR / f"{stem}.pkl"
        dst = DATA_DIR / f"{stem}_pure_backup.pkl"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  {src.name} → {dst.name}")

    print(f"[9] Replacing mqg pkls with winner ({best_model}) ...")
    shutil.copy2(DATA_DIR / f"logkow_{best_model}_model.pkl", DATA_DIR / "logkow_mqg_model.pkl")
    shutil.copy2(DATA_DIR / f"logkoa_{best_model}_model.pkl", DATA_DIR / "logkoa_mqg_model.pkl")
    print("  Done.")

    # ── 10. Save report CSV ───────────────────────────────────────────────
    print("\n[10] Saving performance report ...")
    report_path = OUT_DIR / "ensemble_performance_summary.csv"
    summary_rows: list[dict] = []
    for etype in ENSEMBLE_TYPES:
        for key, endpoint in [("kow", "logKow"), ("koa", "logKoa")]:
            pl = ensemble_payloads[etype][key]
            summary_rows.append({
                "model": etype,
                "endpoint": endpoint,
                "ion_class": "overall",
                "n": pl["n_train"],
                "r2_cv": pl["r2_cv"],
                "rmse_cv": pl["rmse_cv"],
                "is_winner": etype == best_model,
            })

    for _, row in ion_df.iterrows():
        summary_rows.append({
            "model": row["model"],
            "endpoint": row["endpoint"],
            "ion_class": row["ion_class"],
            "n": row["n"],
            "r2_cv": round(float(row["r2"]), 4),
            "rmse_cv": round(float(row["rmse"]), 4),
            "is_winner": row["model"] == best_model,
        })

    pd.DataFrame(summary_rows).to_csv(report_path, index=False)
    print(f"  Saved: {report_path}")

    # ── 11. Plots ─────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        print("\n[11] Generating comparison figures ...")

        # ── Overall comparison ──
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
        colors = {"naef_mqg": "#4c84c4", "crippen_mqg": "#e07b3f", "naef_crippen_mqg": "#5aad6a"}
        hatch = {"naef_mqg": "", "crippen_mqg": "//", "naef_crippen_mqg": "xx"}

        for ax_idx, (endpoint, row_key) in enumerate([("logKow", "kow"), ("logKoa", "koa")]):
            ax = axes[ax_idx]
            r2_vals = [ensemble_payloads[e][row_key]["r2_cv"] for e in ENSEMBLE_TYPES]
            bars = ax.bar(
                ENSEMBLE_TYPES, r2_vals,
                color=[colors[e] for e in ENSEMBLE_TYPES],
                hatch=[hatch[e] for e in ENSEMBLE_TYPES],
                edgecolor="black", linewidth=0.7,
            )
            # Mark winner
            winner_idx = ENSEMBLE_TYPES.index(best_model)
            ax.bar(
                best_model, r2_vals[winner_idx],
                color=colors[best_model], edgecolor="gold",
                linewidth=2.5, hatch=hatch[best_model],
            )
            # Existing MQG reference line
            sub = overall_df[(overall_df["model"] == "mqg") & (overall_df["endpoint"] == endpoint)]
            if not sub.empty:
                ax.axhline(float(sub["r2_cv"].iloc[0]), linestyle="--", color="gray",
                           linewidth=1, label="pure MQG")
                ax.legend(fontsize=8)
            for bar, val in zip(bars, r2_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_title(endpoint, fontsize=11)
            ax.set_ylabel("R² (5-fold CV)")
            ax.set_ylim(max(0, min(r2_vals) - 0.05), min(1, max(r2_vals) + 0.06))
            ax.set_xticklabels([e.replace("_", "\n") for e in ENSEMBLE_TYPES], fontsize=8)

        fig.suptitle("Ensemble Models — Overall CV Performance", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(IMGS_DIR / "ensemble_overall_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {IMGS_DIR / 'ensemble_overall_comparison.png'}")

        # ── Ionisation-stratified comparison ──
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
        ion_colors = {"acid": "#d96c6c", "neutral": "#5b8ec9", "basic": "#4cab6f"}
        ion_classes = ["acid", "neutral", "basic"]

        for row_idx, (endpoint, dataset) in enumerate([("logKow", "S01"), ("logKoa", "S02")]):
            for col_idx, ion_class in enumerate(ion_classes):
                ax = axes[row_idx][col_idx]
                sub = ion_df[
                    (ion_df["endpoint"] == endpoint) & (ion_df["ion_class"] == ion_class)
                ]
                r2_vals = [
                    float(sub[sub["model"] == e]["r2"].iloc[0]) if not sub[sub["model"] == e].empty else float("nan")
                    for e in ENSEMBLE_TYPES
                ]
                ax.bar(
                    ENSEMBLE_TYPES, r2_vals,
                    color=ion_colors[ion_class],
                    edgecolor="black", linewidth=0.7,
                )
                ax.set_title(f"{endpoint} — {ion_class}", fontsize=9)
                ax.set_ylabel("R²", fontsize=8)
                ax.set_xticklabels([e.replace("_", "\n") for e in ENSEMBLE_TYPES], fontsize=7)
                for i, val in enumerate(r2_vals):
                    if np.isfinite(val):
                        ax.text(i, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        fig.suptitle("Ensemble Models — Ionisation-Stratified CV Performance", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(IMGS_DIR / "ensemble_ionization_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {IMGS_DIR / 'ensemble_ionization_comparison.png'}")

    except ImportError:
        print("  matplotlib not available — skipping figures.")

    # ── Print final summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"WINNER: {best_model}")
    w = ensemble_payloads[best_model]
    print(f"  logKow R²={w['kow']['r2_cv']:.4f}  RMSE={w['kow']['rmse_cv']:.4f}")
    print(f"  logKoa R²={w['koa']['r2_cv']:.4f}  RMSE={w['koa']['rmse_cv']:.4f}")
    print(f"\nlogkow/logkoa_mqg_model.pkl now contains the {best_model} ensemble.")
    print("Run 'python scripts/evaluate_by_ionization.py' to regenerate the full ionisation report.")
    print("=" * 70)

    return best_model, ensemble_payloads


if __name__ == "__main__":
    main()
