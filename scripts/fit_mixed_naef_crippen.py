"""Fit a mixed NaefAcree + Crippen SMARTS model and export CSV coefficients.

This script builds two NaefAcree-compatible parameter tables:
- kawow/data/naef2024_logkow_parameters_mixed.csv
- kawow/data/naef2024_logkoa_parameters_mixed.csv

The mixed feature space is:
1) All existing NaefAcree rows from each endpoint table (except Const row as a feature)
2) Appended Crippen SMARTS rows (one feature per Crippen SMARTS line)

Coefficients are fitted with RidgeCV and written back into CSV `Contribution` values,
so predictions can be served by `NaefAcreePartitionCalculator`.

The script also evaluates performance and writes mixed-model correlation plots:
- docs/imgs/corr_mixed_kow_vs_s01.png
- docs/imgs/corr_mixed_koa_vs_s02.png
- docs/imgs/corr_mixed_kow_vs_excel.png
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import SDMolSupplier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from kawow.atom_types import CRIPPEN_ORDER, CRIPPEN_PATTS  # noqa: E402
from kawow.smarts_model import (  # noqa: E402
    NaefAcreePartitionCalculator,
    count_conjugated_neighbor_moieties,
)

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "kawow" / "data"
TEST_DATA_DIR = ROOT / "tests" / "test_data"
OUT_DIR = ROOT / "tests" / "out"
IMG_DIR = ROOT / "docs" / "imgs"

KOW_BASE = DATA_DIR / "naef2024_logkow_parameters.csv"
KOA_BASE = DATA_DIR / "naef2024_logkoa_parameters.csv"
KOW_MIXED = DATA_DIR / "naef2024_logkow_parameters_mixed.csv"
KOA_MIXED = DATA_DIR / "naef2024_logkoa_parameters_mixed.csv"

S01_SDF = TEST_DATA_DIR / "S01. Compounds List for logPow-Parameters Calculations.sdf"
S02_SDF = TEST_DATA_DIR / "S02. Compounds List for logKoa-Parameters Calculations.sdf"
EXCEL_CSV = TEST_DATA_DIR / "vg2c00024_si_001.csv"


def _normalize_name(value: str) -> str:
    value = (value or "").strip().lower()
    return re.sub(r"[^a-z0-9]", "", value)


def _plot_corr(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    mask = np.isfinite(x) & np.isfinite(y)
    xx = x[mask]
    yy = y[mask]

    plt.figure(figsize=(7, 6))
    plt.scatter(xx, yy, s=16, alpha=0.7)

    if len(xx) >= 2:
        p = np.polyfit(xx, yy, 1)
        xfit = np.linspace(float(np.min(xx)), float(np.max(xx)), 200)
        yfit = p[0] * xfit + p[1]
        plt.plot(xfit, yfit, linewidth=1.5, label=f"fit: y={p[0]:.3f}x+{p[1]:.3f}")

    lo = min(float(np.min(xx)), float(np.min(yy))) if len(xx) else -1
    hi = max(float(np.max(xx)), float(np.max(yy))) if len(xx) else 1
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, label="y=x")

    m = _metrics(xx, yy)
    plt.title(f"{title}\nN={m['n']}  R2={m['r2']:.3f}  RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def _metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return {"n": int(mask.sum()), "r2": np.nan, "rmse": np.nan, "mae": np.nan, "pearson": np.nan}
    xx = x[mask]
    yy = y[mask]
    return {
        "n": int(mask.sum()),
        "r2": float(r2_score(xx, yy)),
        "rmse": float(mean_squared_error(xx, yy) ** 0.5),
        "mae": float(mean_absolute_error(xx, yy)),
        "pearson": float(np.corrcoef(xx, yy)[0, 1]),
    }


def _append_crippen_rows(base_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    max_entry = int(pd.to_numeric(base_df["Entry"], errors="coerce").max())
    entry = max_entry

    for type_id in CRIPPEN_ORDER:
        patt_list = CRIPPEN_PATTS[type_id]
        for patt_idx, (smarts_str, _compiled) in enumerate(patt_list, start=1):
            entry += 1
            rows.append(
                {
                    "Entry": entry,
                    "Atom Type": f"Crippen {type_id}",
                    "Neighbours": f"SMARTS {patt_idx}",
                    "Contribution": 0.0,
                    "Occurrences": "",
                    "Molecules": "",
                    "SMARTS": smarts_str,
                    "pi": "",
                    "fnc": "",
                }
            )

    add_df = pd.DataFrame(rows, columns=base_df.columns)
    return pd.concat([base_df.copy(), add_df], ignore_index=True)


def _compile_specs(df: pd.DataFrame) -> tuple[list[dict], int]:
    const_idx = df.index[df["Atom Type"].astype(str) == "Const"]
    if len(const_idx) != 1:
        raise RuntimeError("Expected exactly one Const row")
    const_row = int(const_idx[0])

    specs: list[dict] = []
    for i, row in df.iterrows():
        if i == const_row:
            continue

        smarts_raw = row["SMARTS"]
        fnc_raw = row["fnc"]
        pi_raw = row["pi"]

        smarts = None
        fnc = None
        pi = None

        if pd.notna(smarts_raw) and str(smarts_raw).strip():
            smarts = Chem.MolFromSmarts(str(smarts_raw).strip())
            if smarts is None:
                raise ValueError(f"Invalid SMARTS at row {i}: {smarts_raw}")

        if pd.notna(fnc_raw) and str(fnc_raw).strip():
            fnc_name = str(fnc_raw).strip()
            fnc = getattr(sys.modules["kawow.smarts_model"], fnc_name)

        if pd.notna(pi_raw) and str(pi_raw).strip():
            pi = int(float(pi_raw))

        specs.append({"row_idx": i, "smarts": smarts, "fnc": fnc, "pi": pi})

    return specs, const_row


def _feature_vector(mol: Chem.Mol, specs: list[dict]) -> np.ndarray:
    x = np.zeros(len(specs), dtype=np.float64)

    for j, spec in enumerate(specs):
        smarts = spec["smarts"]
        fnc = spec["fnc"]
        pi = spec["pi"]

        if smarts is not None:
            matches = mol.GetSubstructMatches(smarts)
            if not matches:
                continue
            if pi is None:
                x[j] = float(len(matches))
            else:
                count = 0.0
                for match in matches:
                    center_idx = match[0]
                    n_moieties, _ = count_conjugated_neighbor_moieties(mol, center_idx)
                    if n_moieties == pi:
                        count += 1.0
                x[j] = count
        elif fnc is not None:
            x[j] = float(fnc(mol))
        else:
            x[j] = 0.0

    return x


def _load_training_rows(sdf_path: Path, prop: str) -> list[tuple[Chem.Mol, float]]:
    rows: list[tuple[Chem.Mol, float]] = []
    suppl = SDMolSupplier(str(sdf_path))
    for mol in suppl:
        if mol is None or not mol.HasProp(prop):
            continue
        try:
            y = float(mol.GetProp(prop))
        except Exception:
            continue
        rows.append((mol, y))
    return rows


def _fit_endpoint(df_in: pd.DataFrame, sdf_path: Path, prop: str) -> tuple[pd.DataFrame, dict[str, float], np.ndarray, np.ndarray]:
    df = df_in.copy()
    specs, const_row = _compile_specs(df)

    train_rows = _load_training_rows(sdf_path, prop)
    X = np.vstack([_feature_vector(mol, specs) for mol, _ in train_rows])
    y = np.array([v for _, v in train_rows], dtype=np.float64)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    alphas = np.logspace(-3, 4, 30)

    best_alpha = None
    best_rmse = np.inf
    best_y_cv = None
    for alpha in alphas:
        y_cv_try = np.zeros_like(y, dtype=np.float64)
        for train_idx, test_idx in kf.split(X):
            model_try = make_pipeline(
                StandardScaler(with_mean=False),
                Ridge(alpha=float(alpha), fit_intercept=True, solver="sag", max_iter=8000, tol=1e-4, random_state=42),
            )
            model_try.fit(X[train_idx], y[train_idx])
            y_cv_try[test_idx] = model_try.predict(X[test_idx])

        rmse_try = float(mean_squared_error(y, y_cv_try) ** 0.5)
        if rmse_try < best_rmse:
            best_rmse = rmse_try
            best_alpha = float(alpha)
            best_y_cv = y_cv_try

    if best_alpha is None or best_y_cv is None:
        raise RuntimeError("Unable to fit mixed model: no alpha selected")

    y_cv = best_y_cv
    model = make_pipeline(
        StandardScaler(with_mean=False),
        Ridge(alpha=best_alpha, fit_intercept=True, solver="sag", max_iter=8000, tol=1e-4, random_state=42),
    )
    model.fit(X, y)

    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]
    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    coefs = ridge.coef_ / scale
    intercept = float(ridge.intercept_)

    # Write coefficients back to dataframe
    df.loc[const_row, "Contribution"] = intercept
    for coef, spec in zip(coefs, specs):
        df.loc[spec["row_idx"], "Contribution"] = float(coef)

    # Update Occurrences / Molecules from training counts for transparency
    counts = X
    occ = counts.sum(axis=0)
    mols = (counts > 0).sum(axis=0)
    for j, spec in enumerate(specs):
        df.loc[spec["row_idx"], "Occurrences"] = int(occ[j])
        df.loc[spec["row_idx"], "Molecules"] = int(mols[j])

    metrics = {
        "n": int(len(y)),
        "alpha": float(best_alpha),
        "r2": float(r2_score(y, y_cv)),
        "rmse": float(mean_squared_error(y, y_cv) ** 0.5),
        "mae": float(mean_absolute_error(y, y_cv)),
    }
    return df, metrics, y, y_cv


def _eval_mixed_model() -> dict[str, dict[str, float]]:
    calc = NaefAcreePartitionCalculator(
        logkow_parameter_file=KOW_MIXED.name,
        logkoa_parameter_file=KOA_MIXED.name,
    )

    # S01
    s01_exp = []
    s01_pred = []
    s01_rows = []
    for mol in SDMolSupplier(str(S01_SDF)):
        if mol is None or not mol.HasProp("logP"):
            continue
        exp = float(mol.GetProp("logP"))
        pred = calc.predict(mol)
        s01_exp.append(exp)
        s01_pred.append(float(pred["logKow"]))

        name = mol.GetProp("Alias name") if mol.HasProp("Alias name") else (mol.GetProp("_Name") if mol.HasProp("_Name") else "")
        ik = ""
        try:
            ik = Chem.MolToInchiKey(mol).upper()
        except Exception:
            ik = ""
        s01_rows.append({
            "name": name,
            "name_norm": _normalize_name(name),
            "inchikey": ik,
            "logKow_mixed": float(pred["logKow"]),
            "logKow_exp_s01": exp,
        })

    # S02
    s02_exp = []
    s02_pred = []
    for mol in SDMolSupplier(str(S02_SDF)):
        if mol is None or not mol.HasProp("logKoa"):
            continue
        exp = float(mol.GetProp("logKoa"))
        pred = calc.predict(mol)
        s02_exp.append(exp)
        s02_pred.append(float(pred["logKoa"]))

    # Arp & Hale matching (same logic as existing compare script)
    df_x = pd.read_csv(EXCEL_CSV)
    df_x = df_x[["InChI", "Example Substance name", "logKowlogDow exp est"]].copy()
    df_x.columns = ["inchikey", "name", "logKow_excel"]
    df_x["inchikey"] = df_x["inchikey"].fillna("").astype(str).str.strip().str.upper()
    df_x["name_norm"] = df_x["name"].fillna("").astype(str).map(_normalize_name)
    df_x["logKow_excel"] = pd.to_numeric(df_x["logKow_excel"], errors="coerce")

    df_s01 = pd.DataFrame(s01_rows)
    by_inchi: dict[str, int] = {}
    by_name: dict[str, int] = {}
    for i, row in df_s01.iterrows():
        if row["inchikey"] and row["inchikey"] not in by_inchi:
            by_inchi[row["inchikey"]] = i
        if row["name_norm"] and row["name_norm"] not in by_name:
            by_name[row["name_norm"]] = i

    matched = []
    for _, row in df_x.iterrows():
        idx = None
        mode = ""
        if row["inchikey"] and row["inchikey"] in by_inchi:
            idx = by_inchi[row["inchikey"]]
            mode = "inchikey"
        elif row["name_norm"] and row["name_norm"] in by_name:
            idx = by_name[row["name_norm"]]
            mode = "name"

        if idx is None:
            continue

        srow = df_s01.loc[idx]
        matched.append(
            {
                "name_s01": srow["name"],
                "inchikey": srow["inchikey"],
                "logKow_mixed": srow["logKow_mixed"],
                "logKow_excel": float(row["logKow_excel"]),
                "match_mode": mode,
            }
        )

    df_match = pd.DataFrame(matched)

    s01_m = _metrics(np.array(s01_exp, dtype=float), np.array(s01_pred, dtype=float))
    s02_m = _metrics(np.array(s02_exp, dtype=float), np.array(s02_pred, dtype=float))
    xls_m = _metrics(
        df_match["logKow_excel"].to_numpy(dtype=float),
        df_match["logKow_mixed"].to_numpy(dtype=float),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "mixed_vs_excel_matched.csv").write_text(df_match.to_csv(index=False), encoding="utf-8")

    _plot_corr(
        np.array(s01_exp, dtype=float),
        np.array(s01_pred, dtype=float),
        xlabel="S01 experimental logP",
        ylabel="Mixed model logKow",
        title="Mixed model logKow vs S01",
        out_png=IMG_DIR / "corr_mixed_kow_vs_s01.png",
    )
    _plot_corr(
        np.array(s02_exp, dtype=float),
        np.array(s02_pred, dtype=float),
        xlabel="S02 experimental logKoa",
        ylabel="Mixed model logKoa",
        title="Mixed model logKoa vs S02",
        out_png=IMG_DIR / "corr_mixed_koa_vs_s02.png",
    )
    _plot_corr(
        df_match["logKow_excel"].to_numpy(dtype=float),
        df_match["logKow_mixed"].to_numpy(dtype=float),
        xlabel="Arp & Hale logKow",
        ylabel="Mixed model logKow",
        title="Mixed model logKow vs Arp & Hale",
        out_png=IMG_DIR / "corr_mixed_kow_vs_excel.png",
    )

    return {"s01": s01_m, "s02": s02_m, "excel": xls_m}


def main() -> None:
    kow_df_base = pd.read_csv(KOW_BASE)
    koa_df_base = pd.read_csv(KOA_BASE)

    kow_df_mix = _append_crippen_rows(kow_df_base)
    koa_df_mix = _append_crippen_rows(koa_df_base)

    kow_df_fit, kow_cv, _, _ = _fit_endpoint(kow_df_mix, S01_SDF, "logP")
    koa_df_fit, koa_cv, _, _ = _fit_endpoint(koa_df_mix, S02_SDF, "logKoa")

    kow_df_fit.to_csv(KOW_MIXED, index=False)
    koa_df_fit.to_csv(KOA_MIXED, index=False)

    eval_metrics = _eval_mixed_model()

    summary = {
        "fit_cv": {"logKow": kow_cv, "logKoa": koa_cv},
        "eval": eval_metrics,
        "files": {
            "logkow_csv": str(KOW_MIXED),
            "logkoa_csv": str(KOA_MIXED),
            "plot_kow_s01": str(IMG_DIR / "corr_mixed_kow_vs_s01.png"),
            "plot_koa_s02": str(IMG_DIR / "corr_mixed_koa_vs_s02.png"),
            "plot_kow_excel": str(IMG_DIR / "corr_mixed_kow_vs_excel.png"),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "mixed_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Mixed NaefAcree + Crippen fit complete ===")
    print(f"logKow CV: R2={kow_cv['r2']:.4f}, RMSE={kow_cv['rmse']:.4f}, MAE={kow_cv['mae']:.4f}, alpha={kow_cv['alpha']:.4g}")
    print(f"logKoa CV: R2={koa_cv['r2']:.4f}, RMSE={koa_cv['rmse']:.4f}, MAE={koa_cv['mae']:.4f}, alpha={koa_cv['alpha']:.4g}")
    print(f"S01 eval: R2={eval_metrics['s01']['r2']:.4f}, RMSE={eval_metrics['s01']['rmse']:.4f}, MAE={eval_metrics['s01']['mae']:.4f}")
    print(f"S02 eval: R2={eval_metrics['s02']['r2']:.4f}, RMSE={eval_metrics['s02']['rmse']:.4f}, MAE={eval_metrics['s02']['mae']:.4f}")
    print(f"Excel eval: N={eval_metrics['excel']['n']}, R2={eval_metrics['excel']['r2']:.4f}, RMSE={eval_metrics['excel']['rmse']:.4f}, MAE={eval_metrics['excel']['mae']:.4f}")


if __name__ == "__main__":
    main()
