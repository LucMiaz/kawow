"""Compare SMARTS-model endpoints against S01 (logKow) and S02 (logKoa).

Outputs are written to tests/out/ by default:
- s01_smarts_vs_experimental.csv
- s02_smarts_vs_experimental.csv
- excel_vs_s01_experimental.csv
- smarts_vs_excel_matched.csv
- corr_smarts_kow_vs_s01.png
- corr_smarts_koa_vs_s02.png
- corr_smarts_kow_vs_excel.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import SDMolSupplier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from kawow.smarts_model import NaefAcreePartitionCalculator  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def _normalize_name(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]", "", value)
    return value


def _pick_column(columns: list[str], patterns: list[str]) -> str | None:
    lowered = [(c, str(c).lower().strip()) for c in columns]
    for pat in patterns:
        rgx = re.compile(pat)
        for raw, low in lowered:
            if rgx.search(low):
                return raw
    return None


def _predict_partitions(calc: NaefAcreePartitionCalculator, mol: Chem.Mol) -> dict[str, float]:
    out = calc.parse(mol)
    if isinstance(out, dict):
        hit = out.get(mol)
        if isinstance(hit, dict) and "logKow" in hit and "logKoa" in hit:
            return {"logKow": float(hit["logKow"]), "logKoa": float(hit["logKoa"])}
        if "logKow" in out and "logKoa" in out:
            return {"logKow": float(out["logKow"]), "logKoa": float(out["logKoa"])}
        for value in out.values():
            if isinstance(value, dict) and "logKow" in value and "logKoa" in value:
                return {"logKow": float(value["logKow"]), "logKoa": float(value["logKoa"])}
    raise RuntimeError("Could not extract logKow/logKoa from SMARTS model output")


def _load_endpoint_sdf(sdf_path: Path, exp_prop: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    suppl = SDMolSupplier(str(sdf_path))
    for mol in suppl:
        if mol is None:
            continue

        name = ""
        if mol.HasProp("Alias name"):
            name = mol.GetProp("Alias name")
        elif mol.HasProp("_Name"):
            name = mol.GetProp("_Name")

        inchikey = ""
        try:
            inchikey = Chem.MolToInchiKey(mol)
        except Exception:
            inchikey = ""

        exp_value = np.nan
        if mol.HasProp(exp_prop):
            try:
                exp_value = float(mol.GetProp(exp_prop))
            except Exception:
                exp_value = np.nan

        rows.append(
            {
                "name": name,
                "name_norm": _normalize_name(name),
                "inchikey": inchikey.upper() if inchikey else "",
                "smiles": Chem.MolToSmiles(mol),
                "exp_value": exp_value,
                "mol": mol,
            }
        )
    return rows


def _metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
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
    plt.savefig(out_png, dpi=180)
    plt.close()


def _has_plot_data(x: np.ndarray, y: np.ndarray) -> bool:
    return int((np.isfinite(x) & np.isfinite(y)).sum()) >= 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SMARTS model outputs to S01 (logKow) and S02 (logKoa).")
    parser.add_argument(
        "--s01",
        default=str(Path(__file__).parent.parent / "tests" / "test_data" / "S01. Compounds List for logPow-Parameters Calculations.sdf"),
        help="Path to S01 SDF file with experimental logKow/logP values.",
    )
    parser.add_argument(
        "--s02",
        default=str(Path(__file__).parent.parent / "tests" / "test_data" / "S02. Compounds List for logKoa-Parameters Calculations.sdf"),
        help="Path to S02 SDF file with experimental logKoa values.",
    )
    parser.add_argument(
        "--s01-kow-prop",
        default="logP",
        help="SDF property name for experimental logKow/logP in the S01 file.",
    )
    parser.add_argument(
        "--s02-koa-prop",
        default="logKoa",
        help="SDF property name for experimental logKoa in the S02 file.",
    )
    parser.add_argument(
        "--xlsx",
        default=str(Path(__file__).parent.parent / "tests" / "test_data" / "vg2c00024_si_001.csv"),
        help="Path to supplementary Excel file used for logKow comparison.",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Optional sheet name to read from Excel. If omitted, all sheets are scanned.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).parent.parent / "tests" / "out"),
        help="Output directory for CSVs and plots.",
    )
    args = parser.parse_args()

    s01_path = Path(args.s01)
    s02_path = Path(args.s02)
    xlsx_path = Path(args.xlsx)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not s01_path.exists():
        raise SystemExit(f"S01 file not found: {s01_path}")
    if not s02_path.exists():
        raise SystemExit(f"S02 file not found: {s02_path}")
    if not xlsx_path.exists():
        raise SystemExit(f"Excel file not found: {xlsx_path}")

    calc = NaefAcreePartitionCalculator()

    s01_rows = _load_endpoint_sdf(s01_path, args.s01_kow_prop)
    for row in s01_rows:
        try:
            pred = _predict_partitions(calc, row["mol"])
            row["logKow_smarts"] = pred["logKow"]
        except Exception:
            row["logKow_smarts"] = np.nan

    s02_rows = _load_endpoint_sdf(s02_path, args.s02_koa_prop)
    for row in s02_rows:
        try:
            pred = _predict_partitions(calc, row["mol"])
            row["logKoa_smarts"] = pred["logKoa"]
        except Exception:
            row["logKoa_smarts"] = np.nan

    import pandas as pd

    df_s01 = pd.DataFrame(
        [
            {
                "name": r["name"],
                "name_norm": r["name_norm"],
                "inchikey": r["inchikey"],
                "smiles": r["smiles"],
                "logKow_exp_s01": r["exp_value"],
                "logKow_smarts": r["logKow_smarts"],
            }
            for r in s01_rows
        ]
    )

    df_s02 = pd.DataFrame(
        [
            {
                "name": r["name"],
                "name_norm": r["name_norm"],
                "inchikey": r["inchikey"],
                "smiles": r["smiles"],
                "logKoa_exp_s02": r["exp_value"],
                "logKoa_smarts": r["logKoa_smarts"],
            }
            for r in s02_rows
        ]
    )

    ### mixed model
    calc = NaefAcreePartitionCalculator(logkoa_parameter_file="naef2024_logkoa_parameters_mixed.csv", logkow_parameter_file="naef2024_logkow_parameters_mixed.csv")

    s01_rows_m = _load_endpoint_sdf(s01_path, args.s01_kow_prop)
    for row in s01_rows_m:
        try:
            pred = _predict_partitions(calc, row["mol"])
            row["logKow_smarts"] = pred["logKow"]
        except Exception:
            row["logKow_smarts"] = np.nan

    s02_rows_m = _load_endpoint_sdf(s02_path, args.s02_koa_prop)
    for row in s02_rows_m:
        try:
            pred = _predict_partitions(calc, row["mol"])
            row["logKoa_smarts"] = pred["logKoa"]
        except Exception:
            row["logKoa_smarts"] = np.nan
    df_s01_m = pd.DataFrame(
        [
            {
                "name": r["name"],
                "name_norm": r["name_norm"],
                "inchikey": r["inchikey"],
                "smiles": r["smiles"],
                "logKow_exp_s01": r["exp_value"],
                "logKow_smarts": r["logKow_smarts"],
            }
            for r in s01_rows_m
        ]
    )

    df_s02_m = pd.DataFrame(
        [
            {
                "name": r["name"],
                "name_norm": r["name_norm"],
                "inchikey": r["inchikey"],
                "smiles": r["smiles"],
                "logKoa_exp_s02": r["exp_value"],
                "logKoa_smarts": r["logKoa_smarts"],
            }
            for r in s02_rows_m
        ]
    )

    # Load Excel and detect best sheet/columns for logKow comparison
    df_xlsx = pd.read_csv(xlsx_path)
    col_inchi = "InChI"
    col_name = "Example Substance name"
    col_logkow = "logKowlogDow exp est"


    keep = [c for c in [col_inchi, col_name, col_logkow] if c is not None]
    df_x = df_xlsx[keep].copy()
    df_x = df_x.rename(columns={col_logkow: "logKow_excel"})

    if col_inchi is not None:
        df_x = df_x.rename(columns={col_inchi: "inchikey"})
        df_x["inchikey"] = df_x["inchikey"].fillna("").astype(str).str.strip().str.upper()
    else:
        df_x["inchikey"] = ""

    if col_name is not None:
        df_x = df_x.rename(columns={col_name: "name"})
        df_x["name"] = df_x["name"].fillna("").astype(str)
    else:
        df_x["name"] = ""

    df_x["name_norm"] = df_x["name"].map(_normalize_name)
    df_x["logKow_excel"] = pd.to_numeric(df_x["logKow_excel"], errors="coerce")

    # Match Excel -> S01 (InChIKey first, name fallback)
    s01_by_inchi: dict[str, int] = {}
    s01_by_name: dict[str, int] = {}
    for i, row in df_s01.iterrows():
        if row["inchikey"]:
            s01_by_inchi.setdefault(row["inchikey"], i)
        if row["name_norm"]:
            s01_by_name.setdefault(row["name_norm"], i)

    matched_rows = []
    for _, row in df_x.iterrows():
        s01_idx = None
        match_mode = ""

        ik = row["inchikey"]
        nm = row["name_norm"]
        if ik and ik in s01_by_inchi:
            s01_idx = s01_by_inchi[ik]
            match_mode = "inchikey"
        elif nm and nm in s01_by_name:
            s01_idx = s01_by_name[nm]
            match_mode = "name"

        if s01_idx is None:
            continue

        srow = df_s01.loc[s01_idx]
        matched_rows.append(
            {
                "name_excel": row["name"],
                "name_s01": srow["name"],
                "match_mode": match_mode,
                "inchikey": srow["inchikey"],
                "logKow_excel": row["logKow_excel"],
                "logKow_exp_s01": srow["logKow_exp_s01"],
                "logKow_smarts": srow["logKow_smarts"],
            }
        )

    df_match = pd.DataFrame(matched_rows)

    df_s01.to_csv(out_dir / "s01_smarts_vs_experimental.csv", index=False)
    df_s02.to_csv(out_dir / "s02_smarts_vs_experimental.csv", index=False)
    df_s01_m.to_csv(out_dir / "s01_smarts_m_vs_experimental.csv", index=False)
    df_s02_m.to_csv(out_dir / "s02_smarts_m_vs_experimental.csv", index=False)
    df_match.to_csv(out_dir / "excel_vs_s01_experimental.csv", index=False)

    df_smarts_excel = df_match[["name_s01", "inchikey", "logKow_smarts", "logKow_excel", "match_mode"]].copy()
    df_smarts_excel.to_csv(out_dir / "smarts_vs_excel_matched.csv", index=False)

    s01_kow_x = df_s01["logKow_exp_s01"].to_numpy(dtype=float)
    s01_kow_y = df_s01["logKow_smarts"].to_numpy(dtype=float)
    s02_koa_x = df_s02["logKoa_exp_s02"].to_numpy(dtype=float)
    s02_koa_y = df_s02["logKoa_smarts"].to_numpy(dtype=float)
    s01_m_kow_x = df_s01_m["logKow_exp_s01"].to_numpy(dtype=float)
    s01_m_kow_y = df_s01_m["logKow_smarts"].to_numpy(dtype=float)
    s02_m_koa_x = df_s02_m["logKoa_exp_s02"].to_numpy(dtype=float)
    s02_m_koa_y = df_s02_m["logKoa_smarts"].to_numpy(dtype=float)
    xls_kow_x = df_match["logKow_excel"].to_numpy(dtype=float)
    xls_kow_y = df_match["logKow_smarts"].to_numpy(dtype=float)

    s01_kow_metrics = _metrics(s01_kow_x, s01_kow_y)
    s02_koa_metrics = _metrics(s02_koa_x, s02_koa_y)
    s01_m_kow_metrics = _metrics(s01_m_kow_x, s01_m_kow_y)
    s02_m_koa_metrics = _metrics(s02_m_koa_x, s02_m_koa_y)
    xls_kow_metrics = _metrics(xls_kow_x, xls_kow_y)
    xls_vs_s01_kow_metrics = _metrics(xls_kow_x, df_match["logKow_exp_s01"].to_numpy(dtype=float))

    if _has_plot_data(s01_kow_x, s01_kow_y):
        _plot_corr(
            x=s01_kow_x,
            y=s01_kow_y,
            xlabel=f"S01 experimental {args.s01_kow_prop}",
            ylabel="SMARTS-predicted logKow",
            title="SMARTS logKow vs S01",
            out_png=out_dir / "corr_smarts_kow_vs_s01.png",
        )

    if _has_plot_data(s01_m_kow_x, s01_m_kow_y):
        _plot_corr(
            x=s01_m_kow_x,
            y=s01_m_kow_y,
            xlabel=f"S01 experimental {args.s01_kow_prop}",
            ylabel="SMARTS-predicted logKow",
            title="SMARTS logKow vs S01",
            out_png=out_dir / "corr_smarts_mixed_kow_vs_s01.png",
        )

    if _has_plot_data(s02_koa_x, s02_koa_y):
        _plot_corr(
            x=s02_koa_x,
            y=s02_koa_y,
            xlabel=f"S02 experimental {args.s02_koa_prop}",
            ylabel="SMARTS-predicted logKoa",
            title="SMARTS logKoa vs S02",
            out_png=out_dir / "corr_smarts_koa_vs_s02.png",
        )

    if _has_plot_data(s02_m_koa_x, s02_m_koa_y):
        _plot_corr(
            x=s02_m_koa_x,
            y=s02_m_koa_y,
            xlabel=f"S02 experimental {args.s02_koa_prop}",
            ylabel="SMARTS mixed-predicted logKoa",
            title="SMARTS mixed logKoa vs S02",
            out_png=out_dir / "corr_smarts_mixed_koa_vs_s02.png",
        )
    if _has_plot_data(xls_kow_x, xls_kow_y):
        _plot_corr(
            x=xls_kow_x,
            y=xls_kow_y,
            xlabel="Supplementary Excel logKow",
            ylabel="SMARTS-predicted logKow",
            title="SMARTS logKow vs Supplementary Excel",
            out_png=out_dir / "corr_smarts_kow_vs_excel.png",
        )

    print("=== SMARTS comparison summary ===")
    print(f"S01 rows: {len(df_s01)}")
    print(f"S02 rows: {len(df_s02)}")
    print(f"Excel rows: {len(df_x)}")
    print(f"Matched rows (InChIKey/name): {len(df_match)}")
    print("\nS01 logKow experimental vs SMARTS:")
    print(s01_kow_metrics)
    print("\nS02 logKoa experimental vs SMARTS:")
    print(s02_koa_metrics)
    print("\nExcel logKow vs SMARTS:")
    print(xls_kow_metrics)
    print("\nExcel logKow vs S01 experimental:")
    print(xls_vs_s01_kow_metrics)
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
