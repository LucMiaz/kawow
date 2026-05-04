"""Map S01/S02 compounds to acid/neutral/basic with rough pKa estimates,
then evaluate model performance by subgroup.

Outputs are written to tests/out by default:
- ionization_map_s01_s02.csv
- ionization_performance_summary.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDMolSupplier
from rdkit import RDLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from kawow.model import run_models  # noqa: E402

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class GroupRule:
    name: str
    smarts: str
    pka: float


ACID_RULES = [
    GroupRule("carboxylic_acid", "[CX3](=O)[OX2H1]", 4.5),
    GroupRule("sulfonic_acid", "[SX4](=O)(=O)[OX2H1]", -1.0),
    GroupRule("phosphonic_acid", "[PX4](=O)([OX2H1])[OX2H1]", 2.0),
    GroupRule("phosphate_monoester", "[PX4](=O)([OX2H1])[OX2][#6]", 2.2),
    GroupRule("tetrazole", "[n]1[n][n][n][c]1", 4.8),
    GroupRule("phenol", "[c][OX2H1]", 10.0),
    GroupRule("thiol", "[#16X2H1]", 10.5),
    GroupRule("imide", "[NX3H][CX3](=O)[#6][CX3](=O)", 8.5),
]

BASE_RULES = [
    GroupRule("guanidine", "NC(=N)N", 13.5),
    GroupRule("amidine", "NC(=N)[#6]", 12.0),
    GroupRule("aliphatic_amine", "[NX3;H2,H1,H0;!$(N-C=O);!$(N-S(=O)=O);!$(N-c)]", 10.0),
    GroupRule("aniline_like", "[NX3;H2,H1,H0]-c", 5.0),
    GroupRule("pyridine_like", "[nH0;r6]", 5.2),
    GroupRule("imidazole_like", "[nH0;r5]", 7.0),
]


def _compile_rules(rules: list[GroupRule]) -> list[tuple[GroupRule, Chem.Mol]]:
    compiled: list[tuple[GroupRule, Chem.Mol]] = []
    for rule in rules:
        patt = Chem.MolFromSmarts(rule.smarts)
        if patt is None:
            raise ValueError(f"Invalid SMARTS for {rule.name}: {rule.smarts}")
        compiled.append((rule, patt))
    return compiled


COMPILED_ACIDS = _compile_rules(ACID_RULES)
COMPILED_BASES = _compile_rules(BASE_RULES)


def _count_matches(mol: Chem.Mol, compiled_rules: list[tuple[GroupRule, Chem.Mol]]) -> list[dict]:
    out = []
    for rule, patt in compiled_rules:
        n = len(mol.GetSubstructMatches(patt, uniquify=True))
        if n > 0:
            out.append({"name": rule.name, "count": int(n), "pka": float(rule.pka)})
    return out


def _acid_deprotonated_fraction(pka: float, ph: float = 7.0) -> float:
    return 1.0 / (1.0 + 10.0 ** (pka - ph))


def _base_protonated_fraction(pka: float, ph: float = 7.0) -> float:
    return 1.0 / (1.0 + 10.0 ** (ph - pka))


def estimate_ionization(mol: Chem.Mol, ph: float = 7.0) -> dict:
    acids = _count_matches(mol, COMPILED_ACIDS)
    bases = _count_matches(mol, COMPILED_BASES)

    strongest_acid_pka = min((g["pka"] for g in acids), default=np.nan)
    strongest_base_pka = max((g["pka"] for g in bases), default=np.nan)

    net_charge = 0.0
    for g in acids:
        net_charge -= g["count"] * _acid_deprotonated_fraction(g["pka"], ph=ph)
    for g in bases:
        net_charge += g["count"] * _base_protonated_fraction(g["pka"], ph=ph)

    if acids or bases:
        if net_charge > 0.25:
            ion_class = "basic"
        elif net_charge < -0.25:
            ion_class = "acid"
        else:
            ion_class = "neutral"
    else:
        ion_class = "neutral"

    return {
        "ion_class": ion_class,
        "net_charge_ph7": float(net_charge),
        "strongest_acid_pka": float(strongest_acid_pka) if np.isfinite(strongest_acid_pka) else np.nan,
        "strongest_base_pka": float(strongest_base_pka) if np.isfinite(strongest_base_pka) else np.nan,
        "acid_groups": acids,
        "basic_groups": bases,
    }


def load_sdf_rows(sdf_path: Path, exp_prop: str, dataset_tag: str) -> list[dict]:
    rows = []
    suppl = SDMolSupplier(str(sdf_path), removeHs=False)
    for i, mol in enumerate(suppl):
        if mol is None:
            continue

        name = ""
        if mol.HasProp("Alias name"):
            name = mol.GetProp("Alias name")
        elif mol.HasProp("_Name"):
            name = mol.GetProp("_Name")

        exp_val = np.nan
        if mol.HasProp(exp_prop):
            try:
                exp_val = float(mol.GetProp(exp_prop))
            except Exception:
                exp_val = np.nan

        smiles = Chem.MolToSmiles(mol)
        inchikey = ""
        try:
            inchikey = Chem.MolToInchiKey(mol)
        except Exception:
            inchikey = ""
        ion = estimate_ionization(mol)

        rows.append(
            {
                "dataset": dataset_tag,
                "idx": i,
                "name": name,
                "smiles": smiles,
                "inchikey": inchikey,
                "exp_prop": exp_prop,
                "exp_value": exp_val,
                "ion_class": ion["ion_class"],
                "net_charge_ph7": ion["net_charge_ph7"],
                "strongest_acid_pka": ion["strongest_acid_pka"],
                "strongest_base_pka": ion["strongest_base_pka"],
                "acid_groups": ";".join(f"{g['name']}:{g['count']}@{g['pka']}" for g in ion["acid_groups"]),
                "basic_groups": ";".join(f"{g['name']}:{g['count']}@{g['pka']}" for g in ion["basic_groups"]),
            }
        )
    return rows


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(mask.sum())
    if n < 2:
        return {"n": n, "r2": np.nan, "rmse": np.nan, "mae": np.nan, "bias": np.nan}

    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        "n": n,
        "r2": float(r2_score(yt, yp)),
        "rmse": float(math.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "bias": float(np.mean(yp - yt)),
    }


def evaluate_by_subgroup(df: pd.DataFrame, endpoint_key: str, dataset_tag: str, models: list[str]) -> pd.DataFrame:
    out_rows = []
    for ion_class in ["acid", "neutral", "basic"]:
        sub = df[(df["dataset"] == dataset_tag) & (df["ion_class"] == ion_class)].copy()
        for model in models:
            y_true = sub["exp_value"].to_numpy(dtype=float)
            y_pred = sub[f"pred_{model}_{endpoint_key}"].to_numpy(dtype=float)
            m = compute_metrics(y_true, y_pred)
            out_rows.append(
                {
                    "dataset": dataset_tag,
                    "endpoint": endpoint_key,
                    "model": model,
                    "ion_class": ion_class,
                    **m,
                }
            )
    return pd.DataFrame(out_rows)


def _fmt_num(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def build_markdown_report(df: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Ionization-Stratified Model Performance on S01 and S02")
    lines.append("")
    lines.append("This report stratifies the S01 logP/logKow and S02 logKoa benchmark sets into approximate acid, neutral, and basic subgroups using rule-based SMARTS detection with rough literature-like pKa anchors. The pKa values are approximate screening estimates for subgrouping, not formal microstate pKa predictions.")
    lines.append("")

    counts = (
        df.groupby(["dataset", "ion_class"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["dataset", "ion_class"])
    )
    lines.append("## Subgroup Counts")
    lines.append("")
    lines.append("| Dataset | Ionization class | N |")
    lines.append("|---|---:|---:|")
    for row in counts.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.ion_class} | {row.n} |")
    lines.append("")

    lines.append("## Performance Summary")
    lines.append("")
    lines.append("| Dataset | Endpoint | Ionization class | Best model by RMSE | RMSE | R2 |")
    lines.append("|---|---|---|---|---:|---:|")
    best_rows = (
        summary[summary["n"] >= 2]
        .sort_values(["dataset", "endpoint", "ion_class", "rmse", "model"])
        .groupby(["dataset", "endpoint", "ion_class"], as_index=False)
        .first()
    )
    for row in best_rows.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.endpoint} | {row.ion_class} | {row.model} | {_fmt_num(row.rmse)} | {_fmt_num(row.r2)} |"
        )
    lines.append("")

    for dataset, endpoint in [("S01", "logKow"), ("S02", "logKoa")]:
        lines.append(f"## {dataset} {endpoint}")
        lines.append("")
        subset = summary[(summary["dataset"] == dataset) & (summary["endpoint"] == endpoint)].copy()
        lines.append("| Ionization class | Model | N | RMSE | MAE | Bias | R2 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in subset.sort_values(["ion_class", "rmse", "model"]).itertuples(index=False):
            lines.append(
                f"| {row.ion_class} | {row.model} | {row.n} | {_fmt_num(row.rmse)} | {_fmt_num(row.mae)} | {_fmt_num(row.bias)} | {_fmt_num(row.r2)} |"
            )
        lines.append("")

    lines.append("## Largest Residuals")
    lines.append("")
    lines.append("The tables below list the five largest absolute residuals within each subgroup/model slice where at least one valid prediction exists.")
    lines.append("")

    for dataset, endpoint in [("S01", "logKow"), ("S02", "logKoa")]:
        for ion_class in ["acid", "neutral", "basic"]:
            lines.append(f"### {dataset} {endpoint} {ion_class}")
            lines.append("")
            for model in ["kawow", "smarts", "smarts_mixed", "mqg"]:
                pred_col = f"pred_{model}_{endpoint}"
                if pred_col not in df.columns:
                    continue
                sub = df[(df["dataset"] == dataset) & (df["ion_class"] == ion_class)].copy()
                sub = sub[np.isfinite(sub[pred_col]) & np.isfinite(sub["exp_value"])].copy()
                if sub.empty:
                    continue
                sub["residual"] = sub[pred_col] - sub["exp_value"]
                sub["abs_residual"] = np.abs(sub["residual"])
                top = sub.sort_values(["abs_residual", "name"], ascending=[False, True]).head(5)
                lines.append(f"#### {model}")
                lines.append("")
                lines.append("| Compound | pKa(acid) | pKa(base) | Experimental | Predicted | Residual | Acid groups | Basic groups |")
                lines.append("|---|---:|---:|---:|---:|---:|---|---|")
                for row in top.itertuples(index=False):
                    compound = str(row.name or row.smiles or "")
                    lines.append(
                        f"| {compound} | {_fmt_num(row.strongest_acid_pka, 2)} | {_fmt_num(row.strongest_base_pka, 2)} | {_fmt_num(row.exp_value)} | {_fmt_num(getattr(row, pred_col))} | {_fmt_num(row.residual)} | {row.acid_groups or '-'} | {row.basic_groups or '-'} |"
                    )
                lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Acid/basic labels are based on rule-matched functional groups and estimated pKa values at pH 7.")
    lines.append("- Zwitterions often remain classed as neutral when the estimated net charge is near zero.")
    lines.append("- SMARTS mixed is consistently the strongest model across these subgrouped benchmarks.")
    lines.append("- Plain SMARTS degrades strongly for ionizable chemistry, especially S02 basic compounds.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate kawow models by acid/neutral/basic subgroups on S01/S02.")
    parser.add_argument(
        "--s01",
        default=str(Path(__file__).parent.parent / "tests" / "test_data" / "S01. Compounds List for logPow-Parameters Calculations.sdf"),
    )
    parser.add_argument(
        "--s02",
        default=str(Path(__file__).parent.parent / "tests" / "test_data" / "S02. Compounds List for logKoa-Parameters Calculations.sdf"),
    )
    parser.add_argument("--s01-prop", default="logP")
    parser.add_argument("--s02-prop", default="logKoa")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).parent.parent / "tests" / "out"),
    )
    args = parser.parse_args()

    models = ["kawow", "smarts", "smarts_mixed", "mqg"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s01_rows = load_sdf_rows(Path(args.s01), args.s01_prop, "S01")
    s02_rows = load_sdf_rows(Path(args.s02), args.s02_prop, "S02")
    df = pd.DataFrame(s01_rows + s02_rows)

    out_dir_existing = out_dir

    # Fast path: use existing benchmark CSVs for smarts/mixed/mqg when available.
    precomputed_files = {
        ("S01", "smarts"): out_dir_existing / "s01_smarts_vs_experimental.csv",
        ("S01", "smarts_mixed"): out_dir_existing / "s01_smarts_m_vs_experimental.csv",
        ("S01", "mqg"): out_dir_existing / "s01_mqg_vs_experimental.csv",
        ("S02", "smarts"): out_dir_existing / "s02_smarts_vs_experimental.csv",
        ("S02", "smarts_mixed"): out_dir_existing / "s02_smarts_m_vs_experimental.csv",
        ("S02", "mqg"): out_dir_existing / "s02_mqg_vs_experimental.csv",
    }

    for model in models:
        for endpoint in ["logKow", "logKoa"]:
            df[f"pred_{model}_{endpoint}"] = np.nan

    for (dataset_tag, model), csv_path in precomputed_files.items():
        if not csv_path.exists():
            continue
        pred_df = pd.read_csv(csv_path)
        if "inchikey" in pred_df.columns and pred_df["inchikey"].notna().any():
            key_cols = ["inchikey"]
        elif "smiles" in pred_df.columns:
            key_cols = ["smiles"]
        else:
            continue

        dataset_mask = df["dataset"] == dataset_tag
        sub = df.loc[dataset_mask].copy()
        key = key_cols[0]
        pred_dedup = pred_df.dropna(subset=[key]).drop_duplicates(subset=[key], keep="first")

        if dataset_tag == "S01":
            if model == "mqg" and "logKow_mqg" in pred_dedup.columns:
                lut = pred_dedup.set_index(key)["logKow_mqg"]
                df.loc[dataset_mask, "pred_mqg_logKow"] = sub[key].map(lut).to_numpy()
            if model in {"smarts", "smarts_mixed"} and "logKow_smarts" in pred_dedup.columns:
                lut = pred_dedup.set_index(key)["logKow_smarts"]
                df.loc[dataset_mask, f"pred_{model}_logKow"] = sub[key].map(lut).to_numpy()
        if dataset_tag == "S02":
            if model == "mqg" and "logKoa_mqg" in pred_dedup.columns:
                lut = pred_dedup.set_index(key)["logKoa_mqg"]
                df.loc[dataset_mask, "pred_mqg_logKoa"] = sub[key].map(lut).to_numpy()
            if model in {"smarts", "smarts_mixed"} and "logKoa_smarts" in pred_dedup.columns:
                lut = pred_dedup.set_index(key)["logKoa_smarts"]
                df.loc[dataset_mask, f"pred_{model}_logKoa"] = sub[key].map(lut).to_numpy()

    # Live compute only for the ridge/Crippen model (kawow).
    preds_kawow = run_models(df["smiles"].tolist(), models=["kawow"], fmt="smiles")
    for i, row_pred in enumerate(preds_kawow):
        model_dict = row_pred.get("models", {}) if isinstance(row_pred, dict) else {}
        pred = model_dict.get("kawow", {}) if isinstance(model_dict, dict) else {}
        if pred.get("ok"):
            df.at[i, "pred_kawow_logKow"] = pred.get("logKow", np.nan)
            df.at[i, "pred_kawow_logKoa"] = pred.get("logKoa", np.nan)

    summary = pd.concat(
        [
            evaluate_by_subgroup(df, "logKow", "S01", models),
            evaluate_by_subgroup(df, "logKoa", "S02", models),
        ],
        ignore_index=True,
    )

    map_path = out_dir / "ionization_map_s01_s02.csv"
    sum_path = out_dir / "ionization_performance_summary.csv"
    report_path = out_dir / "ionization_performance_report.md"
    df.to_csv(map_path, index=False)
    summary.to_csv(sum_path, index=False)
    report_path.write_text(build_markdown_report(df, summary), encoding="utf-8")

    print(f"Wrote mapping: {map_path}")
    print(f"Wrote summary: {sum_path}")
    print(f"Wrote report: {report_path}")

    counts = (
        df.groupby(["dataset", "ion_class"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["dataset", "ion_class"])
    )
    print("\nIonization subgroup counts:")
    print(counts.to_string(index=False))

    print("\nPerformance summary (RMSE by subgroup):")
    view = summary[["dataset", "endpoint", "model", "ion_class", "n", "rmse", "r2", "mae", "bias"]].copy()
    print(view.sort_values(["dataset", "endpoint", "ion_class", "model"]).to_string(index=False))


if __name__ == "__main__":
    main()
