"""generate_tables.py
===================
SI tables of the manuscript "A Vanishing Partitioning Gap", and the gap counts
quoted in the main text. Written to outputs/tables/ as CSV, plus tables.md.

  table_S1_exclusions.csv          SI Table S1  (molecules excluded per dataset)
  table_S2_persistence_mapping.csv SI Table S2  (Arp & Hale categories)
  table_S4_benchmark.csv           SI Table S4  (5-fold CV performance)
  table_S6_gap_counts.csv          SI Table S6  (gap counts, UBA and CLP)
  gap_counts_full.csv              all counts behind the text (persistence
                                   categories in the gap, gap shrunk by one
                                   RMSE, band neither vM nor vB)

Usage:
    python generate_tables.py
"""

import pandas as pd

import paths
import generate_figures as gf
from scripts.benchmark_5model import KOA_BASE, KOW_BASE, _build_common_set

UBA_TO_BUCKET = [
    ("vPvM, vPvM & PMT", ["vPvM", "vPvM & PMT"]),
    ("PM, PMT", ["PM", "PMT"]),
    ("Potential PMT/vPvM", ["Potential PMT/vPvM"]),
    ("Not P, Not P & Not M", ["Not P", "Not P, Not M"]),
    ("Not enough data, Not M, Not PMT/vPvM", ["Not enough data", "Not M", "Not PMT/vPvM"]),
]
DATASET_LABELS = {
    "naef_observed": "Naef & Acree",
    "naef_predicted": "Naef & Acree",
    "arphale_lser": "REACH (Arp & Hale)",
    "arphale_naef_experimental": "REACH (Arp & Hale)",
    "arphale_predicted": "REACH (Arp & Hale)",
    "clinventory": "C&L Inventory",
}


def table_s1():
    rows = []
    for name, sdf, prop, endpoint, base in [
        ("S01 (logKow)", paths.S01_SDF, "logP", "logKow", KOW_BASE),
        ("S02 (logKoa)", paths.S02_SDF, "logKoa", "logKoa", KOA_BASE),
    ]:
        entries = sdf.read_text(encoding="utf8", errors="replace").count("$$$$")
        d = _build_common_set(sdf, prop, endpoint, base, base)
        n = len(d["y"])
        rows.append({"Dataset": name, "SDF entries": entries, "RDKit parse failure": entries - d["raw_n"],
                     "Backbone <2": d["raw_n"] - n, "Common evaluation set": n})
    return pd.DataFrame(rows)


def table_s2():
    full = pd.read_csv(paths.ARPHALE_CSV)
    lser = full.dropna(subset=["logKow", "logKoa"])
    rows = []
    for label, cats in UBA_TO_BUCKET:
        rows.append({"vPvM_UBA value(s)": label, "P category": gf._bucket_uba(cats[0]),
                     "n (full dataset)": int(full["vPvM_UBA"].isin(cats).sum()),
                     "n (LSER-available)": int(lser["vPvM_UBA"].isin(cats).sum())})
    # Missing vPvM_UBA values also fall in "no data".
    rows[-1]["n (full dataset)"] += int(full["vPvM_UBA"].isna().sum())
    rows[-1]["n (LSER-available)"] += int(lser["vPvM_UBA"].isna().sum())
    return pd.DataFrame(rows)


def table_s4():
    m = pd.read_csv(paths.BENCHMARK_METRICS_CSV)
    m["model"] = m["model"].map(gf.MODEL_LABELS)
    cols = {"r2": "R2", "rmse": "RMSE", "auc": "AUC", "ap": "AP", "bal_acc": "Bal. Acc.", "mcc": "MCC"}
    return m[["endpoint", "model", "threshold"] + list(cols)].rename(columns=cols)


def gap_count_rows():
    merged = gf.load_benchmark_predictions_all()
    metrics = pd.read_csv(paths.BENCHMARK_METRICS_CSV)
    rmse = {(r.endpoint, r.model): r.rmse for r in metrics.itertuples()}
    df_arphale = gf.load_arphale_persistence()
    df_naef = gf.load_arphale_naef_experimental()
    df_pred = gf.load_arphale_model_predictions()
    try:
        df_cl = gf.load_clinventory_predictions()
    except paths.MissingDataError as exc:
        print(f"  [info] C&L Inventory rows omitted: {exc}")
        df_cl = None
    rows = []
    for key in gf.MOBILITY_CRITERIA:
        gf.use_mobility_criteria(key)
        rows += gf.gap_counts(merged, df_arphale, df_naef, df_pred, df_cl, rmse)
    gf.use_mobility_criteria("uba")
    return pd.DataFrame(rows)


def table_s6(full):
    def fmt(n, total):
        return f"{n:,} ({100 * n / total:.2f}%)".replace(",", " ")

    out = []
    for (dataset, model), g in full.groupby(["dataset", "model"], sort=False):
        uba = g[g["criteria"] == "UBA"].iloc[0]
        clp = g[g["criteria"] == "CLP"].iloc[0]
        source = {"experimental": "experimental (both)", "-": "UFZ-LSER" if dataset == "arphale_lser"
                  else "experimental (Naef & Acree)"}.get(model, gf.MODEL_LABELS.get(model, model))
        core = clp.get("n_gap_core")
        out.append({
            "Dataset": DATASET_LABELS[dataset], "Source of logKow, logKoa": source, "n": int(uba["n_total"]),
            "Gap, UBA": fmt(int(uba["n_gap"]), uba["n_total"]), "Gap, CLP": fmt(int(clp["n_gap"]), clp["n_total"]),
            "CLP gap minus 1 RMSE": "–" if pd.isna(core) or dataset.startswith("arphale")
            else fmt(int(core), clp["n_total"]) if core else "0",
        })
    return pd.DataFrame(out)


def main():
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    print("Table S1 (exclusions)...")
    s1 = table_s1()
    print("Table S2 (persistence mapping)...")
    s2 = table_s2()
    s4 = table_s4()
    print("Gap counts (UBA and CLP)...")
    full = gap_count_rows()
    s6 = table_s6(full)
    tables = {"table_S1_exclusions": s1, "table_S2_persistence_mapping": s2, "table_S4_benchmark": s4,
              "table_S6_gap_counts": s6, "gap_counts_full": full}
    md = []
    for name, df in tables.items():
        df.to_csv(paths.TABLES / f"{name}.csv", index=False)
        md += [f"## {name}", "", df.to_markdown(index=False), ""]
    (paths.TABLES / "tables.md").write_text("\n".join(md), encoding="utf8")
    print(f"Saved {len(tables)} tables and tables.md to {paths.TABLES}")


if __name__ == "__main__":
    main()
