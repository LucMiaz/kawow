# Manuscript: A Vanishing Partitioning Gap

Code to reproduce the figures and SI tables of *A Vanishing Partitioning Gap: Why Persistence Should Come First in Chemical Hazard Screening* (L. T. Miaz and I. T. Cousins), using the five models of this repository (Crippen, Naef & Acree, combined Crippen+Naef & Acree ridge, Random Forest, XGBoost).

The mobility criteria are taken from `kawow.criteria`: the UBA criteria (M: log *K*oc < 4.0, vM: log *K*oc < 3.0, as log *K*ow ≲ 4.5 and ≲ 2.5) and the EU CLP criteria (M: log *K*oc < 3.0, vM: log *K*oc < 2.0, as log *K*ow ≲ 3.5 and ≲ 1.5). Every figure and table that depends on them is produced for both.

## Requirements

The `kawow` package of this repository (`pip install -e .` from the repository root) with its dependencies (RDKit, scikit-learn, XGBoost), plus matplotlib, openpyxl and tabulate. The results in the manuscript were obtained with the versions listed in SI Table S5 (Python 3.14, scikit-learn 1.8.0, XGBoost 3.2.0, RDKit 2025.09.3).

## Data

| Dataset | File | In git | Source |
|---|---|---|---|
| Naef & Acree experimental log *K*ow (S01) and log *K*oa (S02) | `../tests/test_data/S01*.sdf`, `../tests/test_data/S02*.sdf` | yes | Supplementary data of Naef & Acree, *Liquids* **2024**, 4, 231–260, [doi:10.3390/liquids4010011](https://doi.org/10.3390/liquids4010011) |
| Arp & Hale REACH substances with persistence categories and UFZ-LSER estimates | `data/arphale_hale_2022_lser.csv` | yes | See below |
| C&L Inventory structures (181 746 SMILES) | `data/clinventory_molecules.csv` | **no** (~14 MB) | Zenodo, `https://doi.org/10.5281/zenodo.XXXXXXX` (TODO) |

**Arp & Hale.** `data/arphale_hale_2022_lser.csv` has one row per substance of Arp & Hale, *ACS Environ. Au* **2022**, 2, 482–509 ([doi:10.1021/acsenvironau.2c00024](https://doi.org/10.1021/acsenvironau.2c00024)), SI Table S1 (11 614 substances): name, CAS, InChIKey, SMILES, the PMT/vPvM screening outcomes under the UBA (`vPvM_UBA`) and EC (`vPvM_EC`) criteria, the geometric-average QSAR half-life (`P_avg_QSAR`), the acid/base/neutral class, and UFZ-LSER estimates of log *K*ow, log *K*oa and log *K*oc (2017 version of UFZ-LSER). TODO: describe how the UFZ-LSER log *K*oa values were obtained, since they are not part of the Arp & Hale SI.

**C&L Inventory.** The C&L Inventory extract (columns `molecule_id`, `inchikey`, `smiles`, `ec_primary`) was retrieved from ECHA's C&L Inventory and PubChem (SI section S6). It is not stored in git: download `clinventory_molecules.csv` from Zenodo and place it in `data/`. Without it, the scripts stop with a message giving the download link; `build_deployment_predictions.py --skip-clinventory` builds everything else, and `generate_figures.py` and `generate_tables.py` then skip the C&L Inventory figures and rows.

## Steps

Run from this folder:

```bash
python ../scripts/benchmark_5model.py      # optional, ~1 h: 5-fold CV benchmark -> ../tests/out/benchmark_*_5model.csv (in git)
python yrandomization.py                   # ~5 min: SI Figure S1 inputs -> cache/
python build_deployment_predictions.py     # ~15 min: final-model predictions -> cache/
python generate_figures.py                 # figures -> outputs/figures/
python generate_tables.py                  # SI tables and gap counts -> outputs/tables/
```

`cache/` and `outputs/` are not stored in git.

## Outputs

| Manuscript | File (`outputs/`) |
|---|---|
| Figure 1 | `figures/fig1_regulatory_space` |
| Figure 2 | `figures/fig8_gap_zoom_models` |
| Figure 3 | `figures/fig10_clinventory_gap_zoom` |
| Figure 4 | `figures/fig9_arphale_gap_persistence` |
| SI Figure S1 | `figures/fig7_yrandomization` |
| SI Figure S2 | `figures/fig3_model_comparison` |
| SI Figures S3 to S6 | `figures/fig1_*_clp`, `fig8_*_clp`, `fig10_*_clp`, `fig9_*_clp` |
| SI Table S1 | `tables/table_S1_exclusions.csv` |
| SI Table S2 | `tables/table_S2_persistence_mapping.csv` |
| SI Table S4 | `tables/table_S4_benchmark.csv` |
| SI Table S6 | `tables/table_S6_gap_counts.csv` |
| Gap counts in the text | `tables/gap_counts_full.csv` |

SI Tables S3 (model architectures) and S5 (software versions) are not generated.

## Notes on reproducibility

- The benchmark (`../tests/out/benchmark_*_5model.csv`) was regenerated on 2026-09-13; SI Table S4 of the submitted SI predates it, and XGBoost differs slightly (log *K*ow R² 0.9437 and RMSE 0.493 in the table, 0.9443 and 0.491 in the file).
- The final-model predictions of the C&L Inventory for Crippen and XGBoost differ from those of the figures in the first manuscript draft (116 and 69 molecules in the UBA gap, against 107 and 87); the script that produced the earlier figures was not kept. Naef & Acree (105), the combined-ridge model (111) and Random Forest (48) are identical.
- `build_deployment_predictions.py` runs the inner cross-validation of the two ridge models in parallel; `--serial-ridge` uses kawow's serial routine instead, with the same result.
