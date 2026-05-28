# Kawow (under development)

[![CI](https://github.com/LucMiaz/kawow/actions/workflows/ci.yml/badge.svg)](https://github.com/LucMiaz/kawow/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-green.svg)](LICENSE)

![Kawow!](logo/kawow.svg)

Group-additivity prediction of **log*K*ow**, **log*K*oa**, and **log*K*aw** from molecular structure.

*Kawow* implements some models to predict partitioning coefficients (`logKoa`, `logKow` and `logKaw`), in particular the Naef & Acree (2024) group-additivity scheme using RDKit SMARTS pattern matching. Two model families are available depending on how much transparency or accuracy is required.

### Flagging criteria used in outputs

`run_models(...)` returns B/vB, M/vM, and regulatory-gap flags derived from predicted partition values.

**Bioaccumulation** (following doi:[10.1126/science.1138275](https://doi.org/10.1126/science.1138275)):

| Flag | Condition |
|------|-----------|
| `B`  | `logKoa ≥ 6` **and** `logKow ≥ 2` |
| `vB` | `logKoa ≥ 6` **and** `logKow ≥ 5` |

**Mobility** — estimated via `logKoc_est = logKow − 0.4` (UBA drinking-water guidance, [link](https://www.umweltbundesamt.de/en/publikationen/protecting-the-sources-of-our-drinking-water-the)):

| Flag | Condition on `logKoc_est` | Equivalent `logKow` |
|------|--------------------------|---------------------|
| `M`  | `logKoc_est ≤ 4.5`       | `logKow ≤ 4.9`      |
| `vM` | `logKoc_est ≤ 3.5`       | `logKow ≤ 3.9`      |

**Regulatory gaps** (returned as `in_gap1`, `in_gap2`, `in_gap3` booleans and `gap_labels` list):

| Gap | Condition |
|-----|-----------|
| Gap 1 | `3.5 < logKow < 5.0` (between vM and vB on *K*ow axis) |
| Gap 2 | `logKow > 4.5` **and** `logKoa < 6` (non-M, non-B) |
| Gap 3 | `4.5 < logKow < 5.0` **and** `logKoa < 6` (intersection of Gap 1 and Gap 2) |

---

## Installation

```bash
pip install kawow
```

Or from source (requires RDKit ≥ 2022.9):

```bash
git clone https://github.com/LucMiaz/kawow.git
cd kawow
pip install -e ".[dev]"
```

---

## Models at a glance

log*K*ow and log*K*oa R² values are from 5-fold cross-validation on the shared S01∩S02 benchmark (n = 3 319 for log*K*ow; n = 1 956 for log*K*oa), except `smarts` which applies fixed Naef & Acree (2024) parameters directly (no re-fitting). log*K*aw R² is from external validation on the full S03 dataset (Naef & Acree 2024; n = 2 130–2 150; S03 was not used for training).

| Model key | Class | Approach | log*K*ow R² | log*K*oa R² | log*K*aw R² |
|-----------|-------|----------|-------------|-------------|-------------|
| `kawow` | `PartitionCalculator` | Ridge regression on Crippen atom-type counts + Naef special-group features | 0.898 (cv) | 0.937 (cv) | — |
| `smarts` | `NaefAcreePartitionCalculator` | Pure Naef & Acree 2024 group-additivity (no re-fitting, tabulated parameters only) | 0.857 | 0.785 | 0.654 |
| `smarts_mixed` | `NaefAcreeCrippenMixedPartitionCalculator` | Naef & Acree SMARTS contributions + Crippen atom-type Ridge hybrid | 0.938 (cv) | 0.943 (cv) | **0.912** |
| `mqg` | `MQGPartitionCalculator` | Ridge regression ensemble: Naef group contributions + Crippen atom types + Molecular Quantum Graph fingerprints | 0.940 (cv) | 0.942 (cv) | **0.913** |
| `pfasgroups` | `PFASGroupsPartitionCalculator` | Ridge regression on 77-dim PFASGroups halogenated-group descriptor | — (cv) | — (cv) | — |
| `pfasgroups_mixed` | `PFASGroupsPartitionCalculator` | Ridge regression on PFASGroups (77-dim) + Crippen atom-type (77-dim) concatenated features | — (cv) | — (cv) | — |

Use `run_models()` to run several models at once and get per-molecule B/vB and M/vM flags:

```python
import kawow

results = kawow.run_models(
    ["CCCCO", "c1ccccc1", "OC(=O)c1ccccc1"],
    models=["kawow", "smarts_mixed"],
)
for row in results:
    print(row["smiles"], row["models"]["kawow"]["logKow"],
          row["models"]["kawow"]["b_class"])
```

Each element of the returned list is a `dict` with:

| Key | Description |
|-----|-------------|
| `smiles` | canonical SMILES |
| `name` | molecule name from input |
| `models` | `dict` keyed by model name; each value contains `logKow`, `logKoa`, `logKaw`, `b_class`, `m_class`, `ok` |
| `ok` | `True` if at least one model succeeded |

---

## 1 — `PartitionCalculator` (recommended for most uses)

Ridge regression fitted on the same S01/S02 datasets. Coefficients are stored in `kawow/data/*.json` so no re-fitting is needed at import time.

```python
from kawow import PartitionCalculator

calc = PartitionCalculator()           # Ridge (default)

# Single molecule from SMILES
result = calc.predict("CCCCO")        # 1-butanol
print(result)
# {'logKow': 0.88, 'logKoa': 4.12, 'logKaw': -3.24, 'status': 'ok'}

# Batch prediction
smiles = ["c1ccccc1", "CCCCCCCCCC", "OC(=O)c1ccccc1"]
for r in calc.predict_batch(smiles):
    print(r["smiles"], r["logKow"], r["logKoa"], r["logKaw"])
```

**Predict from an InChI string or SDF file:**

```python
r = calc.predict("InChI=1S/C4H10O/c1-2-3-4-5/h5H,2-4H2,1H3")
results = calc.predict("compounds.sdf")   # returns list[dict]
```

**Inspect model metadata:**

```python
info = calc.model_info
print(info["logKow"])
# {'target': 'logKow', 'n_train': 3234, 'alpha': 51.8,
#  'r2_cv': 0.8980, 'rmse_cv': 0.6643,
#  'ccc_cv': 0.9470,          # Lin's concordance correlation coefficient
#  'nrmse_sd_cv': 0.2770,     # RMSE / σ(logKow_train)
#  'nrmse_range_cv': 0.0580,  # RMSE / range(logKow_train)
#  'bf10_log10_cv': '>10^300',# log₁₀ Bayes factor (H₁: ρ ≠ 0)
#  'r_ci95_cv': [0.945, 0.951],  # 95 % CI on Pearson ρ
#  ...}
```

**Re-fit on your own training data:**

```python
import kawow
kawow.fit(
    sdf_logkow="my_logkow.sdf",
    sdf_logkoa="my_logkoa.sdf",
    logkow_prop="logP",
    logkoa_prop="logKoa",
)
calc = kawow.PartitionCalculator()   # reload after fitting
```

### Performance (Naef & Acree benchmark datasets)

All values are from 5-fold cross-validation on the shared S01∩S02 intersection, except `smarts` (no fitting; evaluated on full S01/S02/S03) and the log*K*aw rows (external validation on full S03 — S03 was never used for training).

| Model | Property | n | R² | RMSE | Note |
|-------|----------|---|-----|------|------|
| `kawow` (Crippen Ridge) | log*K*ow | 3 319 | 0.898 | 0.664 | 5-fold CV |
| `kawow` (Crippen Ridge) | log*K*oa | 1 956 | 0.937 | 0.740 | 5-fold CV |
| `smarts` (Naef & Acree) | log*K*ow | 3 344 | 0.857 | 0.786 | external (S01 full) |
| `smarts` (Naef & Acree) | log*K*oa | 1 983 | 0.785 | 1.387 | external (S02 full) |
| `smarts` (Naef & Acree) | log*K*aw | 2 150 | 0.654 | 1.758 | external (S03 full) |
| `smarts_mixed` (hybrid) | log*K*ow | 3 319 | **0.938** | 0.518 | 5-fold CV |
| `smarts_mixed` (hybrid) | log*K*oa | 1 956 | **0.943** | 0.702 | 5-fold CV |
| `smarts_mixed` (hybrid) | log*K*aw | 2 150 | **0.912** | 0.886 | external (S03 full) |
| `mqg` (ensemble) | log*K*ow | 3 319 | **0.940** | 0.510 | 5-fold CV |
| `mqg` (ensemble) | log*K*oa | 1 956 | **0.942** | 0.705 | 5-fold CV |
| `mqg` (ensemble) | log*K*aw | 2 130 | **0.913** | 0.882 | external (S03 full) |

### Regulatory classification performance (F1 scores)-

Binary classification F1 scores on the shared S01∩S02 benchmark (1 083–1 102 molecules with paired experimental log*K*ow and log*K*oa). Flags are applied to **predicted** values using the same thresholds as `run_models()`. `naef_mqg` and `crippen_mqg` are available via `EnsemblePartitionCalculator`.

| Label | Condition | n (+) | `kawow` | `smarts` | `smarts_mixed` | `naef_mqg` | `crippen_mqg` | `mqg` |
|-------|-----------|------:|--------:|---------:|---------------:|-----------:|--------------:|------:|
| G1 | 3.5 < log*K*ow < 5.0 | 178 | 0.67 | 0.74 | **0.77** | **0.77** | 0.69 | 0.55 |
| G2 | log*K*ow > 4.5 and log*K*oa < 6 | 24 | 0.56 | 0.54 | 0.62 | **0.63** | 0.58 | 0.15 |
| G3 | 4.5 < log*K*ow < 5.0 and log*K*oa < 6 | 11 | 0.00 | **0.27** | 0.13 | 0.13 | 0.00 | — |
| M | log*K*oc_est ≤ 4.5 | 797 | 0.97 | 0.98 | **0.98** | 0.98 | 0.97 | 0.95 |
| vM | log*K*oc_est ≤ 3.5 | 677 | 0.95 | 0.96 | **0.97** | **0.97** | 0.95 | 0.95 |
| B | log*K*ow ≥ 2 and log*K*oa ≥ 6 | 503 | 0.94 | 0.94 | **0.96** | 0.95 | 0.95 | 0.94 |
| vB | log*K*ow ≥ 5 and log*K*oa ≥ 6 | 266 | 0.92 | 0.93 | 0.93 | **0.94** | 0.93 | 0.79 |

n (+): true-positive molecule count. — = model makes 0 positive predictions (precision undefined). G3 has only 11 true-positive molecules; most models do not recover this rare class.

---

## 2 — `NaefAcreePartitionCalculator` (SMARTS additivity, full transparency)

Implements the Naef & Acree 2024 method exactly: each SMARTS pattern from the paper's supplementary tables is matched against the molecule and its tabulated contribution added. No matrix regression — every contribution is directly interpretable.

```python
from kawow.smarts_model import NaefAcreePartitionCalculator

calc = NaefAcreePartitionCalculator(smiles="c1ccccc1")
result = calc.predict("c1ccccc1")
# {'logKow': 2.13, 'logKoa': 2.80, 'logKaw': -0.67, 'in_coverage': True}

# Or pass a pre-built RDKit mol:
from rdkit import Chem
mol = Chem.MolFromSmiles("CCCCCCCCCC")
result = calc.predict(mol)

# Batch via constructor:
calc_batch = NaefAcreePartitionCalculator(
    smiles=["c1ccccc1", "CCCCCCCCCC", "OC(=O)c1ccccc1"]
)
for mol, coeffs in calc_batch.results.items():
    print(coeffs)
```

### Performance (Naef & Acree tabulated parameters, evaluated on benchmark sets)

The `smarts` model applies the published Naef & Acree (2024) parameters without any re-fitting. Performance is evaluated on the full individual datasets and on the shared S01∩S02 benchmark intersection.

| Dataset | Property | n | R² | RMSE |
|---------|----------|---|-----|------|
| S01 (Naef 2024, full) | log*K*ow | 3 344 | **0.857** | 0.786 |
| S02 (Naef 2024, full) | log*K*oa | 1 983 | **0.785** | 1.387 |
| S03 (Naef 2024, full) | log*K*aw | 2 150 | **0.654** | 1.758 |
| S01∩S02 intersection | log*K*ow | 3 319 | 0.857 | 0.785 |
| S01∩S02 intersection | log*K*oa | 1 956 | 0.777 | 1.387 |

The remaining error is concentrated in specific chemotypes (notably highly heteroatom-rich agrochemical scaffolds), while the broad SMARTS generalization and pi-environment fixes substantially improved overall log*K*oa performance on S02.

### Correlation plots

| log*K*ow vs Naef S01 | log*K*oa vs Naef S02 | log*K*ow vs Arp & Hale |
|:--------------------:|:--------------------:|:----------------------:|
| ![logKow vs S01](docs/imgs/corr_smarts_kow_vs_s01.png) | ![logKoa vs S02](docs/imgs/corr_smarts_koa_vs_s02.png) | ![logKow vs Excel](docs/imgs/corr_smarts_kow_vs_excel.png) |

---

## Feature engineering

Each molecule is represented by counts of SMARTS atom-type groups from the Naef & Acree parameter tables, plus five special-group descriptors:

- **pi-neighbour moieties** — the number of conjugated systems adjacent to a centre atom (controls which entry in a pi-stratified table applies; computed by `count_conjugated_neighbor_moieties`)
- **H-acceptor binary presence** — 1 if any intramolecular H-bond donor/acceptor pair is within 5 bonds
- **Alkane flag** — 1 if the molecule is a pure saturated hydrocarbon
- **Unsaturated HC flag** — 1 if the molecule is a pure unsaturated hydrocarbon
- **Extra −COOH count** — number of carboxylic acid groups beyond the first
- **Endocyclic C−C single bond count**

The `PartitionCalculator` additionally uses 72 Crippen atom-type features (from RDKit's `Crippen.txt`) on top of the 5 Naef special groups.

### Extended metrics

In addition to R² and RMSE, kawow reports the following metrics for each trained endpoint:

| Metric | Symbol | Description |
|--------|--------|-------------|
| **CCC** | — | Lin's concordance correlation coefficient (Lin 1989): combines precision and accuracy in [−1, 1]. |
| **NRMSE (σ)** | `nrmse_sd_cv` | RMSE / σ(y_train): scale-free error relative to training set spread. |
| **NRMSE (range)** | `nrmse_range_cv` | RMSE / (max − min)(y_train): error as a fraction of the data range. |
| **log₁₀(BF₁₀)** | `bf10_log10_cv` | BIC-approximated Bayes factor for H₁: ρ ≠ 0 (Ly et al. 2015). Values > 1 decisively support the correlation hypothesis. At n > 3000, all models give ">10^300". |
| **95 % CI on ρ** | `r_ci95_cv` | Fisher z-transform confidence interval on the cross-validated Pearson correlation. |

<!-- METRICS_DATA_BEGIN -->

| Model | Property | n | CCC | NRMSE (σ) | NRMSE (range) | log₁₀(BF₁₀) |
|-------|----------|---|-----|-----------|---------------|-------------|
| `kawow` | logKow | 3319 | 0.946 | 0.319 | 0.039 | >10^300 |
| `kawow` | logKoa | 1956 | 0.968 | 0.252 | 0.043 | >10^300 |
| `smarts` | logKow | 3319 | 0.925 | 0.378 | 0.046 | >10^300 |
| `smarts` | logKoa | 1956 | 0.894 | 0.472 | 0.082 | >10^300 |
| `smarts_mixed` | logKow | 3319 | 0.968 | 0.249 | 0.030 | >10^300 |
| `smarts_mixed` | logKoa | 1956 | 0.971 | 0.239 | 0.041 | >10^300 |
| `mqg` | logKow | 3319 | 0.483 | 0.817 | 0.099 | >10^300 |
| `mqg` | logKoa | 1956 | 0.783 | 0.594 | 0.102 | >10^300 |
| `naef_crippen_mqg` | logKow | 3319 | 0.969 | 0.245 | 0.030 | >10^300 |
| `naef_crippen_mqg` | logKoa | 1956 | 0.970 | 0.242 | 0.042 | >10^300 |
| `pfasgroups` | logKow | 3319 | 0.880 | 0.464 | 0.056 | >10^300 |
| `pfasgroups` | logKoa | 1956 | 0.945 | 0.322 | 0.056 | >10^300 |
| `pfasgroups_mixed` | logKow | 3319 | 0.954 | 0.298 | 0.036 | >10^300 |
| `pfasgroups_mixed` | logKoa | 1956 | 0.970 | 0.242 | 0.042 | >10^300 |
| `pfasgroups_naef` | logKow | 3319 | 0.969 | 0.246 | 0.030 | >10^300 |
| `pfasgroups_naef` | logKoa | 1956 | 0.968 | 0.247 | 0.043 | >10^300 |
| `pfasgroups_naef_mixed` | logKow | 3319 | 0.970 | 0.240 | 0.029 | >10^300 |
| `pfasgroups_naef_mixed` | logKoa | 1956 | 0.970 | 0.239 | 0.041 | >10^300 |

<!-- METRICS_DATA_END -->

Access these via `model_info` (see code example above) or via the benchmark script output (`benchmark_results.csv`).

### Y-randomization (permutation test)

Run via `shared_fold_benchmark.py --y-randomization` (default: 1 000 permutations, Ridge 5-fold CV).
Trainable models are tested (kawow, smarts_mixed, mqg, naef_crippen_mqg, pfasgroups,
pfasgroups_mixed, pfasgroups_naef, pfasgroups_naef_mixed); the pure SMARTS lookup is excluded.
Results are saved to `tests/out/y_randomization.csv`.

A non-significant p-value (fraction of permuted R² ≥ observed R²) confirms the model captures genuine structure–property relationships rather than overfitting to label order.
Use the CSV directly for publication numbers, because observed values depend on run settings
(e.g., max-samples/permutation count) while permutation baselines remain near zero.


---

## Reference

Naef, Rudolf, and William E. Acree, Jr. 2024. "Calculation of the Three Partition Coefficients logPow, logKoa and logKaw of Organic Molecules at Standard Conditions at Once by Means of a Generally Applicable Group-Additivity Method." *Liquids* 4, no. 1: 231–260. [10.3390/liquids4010011](https://doi.org/10.3390/liquids4010011)

Arp, H.P.H. and Hale, S.E. 2023. "From Measured Partition Coefficients to the Prediction of Environmental Fate." Supplementary data: `vg2c00024_si_001` (ACS).

Lin, L.I.-K. 1989. "A Concordance Correlation Coefficient to Evaluate Reproducibility." *Biometrics* 45: 255–268. [10.2307/2532051](https://doi.org/10.2307/2532051)

Ly, A., Verhagen, A.J., and Wagenmakers, E.-J. 2015. "Harold Jeffreys's Default Bayes Factor for Testing Point Null Hypotheses from a Continuous Prior Distribution." *Journal of Mathematical Psychology* 28: 71–84. [10.1016/j.jmp.2015.06.004](https://doi.org/10.1016/j.jmp.2015.06.004)

Steiger, J.H. 1980. "Tests for Comparing Elements of a Correlation Matrix." *Psychological Bulletin* 87: 245–251. [10.1037/0033-2909.87.2.245](https://doi.org/10.1037/0033-2909.87.2.245)

Chicco, D., Tötsch, N., and Jurman, G. 2021. "The Matthews Correlation Coefficient (MCC) Is More Reliable Than Balanced Accuracy." *BioData Mining* 14: 13. [10.1186/s13040-021-00244-z](https://doi.org/10.1186/s13040-021-00244-z)

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — Luc T. Miaz, 2026
