# Changelog

All notable changes to kawow are listed here, newest first. Versions follow
[semantic versioning](https://semver.org/); dates are those of the version bump
in `pyproject.toml`.

## Unreleased
- **Corrected training data.** The Naef & Acree (2024) supplementary files S01-S03 in `tests/test_data/` contained 30 compounds whose structure is not the named compound (the value is the one of the named compound): 27 pesticides and drugs (47 records across the three files, e.g. butachlor and heptenophos without their chlorine, hexaflumuron and flurprimidol with CH2 in place of an ether oxygen, phenobarbital drawn as 5,5-diphenylbarbituric acid) and the 6:2, 8:2 and 10:2 fluorotelomer acrylates of S02, drawn as allyl 3-(perfluoroalkyl)propanoates. Their structures are replaced, and five implausible log*K*ow values of S01 are removed (triazolam and alprazolam about 3 log units above PHYSPROP, piretanide 2.4 below, carbosulfan inconsistent with its own S02/S03 values, and 1-azido-4-chlorobenzene at -3.71, probably a sign error). Every change and its reason is in `tests/test_data/naef2024_corrections.csv`; `scripts/correct_naef_data.py` applies them to the published files, leaving every other record byte for byte. S01 now has 3 365 records (common evaluation set n = 3 314).
- **Structures checked against PubChem.** `scripts/check_naef_structures_pubchem.py` resolves every name of S01-S03 with PubChemCompounds and compares the PubChem structure with the published and corrected structures (`tests/test_data/naef2024_pubchem_check.csv`; the name cache it writes is not tracked). All 50 corrected structures are the PubChem structures of the named compounds. Of the 92 other records that differ from PubChem, `tests/test_data/naef2024_pubchem_review.csv` classifies 27 records (20 compounds) as further structure errors and 8 records (4 compounds) as uncertain; they are not yet corrected.
- **All fitted models refitted** on the corrected data, and the five-model benchmark of the manuscript (`tests/out/benchmark_*_5model.csv`) regenerated. `tests/out/shared_fold_benchmark.csv` (the R² of the README model table) is not yet rerun.
- The single azide of S01 (the -3.71 value above) set the azide coefficient of the ridge models, so azides were predicted 6 to 8 log units too low (median shift over the 363 azides of the C&L Inventory). S01 now contains no azide, so azides are outside the models' domain.
- `scripts/fit_ensemble_models.py` failed for the `crippen_mqg` ensemble, which has no Naef parameter file (regression in 1.1.1).
- `scripts/fit_pfasgroups_model.py` and `scripts/fit_pfasgroups_advanced_models.py` wrote the pre-1.0 file names (`*_pfasgroups_mixed_*`, `*_pfasgroups_naef_mixed_*`), so refitting did not replace the models kawow loads. They now write `*_pfasgroups_crippen_*` and `*_pfasgroups_naef_crippen_*`.
- `manuscript/README.md`: XGBoost results depend on the processor architecture (ARM64 and x86-64 give different models with the same data and versions); the four other manuscript models agree to below 1e-5.
- README: the log*K*aw comparison with S03 is not a fully independent validation, because many log*K*oa values of S02 (Ebert et al. 2023) were themselves derived from measured log*K*ow and log*K*aw.

## 1.1.1 (2026-11-12)
- **The `mqg`, `naef_mqg` and `naef_crippen_mqg` models work on other machines.** Their pickles store the absolute Windows path of the Naef parameter CSV on the machine they were fitted on, so loading them raised `FileNotFoundError` everywhere else (including installs from PyPI). The path is now resolved in kawow's data directory, and `scripts/fit_ensemble_models.py` stores only the file name.
- **Invalid inputs are reported, not dropped.** A SMILES or InChI that RDKit cannot parse now gives a row with `status="error"` in its position, in `run_models()` and every calculator's `predict()`. Before, it was skipped silently: `run_models()` returned fewer rows than inputs, `predict()` on a list of two returned a single dict, and `predict()` on one invalid string returned `[]`. `kawow.io.parse_input()` takes `keep_invalid=True` for this.
- Importing kawow no longer silences RDKit for the whole process (`RDLogger.DisableLog("rdApp.*")` in `smarts_model`).
- Importing kawow no longer puts `../../PFASGroups` first on `sys.path`; that sibling checkout is only used when PFASGroups is not installed.
- `kawow.__version__` is read from the package metadata.
- Changelog moved to `CHANGELOG.md` at the repository root.

## 1.1.0 (2026-10-24)
- `pandas` is now a core dependency: `kawow.smarts_model` imports it at module level, so `import kawow` (and the Read the Docs build) failed without it
- Added `kawow.criteria` with two regulatory presets, `"uba"` (default, M as logKoc < 4.0) and `"clp"` (EU CLP Regulation, M as logKoc < 3.0, vM as logKoc < 2.0), and `get_criteria()` for custom thresholds
- `run_models()` and `_classify_partition()` take a `criteria` argument; results report the preset used (`criteria`)
- Added aquatic bioaccumulation flags (`is_B_aquatic`, `is_vB_aquatic`)
- **Behaviour change:** `in_gap` (and `in_gap3`, `in_gap_bm`) now follows the benchmark article: neither M nor B by either route, bounded above by the aquatic B cutoff (logKow ≈ 4.6) instead of vB (5.0)
- Corrected the README/quickstart description of the mobility proxy (Arp & Hale screening zones, not `logKow - 0.4`)
- Updated the README and documentation: benchmark values from `tests/out/shared_fold_benchmark.csv` and `benchmark_metrics_5model.csv`, current outputs of the examples, 78-feature Crippen vector with six special groups, removed the obsolete regulatory F1 table, fixed the API reference (`NaefAcreeCrippenPartitionCalculator`, ensemble, PFASGroups RF/XGB/NN classes, `kawow.criteria`) and the Sphinx version
- Added `manuscript/` with the scripts of the benchmark article

## 1.0.2 (2026-06-08)
- `run_models()` accepts the pre-1.0 model names again (`naefacree` → `naef`, `naefacree_mixed` → `naef_crippen`)
- Fixed the `pfasgroups_crippen` and `pfasgroups_naef_crippen` entries of `run_models()`
- `scikit-learn` requirement relaxed from `>=1.8` to `>=1.3`; CI and test dependencies fixed

## 1.0.0 (2026-06-02)
- renaming of models for clarity (breaking change)

## 0.2.0 (2026-06-01)
- Add random forest, boosting and neural net models based on PFASGroups, Crippen and Naef&Acree embeddings
- Verified gaps and fixed typos

## 0.1.6 (2026-05-28)
- Added support for more variants of PFASGroups models
- Updated benchmark, including y-randomization

## 0.1.5 (2026-05-25)
- Fixed the import of the PFASGroups models

## 0.1.4 (2026-05-25)

- Added `PFASGroupsPartitionCalculator` — Ridge regression on 77-dim PFASGroups halogenated-group descriptor (`pfasgroups`) and on PFASGroups + Crippen concatenated features (`pfasgroups_mixed`)
- Added `scripts/fit_pfasgroups_model.py` to train and save pfasgroups models from SDF files
- Updated `run_models()` to support `"pfasgroups"` and `"pfasgroups_mixed"` model keys
- Updated `docs/features.md` with PFASGroups feature vector documentation (Parts A-D)

## 0.1.3 (2026-05-15)

- Corrected performance metrics throughout README and documentation (values now sourced directly from model files and shared S01∩S02 benchmark)
- Added Gap 1/2/3 regulatory-gap flags to `_classify_partition()` output (`in_gap1`, `in_gap2`, `in_gap3`, `gap_labels`)
- Clarified `mqg` model description: Ridge regression *ensemble* of Naef group contributions + Crippen atom-type features + Molecular Quantum Graph fingerprints (not a standalone random forest)
- Fixed M/vM flagging documentation: M is `logKoc_est ≤ 4.5` (i.e. `logKow ≤ 4.9`), vM is `logKoc_est ≤ 3.5` (i.e. `logKow ≤ 3.9`)

## 0.1.2 (2026-05-05)
- Fixed the return value of `run_models()` for the MQG and mixed models

## 0.1.1 (2026-05-05)

- Added `MQGPartitionCalculator` — random-forest model using Molecular Quantum Graph (MQG) features
- Added `NaefAcreeCrippenMixedPartitionCalculator` — hybrid Naef & Acree additivity combined with Crippen Ridge regression
- Added `run_models()` convenience function to predict with multiple models in one call and return aligned per-molecule results with B/vB and M/vM flags
- Added `fit_mqg()` for re-training the MQG-based model on custom datasets
- Added GitHub repository URL to `pyproject.toml`
- Fixed `__version__` to match `pyproject.toml`

## 0.1.0 (2026-04-28)

Initial release.

- 77-feature group-additivity descriptor (72 Crippen atom types + 5 Naef special groups)
- Ridge regression models for logKow (R² = 0.898, cv) and logKoa (R² = 0.937, cv) on shared S01∩S02 benchmark
- Derived logKaw = logKow − logKoa (Naef Eq. 2, not directly trained)
- `PartitionCalculator` accepting SMILES, InChI, SDF files, RDKit Mol objects
- Pre-fitted JSON model coefficients (Naef & Acree, Liquids 2024 training data)
