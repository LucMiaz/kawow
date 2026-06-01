# Changelog

## 0.2.0 (2026-06-01)
- Add random forest, boosting and neural net models based on PFASGroups, Crippen and Naef&Acree embeddings
- Verified gaps and fixed typos

## 0.1.6 (2026-05-28)
- Added support for more variants of PFASGroups models
- Updated benchmark, including y-randomization

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

## 0.1.1 (2026-05-05)

- Added `MQGPartitionCalculator` — random-forest model using Molecular Quantum Graph (MQG) features
- Added `NaefAcreeCrippenMixedPartitionCalculator` — hybrid Naef & Acree additivity combined with Crippen Ridge regression
- Added `run_models()` convenience function to predict with multiple models in one call and return aligned per-molecule results with B/vB and M/vM flags
- Added `fit_mqg()` for re-training the MQG-based model on custom datasets
- Added GitHub repository URL to `pyproject.toml`
- Fixed `__version__` to match `pyproject.toml`

## 0.1.0 (2025-04-28)

Initial release.

- 77-feature group-additivity descriptor (72 Crippen atom types + 5 Naef special groups)
- Ridge regression models for logKow (R² = 0.898, cv) and logKoa (R² = 0.937, cv) on shared S01∩S02 benchmark
- Derived logKaw = logKow − logKoa (Naef Eq. 2, not directly trained)
- `PartitionCalculator` accepting SMILES, InChI, SDF files, RDKit Mol objects
- Pre-fitted JSON model coefficients (Naef & Acree, Liquids 2024 training data)
