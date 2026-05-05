# Changelog

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
- Ridge regression models for logKow (R²=0.904) and logKoa (R²=0.938)
- Derived logKaw = logKow − logKoa (held-out R²=0.877 on S03)
- `PartitionCalculator` accepting SMILES, InChI, SDF files, RDKit Mol objects
- Pre-fitted JSON model coefficients (Naef & Acree, Liquids 2024 training data)
