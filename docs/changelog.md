# Changelog

## 0.1.0 (2025-04-28)

Initial release.

- 77-feature group-additivity descriptor (72 Crippen atom types + 5 Naef special groups)
- Ridge regression models for logKow (R²=0.904) and logKoa (R²=0.938)
- Derived logKaw = logKow − logKoa (held-out R²=0.877 on S03)
- `PartitionCalculator` accepting SMILES, InChI, SDF files, RDKit Mol objects
- Pre-fitted JSON model coefficients (Naef & Acree, Liquids 2024 training data)
