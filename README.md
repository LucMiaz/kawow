# Kawow

[![CI](https://github.com/LucMiaz/kawow/actions/workflows/ci.yml/badge.svg)](https://github.com/LucMiaz/kawow/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Group-additivity prediction of **log*K*ow**, **log*K*oa**, and **log*K*aw** from molecular structure.

*Kawow* uses RDKit's Crippen atom-type SMARTS to enumerate structural contributions. Ridge regression coefficients are re-fitted on the S01/S02 experimental datasets of Naef & Acree (2024).

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

## Quick start

```python
from kawow import PartitionCalculator

calc = PartitionCalculator()           # loads pre-fitted JSON models

# Single molecule from SMILES
result = calc.predict("CCCCO")        # 1-butanol
print(result)
# {'logKow': 0.88, 'logKoa': 4.12, 'logKaw': -3.24, 'status': 'ok', 'name': '...'}

# Batch prediction
smiles = [
    "c1ccccc1",        # benzene
    "CCCCCCCCCC",      # decane
    "OC(=O)c1ccccc1",  # benzoic acid
]
for r in calc.predict_batch(smiles):
    print(f"{r['smiles']:30s}  logKow={r['logKow']:+.2f}  logKoa={r['logKoa']:+.2f}  logKaw={r['logKaw']:+.2f}")
```

Expected output (approximate):

| SMILES                      | logKow | logKoa | logKaw |
|-----------------------------|--------|--------|--------|
| `c1ccccc1`                  | +2.13  | +2.80  | −0.67  |
| `CCCCCCCCCC`                | +5.40  | +8.23  | −2.83  |
| `OC(=O)c1ccccc1`            | +1.87  | +7.12  | −5.25  |

### Predict from an SDF file

```python
results = calc.predict("compounds.sdf")  # returns list[dict]
```

### Predict from an InChI string

```python
r = calc.predict("InChI=1S/C4H10O/c1-2-3-4-5/h5H,2-4H2,1H3")
```

### Inspect model metadata

```python
info = calc.model_info
print(info["logKow"])
# {'target': 'logKow', 'n_train': 3332, 'alpha': 37.28,
#  'r2_cv': 0.9039, 'rmse_cv': 0.6433, 'intercept': ...}
```

### Access the feature vector directly

```python
from rdkit import Chem
from kawow import compute_features, FEATURE_LABELS

mol = Chem.MolFromSmiles("CCCCO")
vec = compute_features(mol)   # numpy array, shape (77,)

for label, value in zip(FEATURE_LABELS, vec):
    if value:
        print(f"  {label}: {int(value)}")
```

### Re-fitting the model

To fit on your own training data in SDF format:

```python
import kawow
kawow.fit(
    sdf_logkow="my_logkow.sdf",
    sdf_logkoa="my_logkoa.sdf",
    logkow_prop="logP",      # SDF tag name
    logkoa_prop="logKoa",
)
calc = kawow.PartitionCalculator()  # reload updated coefficients
```

## Performance

Evaluated by 5-fold cross-validation on the Naef & Acree (2024) datasets:

| Property | Training set | n    | R²    | RMSE  |
|----------|-------------|------|-------|-------|
| log*K*ow | S01 (Liquids 2024) | 3332 | 0.904 | 0.643 |
| log*K*oa | S02 (Liquids 2024) | 1971 | 0.938 | 0.736 |
| log*K*aw | S03 (held-out)     | 2136 | 0.877 | 1.048 |

RDKit Crippen baseline (log*K*ow): R² = 0.863, RMSE = 0.768.

## Feature engineering

Each molecule is represented by **77 features**:
- **72 Crippen atom-type counts** (from RDKit's `Crippen.txt`, assigned with first-match priority)
- **5 Naef special groups**:
  - `SG_HB_INTRAMOL` — number of intramolecular H-bond donor/acceptor pairs within 5 bonds
  - `SG_ALKANE` — 1 if the molecule is a pure saturated hydrocarbon
  - `SG_UNSAT_HC` — 1 if the molecule is a pure unsaturated hydrocarbon
  - `SG_EXTRA_COOH` — number of −COOH groups beyond the first
  - `SG_ENDOCYCLIC_CC` — number of endocyclic C−C single bonds

## Reference

R. Naef, W.E. Acree Jr., *Liquids* **4**(1):231–260, 2024.
DOI: [10.3390/liquids4010011](https://doi.org/10.3390/liquids4010011)

## License

MIT
