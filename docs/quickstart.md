# Quick start

## Running multiple models at once

The easiest entry point is `run_models()`, which runs any combination of the
four available models and returns aligned per-molecule results with B/vB,
M/vM, and regulatory-gap flags already computed:

```python
import kawow

results = kawow.run_models(
    ["CCCCO", "c1ccccc1", "OC(=O)c1ccccc1"],
    models=["kawow", "smarts_mixed"],   # omit to run all four
)

for row in results:
    kow = row["models"]["kawow"]
    mix = row["models"]["smarts_mixed"]
    print(
        f"{row['smiles']:35s}  "
        f"kawow logKow={kow['logKow']:+.2f}  {kow['b_class']}/{kow['m_class']}"
        f"  |  mixed logKow={mix['logKow']:+.2f}  {mix['b_class']}/{mix['m_class']}"
        f"  gaps={kow.get('gap_labels', [])}"
    )
```

Available model keys: `"kawow"`, `"smarts"`, `"smarts_mixed"`, `"mqg"`.

### Flagging criteria

Each model result dictionary includes:

**Bioaccumulation** (following doi:10.1126/science.1138275):

| Key | Value | Condition |
|-----|-------|-----------|
| `b_class` | `"vB"` | `logKoa ≥ 6` **and** `logKow ≥ 5` |
| `b_class` | `"B"` | `logKoa ≥ 6` **and** `logKow ≥ 2` |
| `b_class` | `"non-B"` | otherwise |

**Mobility** — via `logKoc_est = logKow − 0.4` (UBA drinking-water guidance):

| Key | Value | Condition | Equivalent `logKow` |
|-----|-------|-----------|---------------------|
| `m_class` | `"vM"` | `logKoc_est ≤ 3.5` | `logKow ≤ 3.9` |
| `m_class` | `"M"` | `logKoc_est ≤ 4.5` | `logKow ≤ 4.9` |
| `m_class` | `"non-M"` | otherwise | |

**Regulatory gaps**:

| Key | Condition |
|-----|-----------|
| `in_gap1` | `3.5 < logKow < 5.0` |
| `in_gap2` | `logKow > 4.5` and `logKoa < 6` |
| `in_gap3` | `4.5 < logKow < 5.0` and `logKoa < 6` |
| `gap_labels` | list of applicable gap strings, e.g. `["Gap 1", "Gap 3"]` |

---

## Basic prediction with a single model

```python
from kawow import PartitionCalculator

calc = PartitionCalculator()

# Single SMILES string
result = calc.predict("CCCCO")      # 1-butanol
print(result)
# {'logKow': 0.88, 'logKoa': 4.12, 'logKaw': -3.24, 'status': 'ok', 'name': '...'}
```

The returned dictionary always contains:

| Key      | Type  | Description                                 |
|----------|-------|---------------------------------------------|
| `logKow` | float | Octanol-water partition coefficient (log)   |
| `logKoa` | float | Octanol-air partition coefficient (log)     |
| `logKaw` | float | Air-water partition coefficient (log), = logKow − logKoa |
| `status` | str   | `"ok"` or `"error"`                        |
| `name`   | str   | Molecule name (from input, if available)    |
| `error`  | str   | Error description (only when status="error") |

## Batch prediction from a list of SMILES

```python
smiles = ["c1ccccc1", "CCCCCCCCCC", "OC(=O)c1ccccc1"]
results = calc.predict_batch(smiles)

for r in results:
    print(f"{r['smiles']:30s}  logKow={r['logKow']:+.2f}")
```

## Predict from an SDF file

```python
results = calc.predict("compounds.sdf")   # list[dict]
```

The SDF `_Name` field (or `Alias name` property) is used as the molecule name.

## Predict from an InChI string

```python
result = calc.predict("InChI=1S/C4H10O/c1-2-3-4-5/h5H,2-4H2,1H3")
```

## Predict from a multi-line SMILES string

```python
smiles_block = """
CCCCO  1-butanol
c1ccccc1  benzene
"""
results = calc.predict(smiles_block)  # list[dict]
```

## Mixed input list

```python
from rdkit import Chem
mol = Chem.MolFromSmiles("c1ccccc1")

results = calc.predict(["CCCCO", mol, "InChI=1S/..."])
```

## Feature vectors

Access the underlying 77-dimensional feature vector:

```python
from rdkit import Chem
from kawow import compute_features, FEATURE_LABELS

mol = Chem.MolFromSmiles("CCCCO")
vec = compute_features(mol)    # numpy array, shape (77,)

# Print non-zero features
for label, value in zip(FEATURE_LABELS, vec):
    if value != 0:
        print(f"  {label}: {value:.0f}")
```

## Re-fitting the model

If you have your own SDF training data, you can re-fit the Ridge regression
models and save updated coefficients:

```python
import kawow

kawow.fit(
    sdf_logkow="my_logkow.sdf",
    sdf_logkoa="my_logkoa.sdf",
    logkow_prop="logP",      # SDF property tag containing logKow values
    logkoa_prop="logKoa",    # SDF property tag containing logKoa values
)
# Reload the calculator to use the new coefficients
calc = kawow.PartitionCalculator()
```

The fitted coefficients are saved as JSON files inside the package directory
(`kawow/data/logkow_model.json` and `logkoa_model.json`).

## Model metadata

```python
info = calc.model_info
print(info["logKow"])
# {
#   'target': 'logKow', 'n_train': 3332, 'alpha': 37.28,
#   'r2_cv': 0.9039, 'rmse_cv': 0.6433, 'intercept': -0.412
# }
```
