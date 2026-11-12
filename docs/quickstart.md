# Quick start

## Running several models at once

`run_models()` runs any combination of models and returns aligned per-molecule results with the B/vB, M/vM and regulatory-gap flags already computed:

```python
import kawow

results = kawow.run_models(
    ["CCCCO", "c1ccccc1", "OC(=O)c1ccccc1"],
    models=["crippen", "naef_crippen"],   # omit to run all available models
)

for row in results:
    kow = row["models"]["crippen"]
    mix = row["models"]["naef_crippen"]
    print(
        f"{row['smiles']:20s}"
        f"crippen logKow={kow['logKow']:+.2f} {kow['b_class']}/{kow['m_class']}  |  "
        f"naef_crippen logKow={mix['logKow']:+.2f} {mix['b_class']}/{mix['m_class']}  "
        f"in gap: {mix['in_gap']}"
    )
```

Available model keys: `"crippen"`, `"naef"`, `"naef_crippen"`, `"mqg"`, `"naef_mqg"`, `"crippen_mqg"`, `"pfasgroups"`, `"pfasgroups_crippen"`, `"pfasgroups_naef"`, `"pfasgroups_naef_crippen"`, `"pfasgroups_naef_crippen_rf"`, `"pfasgroups_naef_crippen_xgb"`, `"pfasgroups_naef_crippen_nn"`. The MQG models need the `molecular_quantum_graph` package, and the `_xgb` and `_nn` models need `pip install "kawow[ml]"`; a model that cannot be loaded returns `status="error"` for every molecule.

## Classification criteria

Each model result contains the classification of the predicted values:

**Bioaccumulation**, air-breathing organisms (Kelly et al. 2007, doi:10.1126/science.1138275):

| Key | Value | Condition |
|-----|-------|-----------|
| `b_class` | `"vB"` | `logKoa ≥ 6` **and** `logKow ≥ 5` |
| `b_class` | `"B"` | `logKoa ≥ 6` **and** `logKow ≥ 2` |
| `b_class` | `"non-B"` | otherwise |

**Aquatic bioaccumulation** (`is_B_aquatic`, `is_vB_aquatic`), from BCF ≈ 0.05·Kow: B at `logKow > 4.6` (BCF > 2 000), vB at `logKow > 5.0` (BCF > 5 000).

**Mobility** is defined on log*K*oc and applied through a log*K*ow screening proxy. Two presets are available through the `criteria` argument of `run_models()` (`"uba"` by default), see `kawow/criteria.py`:

| Preset | M (log*K*oc) | vM (log*K*oc) | M (log*K*ow) | Potential M/vM (log*K*ow) | vM (log*K*ow) |
|--------|-------------|---------------|--------------|---------------------------|---------------|
| `uba` (Neumann & Schliebner 2019; Arp & Hale 2019 screening zones) | < 4.0 | < 3.0 | ≤ 4.5 | 3.5–4.5 | ≤ 2.5 |
| `clp` (Delegated Regulation (EU) 2023/707) | < 3.0 | < 2.0 | ≤ 3.5 | 2.5–3.5 | ≤ 1.5 |

`m_class` is `"vM"`, `"Potential M/vM"`, `"M"` or `"non-M"`. The CLP log*K*ow cutoffs are the Arp & Hale zones shifted by the same 1.0 log-unit as the log*K*oc cutoffs. Single thresholds can be changed with `kawow.get_criteria("clp", vm_logkow_threshold=2.5)`, and the result passed as `criteria=`.

**Regulatory gaps**:

| Key | Condition |
|-----|-----------|
| `in_gap` (= `in_gap3`, `in_gap_bm`) | neither M nor B by either route: `logKow` above the M cutoff and below the aquatic B cutoff (≈ 4.6), with `logKoa < 6` |
| `in_gap1` | `logKow` between the vM cutoff and 5.0 (neither vM nor vB) |
| `in_gap2` | `logKow` above the M cutoff and `logKoa < 6` |
| `gap_labels` | list of the gaps that apply, e.g. `["Gap 1", "Gap 3"]` |

The thresholds used are returned in `criteria` and `thresholds`.

---

## Basic prediction with a single model

```python
from kawow import PartitionCalculator

calc = PartitionCalculator()

calc.predict("CCCCO")      # 1-butanol
# {'logKow': 1.095, 'logKoa': 4.368, 'logKaw': -3.273, 'status': 'ok', 'name': 'CCCCO', 'smiles': 'CCCCO'}
```

The returned dictionary contains:

| Key      | Type  | Description                                 |
|----------|-------|---------------------------------------------|
| `logKow` | float | Octanol-water partition coefficient (log)   |
| `logKoa` | float | Octanol-air partition coefficient (log)     |
| `logKaw` | float | Air-water partition coefficient (log), logKow − logKoa |
| `status` | str   | `"ok"`, or `"error"` when no feature vector can be built (fewer than two backbone atoms) |
| `name`   | str   | Molecule name from the input (the SMILES otherwise) |
| `smiles` | str   | Input SMILES                                |

Inputs that RDKit cannot parse are skipped, so the result for an invalid SMILES is an empty list.

## Batch prediction from a list of SMILES

```python
for r in calc.predict_batch(["c1ccccc1", "CCCCCCCCCC", "OC(=O)c1ccccc1"]):
    print(f"{r['smiles']:30s}  logKow={r['logKow']:+.2f}")
```

## Other input formats

```python
calc.predict("compounds.sdf")                                  # SDF file; _Name or "Alias name" as name
calc.predict("InChI=1S/C4H10O/c1-2-3-4-5/h5H,2-4H2,1H3")      # InChI
calc.predict("CCCCO  1-butanol\nc1ccccc1  benzene")           # one "SMILES name" per line

from rdkit import Chem
calc.predict(["CCCCO", Chem.MolFromSmiles("c1ccccc1")])        # mixed list
```

Multiple molecules return a list of dictionaries.

## Feature vectors

The Crippen feature vector has 78 entries (72 atom types and six special groups, see [Features](features.md)):

```python
from rdkit import Chem
from kawow import compute_features, FEATURE_LABELS

vec = compute_features(Chem.MolFromSmiles("CCCCO"))   # numpy array, shape (78,)
for label, value in zip(FEATURE_LABELS, vec):
    if value != 0:
        print(f"  {label}: {value:.0f}")
```

## Re-fitting the model

With your own SDF training data, the ridge models of `PartitionCalculator` can be re-fitted:

```python
import kawow

kawow.fit(
    sdf_logkow="my_logkow.sdf",
    sdf_logkoa="my_logkoa.sdf",
    logkow_prop="logP",      # SDF property holding logKow
    logkoa_prop="logKoa",    # SDF property holding logKoa
)
calc = kawow.PartitionCalculator()   # reload to use the new coefficients
```

The coefficients are written to `kawow/data/logkow_model.json` and `logkoa_model.json` inside the package directory, replacing the shipped ones.

## Model metadata

```python
calc.model_info["logKow"]
# {'target': 'logKow', 'n_train': 3234, 'alpha': 51.79, 'r2_cv': 0.898, 'rmse_cv': 0.6643,
#  'intercept': 0.399, ...}  (plus the fitted coefficients)
```

## Benchmark performance

Cross-validated performance of all models, including classification metrics at the regulatory thresholds, is summarised in the README and stored in `tests/out/shared_fold_benchmark.csv` and `tests/out/benchmark_metrics_5model.csv`.
