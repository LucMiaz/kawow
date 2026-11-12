# Feature engineering

## Crippen feature vector

`kawow.compute_features(mol)` returns a **78-dimensional feature vector**:

- **Features 0–71**: Crippen atom-type counts (72 types)
- **Features 72–77**: special groups following Naef & Acree (2024) (6 values)

It returns `None` for molecules with fewer than two backbone atoms (heavy atoms bound to at least two neighbours, hydrogens included), e.g. methane, water, CF₄ or SOCl₂; these molecules cannot be described by atom types or group contributions.

### Crippen atom types

The 72 atom types are taken from RDKit's `Crippen.txt` data file. Each atom type is identified by an ID (e.g. `C1`, `N3`, `O12`) and a set of atom-centred SMARTS patterns.

Each atom is assigned to the first atom type whose SMARTS matches it, in the order of `Crippen.txt`, as in RDKit's `Crippen._pyGetAtomContribs`. Only the SMARTS patterns are used: RDKit's logP and MR contributions are not, and the ridge coefficients are fitted on the Naef & Acree (2024) experimental datasets.

### Special groups

| Feature label     | Description |
|-------------------|-------------|
| `SG_HB_INTRAMOL`  | Number of intramolecular H-bond donor/acceptor atom pairs within five bonds |
| `SG_ALKANE`       | Number of carbon atoms if the molecule is a pure saturated acyclic hydrocarbon, 0 otherwise |
| `SG_UNSAT_HC`     | Number of carbon atoms if the molecule is a pure unsaturated acyclic hydrocarbon, 0 otherwise |
| `SG_EXTRA_COOH`   | Number of carboxylic-acid groups beyond the first |
| `SG_EXTRA_OH`     | Number of hydroxyl groups beyond the first |
| `SG_ENDOCYCLIC_CC`| Number of endocyclic C–C single bonds |

These groups capture structural features that atom-type counts alone cannot represent (intramolecular H-bonding, hydrocarbon chain length, ring strain), following Naef & Acree (2024).

### Accessing the feature vector

```python
from rdkit import Chem
from kawow import compute_features, FEATURE_LABELS, N_FEATURES

print("Number of features:", N_FEATURES)    # 78

vec = compute_features(Chem.MolFromSmiles("OC(=O)CC(=O)O"))  # malonic acid
for label, val in zip(FEATURE_LABELS, vec):
    if val != 0:
        print(f"  {label}: {val:.0f}")
```

### Inspecting atom-type patterns

```python
from kawow.atom_types import CRIPPEN_PATTS, CRIPPEN_ORDER

# All SMARTS for atom type C18 (aromatic C-H)
for smarts, _ in CRIPPEN_PATTS["C18"]:
    print(smarts)
```

---

## Naef & Acree group counts

The `naef_crippen` model, and the models that combine it with MQG or PFASGroups descriptors, also use the number of matches of each Naef & Acree (2024) group, one vector per endpoint (the log*K*ow and log*K*oa parameter tables differ). Each atom is matched against the SMARTS of the paper's supplementary tables, optionally conditioned on the number of conjugated π-neighbour moieties at the matched centre. The `naef` model sums the tabulated contributions of these groups directly, without fitting.

---

## PFASGroups features (77-dimensional)

The `pfasgroups*` models use a 77-dimensional PFASGroups vector computed by `kawow.pfasgroups_features.compute_pfasgroups_features`. It describes the halogenated structure relevant for the partitioning of fluorinated and chlorinated compounds.

| Part | Indices | Description |
|------|---------|-------------|
| A | 0–10 | 11 molecule-level scalars: total F, Cl, Br, I counts; perfluorination and perchlorination density; CF₂ and CCl₂ density; total branching index; heavy atom count; total carbon count |
| B | 11–13 | 3 aggregate match counts for polyhalogenated groups (PFASGroups IDs 35, 38, 45) |
| C | 14–28 | 15 per-halogen maximum component sizes for perhalogenated groups (IDs 34, 37, 44) × 5 halogens (F, Cl, Br, I, H) |
| D | 29–76 | 48 bag-of-groups counts (generic SMARTS groups, IDs 29–76) |

Parts B–D are computed with `PFASGroups.extract_group_features`.

```python
from rdkit import Chem
from kawow.pfasgroups_features import compute_pfasgroups_features, N_FEATURES

print("PFASGroups feature vector size:", N_FEATURES)   # 77

vec = compute_pfasgroups_features(Chem.MolFromSmiles("FC(F)(F)C(F)(F)C(F)(F)F"))   # perfluorobutane
if vec is not None:
    print("Part A (molecule scalars):", vec[:11])
    print("Part D (non-zero group counts):", [(i, v) for i, v in enumerate(vec[29:], 29) if v > 0])
```

Feature composition of the models:

- `pfasgroups`: PFASGroups (77).
- `pfasgroups_crippen`: PFASGroups (77) + Crippen feature vector (78).
- `pfasgroups_naef`: PFASGroups (77) + Naef & Acree group counts.
- `pfasgroups_naef_crippen` (and its `_rf`, `_xgb`, `_nn` variants): PFASGroups (77) + Naef & Acree group counts + Crippen feature vector (78).
