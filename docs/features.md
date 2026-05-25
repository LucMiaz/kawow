# Feature engineering

## Overview

Each molecule is encoded as a **77-dimensional feature vector**:

- **Features 0–71**: Crippen atom-type counts (72 types)
- **Features 72–76**: Naef special groups (5 values)

## Crippen atom types

The 72 atom types are taken directly from RDKit's `Crippen.txt` data file.
Each atom type is identified by an ID string (e.g. `C1`, `N3`, `O12`) and a
set of atom-centred SMARTS patterns.

Atom assignment follows **first-match priority**: each heavy atom is assigned
to the first atom type whose SMARTS matches it (in the order listed in
`Crippen.txt`). This matches RDKit's `Crippen._pyGetAtomContribs` logic.

The Crippen *logP contributions* (MR and logP weights) are **not used** —
only the SMARTS patterns are used to count atoms. New Ridge regression
coefficients are fitted on the Naef & Acree (2024) experimental datasets.

## Special groups

| Feature label     | Description                                                     |
|-------------------|-----------------------------------------------------------------|
| `SG_HB_INTRAMOL`  | Count of intramolecular H-bond donor/acceptor pairs within 5 bonds |
| `SG_ALKANE`       | 1 if the molecule is a pure saturated hydrocarbon (C, H only, no rings) |
| `SG_UNSAT_HC`     | 1 if the molecule is a pure unsaturated acyclic hydrocarbon    |
| `SG_EXTRA_COOH`   | Number of −COOH groups beyond the first                        |
| `SG_ENDOCYCLIC_CC`| Count of endocyclic C−C single bonds                           |

These special groups capture structural features that the atom-type counts
alone cannot represent (conformational H-bonding, branching geometry, etc.),
following Section 3.4 of Naef & Acree (2024).

## Accessing the feature vector

```python
from rdkit import Chem
from kawow import compute_features, FEATURE_LABELS, N_FEATURES

print("Number of features:", N_FEATURES)    # 77
print("Labels:", FEATURE_LABELS[:5], "...")

mol = Chem.MolFromSmiles("OC(=O)CC(=O)O")  # malonic acid
vec = compute_features(mol)

for label, val in zip(FEATURE_LABELS, vec):
    if val != 0:
        print(f"  {label}: {val:.0f}")
```

## Inspecting atom-type patterns

```python
from kawow.atom_types import CRIPPEN_PATTS, CRIPPEN_ORDER

# All SMARTS for atom type C18 (aromatic C-H)
for smarts, _ in CRIPPEN_PATTS["C18"]:
    print(smarts)
```

---

## PFASGroups features (77-dimensional)

The `pfasgroups` and `pfasgroups_mixed` models use a separate 77-dimensional
feature vector computed by `kawow.pfasgroups_features.compute_pfasgroups_features`.
It is designed to capture the halogenated structural character relevant for
partitioning in fluorinated and chlorinated compounds.

| Part | Indices | Description |
|------|---------|-------------|
| A | 0–10 | 11 molecule-level scalars: total F, Cl, Br, I counts; perfluorination and perchlorination density; CF₂ and CCl₂ density; total branching index; heavy atom count; total carbon count |
| B | 11–13 | 3 aggregate match counts for polyhalogenated groups (PFASGroups IDs 35, 38, 45) |
| C | 14–28 | 15 per-halogen maximum component sizes for perhalogenated groups (IDs 34, 37, 44) × 5 halogens (F, Cl, Br, I, H) |
| D | 29–76 | 48 bag-of-groups counts (generic SMARTS groups, IDs 29–76) |

Parts B–D are computed using `PFASGroups.extract_group_features`.

### Accessing PFASGroups features

```python
from rdkit import Chem
from kawow.pfasgroups_features import compute_pfasgroups_features, N_FEATURES

print("PFASGroups feature vector size:", N_FEATURES)   # 77

mol = Chem.MolFromSmiles("FC(F)(F)C(F)(F)C(F)(F)F")   # perfluorobutane
vec = compute_pfasgroups_features(mol)

if vec is not None:
    print("Part A (mol scalars):", vec[:11])
    print("Part D (group counts, non-zero):", [(i, v) for i, v in enumerate(vec[29:], 29) if v > 0])
```

The `pfasgroups_mixed` model concatenates the 77-dim PFASGroups vector with the
77-dim Crippen atom-type vector (Section above), giving a 154-dimensional input.

