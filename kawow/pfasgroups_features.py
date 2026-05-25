"""PFASGroups-based feature extraction for the kawow benchmark.

Feature vector layout (N_FEATURES = 77, fixed size):

  Part A: 11 molecule-level scalars (pure RDKit)
  Parts B+C+D (66 values): delegated to PFASGroups.extract_group_features
    Part B:  3 aggregate match counts for polyhalogenated groups (ids 35, 38, 45)
    Part C: 15 per-halogen max component sizes for perhalogenated groups
             (ids 34, 37, 44) × halogens (F, Cl, Br, I, H)
    Part D: 48 bag-of-groups counts (wildcard generic groups, ids 29-76)

See PFASGroups.GroupFeatureResult and PFASGroups.extract_group_features for
the full specification of Parts B-D.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

# ---------------------------------------------------------------------------
# PFASGroups import (sibling repo: ../../PFASGroups)
# ---------------------------------------------------------------------------
_PFASGROUPS_PATH = Path(__file__).resolve().parents[2] / "PFASGroups"
if str(_PFASGROUPS_PATH) not in sys.path:
    sys.path.insert(0, str(_PFASGROUPS_PATH))

_PFASGROUPS_AVAILABLE = False
_PFASGROUPS_IMPORT_ERROR: str = ""
try:
    from PFASGroups.group_features import extract_group_features as _extract_group_features  # type: ignore[import]
    _PFASGROUPS_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover
    _PFASGROUPS_IMPORT_ERROR = str(_exc)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pre-compiled SMARTS for CF2 and CCl2 density
_CF2_QUERY = Chem.MolFromSmarts("[C]([F])([F])")
_CCL2_QUERY = Chem.MolFromSmarts("[C]([Cl])([Cl])")

# Feature vector size (must equal sum of Part A-D)
N_FEATURES: int = 11 + 3 + 15 + 48  # = 77

# ---------------------------------------------------------------------------
# Part A — molecule-level scalars (pure RDKit, 11 features)
# ---------------------------------------------------------------------------

def _mol_scalars(mol: Chem.Mol) -> np.ndarray:
    """Return 11 molecule-level scalar features (float32).

    [0]  total_F
    [1]  total_Cl
    [2]  total_Br
    [3]  total_I
    [4]  perfluorination_density  = total_F / max(1, mol_size)
    [5]  perchlorination_density  = total_Cl / max(1, mol_size)
    [6]  cf2_density              = CF2_count / max(1, total_carbons)
    [7]  ccl2_density             = CCl2_count / max(1, total_carbons)
    [8]  total_branching          (PFASGroups formula, 1=linear, <1=branched)
    [9]  mol_size                 (heavy atom count)
    [10] total_carbons
    """
    total_F = total_Cl = total_Br = total_I = total_carbons = 0
    branch_points = 0
    for atom in mol.GetAtoms():
        anum = atom.GetAtomicNum()
        if anum == 9:
            total_F += 1
        elif anum == 17:
            total_Cl += 1
        elif anum == 35:
            total_Br += 1
        elif anum == 53:
            total_I += 1
        elif anum == 6:
            total_carbons += 1
            # Branch point: C with ≥3 C-C bonds
            cc_bonds = sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 6)
            if cc_bonds >= 3:
                branch_points += 1

    mol_size = mol.GetNumAtoms()
    perfluorination_density = total_F / max(1, mol_size)
    perchlorination_density = total_Cl / max(1, mol_size)

    cf2_count = len(mol.GetSubstructMatches(_CF2_QUERY)) if _CF2_QUERY is not None else 0
    ccl2_count = len(mol.GetSubstructMatches(_CCL2_QUERY)) if _CCL2_QUERY is not None else 0
    cf2_density = cf2_count / max(1, total_carbons)
    ccl2_density = ccl2_count / max(1, total_carbons)

    # PFASGroups branching: 1 - 2 * branch_points / n_carbons
    total_branching = 1.0 - 2.0 * branch_points / max(1, total_carbons)

    return np.array([
        total_F, total_Cl, total_Br, total_I,
        perfluorination_density, perchlorination_density,
        cf2_density, ccl2_density,
        total_branching,
        mol_size, total_carbons,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_pfasgroups_features(mol: Chem.Mol) -> np.ndarray | None:
    """Compute the fixed-size PFASGroups feature vector for a molecule.

    Returns a float32 array of shape (N_FEATURES,) = (77,), or None if
    PFASGroups parsing fails (molecule is skipped in the benchmark).

    Raises ImportError if the PFASGroups package cannot be found.

    Feature layout
    --------------
    Part A — molecule-level scalars (indices 0-10, pure RDKit):
      [0]  total_F
      [1]  total_Cl
      [2]  total_Br
      [3]  total_I
      [4]  perfluorination_density    total_F / max(1, mol_size)
      [5]  perchlorination_density    total_Cl / max(1, mol_size)
      [6]  cf2_density                CF₂ count / max(1, total_carbons)
      [7]  ccl2_density               CCl₂ count / max(1, total_carbons)
      [8]  total_branching            1 - 2*branch_points/max(1,n_carbons)
      [9]  mol_size
      [10] total_carbons

    Parts B–D — 66 features from PFASGroups.extract_group_features:
      See :class:`PFASGroups.GroupFeatureResult` and
      :func:`PFASGroups.extract_group_features` for the full specification.
      :meth:`~PFASGroups.GroupFeatureResult.to_array` layout:
        [11-13]  poly_counts        (poly_alkyl, poly_aryl, poly_cyclic)
        [14-28]  per_halogen_sizes  (g34_F … g44_H, 3 groups × 5 halogens)
        [29-76]  generic_groups     (g29 … g76, 48 entries)
    """
    if not _PFASGROUPS_AVAILABLE:
        raise ImportError(
            f"PFASGroups package not found at '{_PFASGROUPS_PATH}'. "
            f"Original import error: {_PFASGROUPS_IMPORT_ERROR}"
        )

    try:
        group_feats = _extract_group_features(mol)
    except Exception:
        return None

    return np.concatenate([_mol_scalars(mol), group_feats.to_array()])
