"""
features.py
===========
Compute the (N_ATOM_TYPES + N_SPECIAL_GROUPS)-dimensional feature vector
for a single RDKit molecule, as required by the Naef group-additivity model.

Atom-type assignment follows RDKit's Crippen logic:
  - each atom is assigned to the FIRST matching pattern in priority order
  - the feature vector counts how many atoms belong to each type
This prevents double-counting when multiple SMARTS for the same type all match.
"""

from __future__ import annotations
import numpy as np
from rdkit import Chem

from .atom_types import (
    CRIPPEN_ORDER,
    CRIPPEN_PATTS,
    N_ATOM_TYPES,
    N_FEATURES,
)

# ── SMARTS helpers for special groups ────────────────────────────────────────
_COOH_SMARTS = Chem.MolFromSmarts("C(=O)[OH]")


def _assign_atom_types(mol: Chem.Mol) -> list[int | None]:
    """
    Assign each atom index to a Crippen type index (or None if unmatched).
    Follows the same first-match priority logic as Crippen._pyGetAtomContribs.
    """
    n_atoms = mol.GetNumAtoms()
    assigned = [None] * n_atoms    # type_index or None
    done     = [False] * n_atoms

    for type_idx, type_id in enumerate(CRIPPEN_ORDER):
        for _smarts, patt in CRIPPEN_PATTS[type_id]:
            for match in mol.GetSubstructMatches(patt, uniquify=False):
                first = match[0]
                if not done[first]:
                    done[first] = True
                    assigned[first] = type_idx
        if all(done):
            break

    return assigned


def _count_intramol_hbonds(mol: Chem.Mol) -> int:
    """Count intramolecular H-bond donor/acceptor atom pairs within 5 bonds."""
    try:
        mol_h = Chem.AddHs(mol)
        donor_atoms = [
            a.GetIdx() for a in mol_h.GetAtoms()
            if a.GetAtomicNum() in (7, 8, 15, 16)
            and any(nb.GetAtomicNum() == 1 for nb in a.GetNeighbors())
        ]
        acceptor_atoms = [
            a.GetIdx() for a in mol_h.GetAtoms()
            if a.GetAtomicNum() in (7, 8)
            and a.GetIdx() not in donor_atoms
        ]
        if not donor_atoms or not acceptor_atoms:
            return 0
        dm = Chem.GetDistanceMatrix(mol_h)
        count = 0
        for d in donor_atoms:
            for a in acceptor_atoms:
                if 2 <= dm[d, a] <= 5:
                    count += 1
        return count
    except Exception:
        return 0


def _is_pure_hc(mol: Chem.Mol) -> tuple[bool, bool]:
    """Return (is_alkane, is_unsat_hc): pure hydrocarbon checks."""
    atom_nums = set(a.GetAtomicNum() for a in mol.GetAtoms())
    if not atom_nums.issubset({1, 6}):
        return False, False
    has_pi = any(
        b.GetBondTypeAsDouble() > 1.0
        for b in mol.GetBonds()
        if b.GetBeginAtom().GetAtomicNum() == 6
        and b.GetEndAtom().GetAtomicNum() == 6
    )
    has_rings = mol.GetRingInfo().NumRings() > 0
    if not has_pi and not has_rings:
        return True, False
    if has_pi and not has_rings:
        return False, True
    return False, False


def _count_extra_cooh(mol: Chem.Mol) -> int:
    if _COOH_SMARTS is None:
        return 0
    return max(0, len(mol.GetSubstructMatches(_COOH_SMARTS)) - 1)


def _count_endocyclic_cc(mol: Chem.Mol) -> int:
    return sum(
        1 for b in mol.GetBonds()
        if b.IsInRing()
        and b.GetBondTypeAsDouble() == 1.0
        and b.GetBeginAtom().GetAtomicNum() == 6
        and b.GetEndAtom().GetAtomicNum() == 6
    )


# ── Public API ────────────────────────────────────────────────────────────────

def compute_features(mol: Chem.Mol) -> np.ndarray | None:
    """
    Return a (N_FEATURES,) float32 array:
      [0..N_ATOM_TYPES-1]  — Crippen atom-type occurrence counts
      [N_ATOM_TYPES..]     — special group values

    Returns None if the molecule has fewer than 2 backbone heavy atoms.
    """
    # Backbone atoms: bound to ≥ 2 heavy neighbours
    backbone = [
        a for a in mol.GetAtoms()
        if sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum() > 1) >= 2
    ]
    if len(backbone) < 1:
        return None

    vec = np.zeros(N_FEATURES, dtype=np.float32)

    # Crippen atom-type counts (first-match priority, Crippen logic)
    assignments = _assign_atom_types(mol)
    for type_idx in assignments:
        if type_idx is not None:
            vec[type_idx] += 1.0

    # Special groups
    n_sg = N_ATOM_TYPES
    vec[n_sg + 0] = _count_intramol_hbonds(mol)
    is_alkane, is_unsat_hc = _is_pure_hc(mol)
    vec[n_sg + 1] = float(is_alkane)
    vec[n_sg + 2] = float(is_unsat_hc)
    vec[n_sg + 3] = _count_extra_cooh(mol)
    vec[n_sg + 4] = _count_endocyclic_cc(mol)

    return vec

