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
_OH_SMARTS   = Chem.MolFromSmarts("[OH]")


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
    """Count intramolecular H-bond donor/acceptor atom pairs within 5 bonds.

    Donor  : acidic H bound to O, N or S
    Acceptor: O, N or F (not already a donor)
    """
    try:
        mol_h = Chem.AddHs(mol)
        donor_atoms = [
            a.GetIdx() for a in mol_h.GetAtoms()
            if a.GetAtomicNum() in (7, 8, 16)
            and any(nb.GetAtomicNum() == 1 for nb in a.GetNeighbors())
        ]
        acceptor_atoms = [
            a.GetIdx() for a in mol_h.GetAtoms()
            if a.GetAtomicNum() in (7, 8, 9)   # N, O or F
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


def _count_hc_carbons(mol: Chem.Mol) -> int:
    """Return the number of carbon atoms in the molecule (pure-HC check done by caller)."""
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)


def _count_extra_cooh(mol: Chem.Mol) -> int:
    if _COOH_SMARTS is None:
        return 0
    return max(0, len(mol.GetSubstructMatches(_COOH_SMARTS)) - 1)


def _count_extra_oh(mol: Chem.Mol) -> int:
    """Count hydroxyl groups beyond the first (the (COH)n special group, n>1)."""
    if _OH_SMARTS is None:
        return 0
    return max(0, len(mol.GetSubstructMatches(_OH_SMARTS)) - 1)


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

    Returns None if the molecule has fewer than 2 backbone heavy atoms
    (backbone atom = heavy atom bound to ≥2 heavy neighbours).
    """
    # Backbone atoms: heavy atoms bound to ≥2 neighbours (any atom, including H).
    # Per Naef: "backbone atoms are characterised in that they are bound to at
    # least two covalently bound neighbour atoms" — i.e. total degree ≥ 2.
    # This correctly rejects single-heavy-atom molecules (CH4, CHCl3, HCN)
    # while accepting CH3COOH (both carbons qualify: CH3 has degree 4, C=O has degree 3).
    mol_h = Chem.AddHs(mol)
    n_backbone = sum(
        1 for a in mol_h.GetAtoms()
        if a.GetAtomicNum() > 1 and a.GetDegree() >= 2
    )
    if n_backbone < 2:
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
    # Naef: Alkane/Unsaturated HC are per-C-atom counts, not binary flags
    n_c = _count_hc_carbons(mol) if (is_alkane or is_unsat_hc) else 0
    vec[n_sg + 1] = float(n_c) if is_alkane   else 0.0
    vec[n_sg + 2] = float(n_c) if is_unsat_hc else 0.0
    vec[n_sg + 3] = _count_extra_cooh(mol)
    vec[n_sg + 4] = _count_extra_oh(mol)
    vec[n_sg + 5] = _count_endocyclic_cc(mol)

    return vec

