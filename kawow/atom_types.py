"""
atom_types.py
=============
Atom type definitions for the Naef group-additivity method.

We use RDKit's Crippen atom-type SMARTS as a proxy for Naef's atom types.
Both methods define atom types via atom-centred SMARTS that encode element,
local hybridization, and neighbour types — the same mathematical structure.
The Crippen *contributions* (logP values) are NOT used; only the SMARTS are
used to count atom occurrences. New ridge-regression coefficients are fitted
on the S01/S02 SDF training data (Naef & Acree, Liquids 2024).

Naef special groups (beyond atom counts):
  SG_HB_INTRAMOL  — number of intramolecular H-bond donor/acceptor pairs
                     within 5 bonds (penalises self-sequestered polarity)
  SG_ALKANE       — 1 if pure saturated hydrocarbon (C/H only, no rings)
  SG_UNSAT_HC     — 1 if pure unsaturated hydrocarbon (C/H only, π bonds)
  SG_EXTRA_COOH   — number of -COOH groups beyond the first
  SG_ENDOCYCLIC_CC — number of endocyclic C-C single bonds

Atom types are parsed from the Crippen.txt data file shipped with RDKit.
Each unique ID (e.g. "C1", "N3") represents one atom type. Multiple SMARTS
rows with the same ID are all used during assignment, with earlier rows having
priority (matching RDKit's _pyGetAtomContribs logic).
"""

import os
from pathlib import Path
from rdkit import Chem, RDConfig

# ── Parse Crippen.txt ─────────────────────────────────────────────────────────
# Try installed RDDataDir first, fall back to rdkit-src in workspace
_CRIPPEN_TXT_CANDIDATES = [
    os.path.join(RDConfig.RDDataDir, "Crippen.txt"),
    r"c:\Users\luc\git\rdkit-src\Data\Crippen.txt",
]

def _parse_crippen_txt() -> tuple[list[str], dict[str, list]]:
    """
    Parse Crippen.txt into:
      - order : list of unique atom-type IDs in priority order
      - patts : dict {id -> [(smarts_str, compiled_mol), ...]}
    """
    txt_path = None
    for candidate in _CRIPPEN_TXT_CANDIDATES:
        if os.path.exists(candidate):
            txt_path = candidate
            break
    if txt_path is None:
        raise FileNotFoundError(
            "Cannot find Crippen.txt. Tried:\n" +
            "\n".join(f"  {c}" for c in _CRIPPEN_TXT_CANDIDATES)
        )

    order: list[str] = []
    patts: dict[str, list] = {}
    with open(txt_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 4 or not cols[0].strip():
                continue
            type_id = cols[0].strip()
            smarts  = cols[1].strip().replace('"', '')
            if smarts in ("", "SMARTS"):
                continue
            mol = Chem.MolFromSmarts(smarts)
            if mol is None:
                continue
            if type_id not in order:
                order.append(type_id)
            patts.setdefault(type_id, []).append((smarts, mol))
    return order, patts


# Parsed at module import time (fast: just reads a ~4 KB text file once)
CRIPPEN_ORDER, CRIPPEN_PATTS = _parse_crippen_txt()

# ── Public constants ──────────────────────────────────────────────────────────
CRIPPEN_LABELS: list[str] = CRIPPEN_ORDER   # e.g. ["C1", "C2", ..., "Me2"]
N_ATOM_TYPES: int = len(CRIPPEN_LABELS)

# Special group names (appended after atom-type counts in feature vector)
SPECIAL_GROUP_LABELS: list[str] = [
    "SG_HB_INTRAMOL",
    "SG_ALKANE",
    "SG_UNSAT_HC",
    "SG_EXTRA_COOH",
    "SG_EXTRA_OH",
    "SG_ENDOCYCLIC_CC",
]

N_SPECIAL_GROUPS: int = len(SPECIAL_GROUP_LABELS)
FEATURE_LABELS: list[str] = CRIPPEN_LABELS + SPECIAL_GROUP_LABELS
N_FEATURES: int = N_ATOM_TYPES + N_SPECIAL_GROUPS
