"""
io.py
=====
Parse molecular input in various formats into a list of (RDKit mol, name) pairs.

Supported inputs:
  - SMILES string (single or multi-line)
  - InChI string (single or multi-line)
  - RDKit Mol object (returned as-is)
  - Path to .sdf file  (reads <name> and <SDF property tags>)
  - List of any of the above (mixed allowed)
"""

from __future__ import annotations
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.inchi import MolFromInchi


def _from_smiles(s: str, name: str = "") -> tuple | None:
    s = s.strip()
    if not s or s.startswith("#"):
        return None
    # Allow "SMILES name" or "SMILES\tname" on same line
    parts = s.split(None, 1)
    smi = parts[0]
    label = parts[1].strip() if len(parts) > 1 else name
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol.SetProp("_Name", label or smi)
    mol.SetProp("_SMILES", smi)
    return mol, label or smi


def _from_inchi(s: str, name: str = "") -> tuple | None:
    s = s.strip()
    if not s:
        return None
    mol = MolFromInchi(s)
    if mol is None:
        return None
    mol.SetProp("_Name", name or s[:40])
    return mol, name or s[:40]


def _read_sdf(path: str | Path, target_prop: str | None = None) -> list[tuple]:
    """
    Read an SDF file. Returns list of (mol, name, value) tuples where
    `value` is the float from `target_prop` (or None if not present/parseable).
    """
    results = []
    supplier = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True)
    for mol in supplier:
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        # Some SDFs embed alias names in the Alias_name property
        if mol.HasProp("Alias name"):
            name = mol.GetProp("Alias name")
        elif mol.HasProp("alias name"):
            name = mol.GetProp("alias name")
        value = None
        if target_prop and mol.HasProp(target_prop):
            try:
                value = float(mol.GetProp(target_prop).strip())
            except ValueError:
                pass
        results.append((mol, name.strip(), value))
    return results


def parse_input(inp, fmt: str = "auto") -> list[tuple]:
    """
    Parse any supported input into a list of (mol, name) tuples.

    Parameters
    ----------
    inp : str | Path | Chem.Mol | list
        The molecular input.
    fmt : str
        'auto' | 'smiles' | 'inchi' | 'sdf' | 'mol'

    Returns
    -------
    list of (Chem.Mol, str) — (mol, name) pairs.
    """
    # ── list / iterable ──────────────────────────────────────────────────────
    if isinstance(inp, (list, tuple)):
        out = []
        for item in inp:
            out.extend(parse_input(item, fmt=fmt))
        return out

    # ── RDKit Mol ─────────────────────────────────────────────────────────────
    if hasattr(inp, "GetNumAtoms"):
        name = inp.GetProp("_Name") if inp.HasProp("_Name") else ""
        return [(inp, name)]

    # ── Path / file ───────────────────────────────────────────────────────────
    if isinstance(inp, (str, Path)) and os.path.exists(str(inp)):
        ext = Path(str(inp)).suffix.lower()
        if ext == ".sdf" or fmt == "sdf":
            rows = _read_sdf(inp)
            return [(mol, name) for mol, name, _ in rows]
        if ext in (".smi", ".smiles", ".txt") or fmt == "smiles":
            out = []
            with open(inp, "r", encoding="utf-8") as f:
                for line in f:
                    r = _from_smiles(line.strip())
                    if r:
                        out.append(r)
            return out

    # ── plain string ──────────────────────────────────────────────────────────
    if isinstance(inp, str):
        s = inp.strip()
        if fmt == "inchi" or s.startswith("InChI="):
            # Multi-line InChI
            results = []
            for line in s.splitlines():
                r = _from_inchi(line.strip())
                if r:
                    results.append(r)
            return results
        # SMILES (possibly multi-line)
        results = []
        for line in s.splitlines():
            r = _from_smiles(line.strip())
            if r:
                results.append(r)
        return results

    return []
