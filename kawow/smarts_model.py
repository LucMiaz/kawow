"""
smarts_model.py
===============
SMARTS-based Naef 2024 additivity model for logKow/logKoa/logKaw.

This module loads SMARTS contributions from the CSV parameter tables and
accumulates contributions for each molecule. Special groups are handled by
callable functions and initialized with practical RDKit-based templates.
"""

from __future__ import annotations

from collections import deque
import re
from pathlib import Path
from typing import Union

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  

DATA_DIR = Path(__file__).parent / "data"

_COOH_SMARTS = Chem.MolFromSmarts("C(=O)[OH]")


def conjugated_moiety_size(mol: Chem.Mol, start_idx: int, forbidden_idx: int) -> int:
    """
    Return heavy-atom size of a conjugation-connected moiety.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule to inspect.
    start_idx : int
        Atom index where traversal starts (typically a neighbor of a matched atom).
    forbidden_idx : int
        Atom index that is excluded from traversal so the walk does not cross back
        through the matched central atom.

    Notes
    -----
    Traversal follows:
    - conjugated bonds, or
    - non-single bonds (double/triple/aromatic)

    Plain non-conjugated single bonds are not traversed.
    """
    visited = {forbidden_idx, start_idx}
    q = deque([start_idx])
    size = 0

    while q:
        i = q.popleft()
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() > 1:
            size += 1

        for b in atom.GetBonds():
            j = b.GetOtherAtomIdx(i)
            if j in visited or j == forbidden_idx:
                continue
            if not b.GetIsConjugated() and b.GetBondType() == Chem.BondType.SINGLE:
                continue
            visited.add(j)
            q.append(j)

    return size


def count_conjugated_neighbor_moieties(mol: Chem.Mol, center_idx: int) -> tuple[int, list[int]]:
    """
    Count distinct conjugated moieties connected to neighbors of a center atom.

    For each neighbor of ``center_idx``, this function checks whether the bond to
    that neighbor can start a conjugated moiety. A bond is accepted when it is
    conjugated, non-single, or when the neighbor atom itself is aromatic. The
    aromatic-neighbor fallback is needed for patterns like S-Ar where RDKit may
    keep the connecting bond as non-conjugated single despite clear pi context.
    It then computes moiety size with ``conjugated_moiety_size`` while forbidding
    traversal back through ``center_idx``.

    Returns
    -------
    tuple[int, list[int]]
        (number_of_neighbor_moieties, list_of_moiety_sizes)
    """
    center = mol.GetAtomWithIdx(center_idx)
    seen_neighbors = set()
    sizes: list[int] = []

    for nbr in center.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx in seen_neighbors:
            continue

        bond = mol.GetBondBetweenAtoms(center_idx, nbr_idx)
        if bond is None:
            continue
        starts_pi_moiety = (
            bond.GetIsConjugated()
            or bond.GetBondType() != Chem.BondType.SINGLE
            or nbr.GetIsAromatic()
        )
        if not starts_pi_moiety:
            continue

        size = conjugated_moiety_size(mol, nbr_idx, center_idx)
        if size > 0:
            sizes.append(size)
            seen_neighbors.add(nbr_idx)

    return len(sizes), sizes


def _count_backbone_atoms(mol: Chem.Mol) -> int:
    """Count heavy atoms with total degree >= 2 in the H-completed graph."""
    mol_h = Chem.AddHs(mol)
    return sum(
        1 for atom in mol_h.GetAtoms()
        if atom.GetAtomicNum() > 1 and atom.GetDegree() >= 2
    )


def _count_carbon_atoms(mol: Chem.Mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)


def _classify_pure_hydrocarbon(mol: Chem.Mol) -> tuple[bool, bool]:
    """Return (is_alkane, is_unsaturated_hc) for pure hydrocarbons only."""
    atom_nums = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
    if not atom_nums.issubset({1, 6}):
        return False, False

    has_pi = any(
        bond.GetBondTypeAsDouble() > 1.0
        and bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
        for bond in mol.GetBonds()
    )
    has_rings = mol.GetRingInfo().NumRings() > 0

    if not has_pi and not has_rings:
        return True, False
    if has_pi:
        return False, True
    return False, False


def _count_intramolecular_hbond_pairs(mol: Chem.Mol) -> float:
    """Count donor/acceptor pairs consistent with the paper's H-acceptor correction."""
    mol_h = Chem.AddHs(mol)
    donor_atoms = [
        atom.GetIdx()
        for atom in mol_h.GetAtoms()
        if atom.GetAtomicNum() in (7, 8, 16)
        and any(neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors())
    ]
    acceptor_atoms = [
        atom.GetIdx()
        for atom in mol_h.GetAtoms()
        if atom.GetAtomicNum() in (7, 8, 9)
        and atom.GetIdx() not in donor_atoms
    ]
    if not donor_atoms or not acceptor_atoms:
        return 0.0

    distance_matrix = Chem.GetDistanceMatrix(mol_h)
    count = 0
    for donor_idx in donor_atoms:
        for acceptor_idx in acceptor_atoms:
            if 2 <= distance_matrix[donor_idx, acceptor_idx] <= 5:
                count += 1
    return float(count)


def _count_hydroxyl_groups(mol: Chem.Mol) -> int:
    """Count non-carboxylic hydroxyl groups attached to carbon."""
    count = 0
    for oxygen in mol.GetAtoms():
        if oxygen.GetAtomicNum() != 8 or oxygen.GetTotalNumHs(includeNeighbors=True) == 0:
            continue

        carbon_neighbors = [neighbor for neighbor in oxygen.GetNeighbors() if neighbor.GetAtomicNum() == 6]
        if not carbon_neighbors:
            continue

        is_carboxylic_oh = False
        for carbon in carbon_neighbors:
            for bond in carbon.GetBonds():
                if bond.GetBondType() != rdchem.BondType.DOUBLE:
                    continue
                other = bond.GetOtherAtom(carbon)
                if other.GetAtomicNum() == 8:
                    is_carboxylic_oh = True
                    break
            if is_carboxylic_oh:
                break

        if not is_carboxylic_oh:
            count += 1

    return count


def _count_alkane_carbons(mol: Chem.Mol) -> float:
    """
    Count carbon atoms only for pure acyclic alkanes.
    """
    is_alkane, _ = _classify_pure_hydrocarbon(mol)
    return float(_count_carbon_atoms(mol) if is_alkane else 0)

def _count_halide(mol: Chem.Mol) -> float:
    """
    Count the Halide special group as a sparse binary correction.

    The paper reports this as a compound-level correction rather than a per-halogen
    atom count; in S01 this matches the low-backbone halogenated compounds.
    """
    has_halogen = any(atom.GetAtomicNum() in (9, 17, 35, 53) for atom in mol.GetAtoms())
    return 1.0 if has_halogen and _count_backbone_atoms(mol) < 2 else 0.0

def _count_unsaturated_hc_carbons(mol: Chem.Mol) -> float:
    """
    Count carbon atoms only for pure unsaturated hydrocarbons.
    """
    _, is_unsaturated_hc = _classify_pure_hydrocarbon(mol)
    return float(_count_carbon_atoms(mol) if is_unsaturated_hc else 0)


def _count_endocyclic_single_bonds(mol: Chem.Mol) -> float:
    """Count single bonds that are part of a ring."""
    count = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() == rdchem.BondType.SINGLE and bond.IsInRing():
            atomStart = bond.GetBeginAtom()
            atomEnd = bond.GetEndAtom()
            # endocyclic single bonds are counted only for C-C bonds according to the paper
            if atomStart.GetAtomicNum() == 6 and atomEnd.GetAtomicNum() ==6:
                count += 1
    return float(count)


def _count_h_acceptor(mol: Chem.Mol) -> float:
    """Count presence of intramolecular H-bond donor/acceptor interaction."""
    return 1.0 if _count_intramolecular_hbond_pairs(mol) > 0 else 0.0

def _count_coh(mol: Chem.Mol) -> float:
    """Count each additional non-carboxylic hydroxyl group beyond the first."""
    return float(max(0, _count_hydroxyl_groups(mol) - 1))

def _count_cooh(mol: Chem.Mol) -> float:
    """Count each additional carboxylic acid group beyond the first."""
    if _COOH_SMARTS is None:
        return 0.0
    return float(max(0, len(mol.GetSubstructMatches(_COOH_SMARTS)) - 1))



class NaefAcreePartitionCalculator:
    """
    computes logKoa, logKow and logKaw contributions based on the Naef & Acree 2024 additivity model.
    """

    def __init__(self, smiles: Union[list,str,None] = None, inchi: Union[list,str,None] = None, mol: Union[list,Chem.Mol,None] = None, logkow_parameter_file: str =  "naef2024_logkow_parameters.csv", logkoa_parameter_file: str = "naef2024_logkoa_parameters.csv") -> None:
        self._logkow_parameter_file = logkow_parameter_file
        self._logkoa_parameter_file = logkoa_parameter_file
        self._initialise_contributions()
        self.allowed_elements = {'H', 'B', 'C', 'N', 'O', 'P', 'S', 'Si', 'F', 'Cl', 'Br', 'I'}
        self._coefficients = {}
        self.mols = []
        self.mols.extend(self._parse_input(smiles, convert_from='smiles'))
        self.mols.extend(self._parse_input(inchi, convert_from='inchi'))
        self.mols.extend(self._parse_input(mol, convert_from=None))
        self._compute_coefficients()
    
    def clear(self):
        self.mols = []
        self._coefficients = {}
    
    def _load_contributions(self, file_name:str) -> tuple[list, float]:
        _coefficients = []
        _constant = 0.0
        reader = pd.read_csv(DATA_DIR / file_name)
        if "kow" in file_name.lower():
            param= 'logKow'
        else:
            param = 'logKoa'
        for i, row in reader.iterrows():
            contribution = float(row['Contribution'])
            _smarts = row['SMARTS']
            atomtype = row['Atom Type']
            pi = int(row['pi']) if not pd.isna(row['pi']) else None
            fnc = globals()[row['fnc']] if not pd.isna(row['fnc']) else None
            if not pd.isna(_smarts):
                try:
                    smarts = Chem.MolFromSmarts(_smarts)
                except Exception as e:
                    raise Exception(f"Invalid SMARTS for ID {row['Entry']} in logKow parameters: {_smarts}. Error: {e}") from e
                else:
                    if smarts is None:
                        raise Exception(f"Invalid SMARTS for ID {row['Entry']} in {param} parameters: {_smarts}. Error: smarts is None after parsing")
                    try:
                        smarts.UpdatePropertyCache(strict=False)
                    except Exception as e:
                        print(f"Error updating property cache for SMARTS ID {row['Entry']} in {param}")
                    try:
                        Chem.GetSymmSSSR(smarts)
                    except Exception as e:
                        print(f"Error computing ring info for SMARTS ID {row['Entry']} in {param} parameters: {_smarts}. Error: {e}")  
                    try:
                        smarts.GetRingInfo().NumRings()
                    except Exception as e:
                        print(f"Error accessing ring info for SMARTS ID {row['Entry']} in {param}.")
            else:        
                smarts = None
            if atomtype == 'Const':
                _constant = contribution
            else:
                _coefficients.append((smarts, pi, fnc, contribution))
        return _coefficients, _constant

    def _initialise_contributions(self):
        # load coefficients from CSV files and prepare for computation
        self._coefficients_kow, self._constant_kow = self._load_contributions(self._logkow_parameter_file)
        self._coefficients_koa, self._constant_koa = self._load_contributions(self._logkoa_parameter_file)

    def call(self, inp):
        return self.parse(inp)
    
    def get(self, inp):
        return self.parse(inp)
    @property
    def dict(self):
        return self._coefficients
    @property
    def results(self):
        return {Chem.MolToSmiles(mol):coeffs for mol, coeffs in self._coefficients.items()}
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        ret = f"NaefAcreePartitionCalculator with partitioning coefficient computed for {len(self._coefficients.keys())} molecules"
        ret += f" and {len(self.mols)} molecules pending contribution calculation" if len(self.mols) > 0 else ""
        return ret

    def _compute_contributions(self, mol: Chem.Mol) -> dict[str, float]:
        contributions = {"logKow": self._constant_kow, "logKoa": self._constant_koa}

        for smarts, pi, fnc, contrib in self._coefficients_kow:
            if smarts is not None:
                matches = mol.GetSubstructMatches(smarts)
                if matches:
                    if pi is not None:
                        for match in matches:
                            center_idx = match[0]
                            n_moieties, _ = count_conjugated_neighbor_moieties(mol, center_idx)
                            if n_moieties == pi:
                                contributions["logKow"] += contrib
                    else:
                        contributions["logKow"] += contrib * len(matches)
            elif fnc is not None:
                contributions["logKow"] += contrib * fnc(mol)
            else:
                # Constant-term row in parameter table.
                contributions["logKow"] += contrib

        for smarts, pi, fnc, contrib in self._coefficients_koa:
            if smarts is not None:
                matches = mol.GetSubstructMatches(smarts)
                if matches:
                    if pi is not None:
                        for match in matches:
                            center_idx = match[0]
                            n_moieties, _ = count_conjugated_neighbor_moieties(mol, center_idx)
                            if n_moieties == pi:
                                contributions["logKoa"] += contrib
                    else:
                        contributions["logKoa"] += contrib * len(matches)
            elif fnc is not None:
                contributions["logKoa"] += contrib * fnc(mol)
            else:
                # Constant-term row in parameter table.
                contributions["logKoa"] += contrib

        contributions["logKaw"] = contributions["logKow"] - contributions["logKoa"]
        contributions["in_coverage"] = self._check_in_region(mol)  # Is the molecule within the tested range of the model?
        return contributions

    def _compute_coefficients(self) -> None:
        coeffs = {}
        while len(self.mols)>0:
            mol = self.mols.pop()
            # compute contributions for this molecule and accumulate into self._coefficients
            if mol in self._coefficients:
                coeff=self._coefficients[mol]
            else:
                coeff = self._compute_contributions(mol)
                self._coefficients[mol] = coeff
            coeffs[mol] = coeff
        return coeffs

    def _ensure_mol(self, inp, convert_from=None) -> Chem.Mol:
        if convert_from == 'smiles':
            mol = Chem.MolFromSmiles(inp)
        elif convert_from == 'inchi':
            mol = Chem.MolFromInchi(inp)
        if isinstance(mol, Chem.Mol):
            return mol
        raise ValueError(f"Cannot convert input to molecule: {inp}")
    
    def _parse_input(self, inp, convert_from=None):
        if isinstance(inp, list):
            mols = []
            for item in inp:
                mols.append(self._ensure_mol(item, convert_from=convert_from))
            return mols
        elif isinstance(inp, str):
            mol = self._ensure_mol(inp, convert_from=convert_from)
            return [mol]
        elif isinstance(inp, Chem.Mol):
            return [inp]
        return []

    def _formula(self, mol: Chem.Mol) -> str:
        formula = CalcMolFormula(mol)
        PAT = r"([A-Z][a-z]?)(\d*)"
        mat = re.findall(PAT,formula)
        formula_dict = {}
        for sym,nb in mat:
            if nb != '':
                formula_dict[sym] = formula_dict.setdefault(sym,0) + int(nb)
            else:
                formula_dict[sym] =formula_dict.setdefault(sym,0) + 1
        return formula_dict
    
    def _check_in_region(self, mol: Chem.Mol)-> bool:
        formula = self._formula(mol)
        # Check if formula contains only allowed elements (H, B, C, N, O, P, S, Si, F, Cl, Br, I)
        return all(elem in self.allowed_elements for elem in formula.keys())

    def parse(self, inp):
        if isinstance(inp, list) and all(isinstance(i, str) for i in inp):
            if all(i.startswith('InChI=') for i in inp):
                mols = self._parse_input(inp, convert_from='inchi')
            else:
                mols = self._parse_input(inp, convert_from='smiles')
        elif isinstance(inp, str):
            if inp.startswith('InChI='):
                mols = self._parse_input(inp, convert_from='inchi')
            else:
                mols = self._parse_input(inp, convert_from='smiles')
        elif isinstance(inp, list) and all(isinstance(i, Chem.Mol) for i in inp):
            mols = self._parse_input(inp, convert_from=None)
        elif isinstance(inp, Chem.Mol):
            mols = [inp]
        self.mols.extend(mols)
        coeffs = self._compute_coefficients()
        return coeffs
    
    def predict(self, inp):
        coeffs = self.parse(inp)
        return coeffs[list(coeffs.keys())[0]]


class NaefAcreeCrippenMixedPartitionCalculator(NaefAcreePartitionCalculator):
    """NaefAcree-compatible mixed model using fitted Naef + Crippen parameter tables."""

    def __init__(
        self,
        smiles: Union[list, str, None] = None,
        inchi: Union[list, str, None] = None,
        mol: Union[list, Chem.Mol, None] = None,
    ) -> None:
        super().__init__(
            smiles=smiles,
            inchi=inchi,
            mol=mol,
            logkow_parameter_file="naef2024_logkow_parameters_mixed.csv",
            logkoa_parameter_file="naef2024_logkoa_parameters_mixed.csv",
        )


__all__ = [
    "NaefAcreePartitionCalculator",
    "NaefAcreeCrippenMixedPartitionCalculator",
]
