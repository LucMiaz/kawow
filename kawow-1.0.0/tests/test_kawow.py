"""
Tests for the kawow package.
"""
import sys
import os

import pytest
import numpy as np

# Allow running tests without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdkit import Chem  # noqa: E402  (after sys.path setup)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def calc():
    """Return a PartitionCalculator loaded once for the whole test session."""
    from kawow import PartitionCalculator
    return PartitionCalculator()


# ── atom_types ────────────────────────────────────────────────────────────────

class TestAtomTypes:
    def test_n_features(self):
        from kawow.atom_types import N_FEATURES, N_ATOM_TYPES, N_SPECIAL_GROUPS
        assert N_FEATURES == N_ATOM_TYPES + N_SPECIAL_GROUPS

    def test_feature_labels_length(self):
        from kawow.atom_types import FEATURE_LABELS, N_FEATURES
        assert len(FEATURE_LABELS) == N_FEATURES

    def test_crippen_labels_nonempty(self):
        from kawow.atom_types import CRIPPEN_LABELS
        assert len(CRIPPEN_LABELS) > 0

    def test_crippen_order_has_c1(self):
        """C1 (methane-like C) must be among the atom types."""
        from kawow.atom_types import CRIPPEN_ORDER
        assert "C1" in CRIPPEN_ORDER

    def test_special_groups_labels(self):
        from kawow.atom_types import SPECIAL_GROUP_LABELS
        assert "SG_HB_INTRAMOL" in SPECIAL_GROUP_LABELS
        assert "SG_ALKANE" in SPECIAL_GROUP_LABELS
        assert "SG_EXTRA_OH" in SPECIAL_GROUP_LABELS
        assert len(SPECIAL_GROUP_LABELS) == 6


# ── features ──────────────────────────────────────────────────────────────────

class TestComputeFeatures:
    def test_shape(self):
        from kawow import compute_features
        from kawow.atom_types import N_FEATURES
        mol = Chem.MolFromSmiles("CCCCO")
        vec = compute_features(mol)
        assert vec is not None
        assert vec.shape == (N_FEATURES,)
        assert vec.dtype == np.float32

    def test_methane_returns_none(self):
        """Methane has only 1 atom; feature computation should return None."""
        from kawow import compute_features
        mol = Chem.MolFromSmiles("C")
        assert compute_features(mol) is None

    def test_nonneg(self):
        """All feature values must be non-negative (counts)."""
        from kawow import compute_features
        for smi in ["CCCCO", "c1ccccc1", "CC(=O)O", "ClCCCl"]:
            mol = Chem.MolFromSmiles(smi)
            vec = compute_features(mol)
            assert vec is not None
            assert (vec >= 0).all(), f"Negative feature for {smi}"

    def test_benzene_aromatic_atom_type(self):
        """Benzene carbons should map to C18 (arom cH), not aliphatic types."""
        from kawow import compute_features, FEATURE_LABELS
        mol = Chem.MolFromSmiles("c1ccccc1")
        vec = compute_features(mol)
        assert vec is not None
        c18_idx = FEATURE_LABELS.index("C18")
        assert vec[c18_idx] == pytest.approx(6.0), "Benzene should have 6 x C18"

    def test_alkane_flag(self):
        from kawow import compute_features, FEATURE_LABELS
        mol = Chem.MolFromSmiles("CCCCCC")  # hexane (6 C atoms)
        vec = compute_features(mol)
        assert vec is not None
        idx = FEATURE_LABELS.index("SG_ALKANE")
        assert vec[idx] == pytest.approx(6.0), "Hexane should set SG_ALKANE to 6 (C count)"

    def test_nonalkane_no_flag(self):
        from kawow import compute_features, FEATURE_LABELS
        mol = Chem.MolFromSmiles("CCCCO")   # 1-butanol
        vec = compute_features(mol)
        assert vec is not None
        idx = FEATURE_LABELS.index("SG_ALKANE")
        assert vec[idx] == pytest.approx(0.0)

    def test_extra_cooh(self):
        from kawow import compute_features, FEATURE_LABELS
        mol = Chem.MolFromSmiles("OC(=O)CC(=O)O")   # malonic acid (2 COOH)
        vec = compute_features(mol)
        assert vec is not None
        idx = FEATURE_LABELS.index("SG_EXTRA_COOH")
        assert vec[idx] == pytest.approx(1.0), "Malonic acid: 1 extra COOH"

    def test_extra_oh(self):
        from kawow import compute_features, FEATURE_LABELS
        mol = Chem.MolFromSmiles("OCC(O)CO")   # glycerol (3 OH)
        vec = compute_features(mol)
        assert vec is not None
        idx = FEATURE_LABELS.index("SG_EXTRA_OH")
        assert vec[idx] == pytest.approx(2.0), "Glycerol: 2 extra OH (3 total - 1)"


class TestSmartsSpecialGroups:
    def test_halide_is_sparse_compound_correction(self):
        from kawow.smarts_model import _count_halide
        assert _count_halide(Chem.MolFromSmiles("CCl")) == pytest.approx(1.0)
        assert _count_halide(Chem.MolFromSmiles("Clc1ccccc1")) == pytest.approx(0.0)

    def test_alkane_requires_pure_acyclic_hydrocarbon(self):
        from kawow.smarts_model import _count_alkane_carbons
        assert _count_alkane_carbons(Chem.MolFromSmiles("CCCCCC")) == pytest.approx(6.0)
        assert _count_alkane_carbons(Chem.MolFromSmiles("CCCCO")) == pytest.approx(0.0)
        assert _count_alkane_carbons(Chem.MolFromSmiles("C1CCCCC1")) == pytest.approx(0.0)

    def test_cooh_and_coh_do_not_double_count_carboxylic_acids(self):
        from kawow.smarts_model import _count_coh, _count_cooh
        malonic = Chem.MolFromSmiles("OC(=O)CC(=O)O")
        glycerol = Chem.MolFromSmiles("OCC(O)CO")
        assert _count_cooh(malonic) == pytest.approx(1.0)
        assert _count_coh(malonic) == pytest.approx(0.0)
        assert _count_coh(glycerol) == pytest.approx(2.0)

    def test_h_acceptor_counts_intramolecular_pair_not_raw_acceptors(self):
        from kawow.smarts_model import _count_h_acceptor
        salicylaldehyde = Chem.MolFromSmiles("O=Cc1ccccc1O")
        acetone = Chem.MolFromSmiles("CC(=O)C")
        assert _count_h_acceptor(salicylaldehyde) == pytest.approx(1.0)
        assert _count_h_acceptor(acetone) == pytest.approx(0.0)

    def test_conjugated_neighbor_moieties_counts_sulfur_aromatic_neighbor(self):
        from kawow.smarts_model import count_conjugated_neighbor_moieties
        mol = Chem.MolFromSmiles("CSc1ccccc1")  # thioanisole
        sulfur_idx = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16)
        n_moieties, sizes = count_conjugated_neighbor_moieties(mol, sulfur_idx)
        assert n_moieties == 1
        assert sizes == [6]

    def test_conjugated_neighbor_moieties_excludes_nonpi_sulfur_neighbors(self):
        from kawow.smarts_model import count_conjugated_neighbor_moieties
        mol = Chem.MolFromSmiles("CSC")  # dimethyl sulfide
        sulfur_idx = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16)
        n_moieties, sizes = count_conjugated_neighbor_moieties(mol, sulfur_idx)
        assert n_moieties == 0
        assert sizes == []


# ── io ────────────────────────────────────────────────────────────────────────

class TestParseInput:
    def test_from_smiles_string(self):
        from kawow.io import parse_input
        pairs = parse_input("CCCCO")
        assert len(pairs) == 1
        mol, name = pairs[0]
        assert mol is not None
        assert mol.GetNumAtoms() == 5  # 4 C + 1 O heavy atoms

    def test_from_multiline_smiles(self):
        from kawow.io import parse_input
        smiles = "CCCCO\nc1ccccc1"
        pairs = parse_input(smiles)
        assert len(pairs) == 2

    def test_from_inchi(self):
        from kawow.io import parse_input
        inchi = "InChI=1S/C4H10O/c1-2-3-4-5/h5H,2-4H2,1H3"
        pairs = parse_input(inchi)
        assert len(pairs) == 1
        mol, _ = pairs[0]
        assert mol is not None

    def test_from_rdkit_mol(self):
        from kawow.io import parse_input
        mol = Chem.MolFromSmiles("CCCCO")
        pairs = parse_input(mol, fmt="mol")
        assert len(pairs) == 1
        assert pairs[0][0] is mol

    def test_from_list(self):
        from kawow.io import parse_input
        pairs = parse_input(["CCCCO", "c1ccccc1"])
        assert len(pairs) == 2

    def test_invalid_smiles_skipped(self):
        from kawow.io import parse_input
        pairs = parse_input("NOT_A_SMILES")
        assert len(pairs) == 0


# ── model / PartitionCalculator ───────────────────────────────────────────────

class TestPartitionCalculator:
    def test_predict_returns_dict(self, calc):
        result = calc.predict("CCCCO")
        assert isinstance(result, dict)

    def test_predict_keys(self, calc):
        result = calc.predict("CCCCO")
        assert "logKow" in result
        assert "logKoa" in result
        assert "logKaw" in result
        assert result["status"] == "ok"

    def test_logkaw_derived(self, calc):
        """logKaw must equal logKow − logKoa (up to float rounding)."""
        result = calc.predict("c1ccccc1")
        assert result["logKaw"] == pytest.approx(
            result["logKow"] - result["logKoa"], abs=0.005
        )

    def test_1butanol_logkow(self, calc):
        """1-butanol: experimental logKow ≈ 0.88 (literature)."""
        result = calc.predict("CCCCO")
        assert result["status"] == "ok"
        assert result["logKow"] == pytest.approx(0.88, abs=0.3)

    def test_benzene_logkow(self, calc):
        """Benzene: experimental logKow ≈ 2.13 (literature)."""
        result = calc.predict("c1ccccc1")
        assert result["status"] == "ok"
        assert result["logKow"] == pytest.approx(2.13, abs=0.4)

    def test_predict_batch(self, calc):
        smiles = ["CCCCO", "c1ccccc1", "CCCCCCCC"]
        results = calc.predict_batch(smiles)
        assert len(results) == 3
        for r in results:
            assert r["status"] == "ok"
            assert "smiles" in r

    def test_predict_batch_invalid(self, calc):
        results = calc.predict_batch(["INVALID_SMILES"])
        assert results[0]["status"] == "error"

    def test_model_info(self, calc):
        info = calc.model_info
        assert "logKow" in info
        assert "logKoa" in info
        assert info["logKow"]["n_train"] > 0
        assert info["logKow"]["r2_cv"] > 0.8

    def test_predict_from_mol_object(self, calc):
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = calc.predict(mol, fmt="mol")
        assert result["status"] == "ok"

    def test_predict_multiline_smiles_returns_list(self, calc):
        result = calc.predict("CCCCO\nc1ccccc1")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_predict_inchi(self, calc):
        """Predict from a full InChI string."""
        inchi = "InChI=1S/C4H10O/c1-2-3-4-5/h5H,2-4H2,1H3"
        result = calc.predict(inchi)
        assert isinstance(result, dict)
        assert result["status"] == "ok"

    def test_logkow_sign_octanol(self, calc):
        """Octanol (logKow ~3.0) should be positive (lipophilic)."""
        result = calc.predict("CCCCCCCCO")
        assert result["logKow"] > 1.5

    def test_logkow_sign_glucose(self, calc):
        """Glucose (logKow ~ −3.2) should be negative (hydrophilic)."""
        result = calc.predict("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")
        assert result["logKow"] < 0.0

    @pytest.mark.parametrize("smi,expected_kow", [
        ("CCCCO",     0.88),   # 1-butanol
        ("c1ccccc1",  2.13),   # benzene
        ("CC(=O)O",   -0.17),  # acetic acid
    ])
    def test_parametrized_kow(self, calc, smi, expected_kow):
        result = calc.predict(smi)
        assert result["status"] == "ok"
        assert result["logKow"] == pytest.approx(expected_kow, abs=0.5)


# ── package-level imports ─────────────────────────────────────────────────────

class TestPackageImports:
    def test_version(self):
        import kawow
        assert hasattr(kawow, "__version__")
        assert kawow.__version__

    def test_all_exports_importable(self):
        import kawow
        for name in kawow.__all__:
            assert hasattr(kawow, name), f"Missing export: {name}"

    def test_n_features_78(self):
        from kawow import N_FEATURES
        assert N_FEATURES == 78
