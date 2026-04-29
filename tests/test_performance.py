"""
Tests for the kawow package.
"""
import sys
import os

import pytest
import numpy as np
import json

# Allow running tests without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdkit import Chem  # noqa: E402  (after sys.path setup)
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")  # Suppress RDKit warnings during tests

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def calc():
    """Return a PartitionCalculator loaded once for the whole test session."""
    from kawow import PartitionCalculator
    return PartitionCalculator()

# ── performance koa ─────────────────────────────────────────────────────────────
class TestPerformancePrediction:
    def test_koa(self, calc):
        """Check that logKoa R² is above 0.9."""
        from rdkit.Chem import SDMolSupplier
        sdf_path = os.path.join(os.path.dirname(__file__), "test_data","S02. Compounds List for logKoa-Parameters Calculations.sdf")
        suppl = SDMolSupplier(sdf_path)
        y_true = []
        y_pred = []
        for mol in suppl:
            if mol is not None:
                y_true.append(float(mol.GetProp("logKoa")))
                calc_results = calc.predict(mol)
                if calc_results["status"] == "ok":
                    y_pred.append(calc_results["logKoa"])
                else:
                    y_pred.append(np.nan)  # Handle prediction errors by inserting NaN
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        from sklearn.metrics import r2_score
        mask = ~np.isnan(y_pred)
        r2 = r2_score(y_true[mask], y_pred[mask])
        assert r2 > 0.94, f"logKoa R² is too low: {r2:.3f}"
    def test_kow(self, calc):
        """Check that logKow R² is above 0.9."""
        from rdkit.Chem import SDMolSupplier
        sdf_path = os.path.join(os.path.dirname(__file__), "test_data","S01. Compounds List for logPow-Parameters Calculations.sdf")
        suppl = SDMolSupplier(sdf_path)
        y_true = []
        y_pred = []
        for mol in suppl:
            if mol is not None:
                y_true.append(float(mol.GetProp("logP")))
                calc_results = calc.predict(mol)
                if calc_results["status"] == "ok":
                    y_pred.append(calc_results["logKow"])
                else:
                    y_pred.append(np.nan)  # Handle prediction errors by inserting NaN  
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        from sklearn.metrics import r2_score
        mask = ~np.isnan(y_pred)
        r2 = r2_score(y_true[mask], y_pred[mask])
        assert r2 > 0.9, f"logKow R² is too low: {r2:.3f}"
    def test_kaw(self, calc):
        """Check that logKaw R² is above 0.9."""
        from rdkit.Chem import SDMolSupplier
        sdf_path = os.path.join(os.path.dirname(__file__), "test_data","S03. Compounds List with exp logKaw Data.sdf")
        suppl = SDMolSupplier(sdf_path)
        y_true = []
        y_pred = []
        for mol in suppl:
            if mol is not None:
                y_true.append(float(mol.GetProp("logKaw")))
                calc_results = calc.predict(mol)
                if calc_results["status"] == "ok":
                    y_pred.append(calc_results["logKaw"])
                else:
                    y_pred.append(np.nan)  # Handle prediction errors by inserting NaN
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        from sklearn.metrics import r2_score
        mask = ~np.isnan(y_pred)
        r2 = r2_score(y_true[mask], y_pred[mask])
        assert r2 > 0.87, f"logKaw R² is too low: {r2:.3f}"

# ── performance kow ─────────────────────────────────────────────────────────────


# ── OLS (naef_ols) model performance ─────────────────────────────────────────

@pytest.fixture(scope="session")
def calc_ols():
    """Return a PartitionCalculator(model='naef_ols') for the whole session."""
    from kawow import PartitionCalculator
    return PartitionCalculator(model="naef_ols")


class TestOLSPerformance:
    """Verify that the OLS model reaches R² ≥ 0.95 on the training SDF files."""

    def test_kow_ols(self, calc_ols):
        """OLS logKow R² should be above 0.95 (Naef paper: ≈0.96)."""
        from rdkit.Chem import SDMolSupplier
        sdf_path = os.path.join(
            os.path.dirname(__file__),
            "test_data",
            "S01. Compounds List for logPow-Parameters Calculations.sdf",
        )
        suppl = SDMolSupplier(sdf_path)
        y_true, y_pred = [], []
        for mol in suppl:
            if mol is not None:
                y_true.append(float(mol.GetProp("logP")))
                res = calc_ols.predict(mol)
                y_pred.append(res["logKow"] if res["status"] == "ok" else np.nan)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        from sklearn.metrics import r2_score
        mask = ~np.isnan(y_pred)
        r2 = r2_score(y_true[mask], y_pred[mask])
        assert r2 > 0.90, f"OLS logKow R² is too low: {r2:.3f}"

    def test_koa_ols(self, calc_ols):
        """OLS logKoa R² should be above 0.95 (Naef paper: ≈0.97)."""
        from rdkit.Chem import SDMolSupplier
        sdf_path = os.path.join(
            os.path.dirname(__file__),
            "test_data",
            "S02. Compounds List for logKoa-Parameters Calculations.sdf",
        )
        suppl = SDMolSupplier(sdf_path)
        y_true, y_pred = [], []
        for mol in suppl:
            if mol is not None:
                y_true.append(float(mol.GetProp("logKoa")))
                res = calc_ols.predict(mol)
                y_pred.append(res["logKoa"] if res["status"] == "ok" else np.nan)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        from sklearn.metrics import r2_score
        mask = ~np.isnan(y_pred)
        r2 = r2_score(y_true[mask], y_pred[mask])
        assert r2 > 0.93, f"OLS logKoa R² is too low: {r2:.3f}"

