"""Accuracy tests for the SMARTS-based Naef/Acree model."""

import os
import sys

import numpy as np
import pytest
from rdkit import RDLogger
from rdkit.Chem import SDMolSupplier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Allow running tests without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kawow.smarts_model import NaefAcreePartitionCalculator  # noqa: E402

RDLogger.DisableLog("rdApp.*")


@pytest.fixture(scope="session")
def smarts_calc():
    """Instantiate the SMARTS model once for the whole test session."""
    return NaefAcreePartitionCalculator()


def _predict_logkow(calc: NaefAcreePartitionCalculator, mol) -> float:
    """Get logKow prediction for one RDKit molecule from parse() output."""
    out = calc.parse(mol)

    # Primary expected shape: {mol_obj: {"logKow": ...}}
    if isinstance(out, dict):
        hit = out.get(mol)
        if isinstance(hit, dict) and "logKow" in hit:
            return float(hit["logKow"])

        # Defensive fallback if output shape changes.
        if "logKow" in out:
            return float(out["logKow"])
        for value in out.values():
            if isinstance(value, dict) and "logKow" in value:
                return float(value["logKow"])

    raise RuntimeError("Could not extract logKow from Naef&Acree model output")


def test_naefacree_logkow_accuracy_on_s01(smarts_calc):
    """
    Validate Crippen-model logKow accuracy against S01 experimental logP values.

    Acceptance criteria (as requested):
    - R2 >= 0.85
    - coverage >= 0.90 (fraction of molecules with successful prediction)
    """
    sdf_path = os.path.join(
        os.path.dirname(__file__),
        "test_data",
        "S01. Compounds List for logPow-Parameters Calculations.sdf",
    )
    suppl = SDMolSupplier(sdf_path)

    y_true = []
    y_pred = []

    for mol in suppl:
        if mol is None or not mol.HasProp("logP"):
            continue
        try:
            y_true.append(float(mol.GetProp("logP")))
            y_pred.append(_predict_logkow(smarts_calc, mol))
        except Exception:
            y_true.append(float(mol.GetProp("logP")))
            y_pred.append(np.nan)

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = ~np.isnan(y_pred)
    coverage = float(mask.mean()) if len(mask) else 0.0

    assert coverage >= 0.90, f"Naef&Acree model coverage too low on S01: {coverage:.3f}"

    r2 = r2_score(y_true[mask], y_pred[mask])
    rmse = mean_squared_error(y_true[mask], y_pred[mask]) ** 0.5
    mae = mean_absolute_error(y_true[mask], y_pred[mask])

    assert r2 >= 0.85, (
        f"Naef&Acree logKow R2 below target on S01: {r2:.3f} "
        f"(RMSE={rmse:.3f}, MAE={mae:.3f}, coverage={coverage:.3f})"
    )
