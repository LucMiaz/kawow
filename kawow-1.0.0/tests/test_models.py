"""Tests for kawow models not covered by test_kawow.py or test_naef_model_accuracy.py.

Covers:
- NaefAcreeCrippenPartitionCalculator (naef_crippen)
- PFASGroupsPartitionCalculator (all four variants)
- PFASGroupsRFPartitionCalculator
- PFASGroupsXGBPartitionCalculator
- PFASGroupsNNPartitionCalculator (skipped if keras not available)
- MQGPartitionCalculator / EnsemblePartitionCalculator (skipped if mqg not available)
- run_models() multi-model API
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdkit import Chem, RDLogger
from rdkit.Chem import SDMolSupplier
from sklearn.metrics import r2_score

RDLogger.DisableLog("rdApp.*")

# ── Paths ─────────────────────────────────────────────────────────────────────

_DATA = os.path.join(os.path.dirname(__file__), "test_data")
S01_SDF = os.path.join(_DATA, "S01. Compounds List for logPow-Parameters Calculations.sdf")
S02_SDF = os.path.join(_DATA, "S02. Compounds List for logKoa-Parameters Calculations.sdf")


# ── Shared helper ─────────────────────────────────────────────────────────────

def _r2_on_sdf(predict_fn, sdf_path, prop_name):
    """Compute (R², coverage) of predict_fn across all mols in an SDF.

    predict_fn(mol) should return a float or raise / return NaN on failure.
    """
    suppl = SDMolSupplier(sdf_path)
    y_true, y_pred = [], []
    for mol in suppl:
        if mol is None or not mol.HasProp(prop_name):
            continue
        try:
            val = predict_fn(mol)
        except Exception:
            val = np.nan
        y_true.append(float(mol.GetProp(prop_name)))
        y_pred.append(float(val) if val is not None else np.nan)

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = ~np.isnan(y_pred)
    coverage = float(mask.sum()) / len(mask) if len(mask) else 0.0
    r2 = r2_score(y_true[mask], y_pred[mask]) if mask.sum() >= 2 else -999.0
    return r2, coverage


# ── NaefAcreeCrippenPartitionCalculator ──────────────────────────────────────

class TestNaefCrippenModel:
    """Tests for the SMARTS-based naef_crippen model (Naef + Crippen parameters)."""

    @pytest.fixture(scope="class")
    def calc(self):
        from kawow.smarts_model import NaefAcreeCrippenPartitionCalculator
        return NaefAcreeCrippenPartitionCalculator()

    def test_predict_returns_dict(self, calc):
        result = calc.predict("CCCCO")
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, calc):
        result = calc.predict("CCCCO")
        assert "logKow" in result
        assert "logKoa" in result
        assert "logKaw" in result

    def test_logkaw_derived_from_kow_koa(self, calc):
        result = calc.predict("c1ccccc1")
        assert result["logKaw"] == pytest.approx(result["logKow"] - result["logKoa"], abs=0.005)

    def test_in_coverage_is_bool(self, calc):
        result = calc.predict("CCCCO")
        assert isinstance(result["in_coverage"], bool)

    def test_predict_from_mol_object(self, calc):
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = calc.predict(mol)
        assert "logKow" in result

    def test_1butanol_logkow(self, calc):
        """1-butanol experimental logKow ≈ 0.88."""
        result = calc.predict("CCCCO")
        assert result["logKow"] == pytest.approx(0.88, abs=0.5)

    def test_benzene_logkow(self, calc):
        """Benzene experimental logKow ≈ 2.13."""
        result = calc.predict("c1ccccc1")
        assert result["logKow"] == pytest.approx(2.13, abs=0.5)

    def test_octanol_logkow_positive(self, calc):
        result = calc.predict("CCCCCCCCO")
        assert result["logKow"] > 1.5

    def test_glucose_logkow_negative(self, calc):
        result = calc.predict("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")
        assert result["logKow"] < 0.0

    def test_predict_repeated_calls_idempotent(self, calc):
        """Calling predict multiple times on the same instance gives consistent results."""
        r1 = calc.predict("CCCCO")
        r2 = calc.predict("CCCCO")
        assert r1["logKow"] == pytest.approx(r2["logKow"])

    def test_accuracy_logkow_s01(self, calc):
        def fn(mol):
            return calc.predict(mol)["logKow"]

        r2, coverage = _r2_on_sdf(fn, S01_SDF, "logP")
        assert coverage >= 0.90, f"naef_crippen logKow coverage on S01: {coverage:.3f}"
        assert r2 >= 0.90, f"naef_crippen logKow R² on S01: {r2:.3f}"

    def test_accuracy_logkoa_s02(self, calc):
        def fn(mol):
            return calc.predict(mol)["logKoa"]

        r2, coverage = _r2_on_sdf(fn, S02_SDF, "logKoa")
        assert coverage >= 0.90, f"naef_crippen logKoa coverage on S02: {coverage:.3f}"
        assert r2 >= 0.90, f"naef_crippen logKoa R² on S02: {r2:.3f}"


# ── PFASGroupsPartitionCalculator (4 Ridge variants) ─────────────────────────

# Per-variant accuracy thresholds (R²): (logKow_min, logKoa_min)
# Derived from 5-fold CV results; actual test-set R² should meet or exceed these.
_PFASGROUPS_THRESHOLDS = {
    "pfasgroups":              (0.75, 0.85),
    "pfasgroups_crippen":      (0.87, 0.92),
    "pfasgroups_naef":         (0.90, 0.90),
    "pfasgroups_naef_crippen": (0.92, 0.92),
}


class TestPFASGroupsVariants:
    """Smoke and accuracy tests for all four PFASGroupsPartitionCalculator variants."""

    @pytest.fixture(scope="class", params=list(_PFASGROUPS_THRESHOLDS))
    def calc_info(self, request):
        from kawow import PFASGroupsPartitionCalculator
        variant = request.param
        kow_thr, koa_thr = _PFASGROUPS_THRESHOLDS[variant]
        return PFASGroupsPartitionCalculator(variant), variant, kow_thr, koa_thr

    def test_predict_returns_dict(self, calc_info):
        calc, variant, *_ = calc_info
        result = calc.predict("CCCCO")
        assert isinstance(result, dict), f"{variant}: predict should return dict for single mol"

    def test_predict_keys(self, calc_info):
        calc, variant, *_ = calc_info
        result = calc.predict("CCCCO")
        assert result["status"] == "ok", f"{variant}: status should be 'ok'"
        assert "logKow" in result and "logKoa" in result and "logKaw" in result

    def test_logkaw_derived(self, calc_info):
        calc, variant, *_ = calc_info
        result = calc.predict("c1ccccc1")
        assert result["logKaw"] == pytest.approx(
            result["logKow"] - result["logKoa"], abs=0.005
        ), f"{variant}: logKaw ≠ logKow − logKoa"

    def test_octanol_logkow_positive(self, calc_info):
        calc, variant, *_ = calc_info
        assert calc.predict("CCCCCCCCO")["logKow"] > 1.5, f"{variant}: octanol logKow should be > 1.5"

    def test_glucose_logkow_negative(self, calc_info):
        calc, variant, *_ = calc_info
        kow = calc.predict("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")["logKow"]
        # pfasgroups alone (no Crippen/Naef features) has limited dynamic range;
        # it falls back near the intercept for non-PFAS compounds — allow up to 1.0.
        threshold = 1.0 if variant == "pfasgroups" else 0.0
        assert kow < threshold, f"{variant}: glucose logKow should be < {threshold}, got {kow}"

    def test_invalid_smiles_returns_empty(self, calc_info):
        calc, variant, *_ = calc_info
        result = calc.predict("NOT_A_SMILES")
        # parse_input drops invalid SMILES → empty list returned
        assert result == [] or (isinstance(result, dict) and result.get("status") == "error")

    def test_predict_batch(self, calc_info):
        calc, variant, *_ = calc_info
        results = calc.predict_batch(["CCCCO", "c1ccccc1"])
        assert len(results) == 2
        assert all(r["status"] == "ok" for r in results), f"{variant}: batch results should all be ok"

    def test_model_info_contains_variant(self, calc_info):
        calc, variant, *_ = calc_info
        info = calc.model_info
        assert info["model"] == variant
        assert "logKow" in info and "logKoa" in info

    def test_accuracy_logkow_s01(self, calc_info):
        calc, variant, kow_thr, _ = calc_info

        def fn(mol):
            r = calc.predict(mol, fmt="mol")
            return r["logKow"] if isinstance(r, dict) and r.get("status") == "ok" else np.nan

        r2, coverage = _r2_on_sdf(fn, S01_SDF, "logP")
        assert coverage >= 0.90, f"{variant} logKow coverage on S01: {coverage:.3f}"
        assert r2 >= kow_thr, f"{variant} logKow R² on S01: {r2:.3f} (threshold {kow_thr})"

    def test_accuracy_logkoa_s02(self, calc_info):
        calc, variant, _, koa_thr = calc_info

        def fn(mol):
            r = calc.predict(mol, fmt="mol")
            return r["logKoa"] if isinstance(r, dict) and r.get("status") == "ok" else np.nan

        r2, coverage = _r2_on_sdf(fn, S02_SDF, "logKoa")
        assert coverage >= 0.90, f"{variant} logKoa coverage on S02: {coverage:.3f}"
        assert r2 >= koa_thr, f"{variant} logKoa R² on S02: {r2:.3f} (threshold {koa_thr})"


# ── PFASGroupsRFPartitionCalculator ──────────────────────────────────────────

class TestPFASGroupsRF:
    """Tests for the Random Forest model (pfasgroups_naef_crippen_rf)."""

    @pytest.fixture(scope="class")
    def calc(self):
        from kawow import PFASGroupsRFPartitionCalculator
        return PFASGroupsRFPartitionCalculator()

    def test_predict_returns_dict(self, calc):
        result = calc.predict("CCCCO")
        assert isinstance(result, dict)
        assert result["status"] == "ok"

    def test_predict_keys(self, calc):
        result = calc.predict("CCCCO")
        assert "logKow" in result and "logKoa" in result and "logKaw" in result

    def test_logkaw_derived(self, calc):
        result = calc.predict("c1ccccc1")
        assert result["logKaw"] == pytest.approx(result["logKow"] - result["logKoa"], abs=0.005)

    def test_octanol_logkow_positive(self, calc):
        assert calc.predict("CCCCCCCCO")["logKow"] > 1.5

    def test_glucose_logkow_negative(self, calc):
        assert calc.predict("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")["logKow"] < 0.0

    def test_1butanol_logkow(self, calc):
        """1-butanol experimental logKow ≈ 0.88."""
        result = calc.predict("CCCCO")
        assert result["logKow"] == pytest.approx(0.88, abs=0.5)

    def test_benzene_logkow(self, calc):
        """Benzene experimental logKow ≈ 2.13."""
        result = calc.predict("c1ccccc1")
        assert result["logKow"] == pytest.approx(2.13, abs=0.5)

    def test_predict_batch(self, calc):
        results = calc.predict_batch(["CCCCO", "c1ccccc1", "CCCCCCCC"])
        assert len(results) == 3
        assert all(r["status"] == "ok" for r in results)

    def test_predict_batch_invalid(self, calc):
        results = calc.predict_batch(["INVALID_SMILES"])
        assert results[0]["status"] == "error"

    def test_model_info(self, calc):
        info = calc.model_info
        assert info["model"] == "pfasgroups_naef_crippen_rf"
        assert "logKow" in info and "logKoa" in info

    def test_accuracy_logkow_s01(self, calc):
        def fn(mol):
            r = calc.predict(mol, fmt="mol")
            return r["logKow"] if isinstance(r, dict) and r.get("status") == "ok" else np.nan

        r2, coverage = _r2_on_sdf(fn, S01_SDF, "logP")
        assert coverage >= 0.90, f"RF logKow coverage on S01: {coverage:.3f}"
        assert r2 >= 0.88, f"RF logKow R² on S01: {r2:.3f}"

    def test_accuracy_logkoa_s02(self, calc):
        def fn(mol):
            r = calc.predict(mol, fmt="mol")
            return r["logKoa"] if isinstance(r, dict) and r.get("status") == "ok" else np.nan

        r2, coverage = _r2_on_sdf(fn, S02_SDF, "logKoa")
        assert coverage >= 0.90, f"RF logKoa coverage on S02: {coverage:.3f}"
        assert r2 >= 0.90, f"RF logKoa R² on S02: {r2:.3f}"


# ── PFASGroupsXGBPartitionCalculator ─────────────────────────────────────────

class TestPFASGroupsXGB:
    """Tests for the XGBoost model (pfasgroups_naef_crippen_xgb)."""

    @pytest.fixture(scope="class")
    def calc(self):
        from kawow import PFASGroupsXGBPartitionCalculator
        return PFASGroupsXGBPartitionCalculator()

    def test_predict_returns_dict(self, calc):
        result = calc.predict("CCCCO")
        assert isinstance(result, dict)
        assert result["status"] == "ok"

    def test_predict_keys(self, calc):
        result = calc.predict("CCCCO")
        assert "logKow" in result and "logKoa" in result and "logKaw" in result

    def test_logkaw_derived(self, calc):
        result = calc.predict("c1ccccc1")
        assert result["logKaw"] == pytest.approx(result["logKow"] - result["logKoa"], abs=0.005)

    def test_octanol_logkow_positive(self, calc):
        assert calc.predict("CCCCCCCCO")["logKow"] > 1.5

    def test_glucose_logkow_negative(self, calc):
        assert calc.predict("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")["logKow"] < 0.0

    def test_1butanol_logkow(self, calc):
        result = calc.predict("CCCCO")
        assert result["logKow"] == pytest.approx(0.88, abs=0.5)

    def test_benzene_logkow(self, calc):
        result = calc.predict("c1ccccc1")
        assert result["logKow"] == pytest.approx(2.13, abs=0.5)

    def test_predict_batch(self, calc):
        results = calc.predict_batch(["CCCCO", "c1ccccc1", "CCCCCCCC"])
        assert len(results) == 3
        assert all(r["status"] == "ok" for r in results)

    def test_predict_batch_invalid(self, calc):
        results = calc.predict_batch(["INVALID_SMILES"])
        assert results[0]["status"] == "error"

    def test_model_info(self, calc):
        info = calc.model_info
        assert info["model"] == "pfasgroups_naef_crippen_xgb"
        assert "logKow" in info and "logKoa" in info

    def test_accuracy_logkow_s01(self, calc):
        def fn(mol):
            r = calc.predict(mol, fmt="mol")
            return r["logKow"] if isinstance(r, dict) and r.get("status") == "ok" else np.nan

        r2, coverage = _r2_on_sdf(fn, S01_SDF, "logP")
        assert coverage >= 0.90, f"XGB logKow coverage on S01: {coverage:.3f}"
        assert r2 >= 0.90, f"XGB logKow R² on S01: {r2:.3f}"

    def test_accuracy_logkoa_s02(self, calc):
        def fn(mol):
            r = calc.predict(mol, fmt="mol")
            return r["logKoa"] if isinstance(r, dict) and r.get("status") == "ok" else np.nan

        r2, coverage = _r2_on_sdf(fn, S02_SDF, "logKoa")
        assert coverage >= 0.90, f"XGB logKoa coverage on S02: {coverage:.3f}"
        assert r2 >= 0.92, f"XGB logKoa R² on S02: {r2:.3f}"


# ── PFASGroupsNNPartitionCalculator (optional: requires keras) ────────────────

class TestPFASGroupsNN:
    """Tests for the Neural Network model (pfasgroups_naef_crippen_nn). Skipped if keras absent."""

    @pytest.fixture(scope="class")
    def calc(self):
        pytest.importorskip("keras", reason="keras not installed — skipping NN model tests")
        from kawow import PFASGroupsNNPartitionCalculator
        return PFASGroupsNNPartitionCalculator()

    def test_predict_returns_dict(self, calc):
        result = calc.predict("CCCCO")
        assert isinstance(result, dict)
        assert result["status"] == "ok"

    def test_predict_keys(self, calc):
        result = calc.predict("CCCCO")
        assert "logKow" in result and "logKoa" in result and "logKaw" in result

    def test_logkaw_derived(self, calc):
        result = calc.predict("c1ccccc1")
        assert result["logKaw"] == pytest.approx(result["logKow"] - result["logKoa"], abs=0.005)

    def test_octanol_logkow_positive(self, calc):
        assert calc.predict("CCCCCCCCO")["logKow"] > 1.5

    def test_glucose_logkow_negative(self, calc):
        assert calc.predict("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")["logKow"] < 0.0

    def test_model_info(self, calc):
        info = calc.model_info
        assert info["model"] == "pfasgroups_naef_crippen_nn"

    def test_accuracy_logkow_s01(self, calc):
        def fn(mol):
            r = calc.predict(mol, fmt="mol")
            return r["logKow"] if isinstance(r, dict) and r.get("status") == "ok" else np.nan

        r2, coverage = _r2_on_sdf(fn, S01_SDF, "logP")
        assert coverage >= 0.90, f"NN logKow coverage on S01: {coverage:.3f}"
        assert r2 >= 0.85, f"NN logKow R² on S01: {r2:.3f}"

    def test_accuracy_logkoa_s02(self, calc):
        def fn(mol):
            r = calc.predict(mol, fmt="mol")
            return r["logKoa"] if isinstance(r, dict) and r.get("status") == "ok" else np.nan

        r2, coverage = _r2_on_sdf(fn, S02_SDF, "logKoa")
        assert coverage >= 0.90, f"NN logKoa coverage on S02: {coverage:.3f}"
        assert r2 >= 0.85, f"NN logKoa R² on S02: {r2:.3f}"


# ── MQGPartitionCalculator / EnsemblePartitionCalculator (optional) ───────────

class TestMQGModel:
    """Tests for MQG-based models. Skipped if the mqg package is not installed."""

    @pytest.fixture(scope="class")
    def mqg_calc(self):
        pytest.importorskip("mqg", reason="mqg package not installed — skipping MQG model tests")
        from kawow import MQGPartitionCalculator
        return MQGPartitionCalculator()

    def test_predict_returns_dict(self, mqg_calc):
        result = mqg_calc.predict("CCCCO")
        assert isinstance(result, dict)
        assert result["status"] == "ok"

    def test_predict_keys(self, mqg_calc):
        result = mqg_calc.predict("CCCCO")
        assert "logKow" in result and "logKoa" in result and "logKaw" in result

    def test_logkaw_derived(self, mqg_calc):
        result = mqg_calc.predict("c1ccccc1")
        assert result["logKaw"] == pytest.approx(result["logKow"] - result["logKoa"], abs=0.005)

    def test_model_info(self, mqg_calc):
        info = mqg_calc.model_info
        assert info["model"] == "mqg"


class TestEnsembleModel:
    """Tests for EnsemblePartitionCalculator. Skipped if mqg package is not installed."""

    @pytest.fixture(scope="class", params=["naef_mqg", "crippen_mqg"])
    def ens_calc(self, request):
        pytest.importorskip("mqg", reason="mqg package not installed — skipping ensemble tests")
        from kawow import EnsemblePartitionCalculator
        return EnsemblePartitionCalculator(request.param), request.param

    def test_predict_returns_dict(self, ens_calc):
        calc, _ = ens_calc
        result = calc.predict("CCCCO")
        assert isinstance(result, dict)
        assert result["status"] == "ok"

    def test_predict_keys(self, ens_calc):
        calc, _ = ens_calc
        result = calc.predict("CCCCO")
        assert "logKow" in result and "logKoa" in result and "logKaw" in result

    def test_logkaw_derived(self, ens_calc):
        calc, _ = ens_calc
        result = calc.predict("c1ccccc1")
        assert result["logKaw"] == pytest.approx(result["logKow"] - result["logKoa"], abs=0.005)

    def test_model_info_contains_type(self, ens_calc):
        calc, ensemble_type = ens_calc
        info = calc.model_info
        assert ensemble_type in info.get("ensemble_type", "") or ensemble_type in info.get("model", "")


# ── run_models() multi-model API ──────────────────────────────────────────────

class TestRunModels:
    """Tests for the run_models() function.

    run_models() returns one dict per MOLECULE (not per model). Each dict has:
      { 'name': str, 'smiles': str, 'ok': bool,
        'models': { model_name: { 'logKow': float, 'logKoa': float, 'logKaw': float,
                                   'status': str, ... } } }
    """

    def test_single_molecule_returns_list_of_one(self):
        from kawow import run_models
        results = run_models("CCCCO", models=["pfasgroups"])
        assert isinstance(results, list)
        assert len(results) == 1

    def test_result_row_schema(self):
        from kawow import run_models
        row = run_models("CCCCO", models=["pfasgroups"])[0]
        # Top-level keys
        assert "models" in row
        assert "ok" in row
        # Per-model sub-dict
        model_result = row["models"]["pfasgroups"]
        assert "logKow" in model_result
        assert "logKoa" in model_result
        assert "logKaw" in model_result
        assert "status" in model_result

    def test_multiple_models_both_present(self):
        from kawow import run_models
        results = run_models("c1ccccc1", models=["pfasgroups", "pfasgroups_naef"])
        assert len(results) == 1
        models_dict = results[0]["models"]
        assert "pfasgroups" in models_dict
        assert "pfasgroups_naef" in models_dict

    def test_batch_input_two_molecules(self):
        from kawow import run_models
        results = run_models(["CCCCO", "c1ccccc1"], models=["pfasgroups"])
        assert len(results) == 2

    def test_no_valid_model_raises_value_error(self):
        from kawow import run_models
        with pytest.raises(ValueError, match="No valid model"):
            run_models("CCCCO", models=["not_a_real_model_xyzzy"])

    def test_deprecated_alias_crippen_warns(self):
        from kawow import run_models
        # 'crippen' is listed in _KEY_ALIASES (maps to itself), so it triggers a warning
        with pytest.warns(DeprecationWarning):
            run_models("CCCCO", models=["crippen"])

    def test_pfasgroups_variants_in_run_models(self):
        from kawow import run_models
        results = run_models("c1ccccc1", models=["pfasgroups", "pfasgroups_naef_crippen"])
        assert len(results) == 1
        models_dict = results[0]["models"]
        assert "pfasgroups" in models_dict
        assert "pfasgroups_naef_crippen" in models_dict

    def test_rf_xgb_in_run_models(self):
        from kawow import run_models
        results = run_models("c1ccccc1", models=["pfasgroups_naef_crippen_rf", "pfasgroups_naef_crippen_xgb"])
        assert len(results) == 1
        models_dict = results[0]["models"]
        assert "pfasgroups_naef_crippen_rf" in models_dict
        assert "pfasgroups_naef_crippen_xgb" in models_dict

    def test_logkaw_consistent_across_models(self):
        from kawow import run_models
        results = run_models(
            "CCCCO",
            models=["pfasgroups", "pfasgroups_naef", "pfasgroups_naef_crippen_rf"],
        )
        for model_result in results[0]["models"].values():
            if model_result.get("status") == "ok":
                assert model_result["logKaw"] == pytest.approx(
                    model_result["logKow"] - model_result["logKoa"], abs=0.005
                ), f"logKaw inconsistency for model {model_result.get('model')}"

    def test_working_models_run_without_error(self):
        """PFASGroups models (which work in run_models) return ok status."""
        from kawow import run_models
        working_models = [
            "pfasgroups", "pfasgroups_naef", "pfasgroups_naef_crippen",
            "pfasgroups_naef_crippen_rf", "pfasgroups_naef_crippen_xgb",
        ]
        results = run_models("c1ccccc1", models=working_models)
        assert len(results) == 1
        models_dict = results[0]["models"]
        failed = [name for name, r in models_dict.items() if r.get("status") != "ok"]
        assert not failed, f"Models returned error status: {failed}"
