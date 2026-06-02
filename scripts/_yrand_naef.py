"""Quick y-randomization test for naef (Naef&Acree standalone) model."""
import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from kawow.metrics import y_randomization_test
import importlib.util

spec = importlib.util.spec_from_file_location(
    "shared_fold_benchmark", pathlib.Path(__file__).parent / "shared_fold_benchmark.py"
)
sfb = importlib.util.module_from_spec(spec)
sys.modules["shared_fold_benchmark"] = sfb
spec.loader.exec_module(sfb)

REPO = pathlib.Path(__file__).parent.parent
S01_SDF = REPO / "tests/test_data/S01. Compounds List for logPow-Parameters Calculations.sdf"
S02_SDF = REPO / "tests/test_data/S02. Compounds List for logKoa-Parameters Calculations.sdf"
KOW_CSV = REPO / "kawow/data/naef2024_logkow_parameters.csv"
KOA_CSV = REPO / "kawow/data/naef2024_logkoa_parameters.csv"

for sdf_path, value_prop, endpoint, naef_csv in [
    (S01_SDF, "logP", "logKow", KOW_CSV),
    (S02_SDF, "logKoa", "logKoa", KOA_CSV),
]:
    print(f"\nProcessing {endpoint}...")
    naef_patterns = sfb._compile_naef_patterns(naef_csv)
    rows = sfb._load_rows(sdf_path, value_prop)
    ys = []
    X_naef = []
    for mol, _name, value in rows:
        x_naef = sfb._compute_naef_group_counts(mol, naef_patterns)
        if x_naef is None or not np.all(np.isfinite(x_naef)):
            continue
        ys.append(value)
        X_naef.append(x_naef)

    y = np.array(ys, dtype=np.float64)
    X = np.array(X_naef, dtype=np.float32)
    print(f"  n={len(y)}, X shape={X.shape}")

    res = y_randomization_test(X, y, n_permutations=500, n_splits=5)
    obs_r2 = round(res["observed_r2"], 4)
    obs_ccc = round(res["observed_ccc"], 4)
    perm_r2_mean = round(res["perm_r2_mean"], 4)
    perm_r2_std = round(res["perm_r2_std"], 4)
    p_value = round(res["p_value"], 4)
    perm_ccc_mean = round(res["perm_ccc_mean"], 4)
    print(f"  RESULT {endpoint}: obs_r2={obs_r2} obs_ccc={obs_ccc} perm_r2_mean={perm_r2_mean} perm_r2_std={perm_r2_std} p_value={p_value} perm_ccc_mean={perm_ccc_mean}")
