"""Invalid inputs are reported in place, and importing kawow has no global side effects."""
import sys

import kawow
from kawow import PartitionCalculator, run_models
from kawow.io import parse_input


def test_parse_input_keeps_invalid_on_request():
    assert [n for _, n in parse_input(["CCO", "not_a_smiles"])] == ["CCO"]
    pairs = parse_input(["CCO", "not_a_smiles", "InChI=bad"], keep_invalid=True)
    assert [m is None for m, _ in pairs] == [False, True, True]
    assert pairs[1][1] == "not_a_smiles"


def test_run_models_one_row_per_input():
    rows = run_models(["CCO", "not_a_smiles", "CCCC"], models=["crippen"])
    assert [r["name"] for r in rows] == ["CCO", "not_a_smiles", "CCCC"]
    assert [r["ok"] for r in rows] == [True, False, True]
    assert rows[1]["models"]["crippen"]["status"] == "error"


def test_predict_list_stays_a_list():
    out = PartitionCalculator().predict(["CCO", "C1CC"])   # C1CC: unclosed ring
    assert isinstance(out, list) and [r["status"] for r in out] == ["ok", "error"]
    single = PartitionCalculator().predict("xyz")
    assert single["status"] == "error" and single["name"] == "xyz"


def test_import_does_not_silence_rdkit():
    # Fresh interpreter: other test modules disable RDKit logging at collection.
    import subprocess
    code = ("import kawow; from rdkit import rdBase; s = rdBase.LogStatus(); "
            "print('rdApp.error:enabled' in s and 'rdApp.warning:enabled' in s)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "True"


def test_import_does_not_prepend_to_sys_path():
    from kawow import pfasgroups_features as pf
    assert sys.path[0] != str(pf._PFASGROUPS_PATH)


def test_shipped_pickles_do_not_depend_on_fitting_machine(tmp_path):
    """The ensemble pickles store C:\\Users\\...\\naef2024_*.csv; resolve it locally."""
    from kawow.model import _resolve_param_file, DATA_DIR
    win = r"C:\Users\someone\git\kawow\kawow\data\naef2024_logkow_parameters.csv"
    assert _resolve_param_file(win) == DATA_DIR / "naef2024_logkow_parameters.csv"
    posix = "/home/someone/kawow/kawow/data/naef2024_logkoa_parameters.csv"
    assert _resolve_param_file(posix) == DATA_DIR / "naef2024_logkoa_parameters.csv"
    own = tmp_path / "custom.csv"; own.write_text("x")
    assert _resolve_param_file(own) == own
