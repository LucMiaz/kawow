"""Paths shared by the manuscript scripts, and a check for datasets that are
not stored in git."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

# Inputs stored in git.
NAEF_DIR = REPO / "tests" / "test_data"
S01_SDF = NAEF_DIR / "S01. Compounds List for logPow-Parameters Calculations.sdf"
S02_SDF = NAEF_DIR / "S02. Compounds List for logKoa-Parameters Calculations.sdf"
BENCHMARK_METRICS_CSV = REPO / "tests" / "out" / "benchmark_metrics_5model.csv"
BENCHMARK_PRED_CSV = REPO / "tests" / "out" / "benchmark_predictions_5model.csv"
DATA = HERE / "data"
ARPHALE_CSV = DATA / "arphale_hale_2022_lser.csv"

# Input not stored in git (see README.md, "Data").
CLINVENTORY_CSV = DATA / "clinventory_molecules.csv"
CLINVENTORY_URL = "https://doi.org/10.5281/zenodo.XXXXXXX"  # TODO: Zenodo record of the C&L Inventory extract

# Generated files (not stored in git).
CACHE = HERE / "cache"
OUTPUTS = HERE / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"


class MissingDataError(FileNotFoundError):
    pass


def require(path, how_to_get):
    """Return path if it exists, otherwise raise an error saying how to obtain it."""
    path = Path(path)
    if not path.exists():
        raise MissingDataError(f"{path} not found. {how_to_get}")
    return path


def require_clinventory():
    return require(
        CLINVENTORY_CSV,
        f"The C&L Inventory extract (181 746 structures, ~14 MB) is not stored in git. Download "
        f"clinventory_molecules.csv from {CLINVENTORY_URL} and place it in {DATA}, or run "
        f"build_deployment_predictions.py with --skip-clinventory.",
    )


def require_cache(name, producer):
    return require(CACHE / name, f"Run {producer} first (see README.md).")
