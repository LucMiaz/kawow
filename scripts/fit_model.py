"""
fit_model.py
============
Fit group-additivity models on the Naef & Acree (Liquids 2024) SDF training data.

Usage
-----
    cd ~/kawow

    # Fit the default Ridge model (overwrites logkow_model.json / logkoa_model.json)
    python scripts/fit_model.py

    # Fit the OLS model (writes logkow_ols_model.json / logkoa_ols_model.json)
    python scripts/fit_model.py --method ols

Reads:
    S01. Compounds List for logPow-Parameters Calculations.sdf  → logKow
    S02. Compounds List for logKoa-Parameters Calculations.sdf  → logKoa

Writes (ridge):
    kawow/data/logkow_model.json
    kawow/data/logkoa_model.json

Writes (ols):
    kawow/data/logkow_ols_model.json
    kawow/data/logkoa_ols_model.json
"""

import argparse
import sys
import os
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from kawow.model import fit

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDF_DIR = os.path.join(MAIN_DIR, "tests", "test_data")

S01 = os.path.join(SDF_DIR, "S01. Compounds List for logPow-Parameters Calculations.sdf")
S02 = os.path.join(SDF_DIR, "S02. Compounds List for logKoa-Parameters Calculations.sdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit kawow partition models.")
    parser.add_argument(
        "--method",
        choices=["ridge", "ols"],
        default="ridge",
        help="Fitting method: 'ridge' (default, L2-regularised) or 'ols' (Naef Gauss-Seidel).",
    )
    parser.add_argument(
        "--sdf_kow",
        default=str(S01),
        help="Path to SDF file with logKow training data.",
    )
    parser.add_argument(
        "--sdf_koa",
        default=str(S02),
        help="Path to SDF file with logKoa training data.",
    )
    args = parser.parse_args()

    s01 = Path(args.sdf_kow)
    s02 = Path(args.sdf_koa)

    if not s01.exists():
        print(f"ERROR: S01 not found at {s01}")
        sys.exit(1)
    if not s02.exists():
        print(f"ERROR: S02 not found at {s02}")
        sys.exit(1)

    fit(
        sdf_logkow=s01,
        sdf_logkoa=s02,
        logkow_prop="logP",
        logkoa_prop="logKoa",
        method=args.method,
    )
