"""
fit_model.py
============
Fit Ridge regression models on the Naef & Acree (Liquids 2024) SDF training data.

Usage
-----
    cd c:\\Users\\luc\\git\\kawow
    C:\\Users\\luc\\Miniforge3\\envs\\chem\\python.exe scripts\\fit_model.py

Reads:
    S01. Compounds List for logPow-Parameters Calculations.sdf  → logKow
    S02. Compounds List for logKoa-Parameters Calculations.sdf  → logKoa

Writes:
    kawow/data/logkow_model.json
    kawow/data/logkoa_model.json
"""

import sys
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from kawow.model import fit

SDF_DIR = Path(
    r"c:\Users\luc\kDrive\Documents\WORK\PhD\vPM-vPB gap\data\liquids-04-00011-s001"
)

S01 = SDF_DIR / "S01. Compounds List for logPow-Parameters Calculations.sdf"
S02 = SDF_DIR / "S02. Compounds List for logKoa-Parameters Calculations.sdf"

if __name__ == "__main__":
    if not S01.exists():
        print(f"ERROR: S01 not found at {S01}")
        sys.exit(1)
    if not S02.exists():
        print(f"ERROR: S02 not found at {S02}")
        sys.exit(1)

    fit(
        sdf_logkow=S01,
        sdf_logkoa=S02,
        logkow_prop="logP",
        logkoa_prop="logKoa",
    )
