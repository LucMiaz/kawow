"""Fit MQG-based partition models on S01/S02 and save .pkl artifacts.

Usage:
    python scripts/fit_mqg_model.py

Writes:
    kawow/data/logkow_mqg_model.pkl
    kawow/data/logkoa_mqg_model.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from kawow.model import fit_mqg

MAIN_DIR = Path(__file__).resolve().parents[1]
SDF_DIR = MAIN_DIR / "tests" / "test_data"

S01 = SDF_DIR / "S01. Compounds List for logPow-Parameters Calculations.sdf"
S02 = SDF_DIR / "S02. Compounds List for logKoa-Parameters Calculations.sdf"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit MQG partition models.")
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
    parser.add_argument(
        "--fp_size",
        type=int,
        default=64,
        help="Number of MQG eigenvalue features.",
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

    fit_mqg(
        sdf_logkow=s01,
        sdf_logkoa=s02,
        logkow_prop="logP",
        logkoa_prop="logKoa",
        fp_size=args.fp_size,
    )
