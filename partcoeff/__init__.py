"""
partcoeff
=========
Group-additivity prediction of logKow, logKoa and logKaw.

Quick start
-----------
>>> import partcoeff
>>> calc = partcoeff.PartitionCalculator()        # loads pre-fitted JSON
>>> calc.predict("CCCCO")
{'logKow': 0.88, 'logKoa': 4.12, 'logKaw': -3.24, 'status': 'ok'}

Re-fitting (needs SDF training files)
--------------------------------------
>>> partcoeff.fit(sdf_logkow="S01.sdf", sdf_logkoa="S02.sdf")
>>> calc = partcoeff.PartitionCalculator()        # reload after fitting

Reference
---------
R. Naef, W.E. Acree Jr., Liquids 4(1):231-260, 2024.
DOI: 10.3390/liquids4010011
"""

from .model import PartitionCalculator, fit
from .io import parse_input
from .features import compute_features
from .atom_types import FEATURE_LABELS, N_FEATURES

__version__ = "0.1.0"
__all__ = [
    "PartitionCalculator",
    "fit",
    "parse_input",
    "compute_features",
    "FEATURE_LABELS",
    "N_FEATURES",
]
