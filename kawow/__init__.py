"""
kawow
=========
Group-additivity prediction of logKow, logKoa and logKaw.

Quick start
-----------
>>> import kawow
>>> calc = kawow.PartitionCalculator()        # loads pre-fitted JSON
>>> calc.predict("CCCCO")
{'logKow': 0.88, 'logKoa': 4.12, 'logKaw': -3.24, 'status': 'ok'}

Re-fitting (needs SDF training files)
--------------------------------------
>>> kawow.fit(sdf_logkow="S01.sdf", sdf_logkoa="S02.sdf")
>>> calc = kawow.PartitionCalculator()        # reload after fitting

Reference
---------
R. Naef, W.E. Acree Jr., Liquids 4(1):231-260, 2024.
DOI: 10.3390/liquids4010011
"""

from .model import PartitionCalculator, MQGPartitionCalculator, EnsemblePartitionCalculator, fit, fit_mqg, run_models
from .smarts_model import NaefAcreePartitionCalculator, NaefAcreeCrippenMixedPartitionCalculator
from .io import parse_input
from .features import compute_features
from .atom_types import FEATURE_LABELS, N_FEATURES

__version__ = "0.1.3"
__all__ = [
    "PartitionCalculator",
    "MQGPartitionCalculator",
    "EnsemblePartitionCalculator",
    "NaefAcreePartitionCalculator",
    "NaefAcreeCrippenMixedPartitionCalculator",
    "fit",
    "fit_mqg",
    "run_models",
    "parse_input",
    "compute_features",
    "FEATURE_LABELS",
    "N_FEATURES",
]
