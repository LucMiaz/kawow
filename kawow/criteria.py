"""Regulatory screening criteria for B/vB and M/vM on the logKow/logKoa scale.

Two mobility presets are available:

``"uba"``
    Criteria first proposed by the German Environment Agency (Neumann &
    Schliebner 2019, UBA TEXTE 127/2019): M as logKoc < 4.0, vM as
    logKoc < 3.0. Projected onto logKow with the screening zones of Arp &
    Hale (2019, UBA TEXTE 126/2019, Fig. 9, neutral compounds): not M above
    logKow 4.5, "Potential M/vM" between 3.5 and 4.5, confident vM below 2.5.
``"clp"``
    Criteria of the EU CLP Regulation (Commission Delegated Regulation (EU)
    2023/707): M as logKoc < 3.0, vM as logKoc < 2.0 (lowest logKoc between
    pH 4 and 9). No logKow screening proxy has been validated for these
    cutoffs; the Arp & Hale zones are shifted by the same 1.0 log-unit as
    the logKoc cutoffs (logKoc-logKow regressions have a slope of ~1, e.g.
    Koc ~ 0.35 Kow, Seth et al. 1999, which also gives logKow ~ 3.5 for M).

B/vB are the same in both presets: Kelly et al. (2007) air-breather
criterion (logKoa >= 6 and logKow >= 2 for B, >= 5 for vB) and the aquatic
(fish) route from BCF ~ 0.05 Kow (B at BCF > 2000, logKow > 4.6; vB at
BCF > 5000, logKow > 5.0).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Criteria:
    """Thresholds used by :func:`kawow.model._classify_partition`."""

    name: str
    b_logkow_threshold: float = 2.0
    vb_logkow_threshold: float = 5.0
    b_logkoa_threshold: float = 6.0
    b_aquatic_logkow_threshold: float = math.log10(2000 / 0.05)
    vb_aquatic_logkow_threshold: float = math.log10(5000 / 0.05)
    m_logkow_threshold: float = 4.5
    m_potential_logkow_threshold: float = 3.5
    vm_logkow_threshold: float = 2.5
    m_logkoc_threshold: float = 4.0
    vm_logkoc_threshold: float = 3.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


CRITERIA: dict[str, Criteria] = {
    "uba": Criteria(name="uba"),
    "clp": Criteria(
        name="clp",
        m_logkow_threshold=3.5,
        m_potential_logkow_threshold=2.5,
        vm_logkow_threshold=1.5,
        m_logkoc_threshold=3.0,
        vm_logkoc_threshold=2.0,
    ),
}

DEFAULT_CRITERIA = "uba"


def get_criteria(criteria: str | Criteria | None = None, **overrides: float) -> Criteria:
    """Return a :class:`Criteria` from a preset name (``"uba"``, ``"clp"``)
    or an existing instance, with optional threshold overrides, e.g.
    ``get_criteria("clp", vm_logkow_threshold=2.5)``."""
    if criteria is None:
        criteria = DEFAULT_CRITERIA
    if isinstance(criteria, str):
        key = criteria.lower()
        if key not in CRITERIA:
            raise ValueError(f"Unknown criteria {criteria!r}; available: {sorted(CRITERIA)}")
        criteria = CRITERIA[key]
    return replace(criteria, **overrides) if overrides else criteria
