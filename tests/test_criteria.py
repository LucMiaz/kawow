"""Tests for the regulatory criteria presets (kawow.criteria) and their use
in the B/M classification."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import kawow
from kawow.criteria import CRITERIA, get_criteria
from kawow.model import _classify_partition


def test_presets_mobility_cutoffs():
    assert CRITERIA["uba"].m_logkoc_threshold == 4.0
    assert CRITERIA["uba"].m_logkow_threshold == 4.5
    assert CRITERIA["clp"].m_logkoc_threshold == 3.0
    assert CRITERIA["clp"].vm_logkoc_threshold == 2.0
    assert CRITERIA["clp"].m_logkow_threshold == 3.5
    assert CRITERIA["clp"].vm_logkow_threshold == 1.5


def test_bioaccumulation_identical_across_presets():
    for field in ("b_logkow_threshold", "vb_logkow_threshold", "b_logkoa_threshold",
                  "b_aquatic_logkow_threshold", "vb_aquatic_logkow_threshold"):
        assert getattr(CRITERIA["uba"], field) == getattr(CRITERIA["clp"], field)
    assert CRITERIA["uba"].b_aquatic_logkow_threshold == pytest.approx(4.602, abs=1e-3)


def test_default_is_uba():
    assert _classify_partition(4.0, 5.0) == _classify_partition(4.0, 5.0, "uba")


def test_get_criteria_overrides_and_errors():
    c = get_criteria("clp", vm_logkow_threshold=2.5)
    assert c.vm_logkow_threshold == 2.5 and c.m_logkow_threshold == 3.5
    with pytest.raises(ValueError):
        get_criteria("reach")


@pytest.mark.parametrize("logkow, logkoa, uba_gap, clp_gap", [
    (4.55, 4.0, True, True),    # in both gaps
    (4.0, 5.0, False, True),    # M under UBA, not under CLP
    (4.0, 6.5, False, False),   # B (Kelly) under both
    (4.8, 4.0, False, False),   # aquatic B
    (3.0, 2.0, False, False),   # M under both
])
def test_gap_bm_under_both_presets(logkow, logkoa, uba_gap, clp_gap):
    assert _classify_partition(logkow, logkoa, "uba")["in_gap_bm"] is uba_gap
    assert _classify_partition(logkow, logkoa, "clp")["in_gap_bm"] is clp_gap


def test_in_gap_is_article_gap():
    # Bounded by the aquatic B cutoff (~4.6), not by vB (5.0).
    r = _classify_partition(4.8, 4.0)
    assert r["in_gap"] is False and r["in_gap3"] is False and r["in_gap_bm"] is False
    r = _classify_partition(4.55, 4.0)
    assert r["in_gap"] is True and r["in_gap3"] is True and "Gap 3" in r["gap_labels"]


def test_run_models_criteria():
    uba = kawow.run_models("CCCCCC", models=["crippen"])[0]["models"]["crippen"]
    clp = kawow.run_models("CCCCCC", models=["crippen"], criteria="clp")[0]["models"]["crippen"]
    assert uba["criteria"] == "uba" and clp["criteria"] == "clp"
    assert clp["thresholds"]["M_logKoc_lt"] == 3.0
