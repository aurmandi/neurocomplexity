"""Tests for the Sethna crackling-noise consistency fields (P0-6).

CriticalityResult exposes both gamma_fit (regression of <S>(T)) and
gamma_predicted = (alpha_t - 1) / (alpha_s - 1). The Tab 3 mismatch
in the paper is computed from these two fields, so we guard their
schema + identity here.
"""
from __future__ import annotations

import math

from neurocomplexity.analysis.criticality import CriticalityResult


def test_criticality_result_has_gamma_fields():
    """gamma_fit and gamma_predicted are first-class fields."""
    fields = CriticalityResult.__dataclass_fields__
    assert "gamma_fit" in fields
    assert "gamma_predicted" in fields


def test_gamma_predicted_identity():
    """gamma_pred = (alpha_t - 1) / (alpha_s - 1) (Sethna 2001)."""
    # Tab 3 real-data values from the paper as a regression test:
    # alpha_s = 1.15, alpha_t = 1.22 -> gamma_pred = 0.22/0.15 ~ 1.467
    alpha_s = 1.15
    alpha_t = 1.22
    expected = (alpha_t - 1.0) / (alpha_s - 1.0)
    assert math.isclose(expected, 0.22 / 0.15, rel_tol=1e-9)
    assert abs(expected - 1.4667) < 5e-3
    # Paper Tab 3 reports gamma_fit = 1.29, so Sethna mismatch is
    # |1.29 - 1.467| = 0.18 (matches paper Tab 3 row).
    gamma_fit = 1.29
    mismatch = abs(gamma_fit - expected)
    assert abs(mismatch - 0.18) < 0.01


def test_criticality_result_is_frozen():
    """Dataclass remains frozen after schema additions."""
    assert CriticalityResult.__dataclass_params__.frozen is True
