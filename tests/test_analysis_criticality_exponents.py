"""Regression tests for the alpha_t / gamma_fit / gamma_predicted bug-fix.

Before the fix, ``criticality()`` set ``alpha_t = 1.0 / slope`` where slope
came from the log_T-vs-log_S regression. That number is the scaling
exponent ``gamma_fit``, not the lifetime power-law exponent ``alpha_t``.
After the fix, ``alpha_t`` is the direct P(T) fit; ``gamma_fit`` is the
regression value; ``gamma_predicted = (alpha_t - 1) / (alpha_s - 1)``;
``kappa = 1 + gamma_predicted``.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurocomplexity.analysis.criticality import (
    CriticalityResult,
    criticality,
    fit_alpha,
    fit_avalanche_exponents,
)
from neurocomplexity.benchmarks.simulators.branching_network import (
    trial_based_avalanches,
)


def _crit():
    rec = trial_based_avalanches(
        n_units=40, n_trials=3000, m=1.0, bin_ms=4.0, seed=0,
    )
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return criticality(rec, populations=["all"], bin_size=(4.0,))


def test_result_carries_new_gamma_fields():
    r = _crit()
    assert hasattr(r, "gamma_fit")
    assert hasattr(r, "gamma_predicted")


def test_alpha_t_is_direct_lifetime_fit_not_regression_slope():
    """alpha_t must equal fit_alpha(lifetimes/bin), NOT 1/slope of log_T vs log_S."""
    r = _crit()
    expected = fit_alpha(r.lifetimes / (r.optimal_bin / 1000.0))
    assert np.isfinite(r.alpha_t)
    assert np.isfinite(expected)
    assert abs(r.alpha_t - expected) < 1e-9


def test_gamma_predicted_matches_formula():
    r = _crit()
    expected = (r.alpha_t - 1.0) / (r.alpha_s - 1.0)
    assert abs(r.gamma_predicted - expected) < 1e-9


def test_gamma_predicted_in_sane_range():
    """Legacy 'kappa' field was removed (duplicated 1+gamma_predicted).
    Guard the surviving canonical gamma_predicted."""
    r = _crit()
    assert np.isfinite(r.gamma_predicted)
    assert 0.5 < r.gamma_predicted < 3.0


def test_gamma_fit_and_gamma_predicted_consistent_at_criticality():
    """Sethna 2001 crackling-noise consistency: gamma_fit ≈ gamma_predicted."""
    r = _crit()
    # Loose tolerance; finite-sample noise from 3000 trials, single seed.
    assert abs(r.gamma_fit - r.gamma_predicted) < 0.5


def test_alpha_t_near_lifetime_target_for_critical_branching():
    """Critical branching theory: alpha_t ≈ 2.0 (Friedman 2012)."""
    r = _crit()
    assert 1.4 < r.alpha_t < 2.6


def test_alpha_t_distinct_from_gamma_fit():
    """The old bug equated alpha_t with gamma_fit. They must now differ."""
    r = _crit()
    # alpha_t should be near 2.0; gamma_fit should be near 1.3-2.0 for critical
    # branching but the two numbers are different statistics.
    # Assert they are independently estimated (not literally equal).
    assert not np.isclose(r.alpha_t, r.gamma_fit, atol=1e-6)


def test_fit_avalanche_exponents_returns_four_values():
    """Helper signature extended: now returns (alpha_s, alpha_t, gamma_fit, r²)."""
    rec = trial_based_avalanches(n_units=30, n_trials=500, m=1.0, bin_ms=4.0, seed=1)
    from neurocomplexity.analysis.criticality import extract_avalanches
    from neurocomplexity.analysis._binning import bin_all_active
    counts = bin_all_active(rec, ["all"], 0.004)
    sizes, lifetimes = extract_avalanches(counts, 0.004)
    out = fit_avalanche_exponents(sizes, lifetimes, 0.004)
    assert len(out) == 4
    a_s, a_t, gf, r2 = out
    assert all(np.isfinite([a_s, a_t, gf, r2]))


def test_degenerate_input_returns_nans_with_new_fields():
    r = CriticalityResult(
        alpha_s=float("nan"), alpha_t=float("nan"),
        optimal_bin=float("nan"), branching=float("nan"),
        sizes=np.array([]), lifetimes=np.array([]),
        r_squared=float("nan"),
        populations=("all",),
        source=None,
    )
    # New fields default to NaN.
    assert np.isnan(r.gamma_fit)
    assert np.isnan(r.gamma_predicted)
