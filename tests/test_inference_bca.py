"""Tests for BCa bootstrap confidence interval (P0-4).

Verifies that ``_ci_from_dist`` correctly implements the
bias-corrected and accelerated (BCa) interval (Efron 1987;
DiCiccio & Efron 1996) and that the legacy BC and percentile
methods remain available.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurocomplexity.inference.bootstrap import _ci_from_dist


def test_percentile_method_basic():
    """Percentile method returns the unadjusted quantiles."""
    rng = np.random.default_rng(0)
    dist = rng.normal(0.5, 0.1, 5000)
    lo, hi = _ci_from_dist(dist, 0.95, observed=0.5, method="percentile")
    expected_lo = float(np.percentile(dist, 2.5))
    expected_hi = float(np.percentile(dist, 97.5))
    assert abs(lo - expected_lo) < 1e-9
    assert abs(hi - expected_hi) < 1e-9


def test_percentile_with_none_observed():
    """observed=None falls back to percentile regardless of method."""
    rng = np.random.default_rng(0)
    dist = rng.normal(0.5, 0.1, 5000)
    lo, hi = _ci_from_dist(dist, 0.95, observed=None, method="bca")
    expected_lo = float(np.percentile(dist, 2.5))
    expected_hi = float(np.percentile(dist, 97.5))
    assert abs(lo - expected_lo) < 1e-9
    assert abs(hi - expected_hi) < 1e-9


def test_bca_symmetric_matches_bc():
    """For a near-symmetric distribution, BCa and BC agree closely."""
    rng = np.random.default_rng(0)
    dist = rng.normal(0.5, 0.05, 5000)
    lo_bc, hi_bc = _ci_from_dist(dist, 0.95, observed=0.5, method="bc")
    lo_bca, hi_bca = _ci_from_dist(dist, 0.95, observed=0.5, method="bca")
    # BCa adds an acceleration term that is ~0 on symmetric distributions
    assert abs(lo_bc - lo_bca) < 0.005
    assert abs(hi_bc - hi_bca) < 0.005


def test_bca_skewed_differs_from_bc():
    """On a skewed bootstrap distribution, BCa and BC do not collapse to
    the same interval — the acceleration term is non-zero."""
    rng = np.random.default_rng(0)
    # Beta(50, 2) is concentrated near 1 with strong left skew —
    # mimics the m -> 1 boundary regime of the branching ratio.
    dist = rng.beta(50.0, 2.0, 5000)
    observed = float(np.mean(dist))
    lo_bc, hi_bc = _ci_from_dist(dist, 0.95, observed=observed, method="bc")
    lo_bca, hi_bca = _ci_from_dist(dist, 0.95, observed=observed, method="bca")
    # The two methods produce numerically distinct intervals on a skewed dist.
    assert (abs(lo_bc - lo_bca) > 1e-4) or (abs(hi_bc - hi_bca) > 1e-4)


def test_invalid_method_raises():
    rng = np.random.default_rng(0)
    dist = rng.normal(0.0, 1.0, 1000)
    with pytest.raises(ValueError, match="method must be one of"):
        _ci_from_dist(dist, 0.95, observed=0.0, method="bogus")


def test_vector_statistic_bca():
    """BCa handles vector-valued statistics component-wise."""
    rng = np.random.default_rng(0)
    # 2-D bootstrap dist: (n_reps, 3)
    dist = rng.normal(loc=[0.5, 1.0, -0.3], scale=[0.05, 0.1, 0.02], size=(2000, 3))
    observed = np.array([0.5, 1.0, -0.3])
    lo, hi = _ci_from_dist(dist, 0.95, observed=observed, method="bca")
    assert lo.shape == (3,)
    assert hi.shape == (3,)
    assert np.all(lo < observed)
    assert np.all(hi > observed)


def test_default_method_is_bca():
    """Default method (no explicit kw) is BCa."""
    rng = np.random.default_rng(0)
    dist = rng.beta(50.0, 2.0, 2000)
    observed = float(np.mean(dist))
    lo_default, hi_default = _ci_from_dist(dist, 0.95, observed=observed)
    lo_bca, hi_bca = _ci_from_dist(dist, 0.95, observed=observed, method="bca")
    assert abs(lo_default - lo_bca) < 1e-9
    assert abs(hi_default - hi_bca) < 1e-9


def test_bca_covers_observed():
    """BCa interval should contain the observed value (well-behaved case)."""
    rng = np.random.default_rng(42)
    dist = rng.normal(0.5, 0.05, 3000)
    observed = 0.5
    lo, hi = _ci_from_dist(dist, 0.95, observed=observed, method="bca")
    assert lo <= observed <= hi


def test_bca_degenerate_dist_returns_nan():
    """Distribution with too few finite values returns NaN bounds."""
    dist = np.array([0.5, float("nan"), float("nan"), 0.5])
    lo, hi = _ci_from_dist(dist, 0.95, observed=0.5, method="bca")
    assert np.isnan(lo) and np.isnan(hi)


def test_branching_bootstrap_default_n_is_2000():
    """P0-4 verification: bootstrap_branching_ratio default n is 2000."""
    import inspect

    from neurocomplexity.inference.bootstrap import bootstrap_branching_ratio
    sig = inspect.signature(bootstrap_branching_ratio)
    assert sig.parameters["n"].default == 2000
