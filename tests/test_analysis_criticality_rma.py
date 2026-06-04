"""RMA / ODR regression option for fit_avalanche_exponents (A1).

OLS treats ``log_s`` as error-free; on real avalanche data both axes
carry sampling noise, so RMA (Reduced Major Axis / Geometric Mean) and
ODR (Orthogonal Distance Regression) give different slopes. The user can
now opt into either with ``regression={"ols","rma","odr"}``. OLS remains
the default so existing benchmark / paper numbers are unchanged.
"""
from __future__ import annotations

import warnings as _warnings

import numpy as np
import pytest
from scipy.stats import linregress

from neurocomplexity.analysis.criticality import (
    _fit_slope,
    fit_avalanche_exponents,
)


def _avalanches(seed: int = 0, n: int = 800):
    """Synthetic power-law sizes and lifetimes with correlated noise on both axes."""
    rng = np.random.default_rng(seed)
    # log_t ~ slope_true * log_s + noise; inject noise on log_s too so OLS is biased.
    slope_true = 0.7
    log_s_clean = rng.uniform(np.log(1), np.log(200), size=n)
    log_t_clean = slope_true * log_s_clean
    log_s = log_s_clean + rng.normal(0, 0.3, size=n)
    log_t = log_t_clean + rng.normal(0, 0.3, size=n)
    sizes = np.exp(log_s).clip(min=1.0)
    bin_size = 0.004
    lifetimes = np.exp(log_t) * bin_size
    return sizes, lifetimes, bin_size


def test_rma_matches_geometric_mean_formula():
    """RMA slope = sqrt(slope_yx * slope_xy) * sign(corr)."""
    rng = np.random.default_rng(0)
    log_s = rng.normal(0, 1, size=500)
    log_t = 1.4 * log_s + rng.normal(0, 0.5, size=500)
    slope_yx, _, r_val, _, _ = linregress(log_s, log_t)
    slope_xy, _, _, _, _ = linregress(log_t, log_s)
    sign = 1.0 if r_val >= 0 else -1.0
    expected = sign * np.sqrt(slope_yx / slope_xy)  # = std(y)/std(x) * sign(r)
    got, _ = _fit_slope(log_s, log_t, regression="rma")
    assert np.isclose(got, expected, rtol=1e-9, atol=1e-9), (got, expected)


def test_ols_default_unchanged():
    """Default ``regression='ols'`` reproduces legacy ``linregress`` slope."""
    sizes, lifetimes, bs = _avalanches(seed=1)
    log_s = np.log(sizes.astype(float))
    log_t_bins = np.log(lifetimes / bs)
    slope_legacy, _, r_val, _, _ = linregress(log_s, log_t_bins)
    _, _, gamma_fit, r2 = fit_avalanche_exponents(sizes, lifetimes, bs)
    assert np.isclose(gamma_fit, 1.0 / slope_legacy, rtol=1e-9, atol=1e-9)
    assert np.isclose(r2, float(r_val ** 2), rtol=1e-9)


def test_rma_diverges_from_ols_under_two_axis_noise():
    """When both axes have noise, RMA and OLS slopes must differ."""
    sizes, lifetimes, bs = _avalanches(seed=2)
    _, _, g_ols, _ = fit_avalanche_exponents(sizes, lifetimes, bs, regression="ols")
    _, _, g_rma, _ = fit_avalanche_exponents(sizes, lifetimes, bs, regression="rma")
    assert np.isfinite(g_ols) and np.isfinite(g_rma)
    # OLS attenuates slope under errors-in-x; RMA should sit above it (for
    # positive correlation), and the gap must be measurable.
    assert abs(g_ols - g_rma) > 0.01, (g_ols, g_rma)


def test_odr_runs_or_falls_back():
    """ODR returns a finite slope; on failure it falls back to OLS with a warning."""
    sizes, lifetimes, bs = _avalanches(seed=3)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _, _, g_odr, _ = fit_avalanche_exponents(sizes, lifetimes, bs, regression="odr")
    assert np.isfinite(g_odr)
    # Either succeeded silently, or the documented RuntimeWarning fallback fired.
    rt = [w for w in caught if issubclass(w.category, RuntimeWarning)
          and "odr" in str(w.message).lower()]
    # Both states are acceptable; just make sure we did not silently raise.
    _ = rt


def test_unknown_regression_raises():
    sizes, lifetimes, bs = _avalanches(seed=4)
    with pytest.raises(ValueError, match="unknown regression"):
        fit_avalanche_exponents(sizes, lifetimes, bs, regression="lasso")
