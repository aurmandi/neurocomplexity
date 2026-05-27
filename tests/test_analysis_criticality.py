"""Regression tests for the alpha_t / gamma fix in analysis.criticality.

The bug: previously ``alpha_t = 1 / slope(log_T ~ log_S)``, which is the
empirical scaling exponent gamma_fit, NOT the lifetime power-law exponent
alpha_t.

The fix: ``alpha_t`` is now obtained from a DIRECT log-spaced histogram fit
of the lifetime distribution (P(T) ~ T^-alpha_t). The old quantity is
preserved as ``gamma_fit`` so the Sethna crackling-noise consistency test
(gamma_fit vs gamma_predicted) can be performed.
"""
from __future__ import annotations

import warnings as _warnings

import numpy as np
import pandas as pd
import pytest

from neurocomplexity.analysis.criticality import (
    CriticalityResult,
    criticality,
    fit_alpha,
    fit_avalanche_exponents,
)
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording


def _critical_branching_rec(duration_s=120.0, n_units=80, m=1.0,
                            bin_ms=4.0, seed=0):
    """Synthesize a branching-process recording at m=1 (critical)."""
    rng = np.random.default_rng(seed)
    bs = bin_ms / 1000.0
    n_bins = int(duration_s / bs)
    rate = np.zeros(n_bins, dtype=np.float64)
    rate[0] = 1.0
    for t in range(1, n_bins):
        rate[t] = rng.poisson(m * rate[t - 1]) + (1 if rng.random() < 0.02 else 0)
    # Spread the rate uniformly across n_units
    spike_times = []
    spike_uids = []
    for t, n in enumerate(rate):
        n_int = int(n)
        if n_int == 0:
            continue
        t0 = t * bs
        units_for_bin = rng.integers(0, n_units, size=n_int)
        offs = rng.uniform(0, bs, size=n_int)
        spike_times.extend(t0 + offs)
        spike_uids.extend(units_for_bin.tolist())
    st = np.asarray(spike_times, dtype=np.float64)
    uid = np.asarray(spike_uids, dtype=np.int64)
    order = np.argsort(st, kind="stable")
    units = pd.DataFrame({"id": np.arange(n_units, dtype=np.int64)})
    return SpikeRecording(
        spike_times=st[order],
        unit_ids=uid[order],
        units=units,
        populations={"all": np.ones(n_units, dtype=bool)},
        duration=float(duration_s),
        sampling_rate=None,
        source=ProvenanceRecord.for_memory("test"),
    )


def _run(rec, **kw):
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        return criticality(rec, populations=["all"],
                           bin_size_ms=(2, 4, 8), **kw)


# ---- the bug-fix assertions ---------------------------------------------

def test_result_has_new_fields():
    rec = _critical_branching_rec()
    r = _run(rec)
    assert hasattr(r, "gamma_fit")
    assert hasattr(r, "gamma_predicted")


def test_alpha_t_matches_direct_pt_fit():
    """alpha_t in result equals fit_alpha on lifetimes/bin_size at the
    chosen optimal bin — the canonical estimator, NOT the regression slope."""
    rec = _critical_branching_rec()
    r = _run(rec)
    if np.isnan(r.alpha_t):
        pytest.skip("not enough avalanches for fit")
    bin_units = r.lifetimes / r.optimal_bin_seconds
    direct = fit_alpha(bin_units)
    assert np.isclose(r.alpha_t, direct, rtol=1e-9, atol=1e-9), (
        f"alpha_t={r.alpha_t} should equal fit_alpha(lifetimes/bs)={direct}"
    )


def test_gamma_fit_equals_inverse_slope():
    """gamma_fit equals 1 / slope of log_T vs log_S — preserves the value
    the buggy code was reporting as alpha_t."""
    from scipy.stats import linregress
    rec = _critical_branching_rec()
    r = _run(rec)
    if np.isnan(r.gamma_fit):
        pytest.skip("not enough avalanches for fit")
    log_s = np.log(r.sizes.astype(float))
    log_t = np.log(r.lifetimes / r.optimal_bin_seconds)
    slope, *_ = linregress(log_s, log_t)
    assert np.isclose(r.gamma_fit, 1.0 / slope, rtol=1e-9, atol=1e-9)


def test_gamma_predicted_formula():
    rec = _critical_branching_rec()
    r = _run(rec)
    if np.isnan(r.gamma_predicted):
        pytest.skip("not enough avalanches for fit")
    expected = (r.alpha_t - 1.0) / (r.alpha_s - 1.0)
    assert np.isclose(r.gamma_predicted, expected, rtol=1e-9)


def test_kappa_equals_one_plus_gamma_predicted():
    rec = _critical_branching_rec()
    r = _run(rec)
    if np.isnan(r.kappa):
        pytest.skip("nan kappa")
    assert np.isclose(r.kappa, 1.0 + r.gamma_predicted, rtol=1e-9)


def test_alpha_t_is_not_equal_to_gamma_fit():
    """The whole point: post-fix alpha_t and gamma_fit are distinct
    quantities. On real avalanche data they differ; this test ensures we
    don't regress to the old conflated value."""
    rec = _critical_branching_rec(seed=2)
    r = _run(rec)
    if np.isnan(r.alpha_t) or np.isnan(r.gamma_fit):
        pytest.skip("nan")
    # Allow tiny chance of accidental equality on degenerate noise; require
    # a fractional gap of at least 5%.
    rel = abs(r.alpha_t - r.gamma_fit) / max(abs(r.alpha_t), 1e-9)
    assert rel > 0.05, (
        f"alpha_t={r.alpha_t:.4f} should not equal gamma_fit={r.gamma_fit:.4f}; "
        "indicates the old conflation bug returned."
    )


# ---- helper-level test ---------------------------------------------------

def test_fit_avalanche_exponents_returns_four_values():
    rec = _critical_branching_rec()
    r = _run(rec)
    alpha_s, alpha_t, gamma_fit, r2 = fit_avalanche_exponents(
        r.sizes, r.lifetimes, r.optimal_bin_seconds)
    assert np.isclose(alpha_s, r.alpha_s, rtol=1e-9, atol=1e-9)
    assert np.isclose(alpha_t, r.alpha_t, rtol=1e-9, atol=1e-9)
    assert np.isclose(gamma_fit, r.gamma_fit, rtol=1e-9, atol=1e-9)
    assert np.isclose(r2, r.r_squared, rtol=1e-9)


# ---- nan path remains intact --------------------------------------------

def test_empty_recording_returns_all_nan_with_new_fields():
    units = pd.DataFrame({"id": [0]})
    rec = SpikeRecording(
        spike_times=np.array([0.1], dtype=np.float64),
        unit_ids=np.array([0], dtype=np.int64),
        units=units,
        populations={"all": np.array([True])},
        duration=2.0,
        sampling_rate=None,
        source=ProvenanceRecord.for_memory("empty"),
    )
    r = _run(rec)
    assert np.isnan(r.alpha_s)
    assert np.isnan(r.alpha_t)
    assert np.isnan(r.gamma_fit)
    assert np.isnan(r.gamma_predicted)
    assert np.isnan(r.kappa)
