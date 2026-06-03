"""Tests for the expanded AutonomyResult schema (P0-5).

After the P0 review revisions, ``AutonomyResult`` exposes the
F-statistic and the BIC-selected lag per population alongside the
Granger-dependency p-value. This guards the new schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurocomplexity.analysis.autonomy import (
    AutonomyResult,
    _autonomy_for,
    autonomy,
)
from neurocomplexity.core.recording import SpikeRecording


def _make_two_pop_recording(seed: int = 0, duration_s: float = 60.0,
                             coupling: float = 0.0):
    """Two-population Poisson recording.

    ``coupling`` in [0, 1] mixes population A's spike timing into B to make
    the Granger F-test non-degenerate. coupling=0 -> fully independent;
    coupling=0.3 -> 30% of B's spikes copy from A with random delay.
    """
    rng = np.random.default_rng(seed)
    n_units = 20
    rate_hz = 50.0
    n_spikes_per_pop = int(rate_hz * (n_units // 2) * duration_s)
    # Pop A: independent Poisson
    times_a = np.sort(rng.uniform(0.0, duration_s, n_spikes_per_pop))
    units_a = rng.integers(0, n_units // 2, n_spikes_per_pop)
    # Pop B: independent Poisson + optional coupling from A
    times_b = np.sort(rng.uniform(0.0, duration_s, n_spikes_per_pop))
    units_b = rng.integers(n_units // 2, n_units, n_spikes_per_pop)
    if coupling > 0:
        n_copy = int(coupling * n_spikes_per_pop)
        idx = rng.choice(n_spikes_per_pop, n_copy, replace=False)
        delayed = times_a[idx] + rng.uniform(0.005, 0.030, n_copy)
        delayed = delayed[delayed < duration_s]
        copy_units = rng.integers(n_units // 2, n_units, delayed.size)
        times_b = np.concatenate([times_b, delayed])
        units_b = np.concatenate([units_b, copy_units])
        order = np.argsort(times_b)
        times_b = times_b[order]
        units_b = units_b[order]
    spike_times = np.concatenate([times_a, times_b])
    unit_ids = np.concatenate([units_a, units_b]).astype(np.int64)
    order = np.argsort(spike_times)
    spike_times = spike_times[order]
    unit_ids = unit_ids[order]
    pop_a = np.zeros(n_units, dtype=bool)
    pop_a[: n_units // 2] = True
    pop_b = ~pop_a
    return SpikeRecording(
        spike_times=spike_times.astype(np.float64),
        unit_ids=unit_ids,
        units=pd.DataFrame({"id": list(range(n_units))}),
        populations={"A": pop_a, "B": pop_b},
        duration=float(duration_s),
        sampling_rate=None,
        source=None,
        intervals={},
    )


def test_autonomy_result_has_f_stats_and_chosen_lags():
    """Result exposes f_stats + chosen_lags dicts keyed by population."""
    rec = _make_two_pop_recording()
    result = autonomy(rec, bin_size_ms=10.0, max_lag=3)
    assert isinstance(result, AutonomyResult)
    assert set(result.values.keys()) == {"A", "B"}
    assert set(result.f_stats.keys()) == {"A", "B"}
    assert set(result.chosen_lags.keys()) == {"A", "B"}


def test_autonomy_f_stat_finite_and_nonnegative():
    """Coupled recording -> F-stat finite + non-negative (numerical edge
    of perfectly independent processes can return NaN by design)."""
    rec = _make_two_pop_recording(coupling=0.3, duration_s=120.0)
    result = autonomy(rec, bin_size_ms=10.0, max_lag=3)
    for name, f in result.f_stats.items():
        assert np.isfinite(f), f"F-stat for {name} not finite"
        assert f >= 0.0, f"F-stat for {name} negative ({f})"


def test_autonomy_chosen_lag_in_valid_range():
    rec = _make_two_pop_recording()
    max_lag = 5
    result = autonomy(rec, bin_size_ms=10.0, max_lag=max_lag)
    for name, lag in result.chosen_lags.items():
        assert isinstance(lag, int)
        assert 1 <= lag <= max_lag, (
            f"chosen_lag {lag} for {name} outside [1, {max_lag}]"
        )


def test_autonomy_p_value_in_unit_interval():
    """p-value either in [0,1] or NaN (numerical edge case)."""
    rec = _make_two_pop_recording(coupling=0.3, duration_s=120.0)
    result = autonomy(rec, bin_size_ms=10.0, max_lag=3)
    for name, p in result.values.items():
        if np.isnan(p):
            continue
        assert 0.0 <= p <= 1.0, f"p-value for {name} = {p} outside [0, 1]"


def test_autonomy_for_returns_triple():
    """Internal _autonomy_for returns (p, f, lag) triple."""
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, size=(1000, 3))
    out = _autonomy_for(counts, target_col=0, max_lag=4)
    assert isinstance(out, tuple)
    assert len(out) == 3
    p, f, lag = out
    assert np.isnan(p) or 0.0 <= p <= 1.0
    assert np.isnan(f) or f >= 0.0
    assert isinstance(lag, int)


def test_autonomy_coupled_pops_reject_null():
    """Coupled pops -> Granger F-test should detect dependence on the
    upstream pop (p < 0.05 on at least one target)."""
    rec = _make_two_pop_recording(seed=7, duration_s=120.0, coupling=0.5)
    result = autonomy(rec, bin_size_ms=10.0, max_lag=3)
    finite_ps = [p for p in result.values.values() if np.isfinite(p)]
    assert len(finite_ps) > 0, "all p-values NaN"
    # At least one population's null should reject (coupling detectable)
    assert min(finite_ps) < 0.05, (
        f"coupled pops did not reject null: ps = {result.values}"
    )


def test_autonomy_result_is_frozen():
    """Schema additions kept the dataclass frozen."""
    rec = _make_two_pop_recording()
    result = autonomy(rec, bin_size_ms=10.0, max_lag=2)
    with pytest.raises((AttributeError, TypeError)):
        result.values = {}  # type: ignore[misc]
