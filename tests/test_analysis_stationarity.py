"""Tests for nc.analysis.stationarity and the StationarityWarning hook."""
from __future__ import annotations

import warnings as _warnings

import numpy as np
import pandas as pd
import pytest

from neurocomplexity._warnings import (
    StationarityWarning,
    _reset_stationarity_dedup,
    _warn_if_nonstationary,
)
from neurocomplexity.analysis.stationarity import stationarity
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording


def setup_function():
    _reset_stationarity_dedup()


def _homog_poisson(n_units=20, rate_hz=10.0, duration=120.0, seed=0):
    rng = np.random.default_rng(seed)
    all_times = []
    all_owners = []
    for u in range(n_units):
        n = rng.poisson(rate_hz * duration)
        t = np.sort(rng.uniform(0.0, duration, size=n))
        all_times.append(t)
        all_owners.append(np.full(n, u, dtype=np.int64))
    st = np.concatenate(all_times)
    uid = np.concatenate(all_owners)
    order = np.argsort(st, kind="stable")
    return SpikeRecording(
        spike_times=st[order].astype(np.float64),
        unit_ids=uid[order],
        units=pd.DataFrame({"id": np.arange(n_units, dtype=np.int64)}),
        populations={"all": np.ones(n_units, dtype=bool)},
        duration=float(duration),
        sampling_rate=None,
        source=ProvenanceRecord.for_memory("test"),
    )


def _ramp_rate(n_units=20, r_start=5.0, r_end=25.0, duration=120.0, seed=1):
    """Population firing rate ramps linearly from r_start to r_end Hz."""
    rng = np.random.default_rng(seed)
    n_bins = int(duration * 100)  # 10ms bins for inhomogeneous Poisson
    dt = duration / n_bins
    centres = (np.arange(n_bins) + 0.5) * dt
    rate_t = r_start + (r_end - r_start) * (centres / duration)
    all_times = []
    all_owners = []
    for u in range(n_units):
        # per-unit thinned Poisson
        lam = rate_t * dt
        counts = rng.poisson(lam)
        bin_starts = centres - dt / 2
        t_unit = []
        for i, c in enumerate(counts):
            if c:
                t_unit.append(rng.uniform(bin_starts[i], bin_starts[i] + dt, size=c))
        if t_unit:
            tt = np.sort(np.concatenate(t_unit))
        else:
            tt = np.empty(0)
        all_times.append(tt)
        all_owners.append(np.full(tt.size, u, dtype=np.int64))
    st = np.concatenate(all_times) if all_times else np.empty(0)
    uid = np.concatenate(all_owners) if all_owners else np.empty(0, dtype=np.int64)
    order = np.argsort(st, kind="stable")
    return SpikeRecording(
        spike_times=st[order].astype(np.float64),
        unit_ids=uid[order],
        units=pd.DataFrame({"id": np.arange(n_units, dtype=np.int64)}),
        populations={"all": np.ones(n_units, dtype=bool)},
        duration=float(duration),
        sampling_rate=None,
        source=ProvenanceRecord.for_memory("test"),
    )


def test_homogeneous_poisson_is_stationary():
    rec = _homog_poisson()
    r = stationarity(rec)
    assert r.is_stationary, r.flags
    assert r.flags == ()


def test_linear_rate_drift_flagged():
    rec = _ramp_rate(r_start=2.0, r_end=30.0, duration=120.0, seed=2)
    r = stationarity(rec)
    assert not r.is_stationary
    assert any("rate_drift" in f for f in r.flags), r.flags
    assert r.rate_drift_slope > 0.0


def test_window_count_correct():
    rec = _homog_poisson(duration=120.0)
    r = stationarity(rec, window_s=30.0)
    assert r.n_windows == 4


def test_short_recording_uses_at_least_two_windows():
    rec = _homog_poisson(duration=10.0)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        r = stationarity(rec, window_s=30.0)
    assert r.n_windows >= 2
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_params_round_trip():
    rec = _homog_poisson()
    r = stationarity(rec, window_s=20.0, cv_threshold=0.5)
    assert r.params["window_s"] == 20.0
    assert r.params["cv_threshold"] == 0.5


def test_warn_helper_emits_for_drift_recording():
    rec = _ramp_rate(r_start=2.0, r_end=30.0)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _warn_if_nonstationary(rec, "transfer_entropy")
    assert any(issubclass(w.category, StationarityWarning) for w in caught)


def test_warn_helper_silent_for_stationary_recording():
    rec = _homog_poisson()
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _warn_if_nonstationary(rec, "transfer_entropy")
    assert not any(issubclass(w.category, StationarityWarning) for w in caught)


def test_warn_helper_deduplicated_per_rec_and_analysis():
    rec = _ramp_rate()
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _warn_if_nonstationary(rec, "transfer_entropy")
        _warn_if_nonstationary(rec, "transfer_entropy")
    n = sum(1 for w in caught if issubclass(w.category, StationarityWarning))
    assert n == 1


def test_warn_helper_separate_analyses_each_warn_once():
    rec = _ramp_rate()
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _warn_if_nonstationary(rec, "transfer_entropy")
        _warn_if_nonstationary(rec, "branching_ratio")
    n = sum(1 for w in caught if issubclass(w.category, StationarityWarning))
    assert n == 2


def test_warning_class_exposed_via_nc_warnings():
    import neurocomplexity as nc
    assert nc.warnings.StationarityWarning is StationarityWarning
