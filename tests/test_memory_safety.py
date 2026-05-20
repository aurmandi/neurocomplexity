"""Tests for chunked bin_spikes + MemoryAllocationWarning + estimate helper."""
from __future__ import annotations

import warnings as _warnings

import numpy as np
import pandas as pd
import pytest

import neurocomplexity as nc
from neurocomplexity.analysis._binning import bin_spikes, estimate_bin_spikes_bytes
from neurocomplexity._warnings import MemoryAllocationWarning
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording


def _rec(duration=60.0, n_units=4, rate_hz=20.0, seed=0):
    rng = np.random.default_rng(seed)
    units = pd.DataFrame({"id": np.arange(n_units, dtype=np.int64)})
    times = []
    owners = []
    for u in range(n_units):
        n = rng.poisson(rate_hz * duration)
        t = np.sort(rng.uniform(0.0, duration, size=n))
        times.append(t)
        owners.append(np.full(n, u, dtype=np.int64))
    st = np.concatenate(times)
    uid = np.concatenate(owners)
    order = np.argsort(st, kind="stable")
    pops = {
        "all": np.ones(n_units, dtype=bool),
        "even": np.array([i % 2 == 0 for i in range(n_units)]),
    }
    return SpikeRecording(
        spike_times=st[order].astype(np.float64),
        unit_ids=uid[order],
        units=units,
        populations=pops,
        duration=duration,
        sampling_rate=None,
        source=ProvenanceRecord.for_memory("test"),
    )


# ---- estimate helper ----------------------------------------------------

def test_estimate_bytes_matches_formula():
    rec = _rec(duration=60.0)
    # 60s / 5ms = 12_000 bins; 2 pops; int32 -> 4 bytes
    assert estimate_bin_spikes_bytes(rec, ["all", "even"], 5.0) == 12_000 * 2 * 4


def test_estimate_bytes_accepts_int_population_count():
    rec = _rec(duration=10.0)
    bytes_list = estimate_bin_spikes_bytes(rec, ["all", "even"], 10.0)
    bytes_int = estimate_bin_spikes_bytes(rec, 2, 10.0)
    assert bytes_list == bytes_int


def test_estimate_bytes_invalid_bin_size_raises():
    rec = _rec()
    with pytest.raises(ValueError):
        estimate_bin_spikes_bytes(rec, ["all"], 0.0)


def test_estimate_exported_at_top_level():
    assert nc.estimate_bin_spikes_bytes is estimate_bin_spikes_bytes


# ---- chunked path -------------------------------------------------------

def test_chunked_matches_unchunked():
    rec = _rec(duration=30.0, seed=42)
    a = bin_spikes(rec, ["all", "even"], 0.005)
    b = bin_spikes(rec, ["all", "even"], 0.005, chunk_seconds=10.0)
    np.testing.assert_array_equal(a, b)


def test_chunked_matches_unchunked_uneven_chunk():
    rec = _rec(duration=30.0, seed=43)
    a = bin_spikes(rec, ["all"], 0.01)
    b = bin_spikes(rec, ["all"], 0.01, chunk_seconds=7.0)  # 30/7 doesn't divide evenly
    np.testing.assert_array_equal(a, b)


def test_chunk_seconds_zero_raises():
    rec = _rec()
    with pytest.raises(ValueError):
        bin_spikes(rec, ["all"], 0.005, chunk_seconds=0.0)


def test_chunk_seconds_negative_raises():
    rec = _rec()
    with pytest.raises(ValueError):
        bin_spikes(rec, ["all"], 0.005, chunk_seconds=-1.0)


def test_chunk_seconds_larger_than_duration_raises():
    rec = _rec(duration=60.0)
    with pytest.raises(ValueError):
        bin_spikes(rec, ["all"], 0.005, chunk_seconds=999.0)


# ---- allocation warning -------------------------------------------------

def test_allocation_warning_fires_when_buffer_exceeds_threshold(monkeypatch):
    rec = _rec(duration=60.0)

    class _FakeVM:
        available = 1024  # 1 KB available -> trivially exceeded

    fake_psutil = type("M", (), {"virtual_memory": staticmethod(lambda: _FakeVM())})
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        bin_spikes(rec, ["all", "even"], 0.005)
    assert any(issubclass(w.category, MemoryAllocationWarning) for w in caught)


def test_allocation_warning_silent_for_small_buffer(monkeypatch):
    rec = _rec(duration=10.0)

    class _FakeVM:
        available = 16 * 1024 ** 3  # 16 GB

    fake_psutil = type("M", (), {"virtual_memory": staticmethod(lambda: _FakeVM())})
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        bin_spikes(rec, ["all"], 0.005)
    assert not any(issubclass(w.category, MemoryAllocationWarning) for w in caught)


def test_allocation_warning_silent_when_chunked(monkeypatch):
    rec = _rec(duration=60.0)

    class _FakeVM:
        available = 1024

    fake_psutil = type("M", (), {"virtual_memory": staticmethod(lambda: _FakeVM())})
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        bin_spikes(rec, ["all", "even"], 0.005, chunk_seconds=10.0)
    assert not any(issubclass(w.category, MemoryAllocationWarning) for w in caught)


def test_warning_class_exposed_on_nc_warnings():
    assert nc.warnings.MemoryAllocationWarning is MemoryAllocationWarning


def test_warning_suppressible_via_filterwarnings(monkeypatch):
    rec = _rec(duration=60.0)

    class _FakeVM:
        available = 1024

    fake_psutil = type("M", (), {"virtual_memory": staticmethod(lambda: _FakeVM())})
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("ignore", MemoryAllocationWarning)
        bin_spikes(rec, ["all", "even"], 0.005)
    assert not any(issubclass(w.category, MemoryAllocationWarning) for w in caught)
