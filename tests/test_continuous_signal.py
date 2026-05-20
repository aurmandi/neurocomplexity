"""Tests for ContinuousSignal and SpikeRecording.signals attachment."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import neurocomplexity as nc
from neurocomplexity.core.continuous import ContinuousSignal
from neurocomplexity.core.exceptions import RecordingValidationError
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording


def _rec(duration=60.0):
    n_units = 4
    units = pd.DataFrame({"id": np.arange(n_units, dtype=np.int64)})
    rng = np.random.default_rng(0)
    times = []
    owners = []
    for u in range(n_units):
        n = rng.poisson(10.0 * duration)
        t = np.sort(rng.uniform(0, duration, size=n))
        times.append(t)
        owners.append(np.full(n, u, dtype=np.int64))
    st = np.concatenate(times)
    uid = np.concatenate(owners)
    order = np.argsort(st, kind="stable")
    return SpikeRecording(
        spike_times=st[order].astype(np.float64),
        unit_ids=uid[order],
        units=units,
        populations={"all": np.ones(n_units, dtype=bool)},
        duration=duration,
        sampling_rate=None,
        source=ProvenanceRecord.for_memory("test"),
    )


def test_continuous_signal_validation_2d():
    with pytest.raises(ValueError):
        ContinuousSignal(values=np.zeros((2, 3)), sampling_rate=100.0)


def test_continuous_signal_validation_empty():
    with pytest.raises(ValueError):
        ContinuousSignal(values=np.array([]), sampling_rate=100.0)


def test_continuous_signal_validation_nan():
    with pytest.raises(ValueError):
        ContinuousSignal(values=np.array([1.0, np.nan]), sampling_rate=100.0)


def test_continuous_signal_validation_sampling_rate():
    with pytest.raises(ValueError):
        ContinuousSignal(values=np.array([1.0]), sampling_rate=0.0)
    with pytest.raises(ValueError):
        ContinuousSignal(values=np.array([1.0]), sampling_rate=-1.0)


def test_continuous_signal_validation_t_start():
    with pytest.raises(ValueError):
        ContinuousSignal(values=np.array([1.0]), sampling_rate=100.0, t_start=-0.1)


def test_duration_property():
    sig = ContinuousSignal(values=np.zeros(200), sampling_rate=100.0)
    assert sig.duration == pytest.approx(2.0)
    assert sig.t_end == pytest.approx(2.0)


def test_signal_attaches_via_with_signal():
    rec = _rec()
    sig = ContinuousSignal(values=np.zeros(6000), sampling_rate=100.0, label="pupil")
    rec2 = rec.with_signal("pupil", sig)
    assert "pupil" not in rec.signals  # original untouched
    assert rec2.signals["pupil"] is sig


def test_signal_must_fit_within_recording_duration():
    rec = _rec(duration=10.0)
    sig = ContinuousSignal(values=np.zeros(2000), sampling_rate=100.0)  # 20s
    with pytest.raises(RecordingValidationError):
        rec.with_signal("speed", sig)


def test_signal_at_exact_duration_ok():
    rec = _rec(duration=10.0)
    sig = ContinuousSignal(values=np.zeros(1000), sampling_rate=100.0)  # exactly 10s
    rec2 = rec.with_signal("speed", sig)
    assert "speed" in rec2.signals


def test_signal_with_t_start_offset():
    rec = _rec(duration=20.0)
    sig = ContinuousSignal(values=np.zeros(500), sampling_rate=100.0, t_start=5.0)
    rec2 = rec.with_signal("late", sig)
    assert rec2.signals["late"].t_end == pytest.approx(10.0)


def test_continuous_signal_exported_at_top_level():
    assert nc.ContinuousSignal is ContinuousSignal


def test_signal_equality():
    a = ContinuousSignal(values=np.arange(10, dtype=float), sampling_rate=100.0)
    b = ContinuousSignal(values=np.arange(10, dtype=float), sampling_rate=100.0)
    assert a == b


def test_signal_inequality_on_values():
    a = ContinuousSignal(values=np.arange(10, dtype=float), sampling_rate=100.0)
    b = ContinuousSignal(values=np.arange(10, dtype=float) + 1, sampling_rate=100.0)
    assert a != b


# ---- binning helpers ---------------------------------------------------------

def test_bin_signal_binary_misaligned_raises():
    from neurocomplexity.analysis._continuous import bin_signal_binary
    sig = ContinuousSignal(values=np.zeros(60), sampling_rate=60.0, label="pupil")
    # 60 Hz signal vs 50 ms bin -> 3 samples per bin (exact integer)
    bin_signal_binary(sig, bin_size_s=0.05, n_bins=20)  # 50 ms -> 3 samples ok
    with pytest.raises(ValueError):
        # 60 Hz signal vs 5 ms bin -> 0.3 samples per bin (not integer)
        bin_signal_binary(sig, bin_size_s=0.005, n_bins=200)


def test_bin_signal_binary_aligned_returns_correct_length():
    from neurocomplexity.analysis._continuous import bin_signal_binary
    sig = ContinuousSignal(values=np.zeros(1000), sampling_rate=100.0)
    out = bin_signal_binary(sig, bin_size_s=0.01, n_bins=1000)
    assert out.shape == (1000,)


def test_bin_signal_binary_median_split_halves():
    from neurocomplexity.analysis._continuous import bin_signal_binary
    rng = np.random.default_rng(0)
    sig = ContinuousSignal(values=rng.normal(size=10000), sampling_rate=100.0)
    out = bin_signal_binary(sig, bin_size_s=0.1, n_bins=1000)  # 10 samples/bin
    ones = int(out.sum())
    assert abs(ones - 500) <= 2  # binary median split


def test_bin_signal_binary_explicit_threshold():
    from neurocomplexity.analysis._continuous import bin_signal_binary
    sig = ContinuousSignal(values=np.linspace(0, 1, 1000), sampling_rate=100.0)
    out_hi = bin_signal_binary(sig, bin_size_s=0.01, n_bins=1000, threshold=999.0)
    assert out_hi.sum() == 0
    out_lo = bin_signal_binary(sig, bin_size_s=0.01, n_bins=1000, threshold=-999.0)
    assert out_lo.sum() == 1000
