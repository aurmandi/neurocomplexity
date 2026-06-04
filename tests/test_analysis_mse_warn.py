"""Short-series RuntimeWarning for multiscale_entropy (A13).

When the coarse-grained series at any scale has fewer than 10**(m+1) template
windows, sample entropy is unreliable. The package now emits a single
RuntimeWarning per ``multiscale_entropy`` call.
"""
from __future__ import annotations

import warnings as _warnings

import numpy as np
import pandas as pd
import pytest

from neurocomplexity.analysis.mse import multiscale_entropy
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording


def _poisson_rec(duration_s: float, n_units: int = 4, rate_hz: float = 10.0,
                 seed: int = 0):
    """Tiny Poisson recording used to provoke the short-series warning."""
    rng = np.random.default_rng(seed)
    spikes_per_unit = [
        np.sort(rng.uniform(0.0, duration_s, size=rng.poisson(rate_hz * duration_s)))
        for _ in range(n_units)
    ]
    times = np.concatenate(spikes_per_unit) if spikes_per_unit else np.array([])
    owners = np.concatenate([
        np.full(len(s), i, dtype=np.int64) for i, s in enumerate(spikes_per_unit)
    ]) if spikes_per_unit else np.array([], dtype=np.int64)
    order = np.argsort(times, kind="stable")
    return SpikeRecording(
        spike_times=times[order].astype(np.float64),
        unit_ids=owners[order],
        units=pd.DataFrame({"id": np.arange(n_units, dtype=np.int64)}),
        populations={"all": np.ones(n_units, dtype=bool)},
        duration=float(duration_s),
        sampling_rate=None,
        source=ProvenanceRecord.for_memory("test"),
    )


def test_mse_short_series_emits_runtime_warning():
    """``K_eff < 10**(m+1)`` at some scale → one ``RuntimeWarning``."""
    # 10 s recording @ 50 ms bins = 200 bins; coarse-graining at scale 20
    # leaves K_eff = floor(200/20) - m = 8 << 10**3 = 1000.
    rec = _poisson_rec(duration_s=10.0, n_units=4, rate_hz=10.0)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        multiscale_entropy(rec, populations=["all"], bin_size_s=0.05,
                          scale_max=20, m=2)
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "K_eff" in str(w.message)]
    assert msgs, [str(w.message) for w in caught]


def test_mse_long_series_no_short_warning():
    """Big enough K_eff → no short-series warning."""
    # 10 s @ 1 ms = 10_000 bins; at scale 5, K_eff ~ 2000 > 10**3.
    rec = _poisson_rec(duration_s=10.0, n_units=4, rate_hz=10.0)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        multiscale_entropy(rec, populations=["all"], bin_size_s=0.001,
                          scale_max=5, m=2)
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "K_eff" in str(w.message)]
    assert not msgs, msgs


def test_mse_short_series_warns_once():
    """One RuntimeWarning per call even if many scales/pops are short."""
    rec = _poisson_rec(duration_s=10.0, n_units=4, rate_hz=10.0)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        multiscale_entropy(rec, populations=["all"], bin_size_s=0.05,
                          scale_max=20, m=2)
    msgs = [w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "K_eff" in str(w.message)]
    assert len(msgs) == 1, [str(w.message) for w in msgs]
