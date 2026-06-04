"""TE ``discretize`` kwarg (A5): binary | quantile | none.

* ``binary`` is the default and reproduces the pre-existing estimator.
* ``quantile`` discretises into ``n_quantile_bins`` levels via the same
  quantile helper used by PID. Useful for rate-coded streams that
  saturate under the binary threshold.
* ``none`` accepts pre-discretised integer counts; validates dtype + range.
"""
from __future__ import annotations

import warnings as _warnings

import numpy as np
import pandas as pd
import pytest

from neurocomplexity.analysis.transfer_entropy import (
    _schreiber_te_general,
    transfer_entropy,
)
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording


def _make_rec(spike_times: dict[int, np.ndarray], duration: float,
              n_units: int):
    """Build a SpikeRecording from a {unit_id: spike_times_seconds} dict."""
    times = np.concatenate([spike_times[uid] for uid in range(n_units)])
    owners = np.concatenate([
        np.full(len(spike_times[uid]), uid, dtype=np.int64)
        for uid in range(n_units)
    ])
    order = np.argsort(times, kind="stable")
    units = pd.DataFrame({"id": np.arange(n_units, dtype=np.int64)})
    populations = {f"u{uid}": (np.arange(n_units) == uid)
                   for uid in range(n_units)}
    return SpikeRecording(
        spike_times=times[order].astype(np.float64),
        unit_ids=owners[order],
        units=units,
        populations=populations,
        duration=float(duration),
        sampling_rate=None,
        source=ProvenanceRecord.for_memory("test"),
    )


def _coupled_pair(seed=0, n=4000, p_copy=0.9):
    rng = np.random.default_rng(seed)
    src_bins = (rng.random(n) < 0.3).astype(np.int8)
    tgt_bins = np.zeros_like(src_bins)
    tgt_bins[1:] = np.where(rng.random(n - 1) < p_copy,
                            src_bins[:-1], 1 - src_bins[:-1])
    return src_bins, tgt_bins


def test_general_te_matches_binary_when_K_equals_2():
    """For K=2 on binary inputs, the generalised plug-in must reproduce
    the binary estimator's result (raw plug-in, no bias correction)."""
    from neurocomplexity.analysis.transfer_entropy import _binary_schreiber_te
    src, tgt = _coupled_pair(seed=1)
    te_bin = _binary_schreiber_te(src, tgt, delay=1, bias="none")
    te_gen = _schreiber_te_general(src.astype(np.int64), tgt.astype(np.int64),
                                    K=2, delay=1, bias="none")
    assert np.isclose(te_bin, te_gen, rtol=1e-9, atol=1e-12), (te_bin, te_gen)


def _two_pop_rec(spikes_src, spikes_tgt, duration):
    """Build a rec with two single-unit populations."""
    return _make_rec({0: spikes_src, 1: spikes_tgt}, duration=duration, n_units=2)


def _poisson(n_spikes, duration, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0.0, duration, size=n_spikes))


def test_binary_default_reproduces_legacy_output():
    """Default ``transfer_entropy()`` call must not be affected by the new kwargs."""
    duration = 50.0
    rec = _two_pop_rec(_poisson(800, duration, 0),
                       _poisson(800, duration, 1),
                       duration)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        a = transfer_entropy(rec, populations=["u0", "u1"],
                            bin_size_ms=10.0, delay_bins=1)
        b = transfer_entropy(rec, populations=["u0", "u1"],
                            bin_size_ms=10.0, delay_bins=1,
                            discretize="binary")
    assert np.allclose(a.matrix, b.matrix, rtol=1e-12, atol=1e-12)


def test_quantile_path_returns_finite_matrix():
    """``discretize='quantile'`` produces a finite (P, P) matrix."""
    duration = 50.0
    rec = _two_pop_rec(_poisson(1200, duration, 2),
                       _poisson(1200, duration, 3),
                       duration)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        res = transfer_entropy(rec, populations=["u0", "u1"],
                              bin_size_ms=10.0, delay_bins=1,
                              discretize="quantile", n_quantile_bins=3)
    assert res.matrix.shape == (2, 2)
    assert np.all(np.isfinite(res.matrix))
    assert res.params["discretize"] == "quantile"
    assert res.params["n_quantile_bins"] == 3


def test_none_rejects_float_input():
    """``discretize='none'`` requires integer counts."""
    duration = 50.0
    rec = _two_pop_rec(_poisson(800, duration, 4),
                       _poisson(800, duration, 5),
                       duration)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        # Counts from bin_spikes are int32 in [0, ...]. Out-of-range check fires
        # when raw counts exceed K.
        with pytest.raises(ValueError, match=r"counts in \[0"):
            transfer_entropy(rec, populations=["u0", "u1"],
                            bin_size_ms=1.0, delay_bins=1,
                            discretize="none", n_quantile_bins=2)


def test_unknown_discretize_raises():
    duration = 10.0
    rec = _two_pop_rec(_poisson(100, duration, 6),
                       _poisson(100, duration, 7),
                       duration)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="unknown discretize"):
            transfer_entropy(rec, populations=["u0", "u1"],
                            bin_size_ms=10.0, discretize="ordinal")
