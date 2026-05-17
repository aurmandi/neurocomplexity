"""Tests for from_spikeinterface (gated on the [spikeinterface] extra)."""
from __future__ import annotations

import sys

import numpy as np
import pytest


def test_from_spikeinterface_roundtrip():
    si = pytest.importorskip("spikeinterface")
    from spikeinterface.core import NumpySorting

    import neurocomplexity as nc

    sample_rate = 30000.0
    trains = {
        0: np.array([0.1, 0.5, 1.2]),
        1: np.array([0.2, 0.9]),
    }
    spike_times_samples = {
        uid: (st * sample_rate).astype(np.int64) for uid, st in trains.items()
    }
    sorting = NumpySorting.from_unit_dict(
        [spike_times_samples], sampling_frequency=sample_rate,
    )

    rec = nc.io.from_spikeinterface(sorting)

    assert rec.n_units == 2
    assert rec.n_spikes == sum(len(v) for v in trains.values())
    assert rec.sampling_rate == sample_rate
    for uid, st in trains.items():
        recovered = np.sort(rec.spike_times[rec.unit_ids == uid])
        np.testing.assert_allclose(recovered, st, atol=1.0 / sample_rate)
    assert rec.source.source_format == "spikeinterface"


def test_from_spikeinterface_missing_pkg_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "spikeinterface", None)
    sys.modules.pop("neurocomplexity.io.spikeinterface", None)
    from neurocomplexity.io import spikeinterface as adapter
    with pytest.raises(ImportError, match=r"neurocomplexity\[spikeinterface\]"):
        adapter.from_spikeinterface(object())
