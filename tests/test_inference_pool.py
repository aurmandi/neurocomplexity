import numpy as np
import pandas as pd
import pytest
from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.inference import SurrogatePool


def _make_rec(seed=0):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, 30, 200))
    return SpikeRecording(
        spike_times=times, unit_ids=np.zeros(200, dtype=np.int64),
        units=pd.DataFrame({"id": [0]}),
        populations={"all": np.array([True])},
        duration=30.0, sampling_rate=None, source=None, intervals={},
    )


def test_pool_len():
    rec = _make_rec()
    pool = SurrogatePool(rec, surrogate="isi_shuffle", n=10, seed=0)
    assert len(pool) == 10


def test_pool_indexing_returns_recording():
    rec = _make_rec()
    pool = SurrogatePool(rec, surrogate="isi_shuffle", n=5, seed=0)
    s0 = pool[0]
    assert s0.spike_times.size == rec.spike_times.size
    assert s0.duration == rec.duration


def test_pool_reproducible_across_calls():
    rec = _make_rec()
    p1 = SurrogatePool(rec, surrogate="spike_dither", n=4, seed=42, delta_ms=5.0)
    p2 = SurrogatePool(rec, surrogate="spike_dither", n=4, seed=42, delta_ms=5.0)
    assert np.array_equal(p1[2].spike_times, p2[2].spike_times)


def test_pool_unknown_method_raises():
    rec = _make_rec()
    with pytest.raises(ValueError):
        SurrogatePool(rec, surrogate="bogus", n=2, seed=0)
