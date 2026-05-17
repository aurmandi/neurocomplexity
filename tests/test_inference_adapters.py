import numpy as np
import pandas as pd
import pytest
from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.analysis.transfer_entropy import transfer_entropy
from neurocomplexity.analysis.branching import wilting_mr
from neurocomplexity.analysis.dimensionality import dimensionality
from neurocomplexity.inference._adapters import (
    adapter_for, observed_statistic, AdapterError,
)


def _two_pop_rec(seed=0, dur=30.0, rate=20.0):
    rng = np.random.default_rng(seed)
    times, uids = [], []
    for uid in range(6):
        n = rng.poisson(rate * dur)
        t = np.sort(rng.uniform(0, dur, n))
        times.append(t); uids.append(np.full(n, uid, dtype=np.int64))
    st = np.concatenate(times); ui = np.concatenate(uids)
    order = np.argsort(st)
    pops = {"A": np.array([True, True, True, False, False, False]),
            "B": np.array([False, False, False, True, True, True])}
    return SpikeRecording(
        spike_times=st[order], unit_ids=ui[order],
        units=pd.DataFrame({"id": list(range(6))}),
        populations=pops, duration=dur,
        sampling_rate=None, source=None, intervals={},
    )


def test_adapter_recomputes_te_matrix():
    rec = _two_pop_rec()
    te = transfer_entropy(rec, populations=["A", "B"], bin_size_ms=20, delay_bins=1)
    adapter = adapter_for(te)
    val = adapter(rec)
    assert val.shape == te.matrix.shape
    np.testing.assert_allclose(val, te.matrix)


def test_adapter_recomputes_branching():
    rec = _two_pop_rec()
    br = wilting_mr(rec, populations=["A", "B"], bin_size_ms=10, k_max=20)
    adapter = adapter_for(br)
    val = adapter(rec)
    if np.isnan(br.m):
        assert np.isnan(val)
    else:
        assert val == pytest.approx(br.m)


def test_adapter_for_unknown_raises():
    class Bogus: ...
    with pytest.raises(AdapterError):
        adapter_for(Bogus())


def test_observed_statistic_matches_adapter():
    rec = _two_pop_rec()
    te = transfer_entropy(rec, populations=["A", "B"], bin_size_ms=20, delay_bins=1)
    obs = observed_statistic(te)
    np.testing.assert_allclose(obs, adapter_for(te)(rec))


def test_all_results_carry_params():
    rec = _two_pop_rec()
    te = transfer_entropy(rec, populations=["A", "B"], bin_size_ms=20, delay_bins=1)
    br = wilting_mr(rec, populations=["A", "B"], bin_size_ms=10, k_max=20)
    dim = dimensionality(rec, populations=["A", "B"], bin_size_ms=20)
    for r in (te, br, dim):
        assert hasattr(r, "params") and isinstance(r.params, dict) and r.params
