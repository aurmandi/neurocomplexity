import numpy as np
import pandas as pd
import pytest
from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.analysis.transfer_entropy import transfer_entropy
from neurocomplexity.analysis.branching import wilting_mr
from neurocomplexity.inference import SurrogatePool, test as inf_test
from neurocomplexity.inference.null_test import (
    pvalue_from_null, effect_size, fdr_bh,
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


def test_pvalue_one_sided_with_phipson_floor():
    null = np.zeros(99)
    p = pvalue_from_null(1.0, null, alternative="greater")
    assert p == pytest.approx(1.0 / 100.0)


def test_pvalue_two_sided_symmetric():
    null = np.random.default_rng(0).normal(0, 1, 1000)
    p = pvalue_from_null(0.0, null, alternative="two-sided")
    assert 0.4 < p <= 1.0


def test_effect_size_zero_when_observed_equals_mean():
    null = np.array([1.0, 2.0, 3.0])
    assert effect_size(2.0, null) == pytest.approx(0.0)


def test_fdr_bh_monotonic():
    p = np.array([0.001, 0.01, 0.02, 0.5])
    q = fdr_bh(p)
    assert q[0] <= q[1] <= q[2] <= q[3]
    assert (q <= 1.0).all() and (q >= 0.0).all()


def test_pvalue_matrix_input():
    null = np.random.default_rng(0).normal(size=(500, 3, 3))
    observed = np.full((3, 3), 0.5)
    p = pvalue_from_null(observed, null, alternative="greater")
    assert p.shape == (3, 3)


def test_test_returns_inference_result():
    rec = _two_pop_rec()
    te = transfer_entropy(rec, populations=["A", "B"], bin_size_ms=20, delay_bins=1)
    inf = inf_test(te, rec, surrogate="isi_shuffle", n=20, seed=0)
    assert inf.statistic_name == "TE"
    assert inf.null_distribution.shape == (20, 2, 2)
    assert inf.method == "isi_shuffle"
    assert inf.n_resamples == 20
    assert inf.seed == 0
    assert np.asarray(inf.p_value).shape == (2, 2)


def test_test_with_explicit_pool():
    rec = _two_pop_rec()
    te = transfer_entropy(rec, populations=["A", "B"], bin_size_ms=20, delay_bins=1)
    pool = SurrogatePool(rec, surrogate="isi_shuffle", n=10, seed=0)
    inf = inf_test(te, rec, pool=pool)
    assert inf.n_resamples == 10


def test_test_rejects_both_pool_and_surrogate():
    rec = _two_pop_rec()
    te = transfer_entropy(rec, populations=["A", "B"], bin_size_ms=20, delay_bins=1)
    pool = SurrogatePool(rec, surrogate="isi_shuffle", n=5, seed=0)
    with pytest.raises(ValueError):
        inf_test(te, rec, pool=pool, surrogate="isi_shuffle", n=5, seed=0)


def test_test_scalar_result_returns_scalar_p():
    rec = _two_pop_rec()
    br = wilting_mr(rec, populations=["A", "B"], bin_size_ms=10, k_max=20)
    inf = inf_test(br, rec, surrogate="isi_shuffle", n=20, seed=0)
    assert isinstance(inf.p_value, float)
    assert 0.0 < inf.p_value <= 1.0
    assert inf.statistic_name == "m"


def test_test_fdr_on_matrix_output():
    rec = _two_pop_rec()
    te = transfer_entropy(rec, populations=["A", "B"], bin_size_ms=20, delay_bins=1)
    inf = inf_test(te, rec, surrogate="isi_shuffle", n=20, seed=0, fdr=True)
    assert inf.p_value_fdr is not None
    assert np.asarray(inf.p_value_fdr).shape == (2, 2)
