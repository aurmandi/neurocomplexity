import numpy as np
import pandas as pd
import pytest
from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.analysis.criticality import criticality
from neurocomplexity.analysis.branching import wilting_mr
from neurocomplexity.analysis.dimensionality import dimensionality
from neurocomplexity.inference import bootstrap as inf_bootstrap
from neurocomplexity.inference.bootstrap import (
    bootstrap_avalanche_exponents, bootstrap_branching_ratio,
)


def _rec_for_avalanches(seed=0):
    rng = np.random.default_rng(seed)
    times, uids = [], []
    for uid in range(40):
        n = rng.poisson(15 * 60)
        t = np.sort(rng.uniform(0, 60, n))
        times.append(t); uids.append(np.full(n, uid, dtype=np.int64))
    st = np.concatenate(times); ui = np.concatenate(uids)
    order = np.argsort(st)
    return SpikeRecording(
        spike_times=st[order], unit_ids=ui[order],
        units=pd.DataFrame({"id": list(range(40))}),
        populations={"all": np.ones(40, dtype=bool)},
        duration=60.0, sampling_rate=None, source=None, intervals={},
    )


def test_bootstrap_avalanche_returns_distribution():
    rec = _rec_for_avalanches()
    result = criticality(rec, bin_size_ms=(4, 8))
    inf = bootstrap_avalanche_exponents(result, rec, n=20, seed=0)
    assert inf.bootstrap_distribution.shape == (20, 2)
    assert np.asarray(inf.ci_lower).shape == (2,)
    assert np.asarray(inf.ci_upper).shape == (2,)
    assert inf.ci_level == 0.95


def test_bootstrap_avalanche_reproducible():
    rec = _rec_for_avalanches()
    result = criticality(rec, bin_size_ms=(4, 8))
    a = bootstrap_avalanche_exponents(result, rec, n=10, seed=42)
    b = bootstrap_avalanche_exponents(result, rec, n=10, seed=42)
    assert np.allclose(a.bootstrap_distribution, b.bootstrap_distribution,
                       equal_nan=True)


def test_bootstrap_branching_ratio_shape():
    rec = _rec_for_avalanches()
    r = wilting_mr(rec, bin_size_ms=4, k_max=30)
    inf = bootstrap_branching_ratio(r, rec, n=15, seed=0, block_seconds=5.0)
    assert inf.bootstrap_distribution.shape == (15,)
    assert isinstance(inf.ci_lower, float)
    assert isinstance(inf.ci_upper, float)


def test_bootstrap_branching_ratio_reproducible():
    rec = _rec_for_avalanches()
    r = wilting_mr(rec, bin_size_ms=4, k_max=30)
    a = bootstrap_branching_ratio(r, rec, n=10, seed=7, block_seconds=5.0)
    b = bootstrap_branching_ratio(r, rec, n=10, seed=7, block_seconds=5.0)
    assert np.allclose(a.bootstrap_distribution, b.bootstrap_distribution,
                       equal_nan=True)


def test_bootstrap_dispatch_pr():
    rec = _rec_for_avalanches()
    r = dimensionality(rec, populations=["all"], bin_size_ms=20)
    inf = inf_bootstrap(r, rec, n=10, seed=0, block_seconds=5.0)
    assert inf.statistic_name == "PR"
    assert inf.bootstrap_distribution.shape == (10,)


def test_bootstrap_warns_when_block_too_large():
    """Tier 2.9 — block bootstrap must warn when too few unique blocks.

    Guard against the silent under-coverage mode flagged by Phase 4
    Reviewer B P0-3. A 60s recording with 20s blocks → 3 unique blocks
    triggers the warning.
    """
    import warnings
    rec = _rec_for_avalanches()
    r = wilting_mr(rec, bin_size_ms=4, k_max=30)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bootstrap_branching_ratio(r, rec, n=5, seed=0, block_seconds=20.0)
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)]
    assert any("unique block" in m for m in msgs), msgs


def test_bootstrap_no_warning_with_reasonable_blocks():
    """No spurious warning when block size is comfortably small."""
    import warnings
    rec = _rec_for_avalanches()
    r = wilting_mr(rec, bin_size_ms=4, k_max=30)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bootstrap_branching_ratio(r, rec, n=5, seed=0, block_seconds=2.0)
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)
            and "unique block" in str(w.message)]
    assert msgs == [], msgs


def test_bootstrap_dispatch_unknown_raises():
    class Foo: ...
    with pytest.raises(TypeError):
        inf_bootstrap(Foo(), None, n=5, seed=0)
