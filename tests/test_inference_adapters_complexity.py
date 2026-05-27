import numpy as np
import pytest

from neurocomplexity.analysis.complexity import lmc_complexity, LMCResult
from neurocomplexity.inference._adapters import adapter_for, observed_statistic
from tests.test_analysis_complexity import _poisson_rec


def test_lmc_adapter_returns_c_per_pop_vector():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    result = lmc_complexity(rec, kind="population", bin_size_s=0.05)
    fn = adapter_for(result)
    stat = fn(rec)
    assert isinstance(stat, np.ndarray)
    assert stat.shape == result.C_per_pop.shape
    assert np.allclose(stat, result.C_per_pop)


def test_lmc_observed_statistic_matches_adapter():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    result = lmc_complexity(rec, kind="population", bin_size_s=0.05)
    fn = adapter_for(result)
    assert np.allclose(observed_statistic(result), fn(rec))


def test_lmc_adapter_works_for_both_mode():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    result = lmc_complexity(rec, kind="both")
    fn = adapter_for(result)
    stat = fn(rec)
    assert stat.shape == result.C_per_pop.shape
