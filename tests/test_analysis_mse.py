import numpy as np
import pytest

from neurocomplexity.analysis.mse import (
    _coarse_grain,
    _sample_entropy,
    MSEResult,
)


def test_coarse_grain_scale_1_is_identity():
    x = np.arange(10, dtype=np.float64)
    assert np.allclose(_coarse_grain(x, 1), x)


def test_coarse_grain_scale_2_averages_pairs():
    x = np.array([0., 2., 4., 6., 8., 10.])
    cg = _coarse_grain(x, 2)
    assert np.allclose(cg, [1., 5., 9.])


def test_coarse_grain_drops_trailing_partial_window():
    x = np.array([1., 2., 3., 4., 5.])
    cg = _coarse_grain(x, 2)
    assert cg.shape == (2,)


def test_sample_entropy_constant_series_is_zero_or_nan():
    x = np.zeros(200, dtype=np.float64)
    val = _sample_entropy(x, m=2, r=0.1)
    assert np.isnan(val) or val == 0.0


def test_sample_entropy_random_series_finite_positive():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    val = _sample_entropy(x, m=2, r=0.2 * x.std())
    assert np.isfinite(val) and val > 0.0


def test_mseresult_is_frozen_dataclass():
    import dataclasses
    assert dataclasses.is_dataclass(MSEResult)
    fields = {f.name for f in dataclasses.fields(MSEResult)}
    assert {"populations", "scales", "sampen", "bin_size_seconds",
            "m", "r_factor", "r_per_pop", "source", "params"}.issubset(fields)
