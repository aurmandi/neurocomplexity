"""cKDTree backend for sample entropy (A4).

The default ``backend='auto'`` switches to scipy's ``cKDTree.query_pairs``
when ``K > 2000``. The numerical result must agree with the legacy NumPy
loop to round-off precision.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurocomplexity.analysis.mse import _sample_entropy


def _ar1(n: int, phi: float = 0.7, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    x = np.empty(n, dtype=np.float64)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


@pytest.mark.parametrize("K", [300, 800])
def test_kdtree_matches_numpy(K):
    """Two backends agree on AR(1) series at modest K (cheap to verify)."""
    x = _ar1(K + 2, phi=0.5, seed=0)
    r = 0.2 * x.std()
    s_np = _sample_entropy(x, m=2, r=r, backend="numpy")
    s_kd = _sample_entropy(x, m=2, r=r, backend="kdtree")
    assert np.isclose(s_np, s_kd, rtol=1e-9, atol=1e-9), (s_np, s_kd)


def test_auto_picks_numpy_below_threshold():
    """Below K=2000 ``auto`` matches the numpy path bit-for-bit."""
    x = _ar1(500, phi=0.5, seed=1)
    r = 0.2 * x.std()
    a = _sample_entropy(x, m=2, r=r, backend="auto")
    b = _sample_entropy(x, m=2, r=r, backend="numpy")
    assert a == b


def test_auto_picks_kdtree_above_threshold():
    """Above K=2000 ``auto`` matches the kdtree path. Uses K=2100 to stay
    below ~16 MB peak for the numpy-loop comparison (3000 was OOMing on
    laptops with tight RAM budgets)."""
    x = _ar1(2100, phi=0.5, seed=2)
    r = 0.2 * x.std()
    a = _sample_entropy(x, m=2, r=r, backend="auto")
    b = _sample_entropy(x, m=2, r=r, backend="kdtree")
    assert a == b


def test_unknown_backend_raises():
    x = _ar1(500, phi=0.5, seed=3)
    with pytest.raises(ValueError, match="unknown backend"):
        _sample_entropy(x, m=2, r=0.1, backend="annoy")
