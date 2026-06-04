"""Dtype-promotion guard for analysis._binning (A11).

Above ``T * P > 2**30`` the count matrix uses ``int64`` instead of
``int32`` so that very long Neuropixels-scale recordings cannot silently
overflow a single column.
"""
from __future__ import annotations

import numpy as np

from neurocomplexity.analysis._binning import _counts_bytes, _counts_dtype


def test_counts_dtype_default_int32():
    """Small (T, P) keeps int32 to minimise memory."""
    assert _counts_dtype(1_000, 4) == np.int32
    assert _counts_bytes(1_000, 4) == 4


def test_counts_dtype_promotes_to_int64_when_large():
    """T * P > 2**30 → int64 (int32 max is ~2.1e9)."""
    T = (1 << 30) + 1
    assert _counts_dtype(T, 1) == np.int64
    assert _counts_bytes(T, 1) == 8


def test_counts_dtype_threshold_is_inclusive_lower_bound():
    """Exactly at 2**30 stays int32; only > triggers the promotion."""
    assert _counts_dtype(1 << 30, 1) == np.int32
    assert _counts_dtype((1 << 30) + 1, 1) == np.int64
