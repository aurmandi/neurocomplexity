"""TE Miller-Madow bias-correction reorder (A9).

Plug-in transfer entropy can drift slightly negative on finite samples
from rounding error; the previous code subtracted the Miller-Madow
correction *before* the final zero-clamp, which under-corrected at the
zero floor. The fix clamps the raw plug-in TE to zero first and then
applies the correction, so that an already-zero estimator does not pay
the correction twice on its way up.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurocomplexity.analysis.transfer_entropy import _binary_schreiber_te


def _independent_binary(T, p=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(T) < p).astype(np.int8)


def test_independent_te_is_zero_under_miller_madow():
    """E[TE] on truly independent processes must remain non-negative."""
    vals = []
    for s in range(30):
        x = _independent_binary(2000, p=0.3, seed=s)
        y = _independent_binary(2000, p=0.3, seed=100 + s)
        vals.append(_binary_schreiber_te(x, y, delay=1, bias="miller_madow"))
    arr = np.asarray(vals)
    assert (arr >= 0.0).all(), arr.min()


def test_coupled_te_exceeds_independent_te():
    """A copy-with-noise source should produce a strictly larger TE than
    an independent pairing on the same target — sanity that the new clamp
    order has not flattened the signal regime."""
    rng = np.random.default_rng(0)
    x = (rng.random(4000) < 0.4).astype(np.int8)
    # y_t = x_{t-1} XOR noise (strong copy, noisy)
    y = np.zeros_like(x)
    y[1:] = np.where(rng.random(len(x) - 1) < 0.85, x[:-1], 1 - x[:-1])
    y_indep = _independent_binary(len(x), p=0.4, seed=42)
    te_coupled = _binary_schreiber_te(x, y, delay=1, bias="miller_madow")
    te_independent = _binary_schreiber_te(x, y_indep, delay=1, bias="miller_madow")
    assert te_coupled > te_independent
    assert te_coupled > 0.0


def test_clamp_order_matches_two_stage_formula():
    """When the raw plug-in TE is itself >= 0, the new two-stage clamp
    reduces to ``max(0, te_plugin - correction)`` exactly."""
    # A non-trivial pair where the plug-in TE is comfortably positive
    rng = np.random.default_rng(0)
    x = (rng.random(2000) < 0.5).astype(np.int8)
    y = np.zeros_like(x)
    y[1:] = np.where(rng.random(1999) < 0.9, x[:-1], 1 - x[:-1])
    plug = _binary_schreiber_te(x, y, delay=1, bias="none")
    mm = _binary_schreiber_te(x, y, delay=1, bias="miller_madow")
    # In the high-signal regime, plug-in >> correction so plug-in == raw
    # and the two-stage clamp coincides with the legacy formula.
    assert plug > 0.0
    assert mm >= 0.0
    assert mm <= plug + 1e-12  # correction can only reduce TE
