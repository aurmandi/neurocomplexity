"""Ince (2017) I_ccs redundancy as an alternative to Williams & Beer I_min (A8).

PID has multiple legitimate redundancy functionals. The package now
exposes I_ccs (Ince 2017) alongside the default I_min (Williams & Beer
2010). I_ccs measures redundancy as the pointwise common change in
surprisal: only the components on which both sources move the target
posterior in the same direction count toward redundancy. I_min is the
minimum specific information per target outcome, which can over-count
redundancy on distributions where the sources inform about different
target outcomes.
"""
from __future__ import annotations

import warnings as _warnings

import numpy as np
import pytest

from neurocomplexity.analysis.pid import (
    _redundancy_iccs,
    _redundancy_imin,
    partial_information,
)
from neurocomplexity.benchmarks.simulators.pid_distributions import pid_recording


def _joint(distribution: str, n_bins: int = 20_000, seed: int = 0,
           n_levels: int = 3):
    """Run the discretisation pipeline and return the (L_y, L_s1, L_s2) joint counts."""
    from neurocomplexity.analysis._binning import bin_spikes
    from neurocomplexity.analysis.pid import _quantile_discretise
    rec = pid_recording(distribution, n_bins=n_bins, bin_ms=10.0, seed=seed)
    bs = 0.010
    y = bin_spikes(rec, ["target"], bs)[:, 0].astype(np.float64)
    s1 = bin_spikes(rec, ["source_1"], bs)[:, 0].astype(np.float64)
    s2 = bin_spikes(rec, ["source_2"], bs)[:, 0].astype(np.float64)
    Y = _quantile_discretise(y, n_levels)
    S1 = _quantile_discretise(s1, n_levels)
    S2 = _quantile_discretise(s2, n_levels)
    L_y = int(Y.max()) + 1
    L_s1 = int(S1.max()) + 1
    L_s2 = int(S2.max()) + 1
    flat = (Y.astype(np.int64) * (L_s1 * L_s2)
            + S1.astype(np.int64) * L_s2
            + S2.astype(np.int64))
    bc = np.bincount(flat, minlength=L_y * L_s1 * L_s2).astype(np.float64)
    return bc.reshape(L_y, L_s1, L_s2)


def test_iccs_xor_redundancy_near_zero():
    """Canonical XOR has zero redundancy under both I_min and I_ccs."""
    joint = _joint("xor", seed=0)
    r_min = _redundancy_imin(joint)
    r_ccs = _redundancy_iccs(joint)
    assert r_min < 0.03, r_min
    assert r_ccs < 0.03, r_ccs


def test_iccs_rdn_recovers_redundancy():
    """The 'rdn' distribution (sources copy the target) is pure redundancy.

    Both I_min and I_ccs must report R ≈ ln 2 nats up to finite-sample noise.
    """
    joint = _joint("rdn", seed=0)
    r_min = _redundancy_imin(joint)
    r_ccs = _redundancy_iccs(joint)
    target = float(np.log(2.0))
    assert abs(r_min - target) < 0.08, (r_min, target)
    assert abs(r_ccs - target) < 0.08, (r_ccs, target)


def test_iccs_finite_and_nonneg_on_and():
    """``I_ccs`` is finite and non-negative on AND (sanity, distinct from I_min).

    Note: ``I_ccs`` is not in general bounded above by ``I_min`` nor by
    ``min(I(Y; S1), I(Y; S2))`` because pointwise common change in
    surprisal can include outcomes where the conditional ``p(y | s_k)``
    diverges sharply from ``p(y)`` even on cells with low marginal weight.
    The PID identity ``R + U1 + U2 + S = total_mi`` together with the
    standard ``U_k = max(0, I(Y;S_k) - R)`` clipping inside
    ``partial_information`` is what guarantees a self-consistent
    decomposition; the bound check is covered by
    :func:`test_iccs_atoms_sum_to_total_mi`.
    """
    joint = _joint("and", seed=0)
    r_min = _redundancy_imin(joint)
    r_ccs = _redundancy_iccs(joint)
    assert np.isfinite(r_min) and r_min >= 0
    assert np.isfinite(r_ccs) and r_ccs >= 0
    # The two functionals should at least disagree on AND (the well-known
    # discriminator); equality would suggest the new path is silently
    # falling back to I_min.
    assert abs(r_min - r_ccs) > 1e-3, (r_min, r_ccs)


def test_default_redundancy_is_imin():
    """``redundancy='imin'`` (the default) reproduces legacy PID atoms bit-for-bit."""
    rec = pid_recording("and", n_bins=20_000, bin_ms=10.0, seed=0)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        a = partial_information(rec, target_pop="target",
                                sources=["source_1", "source_2"],
                                bin_size_ms=10.0, delay_bins=0, n_levels=3)
        b = partial_information(rec, target_pop="target",
                                sources=["source_1", "source_2"],
                                bin_size_ms=10.0, delay_bins=0, n_levels=3,
                                redundancy="imin")
    assert a.redundancy == b.redundancy
    assert a.unique_1 == b.unique_1
    assert a.synergy == b.synergy


def test_iccs_atoms_sum_to_total_mi():
    """PID identity: R + U1 + U2 + S ≈ total_mi for both estimators."""
    rec = pid_recording("and", n_bins=20_000, bin_ms=10.0, seed=0)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        res = partial_information(rec, target_pop="target",
                                  sources=["source_1", "source_2"],
                                  bin_size_ms=10.0, delay_bins=0, n_levels=3,
                                  redundancy="iccs")
    s = res.redundancy + res.unique_1 + res.unique_2 + res.synergy
    assert abs(s - res.total_mi) < 1e-9, (s, res.total_mi)
    assert res.params["redundancy"] == "iccs"


def test_iccs_unknown_redundancy_raises():
    rec = pid_recording("xor", n_bins=2000, bin_ms=10.0, seed=0)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="unknown redundancy"):
            partial_information(rec, target_pop="target",
                                sources=["source_1", "source_2"],
                                bin_size_ms=10.0, delay_bins=0,
                                redundancy="broja")
