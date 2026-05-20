"""Wilting & Priesemann (2018) multi-step regression branching-ratio estimator.

The estimator is robust under subsampling, which is the regime of every real
recording. Given a binned population activity A_t (T,), it computes
    r_k = Cov(A_t, A_{t+k}) / Var(A_t)
for k = 1..k_max and fits log(r_k) = log(b) + k * log(m), so m = exp(slope).

For a critical branching process m = 1; sub-critical < 1; super-critical > 1.

Reference:
    Wilting & Priesemann, Nature Communications 9 (2018) 2325.
    "Inferring collective dynamical states from widely unobserved systems."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import linregress

from neurocomplexity.analysis._binning import bin_all_active
from neurocomplexity.core.recording import SpikeRecording


@dataclass(frozen=True)
class BranchingResult:
    m: float                     # branching ratio
    r_values: np.ndarray         # r_k for k = 1..k_max
    k_lags: np.ndarray           # k indices used in the fit
    r_squared: float
    n_bins: int
    bin_size_seconds: float
    populations: tuple[str, ...]
    source: object
    params: dict = field(default_factory=dict)


def wilting_mr(rec: SpikeRecording,
               populations: Sequence[str] | None = None,
               bin_size_ms: float = 4.0,
               k_max: int = 50,
               k_min: int = 1,
               ) -> BranchingResult:
    """Multi-step regression branching ratio."""
    from neurocomplexity._warnings import _warn_if_uncurated, _warn_if_nonstationary
    _warn_if_uncurated(rec, "branching_ratio")
    _warn_if_nonstationary(rec, "branching_ratio")
    if populations is None:
        populations = list(rec.populations.keys())
    if k_max <= k_min:
        raise ValueError("need k_max > k_min")

    bs = float(bin_size_ms) / 1000.0
    _params = {"populations": list(populations), "bin_size_ms": float(bin_size_ms),
               "k_max": int(k_max), "k_min": int(k_min)}
    A = bin_all_active(rec, populations, bs).astype(np.float64)
    T = A.size
    if T < k_max + 10:
        raise ValueError(f"only {T} bins; need > k_max+10 = {k_max+10}")

    mean = A.mean()
    var = A.var()
    if var == 0:
        return BranchingResult(m=float("nan"), r_values=np.array([]),
                                k_lags=np.array([]), r_squared=float("nan"),
                                n_bins=T, bin_size_seconds=bs,
                                populations=tuple(populations),
                                source=rec.source, params=_params)

    ks = np.arange(k_min, k_max + 1)
    rks = np.empty_like(ks, dtype=np.float64)
    for i, k in enumerate(ks):
        cov_k = np.mean((A[:-k] - mean) * (A[k:] - mean))
        rks[i] = cov_k / var

    # fit log(r_k) = log(b) + k * log(m); drop non-positive r_k.
    nz = rks > 0
    if nz.sum() < 3:
        return BranchingResult(m=float("nan"), r_values=rks, k_lags=ks,
                                r_squared=float("nan"), n_bins=T,
                                bin_size_seconds=bs,
                                populations=tuple(populations),
                                source=rec.source, params=_params)
    slope, _, r_val, _, _ = linregress(ks[nz], np.log(rks[nz]))
    m = float(np.exp(slope))
    return BranchingResult(m=m, r_values=rks, k_lags=ks,
                            r_squared=float(r_val ** 2),
                            n_bins=T, bin_size_seconds=bs,
                            populations=tuple(populations),
                            source=rec.source, params=_params)


def branching_ratio(rec: SpikeRecording,
                    populations: Sequence[str] | None = None,
                    bin_size: float = 0.004,
                    k_max: int = 50,
                    k_min: int = 1,
                    ) -> BranchingResult:
    """Multi-step regression branching ratio (``bin_size`` in seconds).

    Thin alias for :func:`wilting_mr` that takes seconds instead of milliseconds.
    """
    return wilting_mr(rec, populations=populations,
                      bin_size_ms=float(bin_size) * 1000.0,
                      k_max=k_max, k_min=k_min)
