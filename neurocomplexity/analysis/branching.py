"""Wilting & Priesemann (2018) multi-step regression branching-ratio estimator.

The estimator is robust under subsampling, which is the regime of every real
recording. Given a binned population activity A_t (T,), it computes the
lagged autocovariance ``r_k = Cov(A_t, A_{t+k}) / Var(A_t)`` for
k = 1..k_max and fits log(r_k) = log(b) + k * log(m), so m = exp(slope).

For a critical branching process m = 1; sub-critical < 1; super-critical > 1.

Reference:
    Wilting & Priesemann, Nature Communications 9 (2018) 2325.
    "Inferring collective dynamical states from widely unobserved systems."
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import linregress

from neurocomplexity.analysis._binning import bin_all_active
from neurocomplexity.core.recording import SpikeRecording


@dataclass(frozen=True)
class BranchingResult:
    """Output of :func:`wilting_mr` (Wilting & Priesemann 2018).

    Attributes
    ----------
    m
        Estimated branching ratio ``exp(slope)``.

        * ``m ≈ 1`` — critical branching process.
        * ``m < 1`` — sub-critical (activity decays).
        * ``m > 1`` — super-critical (activity diverges).

        Robust to subsampling because the slope of ``log(r_k) ~ k * log(m)``
        cancels the subsampling-induced bias on individual ``r_k`` values.
    r_values
        Empirical regression coefficients ``r_k = Cov(A_t, A_{t+k}) / Var(A_t)``
        for ``k = k_min..k_max``.
    k_lags
        The ``k`` values used in the linear fit (``log(r_k)`` vs ``k``).
    r_squared
        Goodness-of-fit of the log-linear regression. Values close to 1
        indicate clean exponential decay; low values suggest the
        branching-process model does not describe the data.
    n_bins
        Number of population activity bins used.
    bin_size_seconds
        Bin size used for the activity series.
    populations
        Population names whose union was binned into the activity series.
    source
        Back-pointer to the source recording's
        :class:`~neurocomplexity.core.provenance.ProvenanceRecord`.
    params
        Verbatim copy of the keyword arguments passed to :func:`wilting_mr`.
    """

    m: float
    r_values: np.ndarray
    k_lags: np.ndarray
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
    """Estimate the branching ratio ``m`` via multi-step regression.

    Bins the union of ``populations`` into a population activity series
    ``A_t`` at ``bin_size_ms`` resolution, computes the lagged regression
    coefficients ``r_k = Cov(A_t, A_{t+k}) / Var(A_t)`` for ``k_min..k_max``,
    and fits ``log(r_k) = log(b) + k * log(m)``. The branching ratio is
    ``m = exp(slope)``.

    Parameters
    ----------
    rec
        Spike recording.
    populations
        Names of populations to include in the activity series. ``None`` →
        all populations.
    bin_size_ms
        Bin size in milliseconds (default 4 ms, matching Wilting & Priesemann
        2018 for hippocampal recordings).
    k_max, k_min
        Inclusive range of lags used in the log-linear fit.

    Returns
    -------
    :class:`BranchingResult`

    Raises
    ------
    ValueError
        If ``k_max <= k_min`` or the recording is shorter than ``k_max + 10``
        bins.

    Notes
    -----
    The regression cancels the bias introduced by subsampling (only a small
    fraction of neurons recorded), which makes ``m̂`` essentially unbiased
    in the regime of real Neuropixels recordings. See the paper for the
    formal argument.

    Emits a :class:`~neurocomplexity._warnings.QualityControlWarning` if the
    recording has not been quality-curated and a
    :class:`~neurocomplexity._warnings.StationarityWarning` if a rate drift
    or heteroskedasticity is detected.
    """
    from neurocomplexity._warnings import _warn_if_nonstationary, _warn_if_uncurated
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
