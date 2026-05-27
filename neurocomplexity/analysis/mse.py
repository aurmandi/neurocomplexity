"""Multi-Scale Entropy (Costa, Goldberger, Peng 2002).

Coarse-grain a 1-D series at integer scales tau = 1..tau_max and compute the
sample entropy (Richman & Moorman 2000) of each coarse-grained series with a
fixed tolerance r = r_factor * SD(original). One curve per population.

References:
    Costa, Goldberger, Peng. Phys. Rev. Lett. 89 (2002) 068102.
    Richman & Moorman. Am. J. Physiol. Heart Circ. Physiol. 278 (2000) H2039.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MSEResult:
    """Output of :func:`multiscale_entropy` (Costa, Goldberger, Peng 2002).

    Attributes
    ----------
    populations
        Population names (one per row of ``sampen``).
    scales
        ``int64`` array of coarse-graining scales ``tau = 1..scale_max``.
    sampen
        Sample entropy at each scale, shape ``(n_pops, n_scales)``. ``NaN``
        where the coarse-grained series is too short or matches collapse.
    bin_size_seconds
        Bin size used to build the population-rate series.
    m
        Template length used in :func:`_sample_entropy` (default 2).
    r_factor
        Tolerance factor; the actual tolerance is ``r = r_factor * SD(series)``
        per population.
    r_per_pop
        Per-population tolerance ``r`` (in count units, ``float64``).
    source
        Provenance back-pointer.
    params
        Verbatim copy of the keyword arguments passed to
        :func:`multiscale_entropy`.

    See Also
    --------
    lmc_complexity : LMC statistical complexity — different question; see
        ``docs/complexity_measures.md``.
    """

    populations: tuple[str, ...]
    scales: np.ndarray
    sampen: np.ndarray
    bin_size_seconds: float
    m: int
    r_factor: float
    r_per_pop: np.ndarray
    source: object
    params: dict = field(default_factory=dict)


def _coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    """Costa coarse-graining: non-overlapping mean of ``scale`` consecutive samples."""
    if scale < 1:
        raise ValueError("scale must be >= 1")
    x = np.asarray(x, dtype=np.float64)
    n = x.size // scale
    if n == 0:
        return np.empty(0, dtype=np.float64)
    return x[:n * scale].reshape(n, scale).mean(axis=1)


def _sample_entropy(x: np.ndarray, m: int, r: float) -> float:
    """Richman & Moorman sample entropy with template length ``m`` and tolerance ``r``.

    Returns NaN if either A or B is zero (insufficient matches).
    """
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    if N < m + 2:
        return float("nan")
    if r <= 0:
        return float("nan")

    # Use common range K = N - m for both lengths (Richman & Moorman 2000).
    K = N - m
    if K < 2:
        return float("nan")
    from numpy.lib.stride_tricks import sliding_window_view

    def _count_matches(length: int) -> int:
        windows = sliding_window_view(x, length)[:K]  # take first K templates only
        count = 0
        for i in range(K - 1):
            d = np.max(np.abs(windows[i + 1:] - windows[i]), axis=1)
            count += int(np.count_nonzero(d <= r))
        return count

    B = _count_matches(m)
    A = _count_matches(m + 1)
    if B == 0 or A == 0:
        return float("nan")
    return float(-np.log(A / B))


def multiscale_entropy(rec,
                       populations: Sequence[str] | None = None,
                       *,
                       bin_size_s: float = 0.05,
                       scale_max: int = 20,
                       m: int = 2,
                       r_factor: float = 0.2,
                       ) -> MSEResult:
    """Multiscale entropy (Costa 2002) of per-population count series.

    For each population:

    1. Bin spikes at ``bin_size_s``.
    2. For each scale ``tau = 1..scale_max``, coarse-grain the series by
       averaging non-overlapping blocks of ``tau`` consecutive samples.
    3. Compute the sample entropy (Richman & Moorman 2000) of the
       coarse-grained series with template length ``m`` and tolerance
       ``r = r_factor * SD(original)``.

    A signal with long-range temporal structure has sample entropy that
    stays high at coarse scales; an uncorrelated signal (e.g. Poisson)
    has sample entropy that falls toward zero with ``tau``.

    Parameters
    ----------
    rec
        Spike recording.
    populations
        Names of populations. ``None`` → all populations.
    bin_size_s
        Bin size in seconds for the population-rate series.
    scale_max
        Maximum coarse-graining scale. Must be ``>= 2``.
    m
        Sample-entropy template length (default 2).
    r_factor
        Tolerance factor for sample entropy (default 0.2 — Costa 2002).
        Actual tolerance is ``r_factor * SD(original)`` per population.

    Returns
    -------
    :class:`MSEResult`

    Raises
    ------
    ValueError
        If ``bin_size_s <= 0``, ``scale_max < 2``, ``m < 1`` or
        ``r_factor <= 0``.

    Notes
    -----
    Rule of thumb: each coarse-grained series must contain at least
    ``10^(m+1)`` samples for sample entropy to be reliable. At ``scale_max``,
    the coarse-grained length is ``floor(N / scale_max)`` — verify this is
    large enough for your recording.

    Emits :class:`~neurocomplexity._warnings.QualityControlWarning` if the
    recording is uncurated and
    :class:`~neurocomplexity._warnings.StationarityWarning` if a rate drift
    is detected.

    References
    ----------
    * Costa M, Goldberger AL, Peng C-K (2002). *Multiscale entropy analysis
      of complex physiologic time series.* Phys Rev Lett 89:068102.
    * Richman JS, Moorman JR (2000). *Physiological time-series analysis
      using approximate entropy and sample entropy.* Am J Physiol Heart
      Circ Physiol 278:H2039.
    """
    from neurocomplexity._warnings import _warn_if_uncurated, _warn_if_nonstationary
    from neurocomplexity._progress import progress_iter
    from neurocomplexity.analysis._binning import bin_spikes

    _warn_if_uncurated(rec, "multiscale_entropy")
    _warn_if_nonstationary(rec, "multiscale_entropy")
    if bin_size_s <= 0:
        raise ValueError("bin_size_s must be > 0")
    if scale_max < 2:
        raise ValueError("scale_max must be >= 2")
    if m < 1:
        raise ValueError("m must be >= 1")
    if r_factor <= 0:
        raise ValueError("r_factor must be > 0")

    if populations is None:
        populations = list(rec.populations.keys())
    populations = list(populations)

    params = {"populations": list(populations), "bin_size_s": float(bin_size_s),
              "scale_max": int(scale_max), "m": int(m),
              "r_factor": float(r_factor)}

    counts = bin_spikes(rec, populations, bin_size_s).astype(np.float64)  # (T, P)
    T, P = counts.shape
    scales = np.arange(1, scale_max + 1, dtype=np.int64)
    S = scales.size
    sampen = np.full((P, S), np.nan, dtype=np.float64)
    r_per_pop = np.zeros(P, dtype=np.float64)

    total = P * S
    it = iter(progress_iter(range(total), total=total, desc="mse"))
    for p in range(P):
        series = counts[:, p]
        r = r_factor * float(series.std(ddof=0))
        r_per_pop[p] = r
        if r <= 0:
            for _ in scales:
                next(it, None)
            continue
        for si, tau in enumerate(scales):
            cg = _coarse_grain(series, int(tau))
            sampen[p, si] = _sample_entropy(cg, m=m, r=r)
            next(it, None)

    return MSEResult(populations=tuple(populations), scales=scales,
                     sampen=sampen, bin_size_seconds=float(bin_size_s),
                     m=int(m), r_factor=float(r_factor),
                     r_per_pop=r_per_pop, source=rec.source, params=params)
