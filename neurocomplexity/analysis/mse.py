"""Multi-Scale Entropy (Costa, Goldberger, Peng 2002).

Coarse-grain a 1-D series at integer scales tau = 1..tau_max and compute the
sample entropy (Richman & Moorman 2000) of each coarse-grained series with a
fixed tolerance r = r_factor * SD(original). One curve per population.

References:
    Costa, Goldberger, Peng. Phys. Rev. Lett. 89 (2002) 068102.
    Richman & Moorman. Am. J. Physiol. Heart Circ. Physiol. 278 (2000) H2039.
"""
from __future__ import annotations

import warnings as _warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

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


def _sample_entropy(x: np.ndarray, m: int, r: float, *,
                    backend: str = "auto") -> float:
    """Richman & Moorman sample entropy with template length ``m`` and tolerance ``r``.

    Returns NaN if either A or B is zero (insufficient matches).

    Parameters
    ----------
    backend
        Pairwise-distance backend:

        * ``"numpy"`` — pure-NumPy ``O(K**2)`` Chebyshev (max-norm) loop.
          Always available, fast for small ``K``.
        * ``"kdtree"`` — :class:`scipy.spatial.cKDTree.query_pairs` with
          ``p=inf``. ``O(K log K)`` for moderate ``r``; the recommended
          path when ``K`` exceeds a few thousand.
        * ``"auto"`` (default) — picks ``"kdtree"`` when ``K > 2000`` and
          ``scipy`` is importable, else ``"numpy"``.

        Both backends count Chebyshev neighbours within ``r`` and produce
        the same SampEn value up to floating-point round-off.
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

    if backend not in {"numpy", "kdtree", "auto"}:
        raise ValueError(
            f"unknown backend={backend!r}; choose 'numpy', 'kdtree', or 'auto'"
        )

    use_kdtree = backend == "kdtree"
    if backend == "auto":
        use_kdtree = K > 2000
        if use_kdtree:
            try:
                from scipy.spatial import cKDTree  # noqa: F401
            except ImportError:  # pragma: no cover - exercised on no-scipy envs
                use_kdtree = False

    def _count_matches_numpy(length: int) -> int:
        windows = sliding_window_view(x, length)[:K]  # take first K templates only
        count = 0
        for i in range(K - 1):
            d = np.max(np.abs(windows[i + 1:] - windows[i]), axis=1)
            count += int(np.count_nonzero(d <= r))
        return count

    def _count_matches_kdtree(length: int) -> int:
        from scipy.spatial import cKDTree
        windows = sliding_window_view(x, length)[:K]
        # cKDTree.query_pairs returns unordered pairs (i < j) within
        # Chebyshev (p=inf) distance r, matching the upper-triangle count
        # of the numpy loop above. Note: query_pairs uses strict "< r" by
        # default; passing eps=0 keeps it exact at the boundary, but
        # SampEn convention (Richman & Moorman 2000) accepts d <= r. To
        # match, we query at r * (1 + 1e-12) so the boundary is included
        # without changing typical results.
        tree = cKDTree(np.ascontiguousarray(windows))
        pairs = tree.query_pairs(r=r * (1.0 + 1e-12), p=float("inf"),
                                 output_type="set")
        return len(pairs)

    counter = _count_matches_kdtree if use_kdtree else _count_matches_numpy
    B = counter(m)
    A = counter(m + 1)
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
                       backend: str = "auto",
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
        Tolerance factor for sample entropy. Default 0.2 follows the
        Pincus (1991) / Richman & Moorman (2000) convention. Costa et al.
        (2002) reported MSE on physiologic time series using
        ``r_factor=0.15``; pass that value explicitly to reproduce their
        choice. Actual tolerance is ``r_factor * SD(original)`` per
        population.
    backend
        Pairwise-distance backend forwarded to :func:`_sample_entropy`:
        ``"numpy"`` (pure-NumPy ``O(K**2)``), ``"kdtree"`` (scipy
        ``cKDTree.query_pairs``, ``O(K log K)`` for moderate ``r``), or
        ``"auto"`` (default — picks ``kdtree`` when ``K > 2000`` and
        scipy is importable).

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
    from neurocomplexity._progress import progress_iter
    from neurocomplexity._warnings import _warn_if_nonstationary, _warn_if_uncurated
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

    if backend not in {"numpy", "kdtree", "auto"}:
        raise ValueError(
            f"unknown backend={backend!r}; choose 'numpy', 'kdtree', or 'auto'"
        )

    params = {"populations": list(populations), "bin_size_s": float(bin_size_s),
              "scale_max": int(scale_max), "m": int(m),
              "r_factor": float(r_factor), "backend": backend}

    counts = bin_spikes(rec, populations, bin_size_s).astype(np.float64)  # (T, P)
    T, P = counts.shape
    scales = np.arange(1, scale_max + 1, dtype=np.int64)
    S = scales.size
    sampen = np.full((P, S), np.nan, dtype=np.float64)
    r_per_pop = np.zeros(P, dtype=np.float64)

    # Sample-entropy reliability rule of thumb (Richman & Moorman 2000;
    # Pincus 1991): each coarse-grained series needs at least 10**(m+1)
    # template windows for the SampEn match counts to stabilise. We emit
    # at most one warning per mse() call to avoid spamming when many scales
    # or many populations fall below the threshold.
    _min_K = 10 ** (m + 1)
    _short_warned = False

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
            K_eff = max(0, cg.size - m)
            if not _short_warned and K_eff < _min_K:
                _warnings.warn(
                    f"multiscale_entropy: coarse-grained series at "
                    f"scale={int(tau)} has K_eff={K_eff} template windows "
                    f"(< 10**(m+1) = {_min_K} required for reliable "
                    f"sample entropy at m={m}). Sample-entropy values at "
                    f"this and larger scales may be noisy or NaN; consider "
                    f"a longer recording, a smaller scale_max, or m=1.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _short_warned = True
            sampen[p, si] = _sample_entropy(cg, m=m, r=r, backend=backend)
            next(it, None)

    return MSEResult(populations=tuple(populations), scales=scales,
                     sampen=sampen, bin_size_seconds=float(bin_size_s),
                     m=int(m), r_factor=float(r_factor),
                     r_per_pop=r_per_pop, source=rec.source, params=params)
