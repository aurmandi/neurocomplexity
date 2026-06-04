"""Avalanche criticality analysis.

Definitions (Sethna 2001 crackling-noise framework; Friedman 2012; Fontenele
2019; Beggs & Plenz 2003):

- size S = total spike count inside the avalanche burst
- lifetime T = (# consecutive nonzero bins) * bin_size_seconds
- P(S) ~ S^(-alpha_s)
- P(T) ~ T^(-alpha_t) — fit DIRECTLY from lifetime histogram
- <S>(T) ~ T^gamma_fit — empirical scaling exponent (regression)
- gamma_predicted = (alpha_t - 1) / (alpha_s - 1)
- At criticality: gamma_fit ≈ gamma_predicted
History: an earlier version of this module estimated alpha_t as 1/slope of
the log_T-vs-log_S regression. That quantity is gamma_fit, NOT alpha_t.
Fixed: alpha_t now comes from a direct log-spaced histogram fit of the
lifetime distribution; the regression value is preserved as gamma_fit so
the Sethna consistency test can still be performed.
"""
from __future__ import annotations

import warnings as _warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import linregress

from neurocomplexity.analysis._binning import bin_all_active
from neurocomplexity.core.recording import SpikeRecording


@dataclass(frozen=True)
class CriticalityResult:
    """Output of :func:`criticality`.

    Attributes
    ----------
    alpha_s
        Size-distribution exponent ``P(S) ~ S^(-alpha_s)``. At criticality ≈ 1.5.
    alpha_t
        Lifetime-distribution exponent ``P(T) ~ T^(-alpha_t)``, fit directly
        from a log-spaced histogram. At criticality ≈ 2.0. Distinct from
        ``gamma_fit`` (see below).
    optimal_bin
        Bin size (milliseconds) selected by maximising R² of the size-vs-lifetime scaling.
    branching
        Naive ``<A_{t+1} / A_t>`` ratio across non-empty bins. **Deprecated
        for inference** — the naive ratio is biased downward by sub-sampling
        and biased upward by external drive (Wilting & Priesemann 2018).
        Use :func:`~neurocomplexity.analysis.wilting_mr` for any quantitative
        claim about the branching ratio; this field is retained for
        backwards-compatible diagnostic comparison and emits a
        :class:`DeprecationWarning` when populated.
    sizes
        Per-avalanche spike counts (at ``optimal_bin``).
    lifetimes
        Per-avalanche durations in seconds (at ``optimal_bin``).
    r_squared
        R² of the ``<S>(T) ~ T^gamma_fit`` log-linear regression.
    populations
        Populations whose union was binned.
    source
        Provenance back-pointer.
    params
        Keyword arguments passed to :func:`criticality`.
    gamma_fit
        Empirical scaling exponent from ``log <S>(T) = const + gamma_fit · log T``.
        Earlier versions of this module reported this as ``alpha_t`` — that was
        a bug. At criticality, ``gamma_fit ≈ gamma_predicted``.
    gamma_predicted
        Theoretical exponent ``(alpha_t − 1) / (alpha_s − 1)`` (Sethna 2001).
        Sethna consistency check: ``gamma_fit ≈ gamma_predicted``.
    fits
        Per-bin fit table when :func:`criticality` was called with a sequence
        of ``bin_size`` values. Each entry has ``bin_seconds``, ``alpha_s``,
        ``alpha_t``, ``gamma_fit``, ``r_squared``, ``n_avalanches``. Empty for
        single-bin calls.
    """

    alpha_s: float
    alpha_t: float
    optimal_bin: float
    branching: float
    sizes: np.ndarray
    lifetimes: np.ndarray
    r_squared: float
    populations: tuple[str, ...]
    source: object
    params: dict = field(default_factory=dict)
    gamma_fit: float = float("nan")
    gamma_predicted: float = float("nan")
    fits: tuple = ()


def extract_avalanches(counts_1d: np.ndarray, bin_size: float):
    """Extract neuronal avalanches from a 1-D binned count series.

    An avalanche is a maximal run of consecutive non-empty bins.

    Parameters
    ----------
    counts_1d
        Binned spike-count vector for the population (integer counts).
    bin_size
        Bin width in seconds, used to convert lifetimes from bin index to
        seconds.

    Returns
    -------
    sizes
        1-D ``int64`` array of total spike counts per avalanche.
    lifetimes
        1-D ``float64`` array of avalanche lifetimes in **seconds**.
    """
    if counts_1d.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    binary = (counts_1d > 0).astype(np.int8)
    edges = np.diff(np.concatenate(([0], binary, [0])))
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]
    sizes = np.array([int(counts_1d[s:e].sum()) for s, e in zip(starts, stops)],
                     dtype=np.int64)
    lifetimes = (stops - starts) * bin_size
    return sizes, lifetimes


def _power_law(x, a, b):
    return b * (x ** a)


def fit_alpha(data: np.ndarray, xmin: int = 1) -> float:
    """Fit a power-law exponent ``alpha`` such that ``P(x) ~ x^{-alpha}``.

    .. versionchanged:: 1.1.0
        Now the **Clauset–Shalizi–Newman (2009) discrete maximum-likelihood
        estimator** rather than a log-log histogram regression. The MLE is
        the recommended estimator for heavy-tailed count data (avalanche
        sizes, lifetimes): the histogram-slope method is biased and its
        variance depends on the arbitrary binning. The previous
        implementation is preserved verbatim as :func:`fit_alpha_loglog`
        for reproducing pre-1.1.0 numbers.

    Uses the discrete MLE (Clauset et al. 2009, eq. 3.7 approximation)::

        alpha_hat = 1 + n / sum_i ln( x_i / (xmin - 0.5) )

    over the observations ``x_i >= xmin``. ``xmin`` is treated as given
    (the lower cutoff of the scaling regime), not itself optimised.

    Returns ``nan`` if fewer than 5 observations satisfy ``x >= xmin`` or
    the log-sum is non-positive.
    """
    data = np.asarray(data, dtype=float)
    data = data[data >= xmin]
    n = len(data)
    if n < 5:
        return float("nan")
    if float(data.max()) <= xmin:
        return float("nan")
    # Discrete MLE with the continuous-approximation offset (xmin - 0.5),
    # which Clauset et al. show removes most of the small-sample bias of the
    # naive 1 + n / sum ln(x/xmin) form on integer data.
    denom = float(np.sum(np.log(data / (xmin - 0.5))))
    if denom <= 0:
        return float("nan")
    return 1.0 + n / denom


def fit_alpha_loglog(data: np.ndarray, xmin: int = 1) -> float:
    """Legacy log-log histogram power-law fit (pre-1.1.0 ``fit_alpha``).

    Uses log-spaced histogram binning and normalises by bin width so the
    log-log slope recovers the density exponent. Retained for back-compat
    and for reproducing numbers published with versions < 1.1.0. New code
    should use the MLE :func:`fit_alpha`.
    """
    data = np.asarray(data, dtype=float)
    data = data[data >= xmin]
    if len(data) < 5:
        return float("nan")
    x_max = float(data.max())
    if x_max <= xmin:
        return float("nan")
    bins = np.logspace(np.log10(max(xmin, 1.0)), np.log10(x_max + 1.0), 20)
    hist, edges = np.histogram(data, bins=bins)
    widths = np.diff(edges)
    density = hist.astype(float) / np.where(widths > 0, widths, 1.0)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean per log-bin
    nz = density > 0
    if nz.sum() < 3:
        return float("nan")
    try:
        slope, _, _, _, _ = linregress(np.log(centers[nz]), np.log(density[nz]))
    except ValueError:
        return float("nan")
    return -float(slope)


def fit_avalanche_exponents(sizes: np.ndarray, lifetimes: np.ndarray,
                             bin_size: float):
    """Fit (alpha_s, alpha_t, gamma_fit, r_squared) from sizes + lifetimes.

    * ``alpha_s`` from direct P(S) log-spaced histogram fit.
    * ``alpha_t`` from direct P(T) log-spaced histogram fit (lifetimes in
      bin units, matching the alpha_s convention).
    * ``gamma_fit`` from <S>(T) regression: slope of log_T vs log_S → 1/slope.
    * ``r_squared`` is the R² of that scaling regression (used as
      bin-selection criterion).

    Returns ``(nan, nan, nan, nan)`` if inputs are degenerate.
    """
    sizes = np.asarray(sizes); lifetimes = np.asarray(lifetimes)
    if len(sizes) < 10 or np.var(sizes) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    alpha_s = fit_alpha(sizes)
    # P(T) DIRECT fit. Use lifetimes in bin units to mirror alpha_s convention.
    alpha_t = fit_alpha(lifetimes / bin_size)
    log_s = np.log(sizes.astype(float))
    # log of lifetime in *bin units* (dimensionless); divisor matches
    # alpha_t's convention above so the (alpha_t, gamma_fit) pair shares
    # units. Renamed from `log_t` to make the units explicit.
    log_t_bins = np.log(lifetimes / bin_size)
    if np.var(log_s) == 0 or np.var(log_t_bins) == 0:
        return alpha_s, alpha_t, float("nan"), float("nan")
    try:
        slope, _, r_val, _, _ = linregress(log_s, log_t_bins)
    except ValueError:
        return alpha_s, alpha_t, float("nan"), float("nan")
    gamma_fit = 1.0 / slope if slope != 0 else float("nan")
    return alpha_s, alpha_t, gamma_fit, float(r_val ** 2)


def _branching(counts_1d: np.ndarray) -> float:
    """Naive <A_{t+1}/A_t> ratio. **Deprecated** — use ``wilting_mr``.

    Emits a :class:`DeprecationWarning` because this estimator is biased by
    sub-sampling and external drive (Wilting & Priesemann 2018) and should
    not be used for inference. Retained so existing ``CriticalityResult``
    consumers keep working unchanged.
    """
    if counts_1d.size < 2:
        return float("nan")
    a = counts_1d[:-1].astype(np.float64)
    b = counts_1d[1:].astype(np.float64)
    nz = a > 0
    if not nz.any():
        return float("nan")
    _warnings.warn(
        "criticality().branching is the naive <A_{t+1}/A_t> ratio, which "
        "is biased by sub-sampling and external drive. Use "
        "nc.analysis.wilting_mr for quantitative branching-ratio inference. "
        "This field is retained for backwards-compatible diagnostics only.",
        DeprecationWarning,
        stacklevel=2,
    )
    return float(np.mean(b[nz] / a[nz]))


def criticality(rec: SpikeRecording,
                populations: Sequence[str] | None = None,
                bin_size: float | Sequence[float] = 4.0,
                ) -> CriticalityResult:
    """Fit avalanche-size, lifetime, and scaling exponents at a chosen bin.

    By default the analysis runs at a single bin size of 4 ms. If you
    are not sure which bin to use, you can pass a sequence of candidate
    bin sizes. ``criticality`` will then fit each one and report the
    bin that maximises the R² of the size-vs-lifetime regression as the
    "optimal" choice. When you do this, the full per-bin table is
    available on the returned result as ``result.fits`` so you can
    inspect every fit and find optimal value.

    Parameters
    ----------
    rec
        Spike recording.
    populations
        Names of populations whose union is binned. ``None`` → all.
    bin_size
        Bin size in milliseconds. Default is ``4.0``. Pass a single
        ``float`` for a one-shot fit at that bin (recommended when you
        already know an appropriate timescale, e.g. the inter-event
        interval). Pass a ``Sequence[float]`` (e.g. ``[2, 4, 8]``) to
        scan multiple candidates; the per-bin fit table is then
        available on the returned object as ``result.fits``. As a
        standalone alternative that returns only the table without
        picking a winner, use :func:`bin_size_sweep`.

    Returns
    -------
    :class:`CriticalityResult`
        Carries ``alpha_s``, ``alpha_t``, ``gamma_fit``,
        ``optimal_bin``, the avalanche size / lifetime arrays,
        and ``fits`` (the per-bin table when a sequence was passed).
    """
    from neurocomplexity._warnings import _warn_if_nonstationary, _warn_if_uncurated
    _warn_if_uncurated(rec, "criticality")
    _warn_if_nonstationary(rec, "criticality")
    if populations is None:
        populations = list(rec.populations.keys())
    if not populations:
        raise ValueError("no populations to analyse")

    # Normalise bin_size to a sequence; remember whether the user
    # passed a scalar so we can suppress the multi-bin notice and the
    # .fits table in that case.
    if np.isscalar(bin_size):
        bin_size_seq: list[float] = [float(bin_size)]
        single_bin = True
    else:
        bin_size_seq = [float(x) for x in bin_size]
        single_bin = len(bin_size_seq) == 1

    if not single_bin:
        import warnings as _w
        _w.warn(
            "criticality() was called with a sequence of candidate bin "
            "sizes. The bin that maximises R² of the size-vs-lifetime "
            "regression will be reported as `optimal_bin`. The "
            "fit table for every candidate bin is available on the "
            "returned object as `result.fits` so you can inspect each "
            "one rather than only trusting the chosen optimum. If you "
            "want only the table without selecting a winner, call "
            "`nc.analysis.bin_size_sweep(rec, ...)` directly.",
            UserWarning,
            stacklevel=2,
        )

    _params = {"populations": list(populations),
               "bin_size": list(bin_size_seq),
               "bin_selection": "single" if single_bin else "r2_sweep"}

    fits: list[dict] = []
    best = {"r2": -np.inf}
    for bs_ms in bin_size_seq:
        bs = float(bs_ms) / 1000.0
        counts = bin_all_active(rec, populations, bs)
        sizes, lifetimes = extract_avalanches(counts, bs)
        if len(sizes) < 10 or np.var(sizes) == 0:
            continue
        alpha_s = fit_alpha(sizes)
        # alpha_t from DIRECT P(T) fit (lifetimes in bin units).
        alpha_t = fit_alpha(lifetimes / bs)
        log_s = np.log(sizes.astype(float))
        # log of lifetime in *bin units* (dimensionless); units explicit.
        log_t_bins = np.log(lifetimes / bs)
        if np.var(log_s) == 0 or np.var(log_t_bins) == 0:
            continue
        try:
            slope, intercept, r_val, _, _ = linregress(log_s, log_t_bins)
        except ValueError:
            continue
        r2 = float(r_val ** 2)
        gamma_fit = 1.0 / slope if slope != 0 else float("nan")
        fits.append({
            "bin_seconds": bs, "alpha_s": float(alpha_s),
            "alpha_t": float(alpha_t), "gamma_fit": float(gamma_fit),
            "r_squared": r2, "n_avalanches": int(len(sizes)),
        })
        if r2 > best["r2"]:
            best = {"r2": r2, "alpha_s": alpha_s, "alpha_t": alpha_t,
                    "gamma_fit": gamma_fit,
                    "bs": bs, "sizes": sizes, "lifetimes": lifetimes,
                    "counts": counts}

    if "alpha_s" not in best:
        return CriticalityResult(
            alpha_s=float("nan"), alpha_t=float("nan"),
            optimal_bin=float("nan"), branching=float("nan"),
            sizes=np.array([]), lifetimes=np.array([]),
            r_squared=float("nan"),
            populations=tuple(populations),
            source=rec.source,
            params=_params,
            gamma_fit=float("nan"),
            gamma_predicted=float("nan"),
            fits=tuple(fits),
        )

    alpha_s = best["alpha_s"]
    alpha_t = best["alpha_t"]
    if (not np.isnan(alpha_s) and not np.isnan(alpha_t) and alpha_s != 1.0):
        gamma_predicted = (alpha_t - 1.0) / (alpha_s - 1.0)
    else:
        gamma_predicted = float("nan")
    return CriticalityResult(
        alpha_s=alpha_s, alpha_t=alpha_t,
        optimal_bin=best["bs"] * 1000.0,
            branching=_branching(best["counts"]),
        sizes=best["sizes"],
        lifetimes=best["lifetimes"],
        r_squared=best["r2"],
        populations=tuple(populations),
        source=rec.source,
        params=_params,
        gamma_fit=best["gamma_fit"],
        gamma_predicted=gamma_predicted,
        fits=tuple(fits),
    )


def bin_size_sweep(rec: SpikeRecording,
                   populations: Sequence[str] | None = None,
                   bin_size_ms: Sequence[float] = (2, 4, 8, 16, 32),
                   ) -> "CriticalityResult":
    """Sweep bin sizes and return the best-fit :class:`CriticalityResult`.

    Equivalent to ``criticality(rec, populations=populations,
    bin_size=bin_size_ms)``. The winner is the bin that maximises R² of
    the size-vs-lifetime scaling. The full per-bin table is available on
    the returned object as ``result.fits``.

    Parameters
    ----------
    rec
        Spike recording.
    populations
        Population keys to include. ``None`` uses all populations.
    bin_size_ms
        Candidate bin sizes in milliseconds.
    """
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)
        return criticality(rec, populations=populations, bin_size=bin_size_ms)
