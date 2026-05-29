"""Avalanche criticality analysis.

Definitions (Sethna 2001 crackling-noise framework; Friedman 2012; Fontenele
2019; Beggs & Plenz 2003):
  * size S = total spike count inside the avalanche burst
  * lifetime T = (# consecutive nonzero bins) * bin_size_seconds
  * P(S) ~ S^(-alpha_s)
  * P(T) ~ T^(-alpha_t)         ← fit DIRECTLY from lifetime histogram
  * <S>(T) ~ T^gamma_fit         ← empirical scaling exponent (regression)
  * gamma_predicted = (alpha_t - 1) / (alpha_s - 1)
  * At criticality:  gamma_fit ≈ gamma_predicted
  * kappa = 1 + gamma_predicted  (legacy field; kept for API compatibility)

History: an earlier version of this module estimated alpha_t as 1/slope of
the log_T-vs-log_S regression. That quantity is gamma_fit, NOT alpha_t.
Fixed: alpha_t now comes from a direct log-spaced histogram fit of the
lifetime distribution; the regression value is preserved as gamma_fit so
the Sethna consistency test can still be performed.
"""
from __future__ import annotations

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
        Exponent of the avalanche-size distribution ``P(S) ~ S^(-alpha_s)``,
        fit by maximum-likelihood on a log-spaced histogram. At criticality
        for a directed-percolation-class model, ``alpha_s ≈ 1.5``.
    alpha_t
        Exponent of the lifetime distribution ``P(T) ~ T^(-alpha_t)``, fit
        directly from a log-spaced histogram of avalanche lifetimes. At
        criticality, ``alpha_t ≈ 2.0``. **Not** ``1 / slope`` of the size-vs-lifetime
        regression (see ``gamma_fit`` below for that quantity).
    optimal_bin_seconds
        Bin size that maximised the goodness-of-fit of the size-vs-lifetime
        scaling across the swept range. Avalanches are sensitive to bin
        size; this picks the bin that gives the cleanest power law.
    branching
        Branching parameter from the size distribution
        ``branching ≈ <S^2> / <S>^2 - 1``. **Not** the same as
        :func:`~neurocomplexity.analysis.wilting_mr`. Kept for backwards
        compatibility.
    kappa
        Legacy ``kappa = 1 + gamma_predicted``. Kept for API stability; use
        ``gamma_predicted`` directly in new code.
    sizes
        Per-avalanche size counts (at ``optimal_bin_seconds``).
    lifetimes
        Per-avalanche lifetimes in seconds (at ``optimal_bin_seconds``).
    r_squared
        Goodness-of-fit of the ``<S>(T) ~ T^gamma_fit`` log-linear
        regression.
    populations
        Populations whose union was binned into the activity series.
    source
        Provenance back-pointer.
    params
        Verbatim copy of the keyword arguments passed to :func:`criticality`.
    gamma_fit
        Empirical scaling exponent from the regression
        ``log <S>(T) = const + gamma_fit * log T``. Historically misreported
        as ``alpha_t``; we now expose it explicitly.
    gamma_predicted
        Theoretical scaling exponent ``(alpha_t - 1) / (alpha_s - 1)``
        predicted by the Sethna (2001) crackling-noise framework. At
        criticality, ``gamma_fit ≈ gamma_predicted`` — the Sethna
        consistency test.
    """

    alpha_s: float
    alpha_t: float
    optimal_bin_seconds: float
    branching: float
    kappa: float
    sizes: np.ndarray
    lifetimes: np.ndarray
    r_squared: float
    populations: tuple[str, ...]
    source: object  # ProvenanceRecord back-pointer
    params: dict = field(default_factory=dict)
    # New fields (post bug-fix). Defaulted so old pickled results still load.
    gamma_fit: float = float("nan")           # 1/slope of log_T vs log_S
    gamma_predicted: float = float("nan")     # (alpha_t - 1) / (alpha_s - 1)
    # Phase-4 Tier 4 (forking-path lockdown). Every bin's fit, not just the
    # R²-selected one. Empty if a single ``bin_size_ms`` was passed.
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

    Uses log-spaced histogram binning and normalises by bin width so the
    log-log slope recovers the density exponent. Linear binning (the
    previous implementation) systematically biases ``alpha`` upward on
    heavy-tailed data because most tail bins are empty and the lower-x
    bins dominate the fit.
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
    log_t = np.log(lifetimes / bin_size)
    if np.var(log_s) == 0 or np.var(log_t) == 0:
        return alpha_s, alpha_t, float("nan"), float("nan")
    try:
        slope, _, r_val, _, _ = linregress(log_s, log_t)
    except ValueError:
        return alpha_s, alpha_t, float("nan"), float("nan")
    gamma_fit = 1.0 / slope if slope != 0 else float("nan")
    return alpha_s, alpha_t, gamma_fit, float(r_val ** 2)


def _branching(counts_1d: np.ndarray) -> float:
    if counts_1d.size < 2:
        return float("nan")
    a = counts_1d[:-1].astype(np.float64)
    b = counts_1d[1:].astype(np.float64)
    nz = a > 0
    if not nz.any():
        return float("nan")
    return float(np.mean(b[nz] / a[nz]))


def criticality(rec: SpikeRecording,
                populations: Sequence[str] | None = None,
                bin_size_ms: float | Sequence[float] = 4.0,
                ) -> CriticalityResult:
    """Fit avalanche-size, lifetime, and scaling exponents at a chosen bin.

    .. versionchanged:: 1.1.0
        ``bin_size_ms`` now defaults to a *scalar* (4 ms). Passing a
        sequence still works for backwards compatibility but emits a
        :class:`UserWarning` flagging the R²-driven selection as a
        methodological forking-path — see
        ``docs/decisions/2026-05-29-criticality-bin-selection.md`` for
        the rationale.

    Parameters
    ----------
    rec
        Spike recording.
    populations
        Names of populations whose union is binned. ``None`` → all.
    bin_size_ms
        Bin size in milliseconds. Pass a single ``float`` for the
        principled single-bin estimate (recommended). Passing a
        ``Sequence[float]`` triggers the legacy R²-driven sweep; the
        full per-bin table is exposed in :attr:`CriticalityResult.fits`
        so reviewers can inspect every fit. Use
        :func:`bin_size_sweep` if you only want the table without
        choosing a winner.

    Returns
    -------
    :class:`CriticalityResult`
    """
    from neurocomplexity._warnings import _warn_if_nonstationary, _warn_if_uncurated
    _warn_if_uncurated(rec, "criticality")
    _warn_if_nonstationary(rec, "criticality")
    if populations is None:
        populations = list(rec.populations.keys())
    if not populations:
        raise ValueError("no populations to analyse")

    # Normalise bin_size_ms to a sequence; remember whether the user passed
    # a scalar so we can suppress the forking-path warning and the .fits
    # table in that case.
    if np.isscalar(bin_size_ms):
        bin_size_ms_seq: list[float] = [float(bin_size_ms)]
        single_bin = True
    else:
        bin_size_ms_seq = [float(x) for x in bin_size_ms]
        single_bin = len(bin_size_ms_seq) == 1

    if not single_bin:
        import warnings as _w
        _w.warn(
            "criticality() called with a sequence of candidate bin sizes; "
            "the bin that maximises R² of the size-vs-lifetime regression "
            "will be selected and reported as `optimal_bin_seconds`. This "
            "is a methodological forking path — every fit is now exposed "
            "in `CriticalityResult.fits` so a reviewer can audit the "
            "choice. Prefer passing a single `bin_size_ms` value and "
            "justifying it from the autocorrelation time, or call "
            "`nc.analysis.bin_size_sweep(rec, ...)` directly. See "
            "`docs/decisions/2026-05-29-criticality-bin-selection.md`.",
            UserWarning,
            stacklevel=2,
        )

    _params = {"populations": list(populations),
               "bin_size_ms": list(bin_size_ms_seq),
               "bin_selection": "single" if single_bin else "r2_sweep"}

    fits: list[dict] = []
    best = {"r2": -np.inf}
    for bs_ms in bin_size_ms_seq:
        bs = float(bs_ms) / 1000.0
        counts = bin_all_active(rec, populations, bs)
        sizes, lifetimes = extract_avalanches(counts, bs)
        if len(sizes) < 10 or np.var(sizes) == 0:
            continue
        alpha_s = fit_alpha(sizes)
        # alpha_t from DIRECT P(T) fit (lifetimes in bin units).
        alpha_t = fit_alpha(lifetimes / bs)
        log_s = np.log(sizes.astype(float))
        log_t = np.log(lifetimes / bs)
        if np.var(log_s) == 0 or np.var(log_t) == 0:
            continue
        try:
            slope, intercept, r_val, _, _ = linregress(log_s, log_t)
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
            optimal_bin_seconds=float("nan"), branching=float("nan"),
            kappa=float("nan"),
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
        kappa = 1.0 + gamma_predicted
    else:
        gamma_predicted = float("nan")
        kappa = float("nan")
    return CriticalityResult(
        alpha_s=alpha_s, alpha_t=alpha_t,
        optimal_bin_seconds=best["bs"],
        branching=_branching(best["counts"]),
        kappa=kappa,
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
                   ) -> tuple[dict, ...]:
    """Return the avalanche-fit table over a candidate bin-size grid.

    Standalone alternative to :func:`criticality` that exposes the full
    sweep *without* picking a winner. Use this when you want to inspect
    bin-size sensitivity in your manuscript figure or supplement.

    Each row is a dict with keys ``bin_seconds``, ``alpha_s``,
    ``alpha_t``, ``gamma_fit``, ``r_squared``, ``n_avalanches``.

    Bins that produce fewer than 10 avalanches (or otherwise degenerate
    fits) are skipped.

    Notes
    -----
    Added in 1.1.0 to address the Phase-4 Tier 4 forking-path concern.
    """
    from neurocomplexity._warnings import _warn_if_uncurated
    _warn_if_uncurated(rec, "bin_size_sweep")
    if populations is None:
        populations = list(rec.populations.keys())
    if not populations:
        raise ValueError("no populations to analyse")
    rows: list[dict] = []
    for bs_ms in bin_size_ms:
        bs = float(bs_ms) / 1000.0
        counts = bin_all_active(rec, populations, bs)
        sizes, lifetimes = extract_avalanches(counts, bs)
        if len(sizes) < 10 or np.var(sizes) == 0:
            continue
        alpha_s, alpha_t, gamma_fit, r2 = fit_avalanche_exponents(
            sizes, lifetimes, bs)
        if not np.isfinite(r2):
            continue
        rows.append({
            "bin_seconds": bs, "alpha_s": float(alpha_s),
            "alpha_t": float(alpha_t), "gamma_fit": float(gamma_fit),
            "r_squared": float(r2), "n_avalanches": int(len(sizes)),
        })
    return tuple(rows)
