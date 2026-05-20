"""Avalanche criticality analysis.

Bug-fixes preserved from prototype:
  * size = total spike count inside the avalanche burst
  * lifetime = (# consecutive nonzero bins) * bin_size_seconds  → independent of size
  * kappa = 1 + (alpha_t - 1) / (alpha_s - 1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

from neurocomplexity.analysis._binning import bin_all_active
from neurocomplexity.core.recording import SpikeRecording


@dataclass(frozen=True)
class CriticalityResult:
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


def extract_avalanches(counts_1d: np.ndarray, bin_size: float):
    """Return (sizes, lifetimes) for the binned count series."""
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
    """Fit (alpha_s, alpha_t, r_squared) from avalanche sizes and lifetimes.

    `bin_size` is the bin width in seconds used when lifetimes were extracted.
    Returns (nan, nan, nan) if the inputs are degenerate.
    """
    sizes = np.asarray(sizes); lifetimes = np.asarray(lifetimes)
    if len(sizes) < 10 or np.var(sizes) == 0:
        return float("nan"), float("nan"), float("nan")
    alpha_s = fit_alpha(sizes)
    log_s = np.log(sizes.astype(float))
    log_t = np.log(lifetimes / bin_size)
    if np.var(log_s) == 0 or np.var(log_t) == 0:
        return float("nan"), float("nan"), float("nan")
    try:
        slope, _, r_val, _, _ = linregress(log_s, log_t)
    except ValueError:
        return float("nan"), float("nan"), float("nan")
    alpha_t = 1.0 / slope if slope != 0 else float("nan")
    return alpha_s, alpha_t, float(r_val ** 2)


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
                bin_size_ms: Sequence[float] = (2, 4, 8, 16, 32),
                ) -> CriticalityResult:
    """Run the bin-size sweep, fit power laws, return the best fit by R²."""
    from neurocomplexity._warnings import _warn_if_uncurated, _warn_if_nonstationary
    _warn_if_uncurated(rec, "criticality")
    _warn_if_nonstationary(rec, "criticality")
    if populations is None:
        populations = list(rec.populations.keys())
    if not populations:
        raise ValueError("no populations to analyse")
    _params = {"populations": list(populations),
               "bin_size_ms": list(bin_size_ms)}

    best = {"r2": -np.inf}
    for bs_ms in bin_size_ms:
        bs = float(bs_ms) / 1000.0
        counts = bin_all_active(rec, populations, bs)
        sizes, lifetimes = extract_avalanches(counts, bs)
        if len(sizes) < 10 or np.var(sizes) == 0:
            continue
        alpha_s = fit_alpha(sizes)
        log_s = np.log(sizes.astype(float))
        log_t = np.log(lifetimes / bs)  # lifetimes in bin units
        if np.var(log_s) == 0 or np.var(log_t) == 0:
            continue
        try:
            slope, intercept, r_val, _, _ = linregress(log_s, log_t)
        except ValueError:
            continue
        r2 = float(r_val ** 2)
        alpha_t = 1.0 / slope if slope != 0 else float("nan")
        if r2 > best["r2"]:
            best = {"r2": r2, "alpha_s": alpha_s, "alpha_t": alpha_t,
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
        )

    alpha_s = best["alpha_s"]
    alpha_t = best["alpha_t"]
    kappa = (1.0 + (alpha_t - 1.0) / (alpha_s - 1.0)
             if (not np.isnan(alpha_s) and not np.isnan(alpha_t) and alpha_s != 1.0)
             else float("nan"))
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
    )
