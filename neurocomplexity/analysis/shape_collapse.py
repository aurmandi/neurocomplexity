"""Avalanche shape collapse (Friedman et al. 2012, PRL 108, 208102).

Average activity profile a(t,T) of avalanches of duration T is hypothesised to
scale as
        a(t, T) ~ T^(gamma - 1) * F(t / T)
for a single scaling function F and a single exponent gamma. We minimise a
**scale-invariant** residual between rescaled shape curves with a bounded
continuous optimiser (and a grid pre-scan to seed the bracket).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from neurocomplexity.analysis._binning import bin_all_active
from neurocomplexity.core.recording import SpikeRecording


@dataclass(frozen=True)
class ShapeCollapseResult:
    gamma: float
    residual: float
    durations_used: np.ndarray
    mean_shapes: list          # list of arrays, one per duration class
    rescaled_x: np.ndarray
    rescaled_y: np.ndarray
    bin_size_seconds: float
    populations: tuple[str, ...]
    source: object
    params: dict = field(default_factory=dict)


def _extract_avalanche_shapes(counts_1d: np.ndarray):
    binary = (counts_1d > 0).astype(np.int8)
    edges = np.diff(np.concatenate(([0], binary, [0])))
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]
    return [counts_1d[s:e].astype(np.float64) for s, e in zip(starts, stops)]


def _scaled_residual(g: float, interp_shapes: np.ndarray,
                     durations: np.ndarray) -> float:
    """Scale-invariant residual: mean point-wise variance divided by the
    square of the mean curve. Independent of overall amplitude, so the
    optimiser can compare different gammas fairly."""
    scaled = interp_shapes / (durations[:, None] ** (g - 1.0))
    mean_curve = scaled.mean(axis=0)
    var_curve = scaled.var(axis=0)
    denom = float(np.mean(mean_curve ** 2)) + 1e-30
    return float(np.mean(var_curve) / denom)


def shape_collapse(rec: SpikeRecording,
                   populations: Sequence[str] | None = None,
                   bin_size_ms: float = 4.0,
                   min_duration: int = 4,
                   max_duration: int = 60,
                   gamma_range: tuple[float, float] = (0.5, 5.0),
                   n_gamma_seed: int = 41,
                   n_interp: int = 50,
                   ) -> ShapeCollapseResult:
    from neurocomplexity._warnings import _warn_if_uncurated, _warn_if_nonstationary
    _warn_if_uncurated(rec, "shape_collapse")
    _warn_if_nonstationary(rec, "shape_collapse")
    if populations is None:
        populations = list(rec.populations.keys())
    bs = float(bin_size_ms) / 1000.0
    counts = bin_all_active(rec, populations, bs).astype(np.float64)
    shapes = _extract_avalanche_shapes(counts)
    if not shapes:
        raise ValueError("no avalanches detected at this bin size")

    # Group avalanches by duration (in bins); only keep duration classes
    # with enough samples (>=5) to compute a mean shape.
    by_T: dict[int, list[np.ndarray]] = {}
    for s in shapes:
        T = len(s)
        if min_duration <= T <= max_duration:
            by_T.setdefault(T, []).append(s)
    by_T = {T: arrs for T, arrs in by_T.items() if len(arrs) >= 5}
    if len(by_T) < 3:
        raise ValueError(
            f"fewer than 3 duration classes with >=5 avalanches "
            f"(have {len(by_T)}); try a smaller bin or shorter duration range"
        )

    durations = np.array(sorted(by_T.keys()))
    mean_shapes = [np.mean(np.vstack([np.pad(a, (0, T-len(a))) for a in by_T[T]]),
                            axis=0)
                   for T in durations]

    # Common x-axis u = t/T in [0,1]
    u = np.linspace(0, 1, n_interp)
    interp_shapes = []
    for T, shape in zip(durations, mean_shapes):
        x_native = np.arange(T) / max(T - 1, 1)
        interp_shapes.append(np.interp(u, x_native, shape))
    interp_shapes = np.array(interp_shapes)  # (n_durations, n_interp)

    # Coarse grid → bracket → bounded Brent.
    g_lo, g_hi = float(gamma_range[0]), float(gamma_range[1])
    grid = np.linspace(g_lo, g_hi, n_gamma_seed)
    grid_resid = np.array([_scaled_residual(g, interp_shapes, durations)
                           for g in grid])
    k = int(np.argmin(grid_resid))
    # Bracket around the grid minimum, clamped to bounds
    lo = grid[max(k - 1, 0)]
    hi = grid[min(k + 1, len(grid) - 1)]
    if hi - lo < 1e-6:
        lo, hi = g_lo, g_hi

    res = minimize_scalar(
        _scaled_residual,
        bounds=(lo, hi),
        args=(interp_shapes, durations),
        method="bounded",
        options={"xatol": 1e-4},
    )
    best_gamma = float(res.x)
    best_resid = float(res.fun)

    rescaled = interp_shapes / (durations[:, None] ** (best_gamma - 1.0))
    return ShapeCollapseResult(
        gamma=best_gamma,
        residual=best_resid,
        durations_used=durations,
        mean_shapes=mean_shapes,
        rescaled_x=u,
        rescaled_y=rescaled,
        bin_size_seconds=bs,
        populations=tuple(populations),
        source=rec.source,
        params={"populations": list(populations),
                "bin_size_ms": float(bin_size_ms),
                "min_duration": int(min_duration),
                "max_duration": int(max_duration),
                "gamma_range": tuple(gamma_range),
                "n_gamma_seed": int(n_gamma_seed),
                "n_interp": int(n_interp)},
    )
