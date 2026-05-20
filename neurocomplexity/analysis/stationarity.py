"""Population-level stationarity diagnostics for spike recordings.

Reports four metrics over fixed-width time windows:

  * ``population_rate_cv``   — CV of windowed population rate (rate-stability proxy).
  * ``rate_drift_slope``     — OLS slope of rate vs window-centre time, in Hz/s.
  * ``rate_drift_pvalue``    — two-sided slope-vs-zero p-value.
  * ``cv2_mean``             — mean across units of the local-ISI CV2 (Holt et al. 1996).
  * ``rolling_var_ratio``    — max(window variance) / min(window variance).

Any threshold breach produces a human-readable flag string; ``is_stationary`` is
true iff every diagnostic is within bounds.

References:
  * Brody CD (1999). Correlations without synchrony. Neural Comput 11:1537-1551.
  * Holt GR et al. (1996). Comparison of discharge variability in vitro and in
    vivo in cat visual cortex. J Neurophysiol 75:1806-1814.
"""
from __future__ import annotations

import warnings as _pywarnings
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy import stats

from neurocomplexity.core.recording import SpikeRecording


@dataclass(frozen=True)
class StationarityResult:
    population_rate_cv: float
    rate_drift_slope: float
    rate_drift_pvalue: float
    cv2_mean: float
    rolling_var_ratio: float
    n_windows: int
    window_s: float
    params: dict = field(default_factory=dict)
    is_stationary: bool = True
    flags: Tuple[str, ...] = ()


def _population_rate_per_window(rec: SpikeRecording, window_s: float
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Return (rate_per_window_Hz, window_centre_seconds)."""
    duration = float(rec.duration)
    n = max(int(np.floor(duration / window_s)), 2)
    if n * window_s > duration:
        # When duration < 2*window_s we still want two windows for any slope test.
        edges = np.linspace(0.0, duration, n + 1)
    else:
        edges = np.arange(0, (n + 1)) * window_s
        edges[-1] = min(edges[-1], duration)
    counts, _ = np.histogram(rec.spike_times, bins=edges)
    widths = np.diff(edges)
    n_units = max(len(rec.units), 1)
    rate = counts / widths / n_units
    centres = 0.5 * (edges[:-1] + edges[1:])
    return rate.astype(float), centres.astype(float)


def _cv2_for_recording(rec: SpikeRecording) -> float:
    uids = np.asarray(rec.unit_ids, dtype=np.int64)
    if uids.size == 0:
        return float("nan")
    unique = np.unique(uids)
    vals = []
    for u in unique:
        st = rec.spike_times[uids == u]
        if st.size < 3:
            continue
        isi = np.diff(np.sort(st))
        if isi.size < 2:
            continue
        num = 2.0 * np.abs(isi[1:] - isi[:-1])
        den = isi[1:] + isi[:-1]
        ok = den > 0
        if not np.any(ok):
            continue
        vals.append(float(np.mean(num[ok] / den[ok])))
    return float(np.mean(vals)) if vals else float("nan")


def stationarity(rec: SpikeRecording, *,
                 window_s: float = 30.0,
                 cv_threshold: float = 0.30,
                 slope_pvalue_threshold: float = 0.01,
                 var_ratio_threshold: float = 3.0,
                 cv2_threshold: float = 1.5,
                 ) -> StationarityResult:
    """Compute stationarity diagnostics for ``rec``.

    All thresholds are advisory; ``is_stationary`` is True iff none are breached.
    Defaults are deliberately permissive (CV>0.30, slope p<0.01, var-ratio>3,
    CV2>1.5) to avoid spamming warnings on normal experimental recordings.
    """
    if rec.duration < 2.0 * window_s:
        _pywarnings.warn(
            f"recording duration {rec.duration:.1f}s < 2*window_s "
            f"({2*window_s:.1f}s); reducing to 2 windows.",
            UserWarning, stacklevel=2,
        )

    rate, centres = _population_rate_per_window(rec, window_s)
    n_windows = int(rate.size)

    mean_r = float(np.mean(rate)) if rate.size else 0.0
    std_r = float(np.std(rate, ddof=1)) if rate.size > 1 else 0.0
    cv = (std_r / mean_r) if mean_r > 0 else 0.0

    if rate.size >= 2 and np.ptp(centres) > 0:
        lr = stats.linregress(centres, rate)
        slope = float(lr.slope)
        slope_p = float(lr.pvalue)
    else:
        slope = 0.0
        slope_p = 1.0

    var_per_window = np.zeros_like(rate)
    if rate.size >= 2:
        # variance within each window using sub-bin counts (10 sub-bins per window)
        sub_n = 10
        edges_main = np.linspace(0.0, rec.duration, n_windows + 1)
        n_units = max(len(rec.units), 1)
        for i in range(n_windows):
            sub_edges = np.linspace(edges_main[i], edges_main[i + 1], sub_n + 1)
            sub_counts, _ = np.histogram(rec.spike_times, bins=sub_edges)
            sub_widths = np.diff(sub_edges)
            sub_rate = sub_counts / sub_widths / n_units
            var_per_window[i] = float(np.var(sub_rate, ddof=1)) if sub_rate.size > 1 else 0.0
    positive = var_per_window[var_per_window > 0]
    if positive.size >= 2:
        var_ratio = float(np.max(positive) / np.min(positive))
    else:
        var_ratio = 1.0

    cv2 = _cv2_for_recording(rec)

    flags: list[str] = []
    if cv > cv_threshold:
        flags.append(f"population_rate_cv={cv:.3f} > {cv_threshold}")
    if slope_p < slope_pvalue_threshold:
        flags.append(f"rate_drift p={slope_p:.3g} < {slope_pvalue_threshold} "
                     f"(slope={slope:.3g} Hz/s)")
    if var_ratio > var_ratio_threshold:
        flags.append(f"rolling_var_ratio={var_ratio:.2f} > {var_ratio_threshold}")
    if np.isfinite(cv2) and cv2 > cv2_threshold:
        flags.append(f"cv2_mean={cv2:.3f} > {cv2_threshold}")

    return StationarityResult(
        population_rate_cv=cv,
        rate_drift_slope=slope,
        rate_drift_pvalue=slope_p,
        cv2_mean=cv2,
        rolling_var_ratio=var_ratio,
        n_windows=n_windows,
        window_s=float(window_s),
        params={
            "window_s": float(window_s),
            "cv_threshold": float(cv_threshold),
            "slope_pvalue_threshold": float(slope_pvalue_threshold),
            "var_ratio_threshold": float(var_ratio_threshold),
            "cv2_threshold": float(cv2_threshold),
        },
        is_stationary=(len(flags) == 0),
        flags=tuple(flags),
    )
