"""VAR-Granger autonomy index.

Bug-fix preserved: autonomy = p_value of F-test (high p → externals do NOT
significantly improve prediction → high autonomy).

Default bin_size is 10 ms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import f as f_dist
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg

from neurocomplexity.analysis._binning import bin_spikes
from neurocomplexity.core.recording import SpikeRecording


@dataclass(frozen=True)
class AutonomyResult:
    values: dict[str, float]
    bin_size_seconds: float
    max_lag: int
    source: object
    params: dict = field(default_factory=dict)


def _autonomy_for(counts: np.ndarray, target_col: int, max_lag: int) -> float:
    """counts: (T, P) array, target in column target_col, others are externals."""
    T, P = counts.shape
    if T < max_lag + 5 or P < 2:
        return float("nan")

    # reorder so target is column 0 (statsmodels VAR uses all-equations fit; we
    # only inspect the target equation's residuals)
    order = [target_col] + [i for i in range(P) if i != target_col]
    X = counts[:, order].astype(np.float64)
    # VAR requires column variance; guard.
    if np.any(np.var(X, axis=0) == 0):
        return float("nan")

    try:
        var_fit = VAR(X).fit(maxlags=max_lag, ic="bic")
    except Exception:
        return float("nan")
    chosen_lag = max(1, var_fit.k_ar)
    resid_full = var_fit.resid[:, 0]
    ssr_full = float(np.sum(resid_full ** 2))
    df_full = T - chosen_lag * P - 1
    if df_full <= 0 or ssr_full <= 0:
        return float("nan")

    try:
        ar_fit = AutoReg(X[:, 0], lags=chosen_lag, old_names=False).fit()
    except Exception:
        return float("nan")
    resid_red = ar_fit.resid
    ssr_red = float(np.sum(resid_red ** 2))

    df_restr = chosen_lag * (P - 1)
    if df_restr <= 0:
        return float("nan")

    f_stat = ((ssr_red - ssr_full) / df_restr) / (ssr_full / df_full)
    if not np.isfinite(f_stat) or f_stat < 0:
        return float("nan")
    p_val = float(1.0 - f_dist.cdf(f_stat, df_restr, df_full))
    # autonomy = p_value: high p → fail to reject "externals don't help" → autonomous
    return max(0.0, min(1.0, p_val))


def autonomy(rec: SpikeRecording,
             populations: Sequence[str] | None = None,
             bin_size_ms: float = 10.0,
             max_lag: int = 5,
             ) -> AutonomyResult:
    """Autonomy index per population using every other population as externals."""
    from neurocomplexity._warnings import _warn_if_uncurated
    _warn_if_uncurated(rec, "autonomy")
    if populations is None:
        populations = list(rec.populations.keys())
    if len(populations) < 2:
        raise ValueError("need at least 2 populations for autonomy")

    bs = float(bin_size_ms) / 1000.0
    counts = bin_spikes(rec, populations, bs)
    values: dict[str, float] = {}
    for i, name in enumerate(populations):
        values[name] = _autonomy_for(counts, target_col=i, max_lag=max_lag)

    return AutonomyResult(
        values=values,
        bin_size_seconds=bs,
        max_lag=max_lag,
        source=rec.source,
        params={"populations": list(populations),
                "bin_size_ms": float(bin_size_ms),
                "max_lag": int(max_lag)},
    )
