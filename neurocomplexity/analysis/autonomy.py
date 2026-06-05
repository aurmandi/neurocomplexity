"""VAR-Granger autonomy index.

Reports per-population Granger-dependency F-test:
    * p-value   — fail-to-reject p (high p ⇒ data consistent with
                  externals providing no predictive advantage)
    * F-stat    — F-statistic of the nested test
    * chosen_lag — BIC-selected lag of the full VAR (≥ 1)

The fail-to-reject interpretation is documented in the manuscript §2.4.6:
large p is NOT positive evidence of independence; the test may be
underpowered against weak or non-linear alternatives. Effect size
(F-statistic) and chosen lag should be reported alongside p.

Default bin_size is 10 ms.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import f as f_dist
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg

from neurocomplexity.analysis._binning import bin_spikes
from neurocomplexity.core.recording import SpikeRecording


@dataclass(frozen=True)
class AutonomyResult:
    """Output of :func:`autonomy`.

    Attributes
    ----------
    values
        Mapping ``{population_name: granger_dependency_p_value}``. F-test
        p-value for the null hypothesis that adding the other populations
        as predictors does *not* improve forecasting of the target population
        beyond its own past. Large p (e.g. > 0.05) is interpreted as
        fail-to-reject — consistent with no detectable cross-population
        predictive contribution — but is NOT positive evidence of
        independence (the test is underpowered against weak / non-linear
        alternatives).
    f_stats
        Mapping ``{population_name: F_statistic}`` of the nested F-test.
        Reports effect size of the Granger contribution so callers can
        distinguish "small effect, large p" from "no effect, large p".
        ``nan`` when the fit failed (degenerate columns, too few samples,
        or convergence error).
    chosen_lags
        Mapping ``{population_name: lag}`` of the BIC-selected VAR order
        actually used for the full model (minimum 1, capped at ``max_lag``).
        Reports the autoregressive memory the model captured, distinct from
        the requested ``max_lag``.
    bin_size_seconds
        Bin size used to turn spikes into population-count series.
    max_lag
        Maximum VAR lag the BIC search was allowed to consider.
    source
        Back-pointer to the :class:`~neurocomplexity.core.provenance.ProvenanceRecord`
        of the source recording.
    params
        Verbatim copy of the keyword arguments passed to :func:`autonomy`,
        for reproducibility.
    """

    values: dict[str, float]
    f_stats: dict[str, float]
    chosen_lags: dict[str, int]
    bin_size_seconds: float
    max_lag: int
    source: object
    params: dict = field(default_factory=dict)


def _autonomy_for(counts: np.ndarray, target_col: int, max_lag: int
                  ) -> tuple[float, float, int]:
    """counts: (T, P) array, target in column target_col, others are externals.

    Returns
    -------
    (p_value, f_statistic, chosen_lag)
        ``(nan, nan, 0)`` on any failure (degenerate columns, fit error,
        insufficient samples).
    """
    T, P = counts.shape
    if T < max_lag + 5 or P < 2:
        return float("nan"), float("nan"), 0

    # reorder so target is column 0 (statsmodels VAR uses all-equations fit; we
    # only inspect the target equation's residuals)
    order = [target_col] + [i for i in range(P) if i != target_col]
    X = counts[:, order].astype(np.float64)
    # VAR requires column variance; guard.
    if np.any(np.var(X, axis=0) == 0):
        return float("nan"), float("nan"), 0

    try:
        var_fit = VAR(X).fit(maxlags=max_lag, ic="bic")
    except Exception:
        return float("nan"), float("nan"), 0
    chosen_lag = max(1, var_fit.k_ar)
    resid_full = var_fit.resid[:, 0]
    ssr_full = float(np.sum(resid_full ** 2))
    df_full = T - chosen_lag * P - 1
    if df_full <= 0 or ssr_full <= 0:
        return float("nan"), float("nan"), chosen_lag

    try:
        ar_fit = AutoReg(X[:, 0], lags=chosen_lag, old_names=False).fit()
    except Exception:
        return float("nan"), float("nan"), chosen_lag
    resid_red = ar_fit.resid
    ssr_red = float(np.sum(resid_red ** 2))

    df_restr = chosen_lag * (P - 1)
    if df_restr <= 0:
        return float("nan"), float("nan"), chosen_lag

    f_stat = ((ssr_red - ssr_full) / df_restr) / (ssr_full / df_full)
    if not np.isfinite(f_stat) or f_stat < 0:
        return float("nan"), float("nan"), chosen_lag
    p_val = float(1.0 - f_dist.cdf(f_stat, df_restr, df_full))
    # high p → fail to reject "externals don't help"
    # (interpretation: data consistent with no Granger contribution; NOT
    # positive evidence of independence — see module docstring)
    p_val = max(0.0, min(1.0, p_val))
    return p_val, float(f_stat), int(chosen_lag)


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
    f_stats: dict[str, float] = {}
    chosen_lags: dict[str, int] = {}
    from neurocomplexity._progress import progress_iter
    for i, name in progress_iter(list(enumerate(populations)),
                                 total=len(populations), desc="autonomy"):
        p_val, f_val, lag = _autonomy_for(counts, target_col=i, max_lag=max_lag)
        values[name] = p_val
        f_stats[name] = f_val
        chosen_lags[name] = lag

    return AutonomyResult(
        values=values,
        f_stats=f_stats,
        chosen_lags=chosen_lags,
        bin_size_seconds=bs,
        max_lag=max_lag,
        source=rec.source,
        params={"populations": list(populations),
                "bin_size_ms": float(bin_size_ms),
                "max_lag": int(max_lag)},
    )
