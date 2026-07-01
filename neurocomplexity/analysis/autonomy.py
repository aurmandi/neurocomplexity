"""VAR-Granger autonomy index.

Reports per-population Granger-dependency F-test:
    * p-value   — fail-to-reject p (high p ⇒ data consistent with
                  externals providing no predictive advantage)
    * F-stat    — F-statistic of the nested test
    * chosen_lag — BIC-selected lag of the full OLS model (≥ 1)

The full and reduced models are nested OLS fits on the same lagged design
matrix, so the F-statistic is exactly F-distributed under the Gaussian null.
Empirically, on Poisson-binned spike counts the per-population Type-I rate is
mildly anti-conservative (~0.07-0.08 at nominal 0.05); the calibration-free
circular-shift permutation path (``significance="permutation"``) shows the
same offset, indicating it stems from binning and finite-sample VAR structure
rather than non-Gaussianity. Treat p-values near 0.05 cautiously and prefer
the permutation path for publication-grade inference.

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
        Mapping ``{population_name: lag}`` of the BIC-selected lag order
        actually used for the full model (minimum 1, capped at ``max_lag``).
        Reports the autoregressive memory the model captured, distinct from
        the requested ``max_lag``.
    bin_size_seconds
        Bin size used to turn spikes into population-count series.
    max_lag
        Maximum lag the BIC search was allowed to consider.
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


def _lagged_design(X: np.ndarray, lag: int):
    """Target equation design. X: (T, P), target is column 0.

    Returns (y, Z_full, Z_red) aligned to rows t in [lag, T):
    y          - target at time t
    Z_full     - const + lags 1..lag of ALL P populations
    Z_red      - const + lags 1..lag of the target only
    """
    T, P = X.shape
    rows = T - lag
    y = X[lag:, 0]
    full_blocks = [X[lag - j: T - j, :] for j in range(1, lag + 1)]
    red_blocks = [X[lag - j: T - j, 0:1] for j in range(1, lag + 1)]
    const = np.ones((rows, 1))
    Z_full = np.concatenate([const, *full_blocks], axis=1)
    Z_red = np.concatenate([const, *red_blocks], axis=1)
    return y, Z_full, Z_red


def _ols_ssr(y: np.ndarray, Z: np.ndarray) -> tuple[float, int, int]:
    """OLS residual sum of squares, n_parameters, and design-matrix rank.

    ``rank < Z.shape[1]`` signals a rank-deficient (collinear) design, where
    ``lstsq`` returns a least-norm solution whose SSR can be spuriously small;
    callers treat that as a degenerate fit.
    """
    beta, _, rank, _ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    return float(resid @ resid), int(Z.shape[1]), int(rank)


def _autonomy_for(counts: np.ndarray, target_col: int, max_lag: int, *,
                  significance: str = "analytic", n_perm: int = 200,
                  rng: np.random.Generator | None = None
                  ) -> tuple[float, float, int]:
    """Nested-OLS Granger F-test for one target. Same design matrix for full
    and reduced models, so the F-statistic is exactly F-distributed under the
    Gaussian null (analytic path) or calibrated by circular-shift surrogates
    of the external columns (permutation path).

    Returns (p_value, f_statistic, chosen_lag); (nan, nan, 0) on failure.
    """
    T, P = counts.shape
    if T < max_lag + 5 or P < 2:
        return float("nan"), float("nan"), 0
    order = [target_col] + [i for i in range(P) if i != target_col]
    X = counts[:, order].astype(np.float64)
    if np.any(np.var(X, axis=0) == 0):
        return float("nan"), float("nan"), 0

    # BIC lag selection on the full target-equation OLS.
    best = None  # (bic, lag)
    for lag in range(1, max_lag + 1):
        n = T - lag
        if n < (P * lag + 1) + 5:
            break
        y, Zf, _ = _lagged_design(X, lag)
        ssr_f, kf, rank_f = _ols_ssr(y, Zf)
        if ssr_f <= 0 or rank_f < kf:
            continue
        bic = n * np.log(ssr_f / n) + kf * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, lag)
    if best is None:
        return float("nan"), float("nan"), 0
    chosen_lag = best[1]

    y, Zf, Zr = _lagged_design(X, chosen_lag)
    ssr_full, kf, rank_full = _ols_ssr(y, Zf)
    ssr_red, kr, _ = _ols_ssr(y, Zr)
    n = len(y)
    df_full = n - kf
    df_restr = kf - kr  # = chosen_lag * (P - 1) external-lag restrictions
    if df_full <= 0 or df_restr <= 0 or ssr_full <= 0 or rank_full < kf:
        return float("nan"), float("nan"), chosen_lag
    f_stat = ((ssr_red - ssr_full) / df_restr) / (ssr_full / df_full)
    if not np.isfinite(f_stat) or f_stat < 0:
        return float("nan"), float("nan"), chosen_lag

    if significance == "analytic":
        p_val = float(1.0 - f_dist.cdf(f_stat, df_restr, df_full))
    elif significance == "permutation":
        if rng is None:
            rng = np.random.default_rng(0)
        # Only the external columns are shifted, so the target column (and
        # hence the reduced-model design and ssr_red) is invariant across
        # surrogates — reuse ssr_red rather than recompute it each draw. The
        # 1-sample minimum shift preserves short-range autocorrelation, a
        # known conservative bias of circular-shift permutation tests.
        ge = 0
        for _ in range(n_perm):
            Xs = X.copy()
            for c in range(1, P):  # shift each external column independently
                Xs[:, c] = np.roll(X[:, c], int(rng.integers(1, T)))
            ys, Zfs, _ = _lagged_design(Xs, chosen_lag)
            ssr_fs, kfs, rank_fs = _ols_ssr(ys, Zfs)
            if ssr_fs <= 0 or rank_fs < kfs:
                continue
            f_s = ((ssr_red - ssr_fs) / df_restr) / (ssr_fs / df_full)
            if np.isfinite(f_s) and f_s >= f_stat:
                ge += 1
        p_val = (ge + 1) / (n_perm + 1)  # Phipson-Smyth (2010) +1 floor
    else:
        raise ValueError(
            f"significance must be 'analytic' or 'permutation'; got {significance!r}")
    return max(0.0, min(1.0, p_val)), float(f_stat), int(chosen_lag)


def autonomy(rec: SpikeRecording,
             populations: Sequence[str] | None = None,
             bin_size_ms: float = 10.0,
             max_lag: int = 5,
             *,
             significance: str = "analytic",
             n_perm: int = 200,
             seed: int = 0,
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
    rng = np.random.default_rng(seed)
    from neurocomplexity._progress import progress_iter
    for i, name in progress_iter(list(enumerate(populations)),
                                 total=len(populations), desc="autonomy"):
        p_val, f_val, lag = _autonomy_for(
            counts, target_col=i, max_lag=max_lag,
            significance=significance, n_perm=n_perm, rng=rng)
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
                "max_lag": int(max_lag),
                "significance": significance,
                "n_perm": int(n_perm),
                "seed": int(seed)},
    )
