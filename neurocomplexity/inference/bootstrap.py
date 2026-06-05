"""Bootstrap resamplers for analysis results.

Each resampler returns an InferenceResult with `bootstrap_distribution`,
`ci_lower`, `ci_upper`, and `null_distribution=None` so the public
`bootstrap()` dispatch is uniform.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.stats import norm

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

from neurocomplexity.analysis.autonomy import AutonomyResult, autonomy
from neurocomplexity.analysis.branching import BranchingResult, wilting_mr
from neurocomplexity.analysis.criticality import (
    CriticalityResult,
    fit_avalanche_exponents,
)
from neurocomplexity.analysis.dimensionality import (
    DimensionalityResult,
    dimensionality,
)
from neurocomplexity.analysis.shape_collapse import (
    ShapeCollapseResult,
    shape_collapse,
)
from neurocomplexity.inference.results import InferenceResult


def _ci_from_dist(dist: np.ndarray, level: float, observed=None,
                  method: str = "bca"):
    """Bootstrap CI.

    Parameters
    ----------
    dist
        Bootstrap distribution. First axis is the replicate axis; trailing
        axes (if any) carry vector-valued statistics.
    level
        Confidence level (e.g. 0.95).
    observed
        Observed point estimate. Required for ``method`` in {"bca", "bc"};
        ignored (and falls back to percentile) when ``None`` or when
        ``method == "percentile"``.
    method
        ``"bca"`` (default) — bias-corrected and accelerated bootstrap
        (Efron 1987; DiCiccio & Efron 1996). Uses ``z0`` from the
        empirical CDF of the bootstrap dist at ``observed`` and acceleration
        ``a`` from the third / second central-moment ratio of the bootstrap
        distribution (Efron & Tibshirani 1993, eq. 14.15). Recommended for
        near-boundary estimators such as the Wilting--Priesemann m-hat
        approaching 1, where BC under-covers.
        ``"bc"`` — legacy bias-correction only (``z0`` without acceleration).
        ``"percentile"`` — naive percentile (no correction).

    Notes
    -----
    * For a component whose bootstrap replicates have ~zero variance,
      the corrected interval can degenerate to NaN. Callers receive NaN
      bounds for that component rather than a misleadingly narrow interval.
    * Corrections are applied independently per component for vector
      statistics. That assumes component-wise bias estimates are stable;
      with very small ``n_resamples`` (n < 30) consider increasing n or
      falling back to the percentile path.
    """
    lo_q = (1 - level) / 2
    hi_q = (1 + level) / 2
    if observed is None or method == "percentile":
        lo = np.nanpercentile(dist, lo_q * 100, axis=0)
        hi = np.nanpercentile(dist, hi_q * 100, axis=0)
        return lo, hi
    if method not in ("bca", "bc"):
        raise ValueError(f"method must be one of bca/bc/percentile; got {method!r}")

    z_lo = norm.ppf(lo_q)
    z_hi = norm.ppf(hi_q)
    obs = np.asarray(observed, dtype=float)

    def _corrected_one(d, o):
        d = d[~np.isnan(d)]
        if d.size < 5:
            return float("nan"), float("nan")
        # Bias-correction z0: standard-normal quantile of empirical CDF at obs.
        p = float(np.mean(d < o))
        p = min(max(p, 0.5 / d.size), 1 - 0.5 / d.size)
        z0 = norm.ppf(p)
        # Acceleration: skewness of bootstrap dist scaled by 1/6
        # (Efron & Tibshirani 1993, eq. 14.15; bootstrap-based approximation
        # to the jackknife acceleration when only the bootstrap dist is
        # available).
        if method == "bca":
            mu = float(np.nanmean(d))
            diffs = mu - d
            num = float(np.nansum(diffs ** 3))
            den = float(np.nansum(diffs ** 2)) ** 1.5
            a = (num / (6.0 * den)) if den > 1e-12 else 0.0
        else:
            a = 0.0  # plain BC
        def _adj_q(z_q):
            numer = z0 + z_q
            denom = 1.0 - a * numer
            if abs(denom) < 1e-12:
                return float(norm.cdf(z_q))
            return float(norm.cdf(z0 + numer / denom))
        q_lo = _adj_q(z_lo)
        q_hi = _adj_q(z_hi)
        lo_v = float(np.nanpercentile(d, q_lo * 100))
        hi_v = float(np.nanpercentile(d, q_hi * 100))
        return lo_v, hi_v

    if dist.ndim == 1:
        lo_v, hi_v = _corrected_one(dist, float(obs))
        return lo_v, hi_v
    # vector-valued statistic: apply correction independently per component
    out_shape = dist.shape[1:]
    lo_out = np.empty(out_shape)
    hi_out = np.empty(out_shape)
    flat_dist = dist.reshape(dist.shape[0], -1)
    flat_obs = obs.reshape(-1)
    for i in range(flat_dist.shape[1]):
        lo_out.flat[i], hi_out.flat[i] = _corrected_one(
            flat_dist[:, i], float(flat_obs[i]))
    return lo_out, hi_out


def _run(seeds, fn, n_jobs):
    from neurocomplexity._progress import progress_iter
    seeds_list = list(seeds)
    if _HAS_JOBLIB and n_jobs != 1:
        out = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(fn)(s) for s in progress_iter(
                seeds_list, total=len(seeds_list), desc="bootstrap")
        )
    else:
        out = [fn(s) for s in progress_iter(
            seeds_list, total=len(seeds_list), desc="bootstrap")]
    return out


def _child_seeds(seed: int, n: int):
    ss = np.random.SeedSequence(seed)
    return [int(s.generate_state(1)[0]) for s in ss.spawn(n)]


def bootstrap_avalanche_exponents(
    result: CriticalityResult,
    rec,
    *,
    n: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
    n_jobs: int = 1,
) -> InferenceResult:
    """Bootstrap (alpha_s, alpha_t) by resampling avalanches with replacement."""
    sizes = np.asarray(result.sizes, dtype=np.int64)
    lifetimes = np.asarray(result.lifetimes, dtype=float)
    bs = float(result.optimal_bin / 1000.0)
    n_av = sizes.size
    if n_av < 50:
        raise ValueError(f"too few avalanches ({n_av}) to bootstrap")

    def _one(s):
        rng = np.random.default_rng(s)
        idx = rng.integers(0, n_av, n_av)
        a_s, a_t, _gf, _r2 = fit_avalanche_exponents(sizes[idx], lifetimes[idx], bs)
        return np.array([a_s, a_t])

    dist = np.stack(_run(_child_seeds(seed, n), _one, n_jobs), axis=0)
    obs = np.array([result.alpha_s, result.alpha_t])
    lo, hi = _ci_from_dist(dist, ci_level, observed=obs)
    return InferenceResult(
        statistic_name="avalanche_exponents",
        observed=obs,
        null_distribution=None,
        bootstrap_distribution=dist,
        p_value=None, p_value_fdr=None, effect_size=None,
        ci_lower=lo, ci_upper=hi, ci_level=ci_level,
        method="avalanche_resample",
        n_resamples=n, seed=seed,
        metadata={"n_avalanches": n_av, "bin_seconds": bs},
    )


def _warn_if_few_blocks(duration: float, block_seconds: float) -> None:
    """Emit ``UserWarning`` when block bootstrap has too few unique blocks.

    Politis-Romano (1994) require the number of distinct blocks to grow with
    sample size; in practice CIs under-cover badly when ``duration <
    3 * block_seconds`` (≤ 3 unique blocks). This warns before the resample
    proceeds. See `docs/inference.md` § "Block size guidance".
    """
    import warnings as _w
    n_unique = max(1, int(np.ceil(duration / block_seconds)))
    if n_unique < 4:
        _w.warn(
            f"block bootstrap on {duration:.1f}s with block_seconds="
            f"{block_seconds:g}s yields only {n_unique} unique block(s); "
            f"confidence intervals may under-cover. Reduce block_seconds "
            f"to ≤ duration/4 or extend the recording. See "
            f"docs/inference.md § 'Block size guidance'.",
            UserWarning,
            stacklevel=3,
        )


def _block_resampled_recording(rec, *, block_seconds: float, rng):
    """Concatenate randomly-drawn blocks of `block_seconds` from `rec`."""
    n_blocks = int(np.ceil(rec.duration / block_seconds))
    block_starts = np.arange(n_blocks) * block_seconds
    block_ends = np.minimum(block_starts + block_seconds, rec.duration)
    chosen = rng.integers(0, n_blocks, n_blocks)
    new_times, new_owners, offset = [], [], 0.0
    for k in chosen:
        s, e = block_starts[k], block_ends[k]
        m = (rec.spike_times >= s) & (rec.spike_times < e)
        new_times.append(rec.spike_times[m] - s + offset)
        new_owners.append(rec.unit_ids[m])
        offset += (e - s)
    if not new_times:
        st = np.empty(0); ui = np.empty(0, dtype=np.int64)
    else:
        st = np.concatenate(new_times); ui = np.concatenate(new_owners)
    order = np.argsort(st, kind="stable")
    return replace(rec, spike_times=st[order], unit_ids=ui[order],
                   duration=float(offset), intervals={})


def bootstrap_branching_ratio(
    result: BranchingResult,
    rec,
    *,
    n: int = 2000,
    seed: int = 0,
    ci_level: float = 0.95,
    block_seconds: float = 10.0,
    n_jobs: int = 1,
) -> InferenceResult:
    """Block bootstrap for the Wilting-Priesemann branching ratio.

    Resamples non-overlapping ``block_seconds`` time blocks of the
    recording with replacement, re-runs
    :func:`~neurocomplexity.analysis.wilting_mr` on each draw, and returns
    a percentile (or BC) confidence interval around the observed ``m``.

    Parameters
    ----------
    result
        :class:`~neurocomplexity.analysis.BranchingResult` from the real
        data — its ``params`` are reused for every bootstrap draw.
    rec
        The recording the result was computed on.
    n
        Number of bootstrap draws (default 2000 — required for stable BCa
        tail estimates near boundaries such as m=1).
    seed
        Master RNG seed.
    ci_level
        Confidence level (default 0.95).
    block_seconds
        Block length for the moving-block bootstrap (default 10 s — long
        enough to span the autocorrelation in population activity).
    n_jobs
        Parallelism via joblib (default 1, serial).

    Returns
    -------
    :class:`~neurocomplexity.inference.results.InferenceResult`
        With ``observed = result.m``, ``bootstrap_distribution`` of length
        ``n``, and ``ci_lower`` / ``ci_upper`` populated.
    """
    _warn_if_few_blocks(rec.duration, block_seconds)
    kw = dict(result.params)

    def _one(s):
        rng = np.random.default_rng(s)
        boot = _block_resampled_recording(rec, block_seconds=block_seconds, rng=rng)
        try:
            return float(wilting_mr(boot, **kw).m)
        except Exception:
            return float("nan")

    dist = np.array(_run(_child_seeds(seed, n), _one, n_jobs))
    lo, hi = _ci_from_dist(dist, ci_level, observed=float(result.m))
    return InferenceResult(
        statistic_name="m",
        observed=float(result.m),
        null_distribution=None,
        bootstrap_distribution=dist,
        p_value=None, p_value_fdr=None, effect_size=None,
        ci_lower=float(lo), ci_upper=float(hi), ci_level=ci_level,
        method="block_bootstrap",
        n_resamples=n, seed=seed,
        metadata={"block_seconds": float(block_seconds)},
    )


def bootstrap_participation_ratio(
    result: DimensionalityResult,
    rec,
    *,
    n: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
    block_seconds: float = 1.0,
    n_jobs: int = 1,
) -> InferenceResult:
    """Block bootstrap for the participation ratio.

    See :func:`bootstrap_branching_ratio` for parameter semantics.
    ``block_seconds`` defaults to 1 s here because PR is computed on
    short-timescale (10 ms) bins and its autocorrelation is shorter.
    """
    _warn_if_few_blocks(rec.duration, block_seconds)
    kw = dict(result.params)

    def _one(s):
        rng = np.random.default_rng(s)
        boot = _block_resampled_recording(rec, block_seconds=block_seconds, rng=rng)
        try:
            return float(dimensionality(boot, **kw).participation_ratio)
        except Exception:
            return float("nan")

    dist = np.array(_run(_child_seeds(seed, n), _one, n_jobs))
    lo, hi = _ci_from_dist(dist, ci_level,
                           observed=float(result.participation_ratio))
    return InferenceResult(
        statistic_name="PR",
        observed=float(result.participation_ratio),
        null_distribution=None, bootstrap_distribution=dist,
        p_value=None, p_value_fdr=None, effect_size=None,
        ci_lower=float(lo), ci_upper=float(hi), ci_level=ci_level,
        method="block_bootstrap", n_resamples=n, seed=seed,
        metadata={"block_seconds": float(block_seconds)},
    )


def bootstrap_shape_collapse(
    result: ShapeCollapseResult,
    rec,
    *,
    n: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
    block_seconds: float = 1.0,
    n_jobs: int = 1,
) -> InferenceResult:
    """Block bootstrap for the Friedman shape-collapse exponent ``gamma``.

    See :func:`bootstrap_branching_ratio` for parameter semantics.
    """
    _warn_if_few_blocks(rec.duration, block_seconds)
    kw = dict(result.params)

    def _one(s):
        rng = np.random.default_rng(s)
        boot = _block_resampled_recording(rec, block_seconds=block_seconds, rng=rng)
        try:
            return float(shape_collapse(boot, **kw).gamma)
        except Exception:
            return float("nan")

    dist = np.array(_run(_child_seeds(seed, n), _one, n_jobs))
    lo, hi = _ci_from_dist(dist, ci_level, observed=float(result.gamma))
    return InferenceResult(
        statistic_name="gamma",
        observed=float(result.gamma),
        null_distribution=None, bootstrap_distribution=dist,
        p_value=None, p_value_fdr=None, effect_size=None,
        ci_lower=float(lo), ci_upper=float(hi), ci_level=ci_level,
        method="block_bootstrap", n_resamples=n, seed=seed,
        metadata={"block_seconds": float(block_seconds)},
    )


def bootstrap_autonomy(
    result: AutonomyResult,
    rec,
    *,
    n: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
    block_seconds: float = 1.0,
    n_jobs: int = 1,
) -> InferenceResult:
    """Block bootstrap for VAR-Granger autonomy p-values, one per population.

    See :func:`bootstrap_branching_ratio` for parameter semantics. Returns
    a vector ``observed`` with one entry per population (sorted by name),
    matching ``ci_lower`` / ``ci_upper`` arrays of the same shape.
    """
    _warn_if_few_blocks(rec.duration, block_seconds)
    kw = dict(result.params)
    keys = sorted(result.values.keys())
    obs = np.array([result.values[k] for k in keys])

    def _one(s):
        rng = np.random.default_rng(s)
        boot = _block_resampled_recording(rec, block_seconds=block_seconds, rng=rng)
        try:
            r = autonomy(boot, **kw)
            return np.array([r.values.get(k, np.nan) for k in keys])
        except Exception:
            return np.full(len(keys), np.nan)

    dist = np.stack(_run(_child_seeds(seed, n), _one, n_jobs), axis=0)
    lo, hi = _ci_from_dist(dist, ci_level, observed=obs)
    return InferenceResult(
        statistic_name="autonomy_p",
        observed=obs,
        null_distribution=None, bootstrap_distribution=dist,
        p_value=None, p_value_fdr=None, effect_size=None,
        ci_lower=lo, ci_upper=hi, ci_level=ci_level,
        method="block_bootstrap", n_resamples=n, seed=seed,
        metadata={"block_seconds": float(block_seconds), "populations": keys},
    )


_DISPATCH = {
    BranchingResult: bootstrap_branching_ratio,
    CriticalityResult: bootstrap_avalanche_exponents,
    DimensionalityResult: bootstrap_participation_ratio,
    ShapeCollapseResult: bootstrap_shape_collapse,
    AutonomyResult: bootstrap_autonomy,
}


def bootstrap(
    result, rec, *,
    n: int = 1000, seed: int = 0, ci_level: float = 0.95,
    block_seconds=None, n_jobs: int = 1,
) -> InferenceResult:
    """Bootstrap a confidence interval for an analysis result.

    Unified entry-point that dispatches on the result type to the
    appropriate per-statistic block bootstrap
    (``bootstrap_branching_ratio``, ``bootstrap_participation_ratio``,
    ``bootstrap_avalanche_exponents``, ``bootstrap_shape_collapse``,
    ``bootstrap_autonomy``).

    Parameters
    ----------
    result
        A frozen ``*Result`` dataclass from
        :mod:`neurocomplexity.analysis`.
    rec
        The recording that produced ``result``.
    n
        Number of bootstrap draws (default 1000).
    seed
        Master RNG seed.
    ci_level
        Confidence level (default 0.95).
    block_seconds
        Block length for moving-block bootstrap. If ``None``, the dispatched
        function's default is used.
    n_jobs
        Parallelism via joblib (default 1, serial).

    Returns
    -------
    :class:`~neurocomplexity.inference.results.InferenceResult`

    Raises
    ------
    TypeError
        If no bootstrap dispatcher exists for ``type(result)``.

    Notes
    -----
    Confidence intervals use the BCa (bias-corrected and accelerated)
    method when an ``observed`` is provided; otherwise fall back to the
    percentile method. BCa is essential for near-boundary estimators
    (e.g. branching ratio m -> 1) where the bootstrap distribution is
    strongly skewed and plain percentile or BC intervals under-cover.
    BCa may be unstable for vector statistics whose individual
    components have near-zero variance — inspect
    ``bootstrap_distribution`` and pre-screen those entries in your
    pipeline.
    """
    fn = _DISPATCH.get(type(result))
    if fn is None:
        raise TypeError(f"no bootstrap resampler for {type(result).__name__}")
    kw = {"n": n, "seed": seed, "ci_level": ci_level, "n_jobs": n_jobs}
    if block_seconds is not None and "block_seconds" in fn.__code__.co_varnames:
        kw["block_seconds"] = block_seconds
    return fn(result, rec, **kw)
