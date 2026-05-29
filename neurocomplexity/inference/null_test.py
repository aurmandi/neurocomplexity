"""p-value, effect-size, and FDR utilities.

References:
  * Phipson B, Smyth GK (2010). Permutation p-values should never be zero:
    calculating exact p-values when permutations are randomly drawn.
  * Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate.
"""
from __future__ import annotations

import numpy as np

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


def pvalue_from_null(observed, null, *, alternative: str = "greater"):
    """Permutation p-value with Phipson-Smyth +1 floor.

    Parameters
    ----------
    observed
        Scalar or array; matches the trailing shape of ``null``.
    null
        Shape (n,) for scalar ``observed`` or (n, *obs.shape) for array.
    alternative
        ``"greater"`` (default) — right-tail.
        ``"less"`` — left-tail.
        ``"two-sided"`` — robust to skewed nulls: ``2 * min(p_greater, p_less)``
        with each tail floored at ``1/(n+1)`` (Phipson & Smyth 2010) and the
        final value clipped to 1. Avoids the mean-centred ``|null - mu|``
        formulation which under-powers for asymmetric null distributions.
    """
    null = np.asarray(null)
    obs = np.asarray(observed)
    n = null.shape[0]

    def _p_greater():
        if obs.ndim:
            ge = np.sum(null >= obs[None, ...], axis=0)
        else:
            ge = np.sum(null >= obs, axis=0)
        return (1.0 + ge) / (1.0 + n)

    def _p_less():
        if obs.ndim:
            le = np.sum(null <= obs[None, ...], axis=0)
        else:
            le = np.sum(null <= obs, axis=0)
        return (1.0 + le) / (1.0 + n)

    if alternative == "greater":
        return _p_greater()
    if alternative == "less":
        return _p_less()
    if alternative == "two-sided":
        pg = _p_greater()
        pl = _p_less()
        return np.minimum(1.0, 2.0 * np.minimum(pg, pl))
    raise ValueError(
        f"alternative must be 'greater', 'less', or 'two-sided', got {alternative!r}"
    )


def effect_size(observed, null):
    """Standardised effect size of ``observed`` against the null distribution.

    Returns ``z = (observed - mean(null)) / std(null)`` (sample SD,
    ``ddof=1``) element-wise. Entries where ``std(null) == 0`` are
    returned as ``nan`` rather than ``inf``.

    Parameters
    ----------
    observed
        Scalar or array.
    null
        Null distribution. Shape ``(n,)`` for scalar ``observed`` or
        ``(n, *observed.shape)`` for array.

    Returns
    -------
    float or ndarray
        Same shape as ``observed``.
    """
    null = np.asarray(null)
    mu = np.nanmean(null, axis=0)
    sd = np.nanstd(null, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (observed - mu) / sd
    return np.where(sd > 0, z, np.nan)


def fdr_bh(p):
    """Benjamini-Hochberg FDR correction. Flattens, corrects, reshapes."""
    p = np.asarray(p, dtype=float)
    shape = p.shape
    flat = p.ravel()
    m = flat.size
    order = np.argsort(flat)
    ranked = flat[order]
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(flat)
    out[order] = np.clip(q, 0.0, 1.0)
    return out.reshape(shape)


_STAT_NAMES = {
    "TransferEntropyResult": "TE",
    "PIDResult": "PID",
    "BranchingResult": "m",
    "CriticalityResult": "avalanche_exponents",
    "ShapeCollapseResult": "gamma",
    "DimensionalityResult": "PR",
    "AutonomyResult": "autonomy_p",
}


def _statistic_name(result) -> str:
    return _STAT_NAMES.get(type(result).__name__, type(result).__name__)


def test(
    result,
    rec,
    *,
    surrogate: str | None = None,
    n: int = 500,
    seed: int | None = None,
    pool=None,
    alternative: str = "greater",
    fdr: bool = True,
    n_jobs: int = 1,
    **surrogate_kwargs,
):
    """Surrogate-based null-distribution test for an analysis result."""
    from neurocomplexity.inference._adapters import adapter_for, observed_statistic
    from neurocomplexity.inference.pool import SurrogatePool
    from neurocomplexity.inference.results import InferenceResult

    if pool is not None and surrogate is not None:
        raise ValueError("pass either `pool` OR (`surrogate`, `n`, `seed`), not both")
    if pool is None:
        if surrogate is None:
            raise ValueError("must provide `surrogate` if no `pool`")
        if seed is None:
            raise ValueError("`seed` is required for reproducibility")
        pool = SurrogatePool(rec, surrogate=surrogate, n=n, seed=seed,
                             **surrogate_kwargs)

    stat = adapter_for(result)
    obs = observed_statistic(result)
    obs_arr = np.asarray(obs)

    def _one(i):
        return np.asarray(stat(pool[i]))

    from neurocomplexity._progress import progress_iter
    if _HAS_JOBLIB and n_jobs != 1:
        nulls = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one)(i) for i in progress_iter(
                range(len(pool)), total=len(pool), desc="null replicates")
        )
    else:
        nulls = [_one(i) for i in progress_iter(
            range(len(pool)), total=len(pool), desc="null replicates")]
    null = np.stack(nulls, axis=0)

    p = pvalue_from_null(obs_arr, null, alternative=alternative)
    es = effect_size(obs_arr, null)
    p_fdr = fdr_bh(p) if (fdr and obs_arr.ndim >= 1) else None

    return InferenceResult(
        statistic_name=_statistic_name(result),
        observed=obs,
        null_distribution=null,
        bootstrap_distribution=None,
        p_value=p if obs_arr.ndim else float(p),
        p_value_fdr=p_fdr,
        effect_size=es if obs_arr.ndim else (float(es) if not np.isnan(es) else float("nan")),
        ci_lower=None, ci_upper=None, ci_level=0.95,
        method=pool.method,
        n_resamples=pool.n,
        seed=pool.seed,
        metadata={**pool.metadata, "alternative": alternative},
    )
