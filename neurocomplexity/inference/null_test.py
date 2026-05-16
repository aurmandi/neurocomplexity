"""p-value, effect-size, and FDR utilities.

References:
  * Phipson B, Smyth GK (2010). Permutation p-values should never be zero:
    calculating exact p-values when permutations are randomly drawn.
  * Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


def pvalue_from_null(observed, null, *, alternative: str = "greater"):
    """Permutation p-value with Phipson-Smyth +1 floor.

    `null` is shape (n,) for scalar `observed`, or (n, *obs.shape) for arrays.
    """
    null = np.asarray(null)
    obs = np.asarray(observed)
    n = null.shape[0]
    if alternative == "greater":
        if obs.ndim:
            ge = np.sum(null >= obs[None, ...], axis=0)
        else:
            ge = np.sum(null >= obs, axis=0)
        return (1.0 + ge) / (1.0 + n)
    if alternative == "two-sided":
        mu = np.nanmean(null, axis=0)
        if obs.ndim:
            more = np.sum(np.abs(null - mu[None, ...]) >= np.abs(obs - mu)[None, ...], axis=0)
        else:
            more = np.sum(np.abs(null - mu) >= np.abs(obs - mu), axis=0)
        return (1.0 + more) / (1.0 + n)
    raise ValueError(f"alternative must be 'greater' or 'two-sided', got {alternative!r}")


def effect_size(observed, null):
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
    surrogate: Optional[str] = None,
    n: int = 500,
    seed: Optional[int] = None,
    pool=None,
    alternative: str = "greater",
    fdr: bool = True,
    n_jobs: int = 1,
    **surrogate_kwargs,
):
    """Surrogate-based null-distribution test for an analysis result."""
    from neurocomplexity.inference.results import InferenceResult
    from neurocomplexity.inference.pool import SurrogatePool
    from neurocomplexity.inference._adapters import adapter_for, observed_statistic

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

    if _HAS_JOBLIB and n_jobs != 1:
        nulls = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one)(i) for i in range(len(pool))
        )
    else:
        nulls = [_one(i) for i in range(len(pool))]
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
