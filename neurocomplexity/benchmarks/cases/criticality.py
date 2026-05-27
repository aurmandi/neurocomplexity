"""Criticality benchmark cases: branching-ratio m_hat recovery and avalanche exponents."""
from __future__ import annotations
import time
import numpy as np

from neurocomplexity.analysis.branching import wilting_mr
from neurocomplexity.analysis.criticality import criticality, fit_alpha
from neurocomplexity.benchmarks.simulators.branching_network import (
    branching_network, trial_based_avalanches,
)
from neurocomplexity.benchmarks.runner import BenchmarkResult, register


@register("criticality.m_hat")
def bench_m_hat(n_reps: int = 200, seed: int = 0) -> BenchmarkResult:
    """Wilting-Priesemann m_hat recovers true m within 0.03 absolute on average.

    The aggregate tolerance is set from an empirical n_reps=50 calibration
    run (mean abs err 0.024 +/- SE 0.0016 across m in {0.85, 0.90, 0.95,
    0.99}) with a platform-noise buffer; 0.03 ≈ mean + 3·SE. The estimator
    error is known to be m-dependent — per-replicate mean abs err is ~0.048
    at m=0.85 versus ~0.011 at m=0.99 (per_rep metadata exposes this for
    downstream per-m stratification).
    """
    t0 = time.time()
    ms = [0.85, 0.90, 0.95, 0.99]
    rng = np.random.default_rng(seed)
    rng_seeds = rng.integers(0, 2 ** 31 - 1, size=n_reps * len(ms))
    errors: list[float] = []
    per_rep: list[dict] = []
    idx = 0
    for m_true in ms:
        for _ in range(n_reps):
            rec = branching_network(
                n_units=60, m=m_true, duration_s=60.0, bin_ms=4.0,
                seed=int(rng_seeds[idx]),
            )
            idx += 1
            result = wilting_mr(rec, populations=["all"], bin_size_ms=4.0, k_max=50)
            m_hat = float(result.m)
            errors.append(abs(m_hat - m_true))
            per_rep.append({"m_true": m_true, "m_hat": m_hat})
    mean_abs_err = float(np.mean(errors))
    tol = 0.03
    return BenchmarkResult(
        name="criticality.m_hat",
        observed=mean_abs_err,
        expected=0.0,
        tolerance=tol,
        passed=mean_abs_err < tol,
        runtime_s=time.time() - t0,
        n_reps=n_reps,
        metadata={"ms": ms, "per_rep": per_rep},
    )


@register("criticality.exponents")
def bench_avalanche_exponents(n_reps: int = 50, seed: int = 0) -> BenchmarkResult:
    """Mean-field branching exponents alpha (size) ≈ 1.5 and tau (duration) ≈ 2.0.

    Uses the trial-based Galton-Watson simulator (independent avalanche
    trials seeded with a single spike, propagated at m=1 to extinction)
    which produces the canonical heavy-tailed size and duration
    distributions. Size and duration exponents are fit independently from
    the empirical distributions via log-binned histograms.

    Tolerance: alpha within 0.10 of 1.5 (size-tail recovery is robust);
    tau within 0.35 of 2.0 (duration-tail recovery has substantial
    finite-sample bias toward smaller values until n_trials >> 10⁴).

    References
    ----------
    Sethna et al. (2001), "Crackling noise", Nature 410, 242.
    Friedman et al. (2012), "Universal critical dynamics in high
    resolution neuronal avalanche data", PRL 108, 208102.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    rng_seeds = rng.integers(0, 2 ** 31 - 1, size=n_reps)
    alpha_ss: list[float] = []
    tau_ts: list[float] = []
    for s in rng_seeds:
        rec = trial_based_avalanches(
            n_units=40, n_trials=3000, m=1.0, bin_ms=4.0, seed=int(s),
        )
        result = criticality(rec, populations=["all"], bin_size_ms=(4.0,))
        if not np.isnan(result.alpha_s):
            alpha_ss.append(result.alpha_s)
        # Cross-check duration-tail fit (Friedman 2012 tau). Since the
        # alpha_t bug-fix this should agree with result.alpha_t up to
        # numerical noise.
        lifetimes_in_bins = result.lifetimes / result.optimal_bin_seconds
        tau = fit_alpha(lifetimes_in_bins, xmin=1)
        if not np.isnan(tau):
            tau_ts.append(tau)
    mean_alpha_s = float(np.mean(alpha_ss)) if alpha_ss else float("nan")
    mean_tau = float(np.mean(tau_ts)) if tau_ts else float("nan")
    err_s = abs(mean_alpha_s - 1.5)
    err_t = abs(mean_tau - 2.0)
    tol_s, tol_t = 0.10, 0.35
    passed = (err_s < tol_s) and (err_t < tol_t)
    return BenchmarkResult(
        name="criticality.exponents",
        observed=float(max(err_s, err_t)),
        expected=0.0,
        tolerance=max(tol_s, tol_t),
        passed=bool(passed),
        runtime_s=time.time() - t0,
        n_reps=n_reps,
        metadata={
            "alpha_s_mean": mean_alpha_s,
            "tau_mean": mean_tau,
            "err_alpha_s": err_s,
            "err_tau": err_t,
            "tol_alpha_s": tol_s,
            "tol_tau": tol_t,
        },
    )
