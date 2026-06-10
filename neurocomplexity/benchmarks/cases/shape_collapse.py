"""Shape-collapse benchmark: recover the mean-field shape exponent gamma ~ 2.0.

Independent of the criticality.exponents case: shape_collapse() uses its own
scale-invariant residual optimiser, not fit_avalanche_exponents. Drives the
trial-based Galton-Watson simulator at m=1 (mean-field), where the avalanche
shape a(t,T) ~ T^(gamma-1) F(t/T) with gamma = 2.0 (Sethna 2001; Friedman 2012).
"""
from __future__ import annotations

import time

import numpy as np

from neurocomplexity.analysis.shape_collapse import shape_collapse
from neurocomplexity.benchmarks.runner import BenchmarkResult, register
from neurocomplexity.benchmarks.simulators.branching_network import (
    trial_based_avalanches,
)


@register("shape_collapse.gamma")
def bench_gamma_collapse(n_reps: int = 50, seed: int = 0) -> BenchmarkResult:
    """Friedman shape collapse recovers mean-field gamma = 2.0 within 0.40.

    Tolerance 0.40 reflects the finite-sample bias of shape collapse on
    duration-limited avalanche ensembles (the duration tail is short at
    n_trials ~ 3000); the case is a recovery assertion, not a precision claim.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    rng_seeds = rng.integers(0, 2 ** 31 - 1, size=n_reps)
    gammas: list[float] = []
    for s in rng_seeds:
        rec = trial_based_avalanches(
            n_units=40, n_trials=3000, m=1.0, bin_ms=4.0, seed=int(s),
        )
        try:
            res = shape_collapse(rec, populations=["all"], bin_size_ms=4.0,
                                 min_duration=4, max_duration=60)
        except ValueError:
            continue
        if np.isfinite(res.gamma):
            gammas.append(float(res.gamma))
    gamma_mean = float(np.mean(gammas)) if gammas else float("nan")
    err = abs(gamma_mean - 2.0)
    tol = 0.40
    return BenchmarkResult(
        name="shape_collapse.gamma",
        observed=err,
        expected=0.0,
        tolerance=tol,
        passed=bool(np.isfinite(err) and err < tol),
        runtime_s=time.time() - t0,
        n_reps=n_reps,
        metadata={"gamma_mean": gamma_mean, "n_used": len(gammas),
                  "n_skipped": int(n_reps) - len(gammas)},
    )
