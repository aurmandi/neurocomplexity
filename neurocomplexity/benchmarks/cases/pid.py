"""Canonical PID atom decompositions for Williams-Beer I_min validation.

The five canonical bivariate distributions (XOR / AND / COPY / RDN / UNQ)
have closed-form I_min PID atoms. Each case here samples one of these
distributions, runs ``partial_information``, and checks that the four
atoms (redundancy, unique_1, unique_2, synergy) match the ground-truth
values within tolerance.

Reference
---------
Williams PL, Beer RD (2010). "Nonnegative decomposition of multivariate
information." arXiv:1004.2515. Table 2.
"""
from __future__ import annotations
import time
import numpy as np

from neurocomplexity.analysis.pid import partial_information
from neurocomplexity.benchmarks.simulators.pid_distributions import pid_recording
from neurocomplexity.benchmarks.runner import BenchmarkResult, register


# Ground-truth atom values in NATS (natural-log units, matching the analysis
# output). Williams-Beer (2010) Table 2 reports atoms in bits; multiply by
# ln(2) ≈ 0.693 to convert to nats. The simulator's "copy" distribution is
# t = s1 with s1, s2 independent (same as "unq"); it does not implement the
# classical Williams-Beer joint-copy t = (s1, s2). For AND the closed-form
# bit-values (R, U1, U2, S) ≈ (0.311, 0, 0, 0.500) translate to nats
# (0.216, 0, 0, 0.347).
_LN2 = float(np.log(2.0))
_EXPECTED: dict[str, dict[str, float]] = {
    "xor":  {"redundancy": 0.0,        "unique_1": 0.0,  "unique_2": 0.0, "synergy": _LN2},
    "and":  {"redundancy": 0.216,      "unique_1": 0.0,  "unique_2": 0.0, "synergy": 0.347},
    "copy": {"redundancy": 0.0,        "unique_1": _LN2, "unique_2": 0.0, "synergy": 0.0},
    "rdn":  {"redundancy": _LN2,       "unique_1": 0.0,  "unique_2": 0.0, "synergy": 0.0},
    "unq":  {"redundancy": 0.0,        "unique_1": _LN2, "unique_2": 0.0, "synergy": 0.0},
}


def _run_one(distribution: str, n_reps: int, seed: int) -> BenchmarkResult:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    rng_seeds = rng.integers(0, 2 ** 31 - 1, size=n_reps)
    errors: list[float] = []
    per_rep: list[dict] = []
    for s in rng_seeds:
        rec = pid_recording(distribution, n_bins=20_000, bin_ms=10.0, seed=int(s))
        res = partial_information(
            rec, target_pop="target", sources=["source_1", "source_2"],
            bin_size_ms=10.0, delay_bins=0, n_levels=3,
        )
        exp = _EXPECTED[distribution]
        rep_errs = {
            "redundancy": abs(res.redundancy - exp["redundancy"]),
            "unique_1":   abs(res.unique_1   - exp["unique_1"]),
            "unique_2":   abs(res.unique_2   - exp["unique_2"]),
            "synergy":    abs(res.synergy    - exp["synergy"]),
        }
        errors.append(max(rep_errs.values()))
        per_rep.append({
            "distribution": distribution,
            **rep_errs,
            "observed": {k: getattr(res, k) for k in exp},
        })
    max_err = float(np.mean(errors))
    tol = 0.10
    return BenchmarkResult(
        name=f"pid.atoms_{distribution}",
        observed=max_err, expected=0.0, tolerance=tol,
        passed=max_err < tol,
        runtime_s=time.time() - t0, n_reps=n_reps,
        metadata={"per_rep": per_rep, "expected_atoms": _EXPECTED[distribution]},
    )


@register("pid.atoms_xor")
def bench_atoms_xor(n_reps: int = 50, seed: int = 0) -> BenchmarkResult:
    return _run_one("xor", n_reps, seed)


@register("pid.atoms_and")
def bench_atoms_and(n_reps: int = 50, seed: int = 0) -> BenchmarkResult:
    return _run_one("and", n_reps, seed)


@register("pid.atoms_copy")
def bench_atoms_copy(n_reps: int = 50, seed: int = 0) -> BenchmarkResult:
    return _run_one("copy", n_reps, seed)


@register("pid.atoms_rdn")
def bench_atoms_rdn(n_reps: int = 50, seed: int = 0) -> BenchmarkResult:
    return _run_one("rdn", n_reps, seed)


@register("pid.atoms_unq")
def bench_atoms_unq(n_reps: int = 50, seed: int = 0) -> BenchmarkResult:
    return _run_one("unq", n_reps, seed)
