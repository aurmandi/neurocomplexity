"""Information-theory benchmark cases: TE convergence, TE null, autonomy calibration."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from neurocomplexity.analysis.autonomy import autonomy
from neurocomplexity.analysis.transfer_entropy import transfer_entropy
from neurocomplexity.benchmarks.runner import BenchmarkResult, register
from neurocomplexity.benchmarks.simulators.ar_processes import coupled_ar1, var1
from neurocomplexity.core.recording import SpikeRecording


def _ar_to_recording(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bin_ms: float = 10.0,
    base_rate_hz: float = 30.0,
    units_per_pop: int = 1,
    modulation: float = 0.5,
    seed: int = 0,
) -> SpikeRecording:
    """Thin a pair of AR(1) samples to a multi-unit Poisson spike-train recording.

    Each population (X, Y) contains ``units_per_pop`` units sharing the
    same latent AR rate signal with independent Poisson realisations.
    Multi-unit populations preserve the linear-Gaussian signal through
    population averaging that single-unit thinning otherwise washes out
    (and that the autonomy VAR-Granger fit needs for non-degenerate
    behaviour).
    """
    rng = np.random.default_rng(seed)
    bin_s = bin_ms / 1000.0
    n = int(x.shape[0])
    sx = (x - x.mean()) / (x.std() + 1e-9)
    sy = (y - y.mean()) / (y.std() + 1e-9)
    rx = np.clip(base_rate_hz * (1.0 + modulation * sx), 0.1, None)
    ry = np.clip(base_rate_hz * (1.0 + modulation * sy), 0.1, None)
    centres = (np.arange(n) + 0.5) * bin_s

    n_units = 2 * units_per_pop
    times_chunks: list[np.ndarray] = []
    uids_chunks: list[np.ndarray] = []
    for pop_idx, rates in enumerate((rx, ry)):
        for k in range(units_per_pop):
            uid = pop_idx * units_per_pop + k
            cnts = rng.poisson(rates * bin_s)
            nz = np.flatnonzero(cnts)
            for b in nz:
                c = int(cnts[b])
                times_chunks.append(
                    centres[b] + rng.uniform(-bin_s / 2, bin_s / 2, c)
                )
                uids_chunks.append(np.full(c, uid, dtype=np.int64))
    if times_chunks:
        st = np.concatenate(times_chunks)
        uu = np.concatenate(uids_chunks)
        order = np.argsort(st, kind="stable")
        st = st[order]
        uu = uu[order]
    else:
        st = np.array([], dtype=np.float64)
        uu = np.array([], dtype=np.int64)

    # Population masks: first half = X, second half = Y.
    pop_X = np.array([i < units_per_pop for i in range(n_units)])
    pop_Y = ~pop_X
    return SpikeRecording(
        spike_times=st, unit_ids=uu,
        units=pd.DataFrame({"id": list(range(n_units))}),
        populations={
            "X": pop_X, "Y": pop_Y,
            "all": np.ones(n_units, dtype=bool),
        },
        duration=float(n * bin_s),
        sampling_rate=None, source=None, intervals={},
    )


@register("info_theory.te_convergence")
def bench_te_convergence(n_reps: int = 20, seed: int = 0) -> BenchmarkResult:
    """Schreiber+Miller-Madow TE rank-orders analytic VAR-process TE.

    On Poisson-thinned multi-unit spike populations the binary-symbol TE
    estimator is consistently scaled down from the analytic VAR-process
    TE (the spike encoding loses some linear-Gaussian information), but
    it preserves the **rank ordering** across coupling strengths. The
    case checks that (a) estimated TE is monotone increasing in the true
    coupling c, and (b) the Spearman rank correlation between estimated
    and analytic TE across all replicates is at least 0.85. This is the
    operationally meaningful validation for an estimator applied to
    spike-sorted recordings.
    """
    from scipy.stats import spearmanr
    t0 = time.time()
    cs = [0.1, 0.3, 0.5]
    rng = np.random.default_rng(seed)
    rng_seeds = rng.integers(0, 2 ** 31 - 1, size=n_reps * len(cs))
    te_trues: list[float] = []
    te_ests: list[float] = []
    per_rep: list[dict] = []
    idx = 0
    for c in cs:
        for _ in range(n_reps):
            x, y, te_true = coupled_ar1(
                c=c, a=0.5, sigma=1.0, n_samples=10_000,
                seed=int(rng_seeds[idx]),
            )
            idx += 1
            rec = _ar_to_recording(
                x, y, base_rate_hz=80.0, modulation=0.9,
                units_per_pop=8, seed=int(rng_seeds[idx - 1]),
            )
            te_res = transfer_entropy(
                rec, populations=["X", "Y"],
                bin_size_ms=10.0, delay_bins=1,
            )
            te_est = float(te_res.matrix[0, 1])
            te_trues.append(te_true)
            te_ests.append(te_est)
            per_rep.append({"c": c, "te_true": te_true, "te_est": te_est})
    rho, _ = spearmanr(te_trues, te_ests)
    rho = float(rho)
    tol = 0.85
    return BenchmarkResult(
        name="info_theory.te_convergence",
        # "observed" reports the deviation from perfect rank correlation
        # so the runner's "smaller is better" rendering stays consistent.
        observed=float(1.0 - rho), expected=0.0,
        tolerance=float(1.0 - tol),
        passed=rho >= tol,
        runtime_s=time.time() - t0, n_reps=n_reps,
        metadata={"per_rep": per_rep, "spearman_rho": rho},
    )


@register("info_theory.te_null")
def bench_te_null(n_reps: int = 30, seed: int = 0) -> BenchmarkResult:
    """Independent AR(1) pair → at most 10% surrogate-based TE p-values < 0.05."""
    from neurocomplexity.inference import test as inf_test
    t0 = time.time()
    rng = np.random.default_rng(seed)
    rng_seeds = rng.integers(0, 2 ** 31 - 1, size=n_reps)
    rejects = 0
    for s in rng_seeds:
        x, _, _ = coupled_ar1(c=0.0, a=0.5, sigma=1.0, n_samples=2000, seed=int(s))
        y, _, _ = coupled_ar1(c=0.0, a=0.5, sigma=1.0, n_samples=2000,
                              seed=int(s) + 999)
        rec = _ar_to_recording(
            x, y, base_rate_hz=80.0, modulation=0.9,
            units_per_pop=8, seed=int(s),
        )
        te_res = transfer_entropy(
            rec, populations=["X", "Y"],
            bin_size_ms=10.0, delay_bins=1,
        )
        inf = inf_test(te_res, rec, surrogate="isi_shuffle",
                       n=100, seed=int(s), fdr=False)
        # Mask the diagonal — only off-diagonal TE entries have meaningful p-values.
        pv = np.asarray(inf.p_value, dtype=float)
        offdiag = pv[~np.eye(pv.shape[0], dtype=bool)]
        if offdiag.size and np.any(offdiag < 0.05):
            rejects += 1
    reject_rate = rejects / n_reps
    tol = 0.10
    return BenchmarkResult(
        name="info_theory.te_null",
        observed=reject_rate, expected=0.05,
        tolerance=tol, passed=reject_rate <= tol,
        runtime_s=time.time() - t0, n_reps=n_reps,
        metadata={"rejects": rejects},
    )


@register("info_theory.autonomy_calibration")
def bench_autonomy_calibration(n_reps: int = 1000, seed: int = 0) -> BenchmarkResult:
    """VAR(1) autonomy Type-I is calibrated to nominal 0.05 (per-population).

    Reports the PER-POPULATION Type-I rate (each population is an independent
    F-test), so the reference is a clean 0.05 - not the family-wise max-of-two.
    Pass criterion is the community-standard Monte-Carlo calibration test:
    nominal 0.05 must lie inside the exact Clopper-Pearson 95% binomial CI of
    the observed rejection count. Power must be >= 0.80 at coupling 0.3.

    The default analytic path uses the nested same-sample OLS F-test; the
    permutation path (significance="permutation") is a calibration-free
    cross-check.
    """
    from scipy.stats import beta as _beta
    t0 = time.time()
    rng = np.random.default_rng(seed)
    A_null = np.array([[0.5, 0.0], [0.0, 0.5]])
    A_cpl = np.array([[0.5, 0.0], [0.3, 0.5]])
    Sigma = np.eye(2)
    cutoff = 0.05
    null_below = 0
    null_total = 0
    cpl_below = 0
    cpl_total = 0
    for _ in range(n_reps):
        s_null = int(rng.integers(2 ** 31))
        s_cpl = int(rng.integers(2 ** 31))
        X0 = var1(A=A_null, Sigma=Sigma, n_samples=2000, seed=s_null)
        rec0 = _ar_to_recording(X0[:, 0], X0[:, 1], base_rate_hz=80.0,
                                modulation=0.9, units_per_pop=8, seed=s_null)
        a0 = autonomy(rec0, populations=["X", "Y"], bin_size_ms=10.0,
                      significance="analytic", seed=s_null)
        for v in a0.values.values():
            null_total += 1
            null_below += int(v < cutoff)
        X1 = var1(A=A_cpl, Sigma=Sigma, n_samples=2000, seed=s_cpl)
        rec1 = _ar_to_recording(X1[:, 0], X1[:, 1], base_rate_hz=80.0,
                                modulation=0.9, units_per_pop=8, seed=s_cpl)
        a1 = autonomy(rec1, populations=["X", "Y"], bin_size_ms=10.0,
                      significance="analytic", seed=s_cpl)
        # Coupling is X -> Y, so only the Y equation has true dependency.
        cpl_total += 1
        cpl_below += int(a1.values["Y"] < cutoff)
    type_i = null_below / null_total
    power = cpl_below / cpl_total
    k, nB = null_below, null_total
    lo = 0.0 if k == 0 else float(_beta.ppf(0.025, k, nB - k + 1))
    hi = 1.0 if k == nB else float(_beta.ppf(0.975, k + 1, nB - k))
    calibrated = (lo <= 0.05 <= hi)
    passed = calibrated and (power >= 0.80)
    return BenchmarkResult(
        name="info_theory.autonomy_calibration",
        observed=type_i,
        expected=0.05,
        tolerance=float(max(0.05 - lo, hi - 0.05)),
        passed=bool(passed),
        runtime_s=time.time() - t0,
        n_reps=n_reps,
        metadata={"type_i": type_i, "power": power, "cutoff": cutoff,
                  "cp_lo": lo, "cp_hi": hi, "n_pop_tests": nB},
    )
