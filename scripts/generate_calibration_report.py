"""Generate ``docs/calibration_report.md`` from a fresh calibration run.

Re-implements the calibration suite (``tests/test_inference_calibration.py``)
as a script that emits a table of Type-I rate, power, and bootstrap
coverage. Set ``CALIBRATION_FULL=1`` to run the full counts; the default
matches the reduced CI configuration.

Usage::

    python scripts/generate_calibration_report.py
    CALIBRATION_FULL=1 python scripts/generate_calibration_report.py

Output: ``docs/calibration_report.md`` (overwritten) and a print of the
table on stdout.
"""
from __future__ import annotations

import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from neurocomplexity import __version__
from neurocomplexity.analysis.branching import wilting_mr
from neurocomplexity.analysis.transfer_entropy import transfer_entropy
from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.inference import bootstrap as inf_bootstrap
from neurocomplexity.inference import test as inf_test
from neurocomplexity._warnings import StationarityWarning, QualityControlWarning


FULL = bool(int(os.environ.get("CALIBRATION_FULL", "0")))


def _two_indep_poisson(seed: int, rate=15.0, duration=30.0) -> SpikeRecording:
    rng = np.random.default_rng(seed)
    times, uids = [], []
    for uid in range(6):
        n = rng.poisson(rate * duration)
        t = np.sort(rng.uniform(0, duration, n))
        times.append(t); uids.append(np.full(n, uid, dtype=np.int64))
    st = np.concatenate(times); ui = np.concatenate(uids)
    order = np.argsort(st)
    pops = {"A": np.array([True]*3 + [False]*3),
            "B": np.array([False]*3 + [True]*3)}
    return SpikeRecording(
        spike_times=st[order], unit_ids=ui[order],
        units=pd.DataFrame({"id": list(range(6))}),
        populations=pops, duration=duration,
        sampling_rate=None, source=None, intervals={},
    )


def _coupled_poisson(seed: int, coupling=0.5, rate=15.0, duration=30.0,
                     lag_ms=20.0) -> SpikeRecording:
    rng = np.random.default_rng(seed)
    times, uids, a_pool = [], [], []
    for uid in range(3):
        n = rng.poisson(rate * duration)
        t = np.sort(rng.uniform(0, duration, n))
        times.append(t); uids.append(np.full(n, uid, dtype=np.int64))
        a_pool.append(t)
    a_pool_arr = np.concatenate(a_pool)
    for uid in range(3, 6):
        n = rng.poisson((1 - coupling) * rate * duration)
        t_indep = rng.uniform(0, duration, n)
        n_copy = rng.binomial(a_pool_arr.size, coupling / 3.0)
        idx = rng.choice(a_pool_arr.size, n_copy, replace=False)
        t_copy = np.clip(a_pool_arr[idx] + lag_ms / 1000.0, 0, duration - 1e-9)
        t = np.sort(np.concatenate([t_indep, t_copy]))
        times.append(t); uids.append(np.full(t.size, uid, dtype=np.int64))
    st = np.concatenate(times); ui = np.concatenate(uids)
    order = np.argsort(st)
    pops = {"A": np.array([True]*3 + [False]*3),
            "B": np.array([False]*3 + [True]*3)}
    return SpikeRecording(
        spike_times=st[order], unit_ids=ui[order],
        units=pd.DataFrame({"id": list(range(6))}),
        populations=pops, duration=duration,
        sampling_rate=None, source=None, intervals={},
    )


def _branching_network(seed: int, m: float, duration=30.0, dt=0.004,
                       n_units=40, base_rate=5.0) -> SpikeRecording:
    rng = np.random.default_rng(seed)
    n_steps = int(duration / dt)
    counts = np.zeros((n_units, n_steps), dtype=np.int32)
    counts[:, 0] = rng.poisson(base_rate * dt, n_units)
    for t in range(1, n_steps):
        parent_total = counts[:, t-1].sum()
        offspring = rng.poisson(m * parent_total) if parent_total else 0
        if offspring:
            who = rng.integers(0, n_units, offspring)
            np.add.at(counts[:, t], who, 1)
        counts[:, t] += rng.poisson(base_rate * dt, n_units)
    times_list, uids_list = [], []
    for u in range(n_units):
        bins = np.flatnonzero(counts[u])
        for b in bins:
            jitter = rng.uniform(0, dt, counts[u, b])
            times_list.append(b * dt + jitter)
            uids_list.append(np.full(counts[u, b], u, dtype=np.int64))
    st = np.concatenate(times_list) if times_list else np.array([])
    ui = (np.concatenate(uids_list) if uids_list
          else np.array([], dtype=np.int64))
    order = np.argsort(st)
    return SpikeRecording(
        spike_times=st[order], unit_ids=ui[order],
        units=pd.DataFrame({"id": list(range(n_units))}),
        populations={"all": np.ones(n_units, dtype=bool)},
        duration=duration, sampling_rate=None, source=None, intervals={},
    )


def measure_typeI(n_reps: int, n_surr: int) -> tuple[float, float]:
    sig = 0
    for k in range(n_reps):
        rec = _two_indep_poisson(seed=1000 + k)
        te = transfer_entropy(rec, populations=["A", "B"],
                              bin_size_ms=20, delay_bins=1)
        inf = inf_test(te, rec, surrogate="isi_shuffle",
                       n=n_surr, seed=k, n_jobs=1, fdr=False)
        if inf.p_value[0, 1] < 0.05:
            sig += 1
    rate = sig / n_reps
    # Wald 95% CI on a binomial proportion.
    se = (rate * (1 - rate) / max(n_reps, 1)) ** 0.5
    return rate, 1.96 * se


def measure_power(n_reps: int, n_surr: int, coupling: float = 0.5) -> tuple[float, float]:
    sig = 0
    for k in range(n_reps):
        rec = _coupled_poisson(seed=2000 + k, coupling=coupling)
        te = transfer_entropy(rec, populations=["A", "B"],
                              bin_size_ms=20, delay_bins=1)
        inf = inf_test(te, rec, surrogate="isi_shuffle",
                       n=n_surr, seed=k, n_jobs=1, fdr=False)
        if inf.p_value[0, 1] < 0.05:
            sig += 1
    pwr = sig / n_reps
    se = (pwr * (1 - pwr) / max(n_reps, 1)) ** 0.5
    return pwr, 1.96 * se


def measure_coverage(m_true: float, n_reps: int, n_boot: int,
                     duration: float) -> tuple[float, float]:
    covered = 0
    for k in range(n_reps):
        rec = _branching_network(seed=3000 + k, m=m_true, duration=duration)
        r = wilting_mr(rec, bin_size_ms=4)
        inf = inf_bootstrap(r, rec, n=n_boot, seed=k,
                            block_seconds=5.0, n_jobs=1)
        if inf.ci_lower <= m_true <= inf.ci_upper:
            covered += 1
    cov = covered / n_reps
    se = (cov * (1 - cov) / max(n_reps, 1)) ** 0.5
    return cov, 1.96 * se


def main() -> int:
    print(f"neurocomplexity {__version__} — calibration report",
          "(FULL mode)" if FULL else "(reduced CI mode)")
    warnings.simplefilter("ignore", StationarityWarning)
    warnings.simplefilter("ignore", QualityControlWarning)
    warnings.simplefilter("ignore", UserWarning)

    rows = []
    t_total = time.time()

    # Type-I
    n_r, n_s = (200, 200) if FULL else (40, 100)
    t = time.time()
    rate, ci = measure_typeI(n_r, n_s)
    rows.append(("Type-I rate (TE, isi_shuffle, α=0.05)",
                 f"{rate:.3f} ± {ci:.3f}",
                 "[0.025, 0.075]" if FULL else "[0.000, 0.200]",
                 f"{n_r}/{n_s}",
                 f"{time.time() - t:.1f}"))

    # Power
    n_r, n_s = (100, 200) if FULL else (20, 100)
    t = time.time()
    pwr, ci = measure_power(n_r, n_s)
    rows.append(("Power (TE, isi_shuffle, coupling=0.5)",
                 f"{pwr:.3f} ± {ci:.3f}",
                 "≥ 0.80" if FULL else "≥ 0.60",
                 f"{n_r}/{n_s}",
                 f"{time.time() - t:.1f}"))

    # Coverage at three m_true values
    if FULL:
        n_r, n_b, dur, target = 200, 200, 120.0, "≥ 0.85"
    else:
        n_r, n_b, dur, target = 40, 100, 30.0, "≥ 0.60"
    for m_true in (0.85, 0.95, 0.99):
        t = time.time()
        cov, ci = measure_coverage(m_true, n_r, n_b, dur)
        rows.append((f"Coverage (Wilting-Priesemann m̂, m_true={m_true})",
                     f"{cov:.3f} ± {ci:.3f}",
                     target,
                     f"{n_r}/{n_b}",
                     f"{time.time() - t:.1f}"))

    total_min = (time.time() - t_total) / 60.0

    # Compose markdown
    mode = "FULL" if FULL else "reduced (CI)"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md = []
    md.append("# Calibration report — `neurocomplexity` " + __version__)
    md.append("")
    md.append(f"Generated {now} · mode = **{mode}**")
    md.append("")
    md.append("The numbers below are the empirical Type-I rate, statistical")
    md.append("power, and bootstrap coverage of the inference layer measured")
    md.append("on synthetic ground-truth recordings.")
    md.append("")
    md.append("| Metric | Measured (±1.96·SE) | Acceptance target | Reps / surrogates | Wall (s) |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    md.append("")
    md.append(f"Total wall time: **{total_min:.1f} min**")
    md.append("")
    md.append("## Reproducing this report")
    md.append("")
    md.append("```bash")
    md.append("# Reduced (matches the CI gate; ~5 min):")
    md.append("python scripts/generate_calibration_report.py")
    md.append("# Full (matches the published gate; ~30–60 min):")
    md.append("CALIBRATION_FULL=1 python scripts/generate_calibration_report.py")
    md.append("```")
    md.append("")
    md.append("Each row corresponds to a parametrised test case in")
    md.append("`tests/test_inference_calibration.py`. The script re-implements")
    md.append("the test logic so the rates can be reported (the tests")
    md.append("themselves only assert pass/fail).")
    md.append("")
    md.append("## Acceptance gates")
    md.append("")
    md.append("- **Type-I rate** must fall inside the nominal interval for the")
    md.append("  null surrogate to be valid.")
    md.append("- **Power** must clear the lower bound for the estimator to")
    md.append("  detect a true coupling at the published effect size.")
    md.append("- **Bootstrap coverage** below the lower bound means the")
    md.append("  reported CIs under-cover; see")
    md.append("  `docs/inference.md` § \"Block size guidance\" for why this")
    md.append("  worsens as `block_seconds → duration/3`.")
    md.append("")
    md.append("## CI")
    md.append("")
    md.append("`reduced` mode runs on every push as the `calibration` job in")
    md.append("`.github/workflows/test.yml`. `FULL` mode runs nightly or")
    md.append("on release-candidate tags and is enforced as a release gate.")

    out = Path("docs/calibration_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    sys_stdout = __import__("sys").stdout
    try:
        sys_stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    print(f"\nwrote {out}")
    for r in rows:
        print("  " + " | ".join(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
