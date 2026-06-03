"""Run the Tab 2 benchmark suite across 10 seeds to report mean +/- SD per case.

Records to outputs/benchmark_seed_sensitivity.json for paper inclusion.

Usage:
    python examples/benchmark_seed_sensitivity.py

WARNING: Takes ~20-60 min depending on host (11 cases x 200 reps x 10 seeds).
The dimensionality case dominates runtime (~84% of total).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from neurocomplexity.benchmarks.runner import list_cases, run_case

N_SEEDS = 10
N_REPS = 200


def main():
    case_names = list_cases()
    print(f"\nSeed sensitivity sweep: {len(case_names)} cases x {N_SEEDS} seeds x {N_REPS} reps")
    print("=" * 78)

    results_by_case: dict[str, list[float]] = {}
    tols: dict[str, float] = {}
    t_total = time.time()

    for seed in range(N_SEEDS):
        print(f"\n--- Seed {seed}/{N_SEEDS - 1} ---")
        for name in case_names:
            t0 = time.time()
            res = run_case(name, n_reps=N_REPS, seed=seed)
            dt = time.time() - t0
            results_by_case.setdefault(name, []).append(float(res.observed))
            tols[name] = float(res.tolerance)
            ok = "PASS" if res.passed else "FAIL"
            print(f"  {ok}  {name:<45s} obs={res.observed:.4g}  ({dt:.1f}s)")

    print(f"\nTotal wall-clock: {time.time() - t_total:.0f}s")
    print("\n" + "=" * 78)
    print(f"{'Case':<45s} {'Mean':>10s} {'SD':>10s} {'Tol':>10s} {'SD/Tol':>10s}")
    print("-" * 88)

    summary: dict[str, dict] = {}
    for name in case_names:
        vals = np.array(results_by_case[name])
        m = float(vals.mean())
        s = float(vals.std(ddof=1))
        tol = tols[name]
        ratio = s / tol if tol > 0 else float("nan")
        summary[name] = {
            "mean": m,
            "sd": s,
            "tolerance": tol,
            "sd_over_tol": ratio,
            "values": vals.tolist(),
        }
        print(f"{name:<45s} {m:>10.4g} {s:>10.4g} {tol:>10.4g} {ratio:>10.2%}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmark_seed_sensitivity.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
