# Camera-ready regeneration checklist (2026-06-10)

**STATUS 2026-06-12: BOTH RUNS COMPLETE.**
1. Dimensionality benchmark: regenerated as 4×50-rep multi-seed chunks (seeds 0-3,
   single-threaded, sequential/resumable due to repeated machine power-offs).
   200-rep-equivalent observed = 0.6552 (seed SD 0.0061), vs carried-forward 0.6548
   — delta 0.0004, far inside tol 1.0. Merged into `results/benchmarks/v1.1.0.csv`.
   Table 1 ("0.655") and Fig 2 unchanged at display precision; no manuscript rebuild.
2. Bin-sensitivity sweep: complete on the sigma-free 37-pop funnel. Finding: raw
   exponents are NOT bin-invariant (mean tau 1.60/1.42/1.29 at 2/4/8 ms), but the
   crackling-noise *consistency* criterion is robust at/above the 4 ms default
   (37/37 at 4 ms & adaptive, 33/37 at 8 ms, 23/37 at 2 ms). Reported honestly in
   §3.3 + new bin-robustness table; the "bin-robust exponents" framing was dropped.

Original deferral note follows.

Two long runs are deferred from the scientific-rigor hardening pass. Execute both
before camera-ready submission and update the manuscript if numbers move.

## 1. Full dimensionality benchmark (~208 min)

The fast `results/benchmarks/v1.1.0.csv` regeneration ran the changed/added cases
(`shape_collapse.gamma`, `info_theory.autonomy_calibration`) and carried the
`dimensionality.pr_rank` row forward from the prior baseline unchanged (the
dimensionality code was not touched). The rigor-hardening pass ships under the
existing **1.1.0** release (no version bump), so the 12-case suite overwrites the
1.1.0 benchmark artifact in place. For the archival baseline, regenerate the
dimensionality row at full reps:

    py -3 -m neurocomplexity benchmark --case dimensionality.pr_rank --reps 200 -o results/benchmarks/_dim_reps200.csv

Merge the single row into `results/benchmarks/v1.1.0.csv`, replacing the carried-forward
value.

## 2. Steinmetz bin-sensitivity sweep (multi-hour)

Demonstrates the §3.3 exponents are robust to the bin choice (M3), not a 4-ms artifact:

    py -3 datasets/steinmetz_bin_sensitivity.py

Writes `datasets/steinmetz_bin_sensitivity.csv` (tau, alpha, gamma_fit, Sethna delta at
bins {2, 4, 8, adaptive} ms for each of the 37 crackling-consistent populations (sigma-free funnel)). Confirm
tau and alpha are stable across bins (within each population's BCa CI from
`datasets/steinmetz_table3_ci.py`); add a one-sentence robustness statement +
supplementary table to paper §3.3. If exponents move beyond their BCa CIs at any bin,
report it honestly rather than suppressing.

## Notes

- The full benchmark baseline run uses `--reps 200` (the published protocol); the
  autonomy case's Clopper-Pearson calibration is sharper at `--reps 1000` and may be
  run at that level for the autonomy row specifically.
- All deferred scripts are committed and were smoke-checked; only their long execution
  is deferred.
