# Camera-ready regeneration checklist (2026-06-10)

Two long runs are deferred from the scientific-rigor hardening pass. Execute both
before camera-ready submission and update the manuscript if numbers move.

## 1. Full dimensionality benchmark (~208 min)

The fast `results/benchmarks/v1.2.0.csv` regeneration ran the changed/added cases
(`shape_collapse.gamma`, `info_theory.autonomy_calibration`) and carried the
`dimensionality.pr_rank` row forward from v1.1.0 unchanged (the dimensionality code
was not touched). For the archival baseline, regenerate it at full reps:

    py -3 -m neurocomplexity benchmark --case dimensionality.pr_rank --reps 200 -o results/benchmarks/_dim_v120.csv

Merge the single row into `results/benchmarks/v1.2.0.csv`, replacing the carried-forward
value.

## 2. Steinmetz bin-sensitivity sweep (multi-hour)

Demonstrates the §3.3 exponents are robust to the bin choice (M3), not a 4-ms artifact:

    py -3 datasets/steinmetz_bin_sensitivity.py

Writes `datasets/steinmetz_bin_sensitivity.csv` (tau, alpha, gamma_fit, Sethna delta at
bins {2, 4, 8, adaptive} ms for each of the 27 criticality-passing populations). Confirm
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
