# Calibration report — `neurocomplexity` 1.1.0

Generated 2026-05-29T01:13:15+00:00 · mode = **reduced (CI)**

The numbers below are the empirical Type-I rate, statistical
power, and bootstrap coverage of the inference layer measured
on synthetic ground-truth recordings.

| Metric | Measured (±1.96·SE) | Acceptance target | Reps / surrogates | Wall (s) |
|---|---|---|---|---|
| Type-I rate (TE, isi_shuffle, α=0.05) | 0.000 ± 0.000 | [0.000, 0.200] | 40/100 | 4.4 |
| Power (TE, isi_shuffle, coupling=0.5) | 1.000 ± 0.000 | ≥ 0.60 | 20/100 | 2.0 |
| Coverage (Wilting-Priesemann m̂, m_true=0.85) | 0.700 ± 0.142 | ≥ 0.60 | 40/100 | 26.0 |
| Coverage (Wilting-Priesemann m̂, m_true=0.95) | 0.875 ± 0.102 | ≥ 0.60 | 40/100 | 58.8 |
| Coverage (Wilting-Priesemann m̂, m_true=0.99) | 0.650 ± 0.148 | ≥ 0.60 | 40/100 | 196.9 |

Total wall time: **4.8 min**

## Reproducing this report

```bash
# Reduced (matches the CI gate; ~5 min):
python scripts/generate_calibration_report.py
# Full (matches the published gate; ~30–60 min):
CALIBRATION_FULL=1 python scripts/generate_calibration_report.py
```

Each row corresponds to a parametrised test case in
`tests/test_inference_calibration.py`. The script re-implements
the test logic so the rates can be reported (the tests
themselves only assert pass/fail).

## Acceptance gates

- **Type-I rate** must fall inside the nominal interval for the
  null surrogate to be valid.
- **Power** must clear the lower bound for the estimator to
  detect a true coupling at the published effect size.
- **Bootstrap coverage** below the lower bound means the
  reported CIs under-cover; see
  `docs/inference.md` § "Block size guidance" for why this
  worsens as `block_seconds → duration/3`.

## CI

`reduced` mode runs on every push as the `calibration` job in
`.github/workflows/test.yml`. `FULL` mode runs nightly or
on release-candidate tags and is enforced as a release gate.
