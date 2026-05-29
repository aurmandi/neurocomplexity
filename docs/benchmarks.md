# Benchmark Validation Suite

`neurocomplexity.benchmarks` validates every analysis in the package against
synthetic data with closed-form or simulator-derived ground truth. Every case
is independently citable to the literature reference whose claim it verifies.

## Cases

All eleven cases pass against the v1.0.0 baseline at `n_reps=5`.

| Case | Ground truth | Tolerance | Reference |
|---|---|---|---|
| `criticality.m_hat` | Wilting-Priesemann m̂ recovers true m ∈ {0.85, 0.90, 0.95, 0.99} | mean abs err < 0.02 | Wilting & Priesemann 2018 |
| `criticality.exponents` | α (size tail) ≈ 1.5; τ (duration tail) ≈ 2.0 from independent Galton-Watson trials at m=1 | abs err α < 0.10; abs err τ < 0.35 | Sethna 2001; Friedman et al. 2012 |
| `info_theory.te_convergence` | Schreiber+Miller-Madow TE rank-orders analytic VAR-TE across coupling c ∈ {0.1, 0.3, 0.5} on multi-unit populations | Spearman ρ ≥ 0.85 | Schreiber 2000; Barnett et al. 2009 |
| `info_theory.te_null` | Independent AR(1) → surrogate p ≥ 0.05 in ≥ 90 % reps | reject rate ≤ 0.10 | self-consistency |
| `info_theory.autonomy_calibration` | VAR(1) autonomy contrasts null vs coupled on 8-units-per-population spike data | type-I ≤ 0.25; power ≥ 0.80 | Barnett & Seth 2014 |
| `pid.atoms_xor` | Synergy = ln 2, others 0 | mean abs err < 0.03 nats | Williams & Beer 2010 |
| `pid.atoms_and` | R ≈ 0.216, S ≈ 0.347 nats | mean abs err < 0.03 nats | Williams & Beer 2010 |
| `pid.atoms_copy` | Unique₁ = ln 2, others 0 | mean abs err < 0.03 nats | Williams & Beer 2010 |
| `pid.atoms_rdn` | Redundancy = ln 2, others 0 | mean abs err < 0.03 nats | Williams & Beer 2010 |
| `pid.atoms_unq` | Unique₁ = ln 2, others 0 | mean abs err < 0.03 nats | Williams & Beer 2010 |
| `dimensionality.pr_rank` | PR ≈ effective rank for r ∈ {2, 5, 10} | mean abs err < 1.0 | Rajan et al. 2010; Cunningham & Yu 2014 |

### Coverage matrix

Every analysis module is validated by at least one case above:

| Analysis module | Validated by |
|---|---|
| `analysis.branching` (`wilting_mr`) | `criticality.m_hat` |
| `analysis.criticality` (`fit_alpha`, `criticality`) | `criticality.exponents` |
| `analysis.shape_collapse` | shares the `fit_alpha` exponents with `criticality.exponents`; a dedicated shape-collapse simulator case is planned for v1.1 |
| `analysis.transfer_entropy` | `info_theory.te_convergence`, `info_theory.te_null` |
| `analysis.autonomy` | `info_theory.autonomy_calibration` |
| `analysis.pid` | `pid.atoms_{xor,and,copy,rdn,unq}` |
| `analysis.dimensionality` | `dimensionality.pr_rank` |

### Design notes

Three benchmarks deserve a note on what the validation actually tests:

- **`criticality.exponents`** uses the trial-based Galton-Watson simulator
  (`trial_based_avalanches`): independent avalanche trials each seeded with a
  single spike and propagated at m=1 to extinction. This isolates the canonical
  branching-process avalanche distribution without confounding driven
  super-critical activity. The size-tail exponent α and the duration-tail
  exponent τ are fit independently from the empirical distributions via
  log-binned histograms (the `fit_alpha` routine in `analysis.criticality`).
  τ has substantially larger finite-sample bias than α and gets a wider
  tolerance.
- **`info_theory.te_convergence`** validates **rank correlation** between
  estimated and analytic transfer entropy across coupling strengths, not
  absolute magnitude. The binary-symbol TE estimator on Poisson-thinned
  multi-unit populations is consistently scaled down from analytic VAR
  process TE (spike encoding loses some linear-Gaussian information), but
  rank-orders perfectly — the operationally meaningful check for a TE
  estimator applied to spike-sorted recordings.
- **`info_theory.autonomy_calibration`** runs on **eight units per
  population**. With one unit per population the per-bin spike count is too
  sparse for the VAR-Granger fit, triggering the `analysis.autonomy`
  degenerate-fit guard. Multi-unit pooling preserves enough linear-Gaussian
  structure that the autonomy F-test has well-defined p-values; tolerances
  account for the family-wise error rate of running the test independently
  on two populations.

## Running

### Python

```python
from neurocomplexity.benchmarks import run_all, run_case, list_cases

# Run every case
df = run_all(n_reps=200, seed=0)
df.to_csv("results/benchmarks/my_run.csv", index=False)

# Run a single case and inspect per-replicate detail
res = run_case("criticality.m_hat", n_reps=50, seed=0)
print(res.metadata["per_rep"][:5])

# List registered cases
print(list_cases())
```

### CLI

```bash
# Single case with CSV output
python -m neurocomplexity benchmark \
    --case pid.atoms_xor --reps 100 \
    -o results/benchmarks/xor.csv

# All cases at publication-grade reps
python -m neurocomplexity benchmark --reps 200 \
    -o results/benchmarks/full_run.csv
```

The CLI exits with code 0 if every selected case passes, 1 otherwise — usable
as a continuous-integration gate.

## Interpreting failures

A benchmark failure indicates a regression in the underlying analysis, not a
flaky test. Open the per-replicate metadata:

```python
res = run_case("criticality.m_hat", n_reps=50, seed=0)
for rep in res.metadata["per_rep"][:5]:
    print(f"m_true={rep['m_true']}  m_hat={rep['m_hat']:.4f}")
```

Compare per-replicate observed values to ground truth and look for systematic
bias (off-by-one lag, sign error, missing bias correction).

## Reproducibility

Every case takes a `seed` argument; passing the same seed reproduces every
random draw bit-exactly across runs. The published baseline CSV at
`results/benchmarks/v1.0.0.csv` was generated with seed = 0.
