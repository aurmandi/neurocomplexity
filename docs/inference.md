# Inference Methods

This document explains why each surrogate and bootstrap method in
`neurocomplexity.inference` was chosen and which question it tests.

## Surrogates

| Method | Preserves | Destroys | Tests |
|---|---|---|---|
| `spike_dither` (Louis et al. 2010) | rate at scales >> delta | spike timing < delta | fine-timing coupling |
| `isi_shuffle` | per-unit ISI distribution | cross-unit timing | any cross-unit coupling vs. firing statistics |
| `interval_shuffle` | within-trial structure | trial-to-trial unit coupling | true coupling vs. stimulus drive |

All three preserve mean firing rate per unit.

## p-values

We use the Phipson & Smyth (2010) +1 correction:
`p = (1 + #{null >= obs}) / (1 + n)`. Matrix-valued statistics receive
Benjamini-Hochberg FDR correction across entries when `fdr=True`
(the default).

## Bootstrap

| Result | Resampling unit | Default block |
|---|---|---|
| `BranchingResult` | time blocks | 10.0 s |
| `CriticalityResult` (alpha_s, alpha_t) | avalanches | n/a |
| `ShapeCollapseResult` (gamma) | time blocks | 1.0 s |
| `DimensionalityResult` (PR) | time blocks | 1.0 s |
| `AutonomyResult` | time blocks | 1.0 s |

## Public API

```python
from neurocomplexity.inference import test, bootstrap, SurrogatePool

# Surrogate null test
inf = test(result, rec, surrogate="isi_shuffle", n=500, seed=0)
inf.p_value          # scalar or ndarray
inf.p_value_fdr      # FDR-corrected if input is array-shaped
inf.effect_size      # z-score vs. null
inf.null_distribution

# Bootstrap CI
inf = bootstrap(result, rec, n=500, seed=0)
inf.ci_lower, inf.ci_upper
inf.bootstrap_distribution
```

`SurrogatePool` provides a lazy, deterministic, LRU-cached pool of
surrogates that can be passed explicitly to `test()` when you want to
reuse the same null sample across multiple statistics.

## References

- Louis S, Gerstein GL, Grun S (2010). *Analysis of Parallel Spike Trains*, ch. 17.
- Phipson B, Smyth GK (2010). *Stat Appl Genet Mol Biol* 9(1).
- Benjamini Y, Hochberg Y (1995). *J R Stat Soc B* 57(1).
- Wilting J, Priesemann V (2018). *Nat Commun* 9:2325.
