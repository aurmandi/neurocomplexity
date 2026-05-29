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

### Choose your null

| Statistic | Recommended surrogate | Why |
|---|---|---|
| `transfer_entropy` | `isi_shuffle` | Destroys cross-unit timing, preserves each ISI distribution exactly. `spike_dither` preserves only approximate rates and is too soft a null for connectivity — calibration test confirms this. |
| `partial_information` (Williams-Beer) | `isi_shuffle` | Same reasoning — joint distribution depends on cross-unit timing, not just rate. |
| `autonomy` (Granger self-predictability) | `spike_dither` *or* `isi_shuffle` | Either works; dither is faster, ISI is stricter. |
| `criticality` (α, κ, γ) | bootstrap (avalanche resample), not surrogate | These are summary statistics of the avalanche size/lifetime distribution; the question is "how much does the exponent fluctuate over avalanches", which is a bootstrap problem. |
| `wilting_mr.m` | block bootstrap | Block size set by Politis-Romano rule of thumb; see "Bootstrap" below. |
| `shape_collapse.gamma` | block bootstrap | As above. |
| `dimensionality` (PR) | block bootstrap | As above. |
| Trial-aligned statistic (any) | `interval_shuffle` | Preserves the trial structure exactly; tests "is the cross-unit coupling above what trial averaging alone explains?". Requires non-overlapping intervals on the recording. |

## p-values

We use the Phipson & Smyth (2010) +1 correction:
`p = (1 + #{null >= obs}) / (1 + n)`. Matrix-valued statistics receive
Benjamini-Hochberg FDR correction across entries when `fdr=True`
(the default).

### FDR family

For a matrix-valued statistic (e.g. the `(P, P)` transfer-entropy matrix)
the **family** of hypotheses over which BH-FDR controls the false-discovery
rate must be stated explicitly — it changes which entries survive. `test()`
takes a `family=` argument:

| `family`        | Hypotheses corrected together | Use when |
|-----------------|-------------------------------|----------|
| `"global"` (default) | all `P*P` entries as one family | you make one omnibus claim over the whole matrix |
| `"per_row"`     | each row independently (`P` families of `P`) | each row is a separate question — "of all targets, which does source *i* drive?" |
| `"per_column"`  | each column independently | "of all sources, which drives target *j*?" |

The chosen family is recorded in `inf.metadata["fdr_family"]`. Scalar and
1-D statistics are always corrected globally and ignore `family`. Pick the
family **before** looking at the matrix and report it; switching family
after inspecting the result is a forking path. The implementation is
`neurocomplexity.inference.null_test.fdr_bh_family`.

### Alternatives

`pvalue_from_null(..., alternative=...)` accepts:

- `"greater"` (default) — right-tail, ``(1 + #{null >= obs}) / (1 + n)``.
- `"less"` — left-tail, ``(1 + #{null <= obs}) / (1 + n)``.
- `"two-sided"` — `2 * min(p_greater, p_less)` clipped at 1.

The two-sided form is robust to skewed null distributions; an older
mean-centred `|null - mean| >= |obs - mean|` formulation under-powers when
the null is asymmetric and is no longer used.

### Interval overlap safety

`interval_shuffle` will raise a `ValueError` if any two intervals on the
named table overlap (tolerance: 1 µs). Touching intervals
(`stop[i] == start[i+1]`) are allowed. This guards against a silent
corruption mode where a spike inside two overlapping windows would be
reassigned twice.

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

## Block size guidance

The block bootstrap (Politis & Romano 1994) draws non-overlapping time
blocks of length `block_seconds` with replacement. The number of *unique*
blocks is ⌈duration / block_seconds⌉. Coverage degrades sharply when this
count is small:

| `duration / block_seconds` | `block_seconds` for 60 s recording | Recommended? |
|---|---|---|
| < 4 | > 15 s | **No** — `nc.inference.bootstrap` emits `UserWarning` |
| 4 – 20 | 3 – 15 s | acceptable; widen `n` |
| 20 – 100 | 0.6 – 3 s | typical sweet spot |
| > 100 | < 0.6 s | risk: blocks shorter than the autocorrelation time, CIs too narrow |

The block size must be ≥ the population autocorrelation time
(≈ 5–50 ms for cortical spike trains binned at 1–10 ms). A safe heuristic
is `block_seconds ≈ duration / 20` with a floor at 5× the bin size.

When `block_seconds > duration / 4`, `nc.inference.bootstrap` emits a
`UserWarning` ("only N unique block(s); CI may under-cover"). Either
shorten the block or supply more data — do not silence the warning to
publish under-covering intervals.

## Subsampling-robustness scope

This package corrects for subsampling bias **only in one place**:
`nc.analysis.wilting_mr`. The Wilting & Priesemann (2018) multi-step
regression recovers the population branching ratio `m` from a recorded
subset of units because the slope of `log r_k` against `k` is invariant
to the multiplicative subsampling bias on each individual lagged
autocorrelation.

**No other estimator in the package is subsampling-corrected.** The
following statistics are computed *on the observed units only* and the
reported value is biased relative to the population value by the
subsample fraction:

| Estimator | Subsampling-corrected? |
|---|---|
| `nc.analysis.wilting_mr` | **yes** (Wilting-Priesemann) |
| `nc.analysis.criticality` (α_s, α_t, κ, γ_fit) | no |
| `nc.analysis.shape_collapse` | no |
| `nc.analysis.dimensionality` (participation ratio) | no |
| `nc.analysis.transfer_entropy` | no |
| `nc.analysis.partial_information` (Williams-Beer) | no |
| `nc.analysis.multiscale_entropy` | no |
| `nc.analysis.lmc_complexity` | no |
| `nc.analysis.autonomy` | no |

When reporting any quantity other than `wilting_mr.m`, state the unit
subsampling fraction (recorded units / population), and acknowledge it as
a source of bias. In practice the bias direction is statistic-specific —
exponents `α_s`, `α_t`, `γ_fit` are increasingly subsample-invariant only
in the strict scaling regime, and participation ratio under-estimates
the population dimensionality when units are correlated.

## References

- Louis S, Gerstein GL, Grun S (2010). *Analysis of Parallel Spike Trains*, ch. 17.
- Phipson B, Smyth GK (2010). *Stat Appl Genet Mol Biol* 9(1).
- Benjamini Y, Hochberg Y (1995). *J R Stat Soc B* 57(1).
- Wilting J, Priesemann V (2018). *Nat Commun* 9:2325.
