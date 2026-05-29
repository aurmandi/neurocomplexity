# Complexity measures: LMC vs MSE

`neurocomplexity` ships two scalar complexity measures for spike-train
activity. They answer **different questions** and are not interchangeable.
This page explains when to use which.

## TL;DR

| You want to characterise…                                          | Use                 |
|--------------------------------------------------------------------|---------------------|
| Shape of the population-rate distribution at one timescale         | `lmc_complexity`    |
| Temporal multiscale structure (does the signal stay irregular as you coarse-grain?) | `multiscale_entropy` |
| Both — they are complementary, not redundant                       | run both            |

If you have to pick one and your readers come from systems neuroscience,
**`multiscale_entropy` is the more commonly cited measure** (Costa 2002 is the
standard reference, widely used in EEG/MEG/HRV and increasingly in spike-based
work). LMC is theoretically principled but appears less often in the
modern systems-neuro literature.

## `lmc_complexity` — López-Ruiz statistical complexity

**Reference:** López-Ruiz, Mancini, Calbet. *Phys. Lett. A* **209** (1995)
321–326.

For a discrete probability distribution `p` over `N` states:

```
H = -sum p_i log p_i / log N        # normalised Shannon entropy in [0, 1]
D = sum (p_i - 1/N)^2                # LMC disequilibrium (distance from uniform)
C = H * D                            # statistical complexity
```

LMC complexity peaks at **intermediate entropy** — structured but not trivial.
Uniform noise has `H=1, D=0, C=0`; a delta distribution has `H=0, D` large but
`C=0`. Distributions that are neither flat nor concentrated maximise `C`.

**What it measures:** the *shape* of the distribution of population spike
counts at the chosen bin size. One number per population. Optionally a
sliding-window trajectory `H(t), D(t), C(t)`.

**Use when:** you want a single timescale-fixed summary of "how
non-trivial is the distribution of activity levels", e.g. comparing
distributions across conditions, brain states, drug effects.

**Limitations:** single-scale by construction — it cannot distinguish a signal
that is irregular at short timescales but periodic at long timescales from
one that is irregular at every scale. For that, use MSE.

## `multiscale_entropy` — Costa MSE

**References:** Costa, Goldberger, Peng. *Phys. Rev. Lett.* **89** (2002)
068102. Richman & Moorman. *Am. J. Physiol. Heart Circ. Physiol.* **278**
(2000) H2039.

Coarse-grain the population-rate series at integer scales τ = 1…τ_max
(non-overlapping means), then compute the sample entropy of each
coarse-grained series with a fixed tolerance `r = r_factor · SD(original)`.

```
sampen(τ) = -log P(|x_{i+m} - x_{j+m}| < r  |  |x_{i+k} - x_{j+k}| < r for k=0..m-1)
```

The output is a **curve** `sampen(τ)` per population, not a scalar.

**What it measures:** how much irregularity persists in the signal as you
look at coarser and coarser time bins. A purely Poissonian signal becomes
*more* regular under coarse-graining (sampen falls). A signal with structure
at multiple timescales stays irregular (sampen flat or rising).

**Use when:** you want to detect *multi-scale temporal complexity* — the
hallmark of a system poised near criticality, with long-range temporal
correlations. Standard in EEG/MEG/HRV literature; increasingly used for
spike-count time series.

**Limitations:** requires enough data for sample entropy to be well-estimated
at the longest scale (rule of thumb: N ≥ 10^(m+1) at scale τ_max).
Sensitive to the choice of `m` (default 2) and `r_factor` (default 0.2·SD,
the Pincus 1991 / Richman & Moorman 2000 convention; Costa et al. 2002
reported MSE curves with r = 0.15·SD on physiologic time series — pass
`r_factor=0.15` to reproduce the Costa choice).

## When the two disagree

- LMC says "complex", MSE flat: the distribution shape is non-trivial at
  the chosen bin size, but there is no multiscale temporal structure
  (e.g. quasi-stationary mixture of two rate states).
- LMC says "trivial", MSE rising with τ: the marginal distribution is
  uninformative but the temporal correlations carry the signal
  (e.g. broadband noise with slow autocorrelation).

Both situations are scientifically interesting and report different facts.
This is why we expose both rather than picking one.

## Williams-Beer I_min limitation (PID redundancy)

`nc.analysis.partial_information` uses the Williams & Beer (2010) `I_min`
redundancy functional. `I_min` is the *only* closed-form redundancy on
three variables for which the PID identities admit non-negative atoms in
the general discrete case, which is why it is the default. It has one
well-known failure mode that users must understand before reading the
output:

- `I_min` measures redundancy as the *minimum* specific information any
  single source carries about each target outcome and then aggregates
  across outcomes. When the two sources carry information about *different*
  target outcomes — i.e. they are non-overlapping channels — `I_min`
  still reports a positive redundancy because each source non-trivially
  reduces uncertainty about the *outcomes it is informative about*.
- Consequently, **`PIDResult.redundancy` is an upper bound on the true
  shared information**, and `PIDResult.synergy` (derived as the residual
  via the PID identities) is correspondingly biased *downward*.
- This is a property of `I_min`, not of this implementation. Bertschinger
  et al. (2014, *Entropy* 16, 2161) proposed the BROJA decomposition,
  which fixes the issue by optimising over the cone of distributions
  preserving the source-marginal-to-target conditional distributions.
  BROJA is convex but solved numerically; it is on the package roadmap.

**Practical guidance.** If the two source populations plausibly encode
different aspects of the target — common in cortex, where neighbouring
areas often have orthogonal receptive fields — interpret a high
`redundancy` as the *largest* it could be and the corresponding `synergy`
as the *smallest* it could be. Re-run with the populations swapped as a
sanity check; the four atoms are symmetric in the two sources by
construction.

## See also

- `dimensionality.participation_ratio` — geometric complexity (effective
  dimensionality of the per-unit correlation matrix).
- `analysis.manifold` — population-state geometry (PCA / UMAP / t-SNE).
- `criticality` and `wilting_mr` — diverging quantities at a phase transition,
  complementary to the descriptive complexity measures here.
