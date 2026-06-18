# Inference figures

`neurocomplexity.viz` provides four canonical figures for visualising the
output of `neurocomplexity.inference`. All four take an `InferenceResult`,
return a `matplotlib.figure.Figure`, and accept the standard
`palette=`, `panel_label=`, `figsize=`, `ax=` keyword arguments. Save with
`viz.save_publication(fig, path)` for the SVG + TIFF + JPG triplet.

| Figure | What it shows | When to use | InferenceResult fields it needs |
| --- | --- | --- | --- |
| `figure_bootstrap` | Histogram of bootstrap replicates with observed value + shaded confidence interval | One scalar statistic with a CI (`m̂`, `α_s`, PR, γ, autonomy) | `bootstrap_distribution`, `observed`, `ci_lower`, `ci_upper` |
| `figure_null_test` | Histogram of the null/permutation distribution with observed marker + two-sided rejection region + `p` annotation | One scalar statistic tested against a surrogate null (is `m̂ ≠ random?`) | `null_distribution`, `observed`, `p_value`, optionally `p_value_fdr`, `effect_size` |
| `figure_significance_matrix` | Effect-size heatmap with `*`/`**`/`***` markers per cell from `p_FDR` | Pairwise statistics: TE between every pop pair, PID atoms across regions, FC matrices | 2D `effect_size` (or `observed`) and 2D `p_value_fdr` (or `p_value`) |

---

## 1. `figure_bootstrap`

**What it is.** A histogram of the bootstrap distribution of a statistic
`θ̂`, with the point estimate drawn as a vertical line and the
(typically 95 %) confidence interval drawn as a shaded band underneath.

**What it does.** Shows *the sampling variability* of a statistic obtained
by resampling the data (or trials, or blocks of spikes) with replacement.
The width of the histogram is the bootstrap standard error; the shaded
band is the empirical CI.

**Why it is available for our data.** Every analysis in
`neurocomplexity.analysis` that returns a scalar (branching ratio,
avalanche exponents, participation ratio, shape-collapse γ, autonomy index)
is wrapped by a corresponding `bootstrap_*` function in
`neurocomplexity.inference.bootstrap` that uses **block bootstrap** (Künsch
1989; Politis & Romano 1994) on the spike train — preserving the
short-timescale autocorrelation that ordinary bootstrap would destroy.

**Reference figure form.** Efron & Tibshirani (1993)
*An Introduction to the Bootstrap*, Fig 6.3; Pernet, Wilcox & Rousselet
(2012) *Frontiers in Psychology* (the panel used to report robust-statistic
CIs throughout cognitive neuroscience); every modern Neuropixels paper
that reports an inferential CI uses this layout.

---

## 2. `figure_null_test`

**What it is.** A histogram of a statistic computed on *surrogate* data
that share some property of the observed data but break the structure
being tested. The observed value is overlaid as a vertical line; the
two-sided rejection region (default α = 0.05) is shaded.

**What it does.** Answers the question *"is the observed value
distinguishable from chance under the chosen null?"* The annotated
`p`-value is the proportion of surrogates more extreme than the observed
statistic. `p_FDR` is reported separately whenever the call participated
in a multi-test family.

**Why it is available for our data.** `neurocomplexity.inference.test`
calls one of the surrogate generators in
`neurocomplexity.analysis.surrogates` — jitter (preserves rate, breaks
fine timing), ISI shuffle (preserves ISI distribution, breaks
cross-neuron timing), or trial shuffle — to build a null distribution
appropriate for each statistic. The choice is documented in
`docs/inference.md`.

**Reference figure form.** Nichols & Holmes (2002) *Human Brain Mapping*
"Nonparametric permutation tests for functional neuroimaging" Fig. 1 —
the canonical "null distribution + observed marker + critical value"
panel; Maris & Oostenveld (2007) *J. Neurosci. Methods* — the same form
adopted as the field standard for MEG/EEG; Storey & Tibshirani (2003)
*PNAS* for FDR reporting conventions.

*Note on Beggs & Plenz 2003.* Their Figure 2 panels overlay two
histograms (data + shuffled surrogates) on log–log axes — a different
panel form than the single-null-plus-observed convention we follow.
They remain the foundational reference for **the shuffle null itself**
on neuronal avalanches, just not for this particular figure layout.

**Default `alternative`.** One-sided "greater" — the convention for the
non-negative neuroscience statistics this package targets (TE, branching
ratio, autonomy). `inference.test(..., alternative="greater")` stores
the alternative in `result.metadata["alternative"]`, which the figure
reads automatically. Pass `alternative="two-sided"` explicitly when the
statistic supports both tails (e.g. signed correlations).

**`alpha` is explicit.** The shaded rejection region is anchored at the
explicit `alpha=` kwarg (default 0.05), **not** at `1 − result.ci_level`.
`ci_level` belongs to bootstrap CIs; conflating the two would be a
type error.

---

## 3. `figure_significance_matrix`

**What it is.** A heatmap of an `n × n` effect-size matrix (diverging
red↔white↔blue colormap, palette-independent), overlaid with per-cell
significance markers from the FDR-adjusted p-value matrix:
`*` for `p < α`, `**` for `p < 0.01`, `***` for `p < 0.001`. The
diagonal is masked for square self-referential matrices.

**What it does.** Compresses an entire pairwise-statistics table into one
panel that lets a reader see *both* the effect magnitude *and* its
significance at a glance — without having to cross-reference a separate
p-value table.

**Why it is available for our data.** Several analyses in
`neurocomplexity.analysis` return statistics across *pairs* of
populations rather than a single scalar:

- **Transfer entropy** (`transfer_entropy`) — directed information flow
  for every (source, target) pair, returning an `n_pop × n_pop` matrix.
- **Partial information decomposition** (`partial_information`) — atoms
  (R, U_X, U_Y, S) for every (source₁, source₂ → target) triplet, often
  reported as a matrix per atom.
- **Autonomy / Granger-style measures** (`autonomy`) — pairwise
  predictability.

The matching `inference.test(..., fdr=True)` call returns an
`InferenceResult` with 2D `effect_size`, `p_value`, and `p_value_fdr`,
which this figure consumes directly.

**Colormap.** Auto-selected from the data unless ``cmap=`` is passed:
sequential (``magma_r``, ``vmin = 0``) for **non-negative** statistics
(TE, MI, PID atoms — divergence around zero would be meaningless);
diverging (``RdBu_r``, centred at 0) for **signed** statistics
(correlation, signed contrasts).

**Reference figure form.** Vicente et al. (2011) *J. Comput. Neurosci.*
"Transfer entropy — a model-free measure of effective connectivity"
and Wibral, Lizier & Priesemann eds. (2014) *Directed Information
Measures in Neuroscience* — the TE-connectome heatmap form. Wollstadt
et al. (2019) *J. Open Source Software* (IDTxl) documents the same panel
for multi-source PID matrices. *Williams & Beer (2010)* — the original
PID paper — uses Venn-style decompositions and tables for atoms, not
matrices, so it is the reference for the **PID atoms themselves** but
not for the matrix figure form.

---

## Saving + composition

All inference figures behave like the per-result figures: pass them to
`save_publication` for the three-format triplet, or pass `ax=` to embed
them in a custom multi-panel layout you arrange yourself. They do not use
the (removed) `figure_panel` composite builder — for paper figures the
recommendation is to lay panels out by hand in the manuscript figure-prep
tool of choice, because reviewer-requested rearrangements are easier
that way.
