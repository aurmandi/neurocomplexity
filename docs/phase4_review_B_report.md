# Reviewer B — Statistical / reproducible-research review of `neurocomplexity` v1.1.0

## Summary recommendation

**Major Revision.** The numerical machinery of the inference layer is correctly
implemented and well audited (Phipson–Smyth floor, BH-FDR step-up, BC bootstrap,
SeedSequence-based child seeding). The package is unusually careful for an
open-source neural-data project — provenance, dtype hygiene, order-independence,
and explicit two-sided semantics are all in order. However, several user-facing
defaults and documentation choices currently invite misinterpretation in ways
that would surface as statistical objections at JOSS / eLife review:
`spike_dither` as the *quickstart-documented* default for TE; a single global
BH-FDR family with no per-row option; bootstrap block sizes set as
opaque per-statistic constants with no Politis–Romano-style guidance; the
calibration suite excluded from CI without a reported pass/fail; and no
discussion at all of multiple-comparison family definition across a study
(populations × triples × statistics). None of these is wrong code — but the
package ships an inference UX that defends none of these choices to a skeptical
methods reviewer. Fixing the docs and the quickstart defaults is sufficient to
reach Minor Revision.

## Strengths

- Phipson–Smyth +1 floor implemented exactly to spec for both one- and
  two-sided alternatives, with `1/(n+1)` floor preserved on each tail before
  the `2·min` clip (`neurocomplexity/inference/null_test.py:40-61`). The
  retired mean-centred `|null − μ|` form is explicitly documented as rejected
  in `docs/inference.md:31-33`.
- BH-FDR is a textbook step-up implementation: right-to-left running-min over
  ranked `p·m/k`, with monotone clipping (`null_test.py:101-106`). The
  `np.minimum.accumulate(q[::-1])[::-1]` idiom is correct and shape-preserving.
- Effect size uses `ddof=1` (sample SD) and degrades to NaN — not `±inf` —
  when null variance is zero (`null_test.py:88-92`). This is the right
  numerical posture.
- `SurrogatePool` uses `SeedSequence.spawn` with deterministically-stored child
  seeds (`pool.py:81-82`), so a given `(seed, i)` returns the same surrogate
  irrespective of access pattern, cache state, or parallelism. The CI test
  `test_reproducibility_across_n_jobs` (`tests/test_inference_calibration.py:182-188`)
  asserts `n_jobs=1` and `n_jobs=2` give byte-identical null distributions.
- BC (bias-corrected) percentile method is implemented for vector- and scalar-
  valued statistics with a sensible degenerate-case guard (NaN bounds when
  `<5` non-NaN replicates; clip `p` to `[0.5/d.size, 1 − 0.5/d.size]` to
  prevent `norm.ppf(0) = −∞`) — `bootstrap.py:66-80`.
- `interval_shuffle` explicitly rejects overlapping intervals (`surrogates.py:127-137`)
  with a useful error message — a genuine silent-corruption mode is closed.
- Phase 3 reproducibility audit (`docs/phase3_reproducibility_audit.md`) demonstrates
  29/29 determinism, pickle, dtype, and order-independence tests passing.
- Edge cases ("all-zero spike train", "var = 0 in branching") return NaN with
  the failure mode documented (`phase3:84-99`); no silent wrong numbers.

## Findings

### P0 — publication blockers

#### P0-1 — Calibration suite excluded from CI without any published pass/fail

- **Where:** `.github/workflows/test.yml:32` (`--ignore=tests/test_inference_calibration.py`)
  and `tests/test_inference_calibration.py:75-177`.
- **What is wrong:** The package's only Type-I / power / coverage gates live
  in `test_inference_calibration.py` and are explicitly excluded from CI. The
  `FULL` mode (the actual published gate per the file's docstring) is never
  reported anywhere I can find. Without a published, reproducible calibration
  table, the central inferential claim of the package — "our p-values control
  Type-I at the nominal rate; our bootstrap CIs cover at the nominal rate" — is
  unsupported.
- **Why it matters statistically:** A reviewer cannot distinguish between
  "calibration is fine and the slow tests are merely slow" and "calibration is
  unknown because no one ran the full mode and recorded the numbers." The
  in-file note "block bootstrap is known to under-cover near criticality on
  short recordings" (`test_inference_calibration.py:154-162`) further means
  the user-visible default (`block_seconds=10` and `n=1000`) is known to
  under-cover but the under-coverage magnitude is not reported on the
  documented configuration. JOSS reviewer 1 will ask for the table.
- **Proposed fix:** (a) Run `CALIBRATION_FULL=1 pytest -m slow tests/test_inference_calibration.py`
  on a CI-equivalent runner once per release; (b) commit the resulting Type-I
  rate, power, and coverage numbers as a `docs/calibration_report.md`; (c) add
  a nightly-or-on-tag CI job (not on every PR) that re-runs full mode and
  diff-asserts against the committed numbers; (d) reference the report from
  `docs/inference.md`.

#### P0-2 — `docs/quickstart.md` recommends `spike_dither` as the default surrogate for TE

- **Where:** `docs/quickstart.md:94` shows
  `test(te, rec, surrogate="spike_dither", ...)` as *the* TE inference
  example, and the table on lines 105-109 lists `spike_dither` as
  "Default for TE / connectivity".
- **What is wrong:** `spike_dither` only randomises spike times within ±5 ms
  by default (`inference/surrogates.py:18`). For the standard binary-Schreiber
  TE configured with `bin_size_ms=10` (per the quickstart, line 65), a ±5 ms
  jitter preserves the *bin* of nearly every spike most of the time. The
  empirical null under this surrogate is therefore *not* a null of "no
  directed coupling at the binning scale of the analysis"; it is closer to a
  null of "no sub-bin coupling", which the TE estimator at this bin size
  cannot detect anyway. Type-I will look fine — but power against any
  binwise-aligned coupling will be over-estimated.
- **Why it matters statistically:** The null hypothesis a user *thinks* they
  are testing ("does A drive B beyond what their marginal rates predict?") is
  not what `spike_dither` tests. `isi_shuffle` is the right default for TE
  (preserves the marginal rate distribution per unit exactly, destroys
  cross-unit coupling); the calibration file itself uses `isi_shuffle` for
  every Type-I and power test (`test_inference_calibration.py:84, 104`).
- **Proposed fix:** Either (i) change the quickstart default to `isi_shuffle`
  and add a paragraph explaining that `spike_dither(delta_ms ≤ bin_size_ms/2)`
  tests a fine-timing-only hypothesis; or (ii) issue a `UserWarning` from
  `test()` when `surrogate="spike_dither"` is chosen with
  `delta_ms < adapter_bin_size_ms / 2`. Independently update the
  `docs/quickstart.md` table to make per-statistic recommendations explicit:
  `isi_shuffle` for TE/PID/connectivity, `spike_dither` for fine-timing/STA,
  `interval_shuffle` for trial-structured experiments.

#### P0-3 — Block bootstrap degenerates silently when `block_seconds > duration / 3`

- **Where:** `inference/bootstrap.py:154-173` (`_block_resampled_recording`),
  no validation.
- **What is wrong:** With duration `T` and `block_seconds = B`,
  `n_blocks = ceil(T/B)`. For a 30 s recording with the dispatcher default
  `block_seconds=10.0` for branching (`bootstrap.py:183, 250`), the user gets
  only 3 unique blocks and a bootstrap distribution of 3³ = 27 effective
  configurations. The CI bounds are then *not* asymptotically valid; they are
  empirical quantiles of a vanishingly small atomic distribution.
  No warning, no error. The block-bootstrap literature (Politis & Romano 1994)
  treats `n_blocks ≥ 40` as a soft floor for reliable percentile CIs.
- **Why it matters statistically:** A user on Allen-style data who runs
  `wilting_mr` on a 60 s trial epoch and calls `bootstrap(...)` will receive a
  numerically-valid-looking 95% CI that under-covers severely. The Phase-3 audit
  did not catch this because all tests use long synthetic recordings.
- **Proposed fix:** In `_block_resampled_recording` (or the dispatcher), raise
  `ValueError` if `n_blocks < 10` and emit a `UserWarning` if
  `10 ≤ n_blocks < 40` advising the user to either shorten `block_seconds` or
  acknowledge wide-CI degeneracy. Document the rule of thumb in
  `docs/inference.md` Bootstrap section.

### P1 — must address

#### P1-1 — No per-statistic surrogate recommendation enforced at the API level

- **Where:** `inference/null_test.py:125-188`; `pool.py:60-83`.
- **What:** `test()` accepts any of three surrogates for any of nine result
  types. There is no per-statistic recommendation, no warning when a
  questionable pairing is chosen (e.g., `spike_dither` for branching ratio,
  which depends only on the binned population rate and to which `spike_dither`
  is approximately invariant by design; the null will be degenerate and
  p-values will collapse to the floor).
- **Why:** The package's surrogate choice *is* its null-hypothesis choice. A
  free-choice API forces the user to be the statistician.
- **Fix:** Add a per-result-type "recommended surrogate" registry adjacent to
  `_STAT_NAMES` (`null_test.py:110-118`). If the user passes a non-recommended
  surrogate, emit a `UserWarning` explaining the mismatch. Document the
  recommendations in a new section of `docs/inference.md`.

#### P1-2 — Global BH-FDR family with no per-row / per-column option

- **Where:** `inference/null_test.py:95-107`; `null_test.py:173` (`fdr_bh(p)`).
- **What:** For a TE matrix `p` of shape `(N, N)`, BH is flattened across all
  `N²` entries. The "family" is therefore "all directed pairs in this matrix".
  For neuroscientific use (testing whether area X drives any of N other
  areas), a per-row FDR (one hypothesis family per source) is often the more
  power-preserving and interpretable choice; per-column likewise (one family
  per target). The current API offers no toggle.
- **Why:** Statistical power scales inversely with family size. Defaulting to
  global is the *conservative* choice and so defensible, but only if the user
  is told that explicitly.
- **Fix:** (a) Document the chosen family definition prominently in
  `docs/inference.md`; (b) extend `test(..., fdr_axis=None|"rows"|"cols")` or
  ship a separate `inference.fdr_per_axis(p, axis)` helper. Note that the
  current PID atom case (`obs_arr.shape == (4,)`) is FDR'd across the 4 atoms
  globally — almost certainly *not* what the user wants; per-atom raw
  p-values are usually what's reported.

#### P1-3 — Bootstrap block size: defaults are magic numbers, no Politis–Romano guidance

- **Where:** `bootstrap.py:183, 250, 290, 327` (`block_seconds=10.0|1.0`);
  `docs/inference.md:46-51`.
- **What:** The defaults (10 s for branching, 1 s for everything else) are
  stated without derivation. There is no implementation of any data-driven
  block-size selector (Politis & White 2004; Patton, Politis & White 2009),
  no documentation of how a user with a long autocorrelation should choose,
  and no diagnostic that flags `block_seconds < estimated_autocorrelation`.
- **Why:** Block size is the single largest source of bias/variance trade-off
  in the block bootstrap; a fixed default is essentially a hidden parameter.
- **Fix:** Either (a) ship a `inference.suggest_block_size(rec, statistic)`
  helper that runs an integrated-autocorrelation estimate on the population
  rate and returns a recommendation; or (b) document in `docs/inference.md`
  a paragraph on how to choose, with a worked example on Allen data. Either
  way, mention Politis–Romano (1994) and Künsch (1989) in the references.

#### P1-4 — `effect_size` is a z-score under the null, not Cohen's d — currently undocumented

- **Where:** `null_test.py:67-92`; docstring lines 67-86 describe it as
  "standardised effect size" without committing to a name.
- **What:** A reader skimming the docs will assume Cohen's d, which has a
  pooled-SD denominator. Here the denominator is the *null* SD only. Both are
  defensible; the package's choice is correct (it is what a methods paper
  using surrogate-null analysis would report), but it is not what "effect
  size" connotes in psych / clinical conventions.
- **Fix:** Rename docstring header to "z-score against null" and add one
  line: "Interpret as a standard-normal quantile, not Cohen's d. To convert
  to Cohen's d the user must pool the null SD with the bootstrap SD of the
  observed."

#### P1-5 — `SurrogatePool` shared across statistics induces correlated p-values

- **Where:** `pool.py:60-97` (`SurrogatePool` reuse pattern);
  `docs/inference.md:71-73` (recommended usage).
- **What:** When the same pool is passed to `test(TE, ...)` and `test(PID, ...)`,
  both null distributions are functions of the same surrogate recordings.
  Their p-values are positively correlated, which silently breaks the
  independence assumption of any downstream combination (e.g., Stouffer's,
  Fisher's). The docs explicitly recommend this re-use pattern.
- **Why:** Reusing a pool is good for compute but invalidates any
  meta-analytic combination of the resulting p-values without explicit
  dependency correction.
- **Fix:** Add a paragraph to `docs/inference.md` warning that pool-sharing
  induces dependency among the resulting p-values, and recommending a fresh
  pool per statistic when the p-values will be combined downstream.

#### P1-6 — `StationarityWarning` is non-blocking; bootstrap CI validity assumes stationarity

- **Where:** `_warnings.py:91-108`; `bootstrap.py:154-173`.
- **What:** Block bootstrap consistency (Künsch 1989; Politis–Romano 1994)
  requires the underlying process to be (at least) weakly stationary. The
  package warns but does not block; a user can compute a 95% CI on a recording
  the package itself flagged as non-stationary, and nothing in
  `InferenceResult` records the warning's status. This is recoverable but
  needs documentation and provenance.
- **Fix:** (a) Record `metadata["stationarity_passed"]` on every
  `InferenceResult` returned by `bootstrap()` and `test()`, set from
  `stationarity(rec).is_stationary`; (b) document in `docs/inference.md` that
  CIs from non-stationary recordings should be reported with a caveat.

### P2 — nice-to-have

#### P2-1 — `pvalue_from_null` two-sided form is `2·min(p_g, p_l)`, not the asymmetric Phipson–Smyth proposal

- **Where:** `null_test.py:58-61`.
- **What:** Phipson & Smyth (2010, §3) note that for asymmetric nulls the
  `2·min` Bonferroni-style form is conservative and that an alternative,
  using the proportion of null replicates "at least as extreme" defined by
  rank in the empirical null CDF, has better power. The current package's
  `2·min(p_g, p_l)` is the most common practice and is documented as
  intentional, but a one-line note acknowledging the alternative would
  pre-empt referee objections.
- **Fix:** One sentence in `docs/inference.md:31-33`.

#### P2-2 — BC is silently degraded on small `n_resamples`

- **Where:** `bootstrap.py:68-69` (`if d.size < 5: return nan, nan`).
- **What:** BCa (the accelerated form) would be the more principled choice for
  finite-sample skewness; the current BC method does only the median-bias
  correction. A user passing `n=50` (the dispatch defaults are 1000, but
  nothing prevents `n=50`) gets nominally-valid but in practice unstable
  intervals. Worth one line in the docstring.
- **Fix:** Document the BC vs BCa distinction; consider implementing
  acceleration via the jackknife-on-blocks for v1.2.

#### P2-3 — `fdr_bh` silent NaN pass-through

- **Where:** `null_test.py:101-104`; documented as deliberate in
  `phase2_math_audit.md:243-246`.
- **What:** NaN p-values pass through `np.minimum.accumulate` unchanged and
  bypass the BH adjustment entirely. For a TE matrix this happens whenever
  the diagonal (`TE(X→X)=0`) yields a degenerate null and therefore NaN p.
  Not wrong, but a user who flattens and aggregates will silently include the
  NaNs in their denominator.
- **Fix:** Add `fdr_bh(..., nan_policy="ignore"|"raise")` and document that
  `ignore` (current behaviour) drops NaNs from the family count, while
  `raise` matches strict-mode statistical packages.

#### P2-4 — No primitive for combining p-values across populations

- **Where:** Not present in `inference/__init__.py`.
- **What:** A standard cortical-population study tests directed flow between
  ~6 populations + a PID atom per triple + branching per population. The
  package offers no Stouffer / Fisher / Cauchy primitive for combining
  these. Even if the answer is "out of scope, refer users to scipy", say so.
- **Fix:** Add a one-paragraph note to `docs/inference.md` discussing
  family-of-studies multiple-comparison policy: package handles within-matrix
  FDR, leaves cross-statistic correction to the user, recommends a Bonferroni
  factor of (number of statistics × number of populations).

#### P2-5 — `p_value_fdr` shape inconsistency for PID

- **Where:** `null_test.py:173` (`p_fdr = fdr_bh(p) if (fdr and obs_arr.ndim >= 1) else None`)
  combined with `_pid_adapter` returning a length-4 vector
  (`_adapters.py:42-47`).
- **What:** PID gets FDR'd over its 4 atoms. This is almost never what a user
  reporting redundancy / unique / synergy wants — each atom has its own
  scientific question and an FDR family of 4 is meaninglessly small.
- **Fix:** In `test()`, treat `PIDResult` specially: return raw p-values
  per atom and skip FDR, with a comment in the metadata.

## Specific responses to Phase-2 caveats

- **LMC `N_states` data-dependent:** Concur. Cross-population LMC comparison
  with per-population state space is a real foot-gun. Prefer Phase-2's option
  (b): add an optional `n_states` argument that fixes a global state-space
  size (e.g., max over populations + 1). The package can still infer per-pop
  by default but the cross-pop figure code should explicitly pass the global
  value. Documentation alone is insufficient — users will not read it before
  they generate the cross-pop figure.

- **MSE `r_factor = 0.2` mis-cited as Costa 2002:** Concur with Phase-2
  recommendation (b) for v1.x (correct the citation, document both 0.2 and
  0.15), and (a) for v2.0 (change default to canonical 0.15). The doc fix is
  free and must ship now; the default change is breaking and belongs in a
  versioned migration.

- **Schreiber TE simplified Miller–Madow:** Concur that the simplified
  `(m_joint − 1)/(2N)` is defensible at typical Neuropixels N (≥10⁴ bins) and
  documented. But I would go further: add a `bias_correction` parameter to
  `transfer_entropy(..., bias="mm" | "roulston" | "none")` so the principled
  Roulston (1999) form is available to users who run on short epochs. The
  current behaviour can remain default. Mark Roulston "Experimental" in the
  v1.2 release until validated.

- **Williams–Beer I_min PID benchmark tolerance loose at 0.10 nats:**
  Concur. The 0.10-nat tolerance accepts a 29% relative error on the AND case
  expected value of 0.347 nats. Tighten to 0.03 nats per Phase-2's own
  recommendation. Additionally, the I_min over-redundancy known limitation
  should be a docstring header on `partial_information`, not buried in
  `docs/complexity_measures.md` — users who only read the function help will
  miss it.

## Inference-validity statement

The inference layer of `neurocomplexity` v1.1.0 will produce a publishable
p-value or CI on a real Neuropixels session **provided** the following
conditions are met, none of which are currently enforced by the package
itself: (a) the analyst chooses `isi_shuffle`, not `spike_dither`, as the
TE / PID null surrogate (the documented default in the quickstart is wrong
for the bin sizes typically used); (b) the recording is at least 5× the
chosen `block_seconds` long, so the block bootstrap has ≥ 40 unique blocks
and gives an asymptotically-valid CI rather than an atomic distribution over
a handful of configurations; (c) `stationarity(rec).is_stationary` is `True`,
or the user has cropped to a stationary epoch (block bootstrap is not
distributionally robust to rate drift); (d) the user does not combine
multiple p-values produced by the same `SurrogatePool` instance without
acknowledging the dependence; (e) the methods section reports FDR family as
"all entries in the TE matrix" and states the multiple-comparison policy
across populations explicitly. The Methods section of a manuscript using
this package should state: surrogate method, n_surrogates, block_seconds,
n_bootstrap, FDR family definition, and the package's calibration-table
reference (which the package does not currently provide — see P0-1).

## What's missing

Standard inference primitives I would expect from a complexity-science
package that `neurocomplexity` does not yet provide. None are blockers; all
are roadmap items.

- **Data-driven block-size selection** (Politis & White 2004; Patton-Politis-
  White 2009). The package commits to block bootstrap but ships only fixed
  block-seconds defaults.
- **BCa (accelerated bootstrap)** — would replace BC for vector statistics
  with non-trivial skewness. Jackknife-on-blocks gives the acceleration term.
- **Calibration report committed to the repository** — see P0-1.
- **A Stouffer / Fisher / Cauchy combiner** for cross-population /
  cross-statistic meta-tests, with an explicit warning about dependence.
- **A `family=...` argument on `fdr_bh`** so users can do per-row / per-column
  / global / custom-grouped FDR without leaving the package.
- **`equitable null` benchmarks** — a reproducible figure (committed to docs)
  showing the package's Type-I rate vs nominal-α on a grid of (n_surrogates,
  recording_duration, statistic) combinations. This is the figure JOSS
  Reviewer 1 will demand.
- **A Kraskov-style continuous-signal TE estimator** as an alternative to
  binary-Schreiber on LFP / calcium signals (currently only `estimator="binary"`
  is implemented per Phase-2 audit; the code already has the hook).
- **A surrogate "diagnostic" method** that, given a surrogate-null sample and
  the observed statistic, plots and quantifies whether the null is symmetric,
  unimodal, and well-separated from the observed — needed before any one-
  sided p-value is trusted.
- **A `n_blocks_effective` field** on `InferenceResult` so reviewers can
  immediately judge whether the bootstrap was numerically degenerate.

End of report.
