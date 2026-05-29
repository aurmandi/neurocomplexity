# Reviewer A — Criticality / dynamical-systems review of `neurocomplexity` v1.1.0

## Summary recommendation

**Major Revision.** The numerical core is sound — Wilting–Priesemann `m̂`
recovers known branching ratios at 10⁻³ bias on the bundled simulator
(`docs/phase2_math_audit.md:696-702`), the Sethna identity holds to
floating-point precision (`docs/phase2_math_audit.md:746-757`), the FDR /
Phipson–Smyth machinery is canonical, and the package's "fail loud, never
silently wrong" discipline shows in 29/29 reproducibility tests
(`docs/phase3_reproducibility_audit.md:18`). The criticality outputs would be
usable on Neuropixels data **as point estimates with caveats**. What blocks
publication today is the **scientific framing in the user-facing layer**:
(i) R²-driven bin-size selection is a documented forking path on the headline
estimator (`analysis/criticality.py:209-283`) with no published-range
constraint and no warning to the user; (ii) the package literature consistently
implies that "Wilting subsampling robustness" protects everything in the
toolbox, when it formally protects only the branching slope; (iii) MSE,
`I_min`, and Miller–Madow caveats already identified in Phase 2 have not
propagated into the documentation a domain reader actually reads; (iv) binary,
`k = l = 1` TE is called "effective connectivity" with no scope statement.
None of these are math bugs. All are correctable in docs + a handful of
warnings without changing the v1.x API. With the P0 items below addressed,
this is a publishable software paper.

## Strengths

- Wilting `m̂` validated against an in-tree branching-process simulator at
  four sub-critical points; documented bias < 0.005 at all four
  (`docs/phase2_math_audit.md:696-702`). This is the right way to defend the
  estimator and rare in toolboxes of this scope.
- Sethna identity coded as a literal one-liner with a divide-by-zero guard
  (`neurocomplexity/analysis/criticality.py:265-267`) and locked in as a
  regression invariant (`docs/phase2_math_audit.md:756-757`).
- `fit_alpha` switched from linear- to log-spaced histogram binning with an
  explicit comment about the upward-bias failure mode of the previous
  implementation (`neurocomplexity/analysis/criticality.py:140-164`). That
  history note is exactly what a reviewer wants to see.
- A documented history line on the `alpha_t` regression-vs-direct-fit
  confusion (`neurocomplexity/analysis/criticality.py:13-18`), with
  `gamma_fit` preserved as a distinct field. This is the correct fix for a
  bug that exists in many published criticality pipelines.
- Subsampling-bias correction is wired in at the level a real Neuropixels
  user needs it: the Wilting slope estimator
  (`neurocomplexity/analysis/branching.py:148-163`), not just toy examples.
- Stationarity-sensitive analyses (`criticality`, `branching`,
  `transfer_entropy`, `shape_collapse`) all funnel through
  `_warn_if_nonstationary` (`neurocomplexity/_warnings.py:91-108`,
  `neurocomplexity/analysis/criticality.py:214-216`,
  `neurocomplexity/analysis/branching.py:123-125`). The warning system
  itself is well-designed (per-`(rec, analysis)` dedup, suppressible via
  standard `warnings.filterwarnings`).
- Shape collapse uses a *scale-invariant* residual rather than raw MSE
  (`neurocomplexity/analysis/shape_collapse.py:80-89`); this is the right
  choice — naive residual minimisation pushes `γ` to whatever value flattens
  the rescaled amplitude, not to the universal exponent.
- Edge cases on `var = 0`, log(0), single-event avalanche, empty input are
  all unit-tested (`docs/phase3_reproducibility_audit.md:87-99`); these are
  exactly the silent-NaN traps that bite criticality analyses.

## Findings

### P0 — publication blockers

#### P0-1. R²-selection of `optimal_bin_seconds` is an unguarded forking path on the headline result.

- Where: `neurocomplexity/analysis/criticality.py:209-248`.
- What is wrong: the bin-size sweep `(2, 4, 8, 16, 32)` ms is searched and
  the bin that maximises the R² of the `log <S>(T)` regression is selected
  and returned as `optimal_bin_seconds`; the exponents `alpha_s`, `alpha_t`,
  `gamma_fit` reported are conditional on that maximisation. Bin choice is
  the single most consequential parameter in avalanche analysis (Beggs &
  Plenz 2003, Priesemann et al. 2014 explicitly discuss bin–exponent
  coupling). Maximising the goodness-of-fit of the very regression whose
  exponents are reported is a *forking path*: it biases reported R² upward
  and biases `(α_s, α_t, γ_fit)` toward whatever bin happens to look cleanest
  on this particular recording. There is no warning, no diagnostic table of
  per-bin exponents in the returned dataclass, no way for a downstream
  reviewer to see the sensitivity.
- Why it matters: a Phase-4 paper reviewer asking "show me the exponents at
  fixed 4 ms bins and at fixed 8 ms bins" cannot get an answer from the
  Result object as returned. The current design quietly does inferential
  model-selection inside what users perceive as an estimator.
- Proposed fix: (a) Surface the per-bin sweep as a field on
  `CriticalityResult` (`per_bin_alpha_s`, `per_bin_alpha_t`,
  `per_bin_gamma_fit`, `per_bin_r2`, indexed by `bin_size_ms`). (b) Emit a
  warning (e.g. `CriticalityBinSelectionWarning`) on every call,
  documenting that bin selection happened and pointing at the per-bin
  fields. (c) Expose `criticality(..., bin_size_ms=4.0)` scalar path that
  bypasses the sweep, for users who want to lock the bin. (d) In
  `docs/complexity_measures.md` (or a new criticality docs page), add a
  paragraph stating: "The reported `optimal_bin_seconds` is the bin chosen
  by R²; report it alongside exponents and reproduce results at a
  paper-locked bin for publication."

#### P0-2. Williams–Beer `I_min` over-redundancy is not disclosed in user-facing docs.

- Where: `neurocomplexity/analysis/pid.py:1-22`,
  `neurocomplexity/analysis/pid.py:39-75`, `docs/complexity_measures.md`
  (entirely absent), `docs/quickstart.md` (entirely absent).
- What is wrong: Phase 2 audit acknowledges that `I_min` does not
  distinguish same-content from same-magnitude redundancy (Bertschinger et
  al. 2014, recommending `I_BROJA`) and flagged this as a Phase-4 item
  (`docs/phase2_math_audit.md:594-601`). None of this has been propagated
  into any docstring or user-facing doc. The PID module docstring
  (`pid.py:1-22`) cites Williams & Beer 2010 as authoritative and makes no
  mention of `I_BROJA`, `I_PM`, or any structural limitation. A naive user
  reading the returned `redundancy` field will treat it as ground truth.
- Why it matters: PID atoms are routinely over-interpreted in systems-neuro
  papers ("synergy means non-linear coding", etc.); the `I_min` redundancy
  inflation translates directly into deflated unique-information atoms,
  which is the substantive claim users build figures on.
- Proposed fix: (a) Add a "Limitations" paragraph to the
  `partial_information` docstring stating that `I_min` is known to
  over-estimate redundancy (and therefore under-estimate uniques) and that
  `I_BROJA` / `I_PM` are more principled. (b) Add an
  `docs/information_decomposition.md` page (or extend
  `docs/complexity_measures.md`) with the same disclosure. (c) Tag the
  `redundancy` field's docstring in `PIDResult` (`pid.py:47-49`) with the
  caveat. The package can keep `I_min` as the default for v1.x.

#### P0-3. "Subsampling-robust" claim is implicitly generalised beyond the branching ratio.

- Where: `docs/quickstart.md:30-32` (frames `QualityControlWarning` /
  uncurated-data discipline broadly), `docs/inference.md:80`
  (Wilting–Priesemann 2018 listed as a generic reference for the
  inference module), and the package narrative in general. Concretely:
  `neurocomplexity/analysis/branching.py:111-116` correctly scopes
  subsampling-bias correction to the branching slope, but no
  corresponding "TE / PID / PR / MSE / LMC are NOT subsampling-corrected"
  statement appears anywhere in the user-facing docs.
- What is wrong: Wilting & Priesemann (2018) prove the slope of
  `log r_k` vs `k` is invariant to subsampling-induced rescaling on each
  `r_k`. That argument does not transfer to TE, PID, PR, MSE, or LMC.
  Subsampling biases binary-TE downward (because both x_past and y_past
  marginals are deflated), biases PR upward toward `n_recorded_units`,
  biases MSE depending on rate structure, biases `I_min` redundancy in
  ways nobody has fully characterised. The current docs read as though the
  package's "Wilting-flavour" robustness is the package's identity.
- Why it matters: Neuropixels sessions sample 1-10% of the local population.
  A user is almost certain to apply TE to a 100-unit subset of a 10⁴-unit
  cortical area and report it as "effective connectivity," with no flag
  from the package that the estimate is biased by subsampling.
- Proposed fix: Add a new section to `docs/inference.md` titled
  "Subsampling: what is corrected and what is not." Spell out: only Wilting
  `m̂` is subsampling-bias-corrected; α_s, α_t, γ_fit on subsampled data
  are biased downward by truncation (already noted in Phase 2 audit
  `docs/phase2_math_audit.md:760-773` but only in internal docs); TE / PID
  / PR / MSE / LMC have no subsampling correction and should be reported
  with the recording fraction stated.

### P1 — must address in revision

#### P1-1. Binary, `k = l = 1` TE is reported under the "effective connectivity" label.

- Where: `neurocomplexity/analysis/transfer_entropy.py:1-6, 22-30`,
  `docs/quickstart.md:64-65` ("Effective connectivity"),
  `docs/inference.md` (TE framed without scope statement).
- What is wrong: `_binary_schreiber_te` (`transfer_entropy.py:58-98`) binarises
  spike counts (`> 0`) and uses 1-bin history on both source and target.
  This is a valid Schreiber-2000 implementation, but: (a) "effective
  connectivity" in the Friston sense implies a causal-model-fitted weight;
  Schreiber TE on binary spike counts is a model-free statistical
  dependency, not a connectivity weight; (b) `k = l = 1` is by far the
  most popular convention but is known to mis-attribute TE when the true
  history length is larger (e.g. on bursting populations); (c) binarisation
  destroys the rate code, which is exactly the signal carrying TE on most
  Neuropixels recordings; (d) the simplified Miller–Madow correction
  `(m_joint - 1)/(2N)` is a practical choice but loose vs Roulston 1999 as
  Phase 2 noted (`docs/phase2_math_audit.md:476-483`).
- Why it matters: a reader of a paper citing
  `nc.transfer_entropy(rec).matrix[A, B] > 0` will (correctly, given the
  word "effective connectivity" in the docs) write "A drives B." That
  inference is not licensed by binary `k = l = 1` Schreiber TE on
  subsampled spike trains.
- Proposed fix: In the user-facing string, replace "effective connectivity"
  with "pairwise transfer entropy" (mathematical name) everywhere outside
  the module's own internal narrative. Add a Limitations paragraph in
  `transfer_entropy.py` docstring and `docs/inference.md` listing:
  binarisation discards rate, `k = l = 1` is the convention not the
  optimum, no source-history-length selection, no conditional-TE
  multivariate correction, no causal-graph interpretation. Cite Lizier et
  al. 2014 on TE interpretation pitfalls; cite Wibral et al. 2014 (book)
  on `k` / `l` selection. Mention that a Kraskov k-NN estimator is on the
  roadmap.

#### P1-2. MSE `r_factor = 0.2` mis-cited as Costa 2002 in code, contradicted by docs.

- Where: `neurocomplexity/analysis/mse.py:211` and its docstring (Phase 2
  audit at `docs/phase2_math_audit.md:384-388` quotes the docstring as
  `"default 0.2 — Costa 2002"`); but
  `docs/complexity_measures.md:78` reads `"default 0.15·SD"`.
- What is wrong: code default is 0.2; user-facing docs page says 0.15. They
  cannot both be right. Phase 2 already pinned the canonical Costa 2002
  value at 0.15 (`docs/phase2_math_audit.md:386-388`).
- Why it matters: a user reading `docs/complexity_measures.md` will assume
  Costa-canonical `r = 0.15 σ`; the code silently runs `r = 0.20 σ`; their
  reproductions will not match. This is a reproducibility bug at the
  level of the headline complexity measure.
- Proposed fix: Pick one. (a) Change the code default to 0.15 and add a
  CHANGELOG note (preferred — matches docs and canonical paper). (b) Keep
  0.2 and fix the docs and code docstring to state "0.2 (Pincus 1991
  ApEn convention)" — but then `docs/complexity_measures.md:78` must be
  amended in lockstep. Choosing (a) is consistent with the publication
  plan's "no analysis without a citation" principle
  (`docs/publication_plan.md:11-13`).

#### P1-3. LMC `n_states` is per-population by design; cross-population comparability foot-gun is undocumented at the API surface.

- Where: `neurocomplexity/analysis/complexity.py:108-117` (state count =
  `max(counts) + 1`); Phase 2 already flagged this
  (`docs/phase2_math_audit.md:127-138`). The user-facing
  `docs/complexity_measures.md` does *not* mention the per-population
  state-space.
- What is wrong: users will compare LMC `C` across populations of different
  mean firing rate and silently be comparing two different normalisations
  (`H` is normalised by `log N` with different `N`). The result-level
  `n_states_per_pop` field is a fix but only if the user knows to inspect it.
- Why it matters: comparing LMC across cortical areas is a *first-thing-you-do*
  use case. The current design returns plausible numbers that are not
  on the same scale.
- Proposed fix: (a) Add an explicit `n_states` (or `n_states_mode={"per_pop",
  "global"}`) keyword to `lmc_complexity` with `"global"` being
  `max over all populations + 1`. (b) Document the trade-off prominently in
  `docs/complexity_measures.md`. (c) When the per-population `n_states_per_pop`
  values differ by more than a factor of two between populations in a
  single call, emit a warning.

#### P1-4. `kappa = 1 + γ_predicted` legacy alias is an interpretive trap.

- Where: `neurocomplexity/analysis/criticality.py:57-59`,
  `neurocomplexity/analysis/criticality.py:267`,
  `neurocomplexity/analysis/criticality.py:275`.
- What is wrong: the literature `κ` of Shew, Yang, Petermann et al. 2009
  is a *different* statistic — the integrated deviation of the avalanche
  size CDF from the theoretical near-critical CDF, not `1 + γ_pred`. The
  docstring honestly calls this a "legacy" field, but legacy users
  importing under the literature name will get a number that has nothing
  to do with the Shew κ they expect.
- Why it matters: at minimum it generates confusion in code review of
  downstream user scripts; worst case it leaks into a methods section
  attributed to Shew 2009.
- Proposed fix: (a) Deprecate the field with a `DeprecationWarning` on
  attribute access (use `__post_init__` + custom `__getattribute__` on the
  frozen dataclass, or expose as a property). (b) Rename it
  `kappa_legacy_1_plus_gamma_predicted` for v2.0. (c) Add a one-line
  docstring note pointing to Shew 2009 and stating that this field is *not*
  Shew κ.

#### P1-5. Power-law fitting uses log-binned histogram regression rather than Clauset–Shalizi–Newman MLE.

- Where: `neurocomplexity/analysis/criticality.py:136-164`.
- What is wrong: `fit_alpha` fits 20 log-spaced histogram bins with OLS in
  log–log space. Clauset, Shalizi, Newman (SIAM Review 2009) showed this
  procedure is biased and the modern community standard is the MLE
  (`α̂ = 1 + N / Σ ln(x_i / x_min)`) with a Kolmogorov–Smirnov-minimising
  `x_min`. The log-binned fit is empirically reasonable on data this
  package was validated against (Phase 2 reports `α_s = 1.49` vs theory
  1.5), but at the publication-defensibility bar a reviewer will ask why
  the package chose log-binning over MLE.
- Why it matters: defensibility, not correctness. A paper-grade analysis
  is expected to use Clauset MLE with a goodness-of-fit p-value and an
  alternative-distribution (lognormal, truncated power law) comparison.
- Proposed fix: (a) Add a `method={"loghist", "mle"}` keyword to
  `fit_alpha` and surface it on `criticality(...)`. (b) Implement
  `x_min` selection per Clauset (KS-minimising sweep) and the
  log-likelihood ratio vs lognormal. (c) Document the default choice and
  rationale.

### P2 — nice-to-have

#### P2-1. Stationarity warning logic does not block analysis; document the policy.

- Where: `neurocomplexity/_warnings.py:91-108`.
- What it is: `StationarityWarning` is fired once per `(rec, analysis)`
  but the analysis proceeds and the result is returned with no
  `is_stationary` flag inside the Result dataclass. Policy is
  defensible (it lets users mute the warning) but should be stated
  explicitly in the docs.
- Fix: Add `stationarity_flagged: bool` field to every stationarity-sensitive
  Result; document the warn-and-proceed policy in `docs/quickstart.md`.

#### P2-2. `extract_avalanches` uses "any nonzero bin" definition without exposing a threshold.

- Where: `neurocomplexity/analysis/criticality.py:120-129`.
- What it is: avalanche = maximal run of bins with `count > 0`. Standard
  Beggs–Plenz definition. But many modern papers (Fontenele 2019,
  Wilting–Priesemann 2018) threshold population activity at a non-zero
  level (e.g. median). The current API does not expose `threshold`.
- Fix: Add a `threshold_quantile` (or `threshold_counts`) keyword;
  default to 0 (current behaviour); document the choice in the criticality
  docs page.

#### P2-3. Phase 2 documented finite-size truncation bias on `α_t` (`α_t ≈ 1.83` vs theory 2.0); not surfaced to users.

- Where: `docs/phase2_math_audit.md:760-773` discusses this internally;
  `docs/complexity_measures.md` and `docs/quickstart.md` do not mention it.
- Fix: Add to the criticality docs page: typical recovered α_t on
  finite-data Neuropixels sessions falls in [1.6, 2.2]; values near 2.0
  require very long recordings or unrestricted trial durations.

#### P2-4. `branching` field on `CriticalityResult` is the Beggs 2003 single-step ratio, kept "for backwards compatibility" (`criticality.py:54-56`).

- Where: `neurocomplexity/analysis/criticality.py:198-206`, `:274`.
- What it is: this single-step ratio is what every modern paper has
  *replaced* with Wilting–Priesemann. Returning it alongside the headline
  result invites users to cite it.
- Fix: Deprecate (`DeprecationWarning` on access) for v2.0; or rename to
  `branching_singlestep_legacy`. Point at `nc.wilting_mr` in the docstring.

## Specific responses to Phase-2 caveats

- **LMC `N_states` data-dependent → cross-population comparability foot-gun.**
  Verdict: P1 (see P1-3 above). Resolution: add a `n_states={"per_pop",
  "global"}` keyword; document the trade-off in
  `docs/complexity_measures.md`; emit a warning when per-population state
  counts diverge by >2× within a single call.

- **MSE default `r_factor = 0.2` mis-cited as Costa 2002.**
  Verdict: P1 (see P1-2 above). Compounded by the fact that
  `docs/complexity_measures.md:78` *already states* `0.15`, contradicting
  the code default — so the docs/code disagreement is a live reproducibility
  bug. Resolution: change the code default to 0.15 (Phase 2 audit's
  preferred reading) and CHANGELOG it; alternatively, keep 0.2 and amend
  both code docstring and `docs/complexity_measures.md:78` in lockstep.
  Prefer (a).

- **Schreiber TE Miller–Madow simplified `(m_joint - 1)/(2N)`.**
  Verdict: P2 in isolation, P1 in combination with the binarisation /
  `k = l = 1` framing issue (see P1-1 above). The simplified MM is a
  community standard (matches `pyinform`); for v1.x, the right fix is
  documentation: state the simplification and the Roulston alternative in
  the `transfer_entropy` docstring and `docs/inference.md`. v2.0 should
  offer Roulston as a keyword.

- **WB `I_min` PID benchmark tolerance loose at 0.10 nats; `I_min` known to
  over-estimate redundancy.**
  Verdict: P0 on the user-facing disclosure side (see P0-2 above), P2 on
  the benchmark tolerance side. Tightening the AND benchmark from 0.10 to
  ~0.03 with `n_bins = 20000` is a quick win and turns the benchmark from
  defensive to informative.

## Domain-defensibility statement

Would I, as a domain expert, trust the package's criticality outputs in a
published paper on a Neuropixels dataset? **Yes, conditionally.** The Wilting
`m̂` is the most carefully validated estimator in the package and is the one
I would cite without qualification. The Sethna identity is exact. The
log-spaced `fit_alpha` is reasonable, and the documented `α_s ≈ 1.49` recovery
against theory 1.5 is publication-grade. I would attach the following
disclaimers in any methods section using v1.1.0: (i) bin size for criticality
was selected by R² on `(2, 4, 8, 16, 32)` ms — report exponents at a
paper-locked single bin in a supplementary table; (ii) α_t will be biased
downward by recording-length truncation, expect values in [1.6, 2.2] on real
data; (iii) the `kappa` field is *not* Shew 2009 κ — use `gamma_predicted`;
(iv) transfer entropy is binary, `k = l = 1` Schreiber and does not infer
causal connectivity — call it "pairwise TE" not "effective connectivity";
(v) PID uses `I_min`, which over-estimates redundancy; treat the redundancy
atom as an upper bound. With these in a methods box, a paper using
`neurocomplexity` v1.1.0 is defensible at a competitive systems-neuro venue.

## What's missing (out-of-scope-but-noted)

- **Clauset–Shalizi–Newman MLE with KS-based `x_min`** — modern standard for
  power-law fitting; see P1-5.
- **Goodness-of-fit p-value via parametric bootstrap (Clauset 2009 §IV)** —
  the right way to ask "is this actually a power law?" rather than "what's
  the slope of the log–log fit?"
- **Lognormal / truncated power law alternative comparison** — the
  log-likelihood ratio test of Vuong (1989) as applied by Clauset. Required
  for any "we see scale-free avalanches" claim.
- **Finite-size data collapse on `P(S, L)`** — the two-variable scaling
  collapse from Lübeck 2004 / Sethna 2001. Shape collapse is a 1-D
  projection of this.
- **`α_s + α_t` consistency check via Friedman 2012 universality class** —
  the package reports both exponents but does not flag inconsistency with
  the directed-percolation MFT class (`α_s = 1.5`, `α_t = 2.0`,
  `γ = 2.0`). A `criticality_class_check(result)` helper would be cheap.
- **Kraskov k-NN TE estimator** — already hinted at by the `estimator=`
  parameter (`transfer_entropy.py:117`) but only `"binary"` is implemented.
- **Conditional TE / multivariate TE** — pairwise TE on `N > 2` populations
  cannot disentangle direct from common-driver TE; the docs do not warn
  about this.
- **Detrended fluctuation analysis (DFA) / long-range temporal correlation
  (LRTC) exponent** — the other half of the "near-critical cortex" toolkit
  (Linkenkaer-Hansen 2001) is absent.
- **Branching-ratio uncertainty under noise floor** — Wilting estimate
  needs a CI; `nc.inference.bootstrap` provides one but the criticality
  pipeline does not auto-attach.
- **Subsampling-bias quantification on TE / PID / PR** — even just
  documenting the empirical scaling vs recording fraction on the bundled
  simulator would be a community service.
