# Criticality bin selection — methodological forking-path lockdown

**Date:** 2026-05-29
**Status:** Accepted (landed in 1.1.0, Phase 4 punch-list Tier 4)
**Tracks:** `docs/phase4_review_A_report.md` § P0-1;
`docs/phase4_review_panel.md` § Tier 4 item 14.

## Context

`neurocomplexity.analysis.criticality` historically swept a candidate
grid of bin sizes (e.g. `[2, 4, 8, 16, 32]` ms), fitted the
size-vs-lifetime regression at each, and returned the fit at the bin
with the highest R². That bin was reported as `optimal_bin_seconds`
and the exponents (α_s, α_t, γ_fit) from that bin were the headline
numbers.

Reviewer A (Phase 4) raised this as an unguarded forking path on the
headline criticality result:

> The package returns the `(α_s, α_t)` at whichever bin happens to
> yield the highest R² on this recording. That bin is rarely the bin
> the user would pre-register. The pipeline does not announce the
> choice, the alternative bins' fits are discarded, and the reader of
> a published result has no way to tell whether the reported numbers
> are robust or whether they survive only at one cherry-picked bin.

The cleanest fix would be to remove the sweep entirely. That is a
breaking change for downstream notebooks and the CLI default. We chose
a path that is forwards-compatible *and* takes the forking path out of
the headline result.

## Decision

1. **Default `bin_size_ms` is a scalar (4 ms).** `criticality(rec)` now
   runs at one bin and reports the fit at that bin. The methodological
   choice is "the bin you pre-registered or justified from the
   autocorrelation time", not "the bin that maximised R²".

2. **Sequence input still works, with a forking-path warning.** Callers
   that pass `bin_size_ms=[2, 4, 8, 16, 32]` get the legacy R²-driven
   selection *and* a `UserWarning` pointing at this document. The full
   per-bin fit table is now exposed in `CriticalityResult.fits` so a
   reviewer can audit every fit, not just the winner.

3. **New standalone `bin_size_sweep` function.** Use this in
   manuscripts and supplements when you want to show the bin-size
   sensitivity of the fit without picking a winner. It returns a list
   of dicts, one per bin, with the same fields as `fits`.

4. **CLI changes.** `--crit-bins` is replaced by `--crit-bin-ms`
   (single value, default 4) and an opt-in `--crit-bin-sweep` for the
   legacy behaviour. The default `analyze` invocation no longer R²-shops.

## Alternatives considered

- **Hard-fail on sequence input.** Rejected: breaks downstream
  notebooks unnecessarily; the legacy behaviour is still valid for
  exploratory work as long as it is documented and audited via `fits`.
- **Pick by AIC / BIC.** Rejected: AIC/BIC over candidate bin sizes
  has the same forking-path concern, just dressed in better statistical
  language. The underlying issue is that the user is doing model
  selection on the very quantity they want to report.
- **Always return the full sweep and force the user to pick.**
  Rejected as a default: most users want a single number and a sensible
  default; we now provide that via the scalar path.

## Acceptance test

- `tests/test_analysis_criticality.py::test_scalar_bin_no_sweep` —
  scalar `bin_size_ms` returns a `CriticalityResult` with
  `optimal_bin_seconds == bin_size_ms / 1000` and `fits` of length 1.
- `tests/test_analysis_criticality.py::test_sequence_bin_warns` —
  sequence input emits the forking-path `UserWarning`.
- `tests/test_analysis_criticality.py::test_fits_table_complete` —
  every bin attempted is represented in `fits`.

## Re-review notes

- This change is non-breaking for users who passed a single bin via the
  old API. Users who relied on the default sequence sweep will see (a)
  a different headline result (because there is no R² shopping any
  more) and (b) a `UserWarning` if they restore the old behaviour.
- The criticality regression invariants in
  `tests/test_invariants.py` continue to pass: they were always
  parameterised on a chosen bin and tested the regression identities at
  that bin, not the bin-selection logic.
- Reviewer A's P0-1 is therefore resolved with the change documented in
  user-facing docstrings + `docs/inference.md` and re-review can verify
  the change by inspecting `params["bin_selection"]` in any
  `CriticalityResult` produced after this commit.
