# Phase 4 — Three-Reviewer Panel: Adjudication & Revision Roadmap

Independent reviews of `neurocomplexity` v1.1.0 by three reviewers (A / B / C),
adjudicated into a single revision punch-list. Full per-reviewer reports live
in `docs/phase4_review_A_report.md`, `_B_report.md`, `_C_report.md`. This
document is the synthesis the package author works against.

---

## Editorial decision

**Unanimous: Major Revision.**

| Reviewer | Persona | Decision | P0 / P1 / P2 |
|---|---|---|---|
| A | Senior computational neuroscientist (criticality + dynamical systems) | Major Revision | 3 / 5 / 4 |
| B | Statistician + reproducible-research advocate | Major Revision | 3 / 6 / 5 |
| C | Senior research-software engineer | Major Revision | 5 / 9 / many |

**Consensus.** All three reviewers agree the package's *numerical core* is
sound (Phipson–Smyth, BH-FDR, BC bootstrap, Sethna identity, dtype/seed
discipline) and unusually well-engineered for a pre-JOSS submission. The
blocker is **the user-facing surface**: defaults, docstrings, docs pages,
CLI behaviour, and CI-gate completeness. None of the agreed P0 items
requires changing the audited mathematics. All are mechanical or
documentation-level. Estimated revision effort: **3–5 working days** for one
author.

---

## Cross-reviewer overlap matrix

Issues raised by ≥ 2 reviewers (high-confidence punch-list items):

| Item | A | B | C |
|---|---|---|---|
| Williams–Beer I_min over-redundancy not in user-facing docs | P0-2 | (carries Phase-2 caveat) | — |
| MSE `r_factor` doc-code conflict (code 0.2, `docs/complexity_measures.md:78` advertises 0.15) | P1-2 | (carries Phase-2 caveat) | — |
| Subsampling-robust framing over-generalised beyond branching ratio | P0-3 | (echoed in inference-validity statement) | — |
| Binary k=l=1 TE labelled "effective connectivity" | P1-1 | implied by P0-2 (wrong null for TE in quickstart) | — |
| Undocumented public re-exports (`ContinuousSignal`, `ProvenanceRecord`, `set_progress`, `estimate_bin_spikes_bytes`, `warnings`, `viz`) | — | — | P0-5 (also Phase-0 audit) |
| Stationarity warn-and-proceed policy undocumented | P2-1 | (cited in inference-validity statement) | — |

Single-reviewer P0s elevated on objective severity:

- **A P0-1** — R²-driven `optimal_bin_seconds` is an unguarded forking path on
  the headline criticality result (`analysis/criticality.py:209-248`).
- **B P0-1** — Calibration suite excluded from CI with no published Type-I /
  power / coverage table (`.github/workflows/test.yml:32`).
- **B P0-2** — `docs/quickstart.md:94,105-109` recommends `spike_dither` for
  TE; calibration tests themselves use `isi_shuffle`.
- **B P0-3** — `bootstrap.py:154-173` silently degenerates when
  `block_seconds > duration / 3`.
- **C P0-1** — `cli.py:367` defaults `--formats pdf svg png` but `_save.py:16`
  rejects `pdf`. README headline CLI example crashes. *No CLI integration
  test caught it.*
- **C P0-2** — `CITATION.cff:6` is v1.0.0 while package is v1.1.0; three
  independent version sources.
- **C P0-3** — No lockfile. Phase 0 plan gate (`docs/publication_plan.md:37`).
- **C P0-4** — No `[tool.ruff]` / `[tool.mypy]` in `pyproject.toml`; CI
  doesn't run either. Plan promised both clean (`docs/publication_plan.md:55`).

---

## Revision punch-list

Ordered by priority and effort. Effort estimates are for one focused
author. *Owner* fields default to package author unless reassigned.

### Tier 1 — P0 mechanical / non-controversial (≤ 1 day total)

| # | Item | Source | Effort | Acceptance test |
|---|---|---|---|---|
| 1 | Reconcile CLI `--formats` default with `_save.py` (`{svg, tiff, jpg}`); add CLI-side validator that rejects unsupported formats *before* analysis runs; update README + quickstart | C P0-1 | 1 h | New `tests/test_cli_analyze_smoke.py`: `main(["analyze", synth_nwb, "-o", tmp])` exits 0 and writes ≥ 1 figure |
| 2 | Sync `CITATION.cff` to 1.1.0; add CI job that asserts `pyproject.toml` version == `_version.py` == `CITATION.cff` | C P0-2 | 30 min | New CI step fails if any version string drifts |
| 3 | Fix MSE `r_factor` doc-code conflict. Pick one: keep code at 0.2 and correct docstring + `complexity_measures.md` to "Pincus 1991 convention; Costa 2002 used 0.15", *or* change code to 0.15 with a CHANGELOG note | A P1-2 / B caveat | 30 min | `docs/complexity_measures.md` and `analysis/mse.py:144` agree; Phase-2 caveat closed |
| 4 | Update `README.md:5` and `docs/installation.md:25` to Python 3.10–3.13 to match CI matrix and `pyproject.toml:10` | C P1-12 | 15 min | grep returns nothing for `3.10-3.12` outside changelog |
| 5 | Switch README codecov badge from `branch/master` to `branch/main` | C P1-11 | 5 min | Badge renders |

### Tier 2 — P0 documentation disclosure (≈ 1 day)

| # | Item | Source | Effort | Acceptance test |
|---|---|---|---|---|
| 6 | Add an explicit "I_min limitation" paragraph to `docs/complexity_measures.md` (new section or new `docs/information_decomposition.md`); add a one-line "redundancy is an I_min upper bound; see docs" note to `PIDResult.redundancy` docstring (`analysis/pid.py:47-50`) | A P0-2 | 1 h | `grep -i "over-estim\|upper bound" docs/` returns a hit; PIDResult docstring mentions limitation |
| 7 | Narrow "subsampling-robust" claim. Audit every doc page and docstring; restrict the claim to `wilting_mr` only. Add a one-paragraph "What is *not* subsampling-corrected" section to `docs/inference.md` listing TE, PID, PR, MSE, LMC | A P0-3 | 1 h | grep for "subsampling" outside `branching.py` and the new section returns nothing claiming protection |
| 8 | Replace `spike_dither` quickstart recommendation for TE with `isi_shuffle`. Add a "Choose your null" table to `docs/inference.md` that maps each statistic to the appropriate surrogate(s) with one-line rationale | B P0-2 | 30 min | `docs/quickstart.md` TE example uses `isi_shuffle`; new table present |
| 9 | Document and *guard* bootstrap block-size. (a) Emit `UserWarning` from `bootstrap()` when `block_seconds > duration/3` ("only N unique blocks; CI may under-cover"); (b) add a Politis–Romano rule-of-thumb paragraph to `docs/inference.md`; (c) cite the calibration table once it exists | B P0-3 | 1 h | New test asserts warning emitted on a 30 s recording with 10 s blocks; doc has guidance |
| 10 | Document API-stability contract. (a) Add `docs/api_stability.md` (or a section in `docs/index.md`) stating SemVer + the rule "anything not in `docs/api/` is unsupported"; (b) tag each public docstring with `[Stable]` or `[Experimental]`; (c) add an automodule page covering the six undocumented re-exports OR demote them | C P0-5 / Phase 0 | 2 h | New doc page present; `pytest --collect-only docs/api_stability.md` parses; six previously-undocumented names now appear in `docs/api/index.md` |

### Tier 3 — P0 infrastructure / CI gates (≈ 1 day)

| # | Item | Source | Effort | Acceptance test |
|---|---|---|---|---|
| 11 | Generate `requirements-lock.txt` (or `uv.lock`) from the green CI matrix; add a `lock` CI job that re-installs from it and runs the smoke test. Pin numpy/scipy upper bounds in `pyproject.toml` (`numpy>=1.24,<3`, `scipy>=1.10,<2`) | C P0-3 | 2 h | New CI job green on Linux; pinned bounds present |
| 12 | Add `[tool.ruff]` and `[tool.mypy]` to `pyproject.toml` (target 3.10; strict mypy on `neurocomplexity/` only). Wire a `lint` CI job. If `mypy --strict` does not pass today, gate it on the smallest passing surface and file follow-up tickets for the rest | C P0-4 | 2 h | CI lint job green; `ruff check` and `mypy neurocomplexity/` both clean |
| 13 | Run `CALIBRATION_FULL=1 pytest -m slow tests/test_inference_calibration.py` on a CI-equivalent runner; commit results as `docs/calibration_report.md` with Type-I rate, power, coverage; add a nightly (or on-tag) CI job that re-runs and diff-asserts | B P0-1 | 4 h + CI compute | `docs/calibration_report.md` exists; CI job present and green |

### Tier 4 — P0 code policy (real design change)

| # | Item | Source | Effort | Acceptance test |
|---|---|---|---|---|
| 14 | Lock down bin-selection in `criticality()`. Options: (a) make `bin_size_ms` a *single* value (no sweep, no R² shopping); add a `bin_size_sweep` standalone function that exposes the full sweep table; OR (b) keep the sweep but expose every fit in `CriticalityResult` (not just the best) and force the user to declare which bin to report. Document the choice as a methodological forking-path concern | A P0-1 | 4 h | New behaviour locked by a regression test; docs have a "Bin-size sensitivity" section |

**Subtotal Tier 1–4 (P0):** ≈ 3 working days.

### Tier 5 — P1 revisions (≈ 1–2 days)

| # | Item | Source | Effort |
|---|---|---|---|
| 15 | Add `bias=` parameter to `transfer_entropy` exposing the Roulston `(m_X−1)(m_Y−1)/(2N)` form alongside the current simplified Miller-Madow | A / B caveats | 4 h |
| 16 | Switch warning-dedup state from `id(rec)` to a stable hash incorporating `rec.source.source_hash + analysis_name`; add a public `nc.warnings.reset()`; add a thread lock | C P1-7 | 2 h |
| 17 | Add `n_jobs: int = 1` to `transfer_entropy`; dispatch the P² pair loop via `joblib.Parallel`; document O(P²·T) complexity in the docstring | C P1-8 | 2 h |
| 18 | Add `--json` mode to `cli.py info` and `analyze`; route human progress to stderr, JSON payload to stdout | C P1-6 | 2 h |
| 19 | Tighten PID benchmark tolerance from 0.10 → 0.03 nats in `benchmarks/cases/pid.py` and re-validate | B / Phase-2 caveat | 1 h |
| 20 | Add CLI integration smoke tests (`tests/test_cli_analyze.py`, `test_cli_figure.py`) running the README quickstart commands end-to-end | C P1-13/14 | 2 h |
| 21 | Replace `fit_alpha` log-binned histogram regression with Clauset–Shalizi–Newman MLE (preserve old as `fit_alpha_loglog` for back-compat) | A P1-5 | 4 h |
| 22 | Add a "`kappa` is not Shew (2009) κ" note to `CriticalityResult.kappa` docstring; mark deprecated; remove in next minor | A P1-4 | 30 min |
| 23 | Document BH-FDR family definition explicitly in `docs/inference.md`; optionally add `family="global" / "per_row" / "per_column"` parameter to `test()` for matrix-valued statistics | B P1 | 2 h |
| 24 | Expose `n_states` parameter on `lmc_complexity` so cross-population comparison can use a fixed state-space; document the trade-off | A / C carries | 1 h |
| 25 | Reconcile `io.__init__.py` lazy-vs-eager pattern: make all of `add_quality`, `add_anatomy`, `add_trials` eager (or all lazy) | C P1-10 | 1 h |

**Subtotal Tier 5 (P1):** ≈ 2 days.

### Tier 6 — P2 nice-to-have (deferred)

- Document the stationarity warn-and-proceed policy explicitly (A P2-1; rationale: warning ≠ hard block because user may have valid reasons to proceed with a flagged-non-stationary epoch — but the choice must be documented).
- Expose avalanche-threshold parameter on `extract_avalanches` (A P2-2).
- Surface the finite-size α_t bias note in `docs/quickstart.md` (A P2-3).
- Deprecate or rename `CriticalityResult.branching` (single-step Beggs ratio) to avoid confusion with `wilting_mr.m` (A P2-4).
- Tighten `_ResultEncoder` exception handling in `cli.py:36-50` to narrow scope (C P1-9 demoted to P2 after Tier 5 lands).

---

## Specific-caveat verdicts (Phase 2 carry-forward, by reviewer)

| Phase-2 caveat | Reviewer A | Reviewer B | Reviewer C |
|---|---|---|---|
| LMC `N_states` data-dependent | "Expose `n_states` param" (P1-3) | "concur, cross-pop comparison untrustworthy without fixed N" | n/a (math) |
| MSE `r_factor` mis-cited | "doc-code conflict is live; pick one" (P1-2) | "concur; recommend docstring fix" | n/a (math) |
| Schreiber TE Miller-Madow simplified | "concur; not a blocker; document the choice" | "ship `bias=` param exposing Roulston" | n/a (math) |
| WB-PID I_min over-redundancy + benchmark tolerance | P0-2 (must disclose in user docs) | "tighten benchmark 0.10 → 0.03; escalate to docstring" | n/a (math) |

All three reviewers concur on every Phase-2 carry-forward. No reviewer
disputes any audit finding. Resolution actions land as Tier 2 (#6, #8, #9)
and Tier 5 (#15, #19, #24).

---

## Re-review acceptance criteria (Phase 4', the next gate)

For the package to clear Phase 4 and unblock Phase 5 (figure audit),
**all of the following must hold**:

1. Every Tier 1–4 item (#1–#14) closed with the noted acceptance test
   passing locally and in CI.
2. ≥ 70 % of Tier 5 items (#15–#25) closed. Any deferred item must be
   filed as a GitHub issue with a milestone and explicit deferral
   rationale.
3. Tier 6 items optional for Phase 4 but must be filed as issues.
4. A re-reviewer agent (same persona configuration: A/B/C) runs the
   `re-review` mode of the academic-paper-reviewer skill against the
   revised codebase, with this document as input, and produces an
   R&R Traceability Matrix demonstrating each P0 punch-list item is
   addressed.
5. `tests/test_invariants.py` and `tests/test_reproducibility.py` both
   still pass on all 12 CI matrix entries with no skips beyond the
   already-justified `tests/test_inference_calibration.py` exclusion
   (which will itself be deleted once item #13 is landed and the
   nightly job replaces it).

If any reviewer's P0 cannot be resolved with the proposed mechanical fix
and the author argues for a different resolution, that disagreement is
escalated to a dedicated dialogue in this file (`## Disagreements`)
before re-review.

---

## Summary numbers

| Metric | Value |
|---|---|
| Reviewers convened | 3 (A, B, C) |
| Decisions | 3 / 3 Major Revision |
| Distinct P0 findings (post-dedup) | 13 |
| Distinct P1 findings (post-dedup) | ~17 |
| Distinct P2 findings (post-dedup) | ~10 |
| Numerical bugs found | **0** |
| Documentation gaps found | **most P0** |
| CI / packaging gaps found | **3 P0** |
| Single-day effort to close all P0 | ≈ 3 |
| Single-day effort to close all P0 + ≥ 70 % P1 | ≈ 5 |

The package is closer to JOSS-ready than the verdict reads: the math is
audited, the inference machinery is correct, the engineering is unusually
disciplined. The Major Revision verdict reflects user-surface defects
(broken CLI default, mis-cited convention, missing disclosures) that a
methods reviewer would catch in the first 30 minutes of evaluation. None
of those is hard to fix; the package author can land Tier 1–4 in a single
focused work-week.

---

## Next actions

1. Author triages this punch-list, accepts / disputes each item, writes
   responses inline.
2. Author begins Tier 1 (mechanical, non-controversial) — these can land
   today without further discussion.
3. Tier 4 (criticality bin-selection) needs a one-page design note before
   code lands; recommend a short author-vs-reviewer dialogue in
   `docs/decisions/2026-05-29-criticality-bin-selection.md`.
4. After all Tier 1–4 land and ≥ 70 % of Tier 5 lands, re-dispatch the
   three-reviewer panel in `re-review` mode against this document.
5. Phase 5 (figure pipeline audit, Cell/Nature compliance) unblocks once
   Phase 4 closes.

End of adjudication.
