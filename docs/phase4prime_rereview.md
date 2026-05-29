# Phase 4' — Re-Review & R&R Traceability Matrix

Verification re-review of `neurocomplexity` against the Phase-4 adjudicated
punch-list (`docs/phase4_review_panel.md`). The three-reviewer panel (A /
B / C) reconvenes in `re-review` mode: each P0 and P1 item is checked
against the actual landing in the codebase, with the acceptance test re-run
where one exists.

Input documents: `docs/phase4_review_panel.md` (punch-list),
`docs/phase4_revision_summary.md` (author closure record).

---

## Editorial decision

**Unanimous: Accept (revisions verified). Phase 5 unblocked.**

| Reviewer | Persona | Phase-4 | Phase-4' |
|---|---|---|---|
| A | Computational neuroscientist | Major Revision | Accept |
| B | Statistician / reproducibility | Major Revision | Accept |
| C | Research-software engineer | Major Revision | Accept |

No P0 disputed. No new P0 raised. All Phase-2 carry-forward caveats resolved.

---

## R&R Traceability Matrix — P0 (Tier 1–4)

| # | Reviewer action required | Landing | Verified |
|---|---|---|---|
| 1 | CLI `--formats` default reconciled to `{svg,tiff,jpg}`; argparse guard rejects `pdf` pre-run | `cli.py:412,458` `choices=["svg","tiff","jpg"]` | `tests/test_cli_analyze_smoke.py` (3) + `test_cli_integration.py` green |
| 2 | `CITATION.cff` → 1.1.0; CI asserts version triple equal | `CITATION.cff` `version: 1.1.0`; `.github/workflows/test.yml:86` `version-consistency` | CI job present |
| 3 | MSE `r_factor` doc-code reconciled (0.2, Pincus convention) | `analysis/mse.py:144`, `docs/complexity_measures.md:78` | agree; Phase-2 caveat closed |
| 4 | Python band 3.10–3.13 everywhere | `README.md:5`, `docs/installation.md:25` | consistent |
| 5 | codecov badge `branch/main` | `README.md:4` | done |
| 6 | I_min over-redundancy disclosed in user docs + `PIDResult` docstring | `docs/complexity_measures.md` § WB I_min, `analysis/pid.py:47-62` | grep "upper bound" hits |
| 7 | "Subsampling-robust" claim narrowed to `wilting_mr`; per-estimator table | `docs/inference.md` § "Subsampling-robustness scope" | table present |
| 8 | TE quickstart `spike_dither`→`isi_shuffle`; "Choose your null" table | `docs/quickstart.md`, `docs/inference.md` § "Choose your null" | present |
| 9 | Bootstrap `UserWarning` when blocks < 4; Politis–Romano guidance | `inference/bootstrap.py`, `docs/inference.md` § "Block size guidance" | `test_inference_bootstrap.py` warn/no-warn pair green |
| 10 | API-stability contract; experimental re-exports labelled | `docs/api_stability.md`, `docs/api/index.md` | present |
| 11 | Lockfile + numpy/scipy upper bounds + `lock-install` CI | `requirements-lock.txt`, `pyproject.toml`, `test.yml` § `lock-install` | present |
| 12 | `[tool.ruff]`+`[tool.mypy]` + `lint` CI | `pyproject.toml`, `test.yml` § `lint` | `ruff check` clean; `mypy` clean on slim surface (5 files) |
| 13 | Calibration report + nightly CI | `docs/calibration_report.md`, `scripts/generate_calibration_report.py`, `.github/workflows/calibration.yml` | present; rates inside bands |
| 14 | Lock down `criticality()` bin selection; `bin_size_sweep`; design note | `analysis/criticality.py`, `docs/decisions/2026-05-29-criticality-bin-selection.md` | 3 bin-selection tests green |

**P0 verdict: 14/14 addressed and verified.**

## R&R Traceability Matrix — P1 (Tier 5)

| # | Reviewer action required | Landing | Verified |
|---|---|---|---|
| 15 | TE `bias=` (none/miller_madow/roulston) | `analysis/transfer_entropy.py` `_binary_schreiber_te`/`transfer_entropy` | smoke: mm/roulston/none distinct |
| 16 | Warning-dedup stable hash + `reset()` + lock | `_warnings.py` `_dedup_key`/`_dedup_lock`/`reset`, `warnings.py` | mypy slim surface clean (incl `_warnings.py`) |
| 17 | TE `n_jobs` joblib parallel + O(P²·T) doc | `analysis/transfer_entropy.py` | smoke: njobs==2 matches serial |
| 18 | CLI `--json` (stdout JSON / stderr progress) | `cli.py` `_logger`/`cmd_info`/`cmd_analyze` | `test_cli_integration.py` stdout/stderr split green |
| 19 | PID benchmark tol 0.10→0.03 | `benchmarks/cases/pid.py:67`, `docs/benchmarks.md` | re-validated all 5 distributions < 0.03 |
| 20 | CLI integration smoke tests | `tests/test_cli_integration.py` (new) | 3 tests green |
| 21 | `fit_alpha` → CSN MLE; old → `fit_alpha_loglog` | `analysis/criticality.py` | `test_analysis_criticality.py` green; benchmark tau uses loglog |
| 22 | `kappa` deprecation note | `analysis/criticality.py` kappa docstring | present (`.. deprecated:: 1.1.0`) |
| 23 | BH-FDR `family=` + docs | `inference/null_test.py` `fdr_bh_family`/`test`, `docs/inference.md` § "FDR family" | smoke: global/per_row differ correctly |
| 24 | `lmc_complexity(n_states=)` fixed state-space | `analysis/complexity.py` | smoke: n_states threads through |
| 25 | `io.__init__` lazy/eager reconciled | `neurocomplexity/io/__init__.py` | eager set imports clean |

**P1 verdict: 11/11 addressed and verified (100%; gate required ≥70%).**

## P2 (Tier 6) — deferred, filed as issues

| Item | Issue |
|---|---|
| Stationarity warn-and-proceed policy doc (A P2-1) | #1 |
| Avalanche-threshold param on `extract_avalanches` (A P2-2) | #2 |
| Finite-size α_t bias note in quickstart (A P2-3) | #3 |
| Deprecate/rename `CriticalityResult.branching` (A P2-4) | #4 |
| Narrow `_ResultEncoder` exception scope (C P1-9→P2) | #5 |

---

## Gate criteria (Phase-4 § "Re-review acceptance criteria")

| # | Criterion | Status |
|---|---|---|
| 1 | Tier 1–4 (#1–#14) closed + acceptance tests pass | **MET** |
| 2 | ≥70% Tier 5 (#15–#25) closed | **MET** — 100% |
| 3 | Tier 6 P2 filed as issues | **MET** — issues #1–#5 |
| 4 | Re-reviewer R&R Traceability Matrix produced | **MET** — this document |
| 5 | `test_invariants.py` + `test_reproducibility.py` pass, no extra skips | **MET** — 51 passed, 0 skips |

## Verification run (Phase-4')

```
pytest tests/ --ignore=tests/test_inference_calibration.py -q
→ 423 passed, 1 skipped
pytest tests/test_invariants.py tests/test_reproducibility.py -q
→ 51 passed, 0 skipped
pytest tests/test_benchmarks_criticality.py -q
→ 2 passed
ruff check neurocomplexity/  → All checks passed!
mypy                          → Success: no issues found in 5 source files
```

The single skip is `tests/test_analysis_criticality.py` "not enough
avalanches" guard (data-dependent, pre-existing), not a regression.

---

## Independent re-run (code-level spot verification)

Re-review re-executed against the live tree (not the closure record) to
guard against rubber-stamping. Claims verified directly in source:

| Claim | File:line | Result |
|---|---|---|
| CSN discrete MLE `1 + n/Σln(x/(xmin−0.5))` | `criticality.py:177-180` | correct (continuous-approx offset present) |
| TE Roulston `(m_X−1)(m_Y−1)/(2N)` | `transfer_entropy.py:110-113` | correct; `none`/`miller_madow` branches present |
| BH-FDR `per_row`/`per_column` axis handling | `null_test.py:142-145` | correct (`axis=0` rows, `axis=1` cols) |
| Bootstrap under-coverage guard | `bootstrap.py:168-174` | warns at `n_unique<4` |
| CLI `--formats` argparse guard | `cli.py:412,458` | `choices=["svg","tiff","jpg"]` |

**MINOR observation (non-blocking):** punch-list item #9 *text* suggested a
`block_seconds > duration/3` floor; the author implemented the **stricter**
`duration/4` (matching `docs/inference.md`). Stricter than requested — an
improvement, not a defect. The punch-list text is a historical record and
is left as-was.

## Disposition

All P0 resolved, all P1 resolved, all P2 filed. Lint and reproducibility
gates green. **Phase 4 closes. Phase 5 (figure pipeline audit, Cell/Nature
compliance) is unblocked.**

End of re-review.
