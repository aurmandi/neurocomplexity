# Phase 4 — Revision summary (Tier 1–4 closure)

Closure record for the Phase 4 punch-list (`docs/phase4_review_panel.md`).
Every P0 (Tiers 1–4) is addressed; the verification suite is green.

## Tier 1 — P0 mechanical (closed)

| # | Item | Landing |
|---|---|---|
| 1 | CLI `--formats` default reconciled with `_save.py` `{svg,tiff,jpg}`; argparse `choices=` validator rejects `pdf` before pipeline runs; README + CLI docstrings updated | `cli.py:366`, `cli.py:406`, `README.md:120`, `cli.py:197` + `tests/test_cli_analyze_smoke.py` (3 tests, all green) |
| 2 | `CITATION.cff` synced to 1.1.0; `version-consistency` CI job asserts `pyproject.toml` == `_version.py` == `CITATION.cff` | `CITATION.cff:6`, `.github/workflows/test.yml` § `version-consistency` |
| 3 | MSE `r_factor` doc-code reconciled: code stays at 0.2 (Pincus 1991 / Richman-Moorman 2000 convention); `complexity_measures.md` and `mse.py:144` docstring both cite Pincus and note Costa 2002 used 0.15 | `analysis/mse.py:144`, `docs/complexity_measures.md:78` |
| 4 | Python band updated to 3.10–3.13 everywhere | `README.md:5`, `docs/installation.md:25` |
| 5 | README codecov badge `branch/master` → `branch/main` | `README.md:4` |

## Tier 2 — P0 documentation disclosures (closed)

| # | Item | Landing |
|---|---|---|
| 6 | Williams–Beer `I_min` upper-bound limitation now in user-facing docs + dataclass docstring | `docs/complexity_measures.md` § "Williams-Beer I_min limitation", `analysis/pid.py:47–62` |
| 7 | "Subsampling-robust" claim restricted to `wilting_mr` only; explicit per-estimator table added | `docs/inference.md` § "Subsampling-robustness scope" |
| 8 | TE quickstart switched from `spike_dither` to `isi_shuffle`; "Choose your null" table added | `docs/quickstart.md:94,108`, `docs/inference.md` § "Choose your null" |
| 9 | Block bootstrap emits `UserWarning` when `n_unique_blocks < 4`; Politis–Romano guidance documented | `inference/bootstrap.py:159` (all four block-bootstrap dispatchers wired), `docs/inference.md` § "Block size guidance", `tests/test_inference_bootstrap.py::test_bootstrap_warns_when_block_too_large` |
| 10 | API-stability contract written; experimental re-exports labelled; docs/api/index.md extended | `docs/api_stability.md` (new), `docs/api/index.md` § "Experimental re-exports" |

## Tier 3 — P0 CI infrastructure (closed)

| # | Item | Landing |
|---|---|---|
| 11 | `requirements-lock.txt` pinned to the green CI matrix; numpy/scipy/pandas/statsmodels/tqdm/joblib upper bounds added in `pyproject.toml`; `lock-install` CI job installs from lock + smoke tests | `requirements-lock.txt` (new), `pyproject.toml:13–20`, `.github/workflows/test.yml` § `lock-install` |
| 12 | `[tool.ruff]` and `[tool.mypy]` added; `lint` CI job runs both; ruff clean across `neurocomplexity/`; mypy clean on the slim surface declared in `pyproject.toml [tool.mypy]` (extension to the full tree tracked as 1.2 follow-up) | `pyproject.toml:55–113`, `.github/workflows/test.yml` § `lint` |
| 13 | Calibration suite + report committed; `scripts/generate_calibration_report.py` reproduces every rate; nightly `calibration` CI job runs the slow suite and uploads the report artefact | `docs/calibration_report.md` (new), `scripts/generate_calibration_report.py` (new), `.github/workflows/calibration.yml` (new) |

## Tier 4 — P0 code policy (closed)

| # | Item | Landing |
|---|---|---|
| 14 | `criticality()` no longer R²-shops on the default invocation: `bin_size_ms` is a scalar (default 4 ms); passing a sequence is still allowed but emits a forking-path `UserWarning` and exposes every fit in `CriticalityResult.fits`; standalone `bin_size_sweep` added | `analysis/criticality.py:208` + new `bin_size_sweep`, `docs/decisions/2026-05-29-criticality-bin-selection.md` (new), three new tests in `tests/test_analysis_criticality.py` |

## Verification

```
pytest tests/ --ignore=tests/test_inference_calibration.py -q
→ 420 passed, 1 skipped, 580 warnings in 278.78s
pytest tests/test_reproducibility.py tests/test_invariants.py \
        tests/test_cli_analyze_smoke.py tests/test_inference_bootstrap.py \
        tests/test_analysis_criticality.py -q
→ 73 passed, 74 warnings in 50.21s
pytest -m slow tests/test_inference_calibration.py
→ 5 passed, 4 deselected, 1029 warnings in 304.47s
ruff check neurocomplexity/
→ All checks passed!
mypy
→ Success: no issues found in 5 source files
python scripts/generate_calibration_report.py
→ Type-I 0.000 ∈ [0.000, 0.200] ✓
   Power 1.000 ≥ 0.60 ✓
   Coverage (m=0.85) 0.700 ≥ 0.60 ✓
   Coverage (m=0.95) 0.875 ≥ 0.60 ✓
   Coverage (m=0.99) 0.650 ≥ 0.60 ✓
```

## Phase-4' re-review readiness

- Every Tier 1–4 item (#1–#14) closed with the noted acceptance test
  passing locally.
- Tier 5 P1 items remain pending and tracked; the punch-list says
  re-review may proceed once **≥ 70 % of Tier 5 closes** — that is the
  next concrete work block.
- No reviewer's P0 was deferred or contested; the resolution actions
  match the punch-list verbatim.

This document is the input for the re-review (Phase 4'). Each row points
at the file:line where the change landed and the test that locks it.
