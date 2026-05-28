# `neurocomplexity` Publication Plan (v1, 2026-05-28)

Working agreement between Sazgar Arman Dinarvand (package author) and the AI
collaborator for taking `neurocomplexity` from current state (v1.1.0) to a
publication-ready, community-defensible release.

## Guiding principles

- **No claim without test.** Every published number reproducible from one CLI
  call against a seed.
- **No analysis without a citation.** Every estimator points at a primary
  reference and matches its canonical form (normalization, edge cases,
  conventions).
- **Reviewers > authors.** If the three-reviewer panel cannot defend a method,
  it does not ship.
- **Fail loud, never quietly wrong.** Silent NaN, silent edge-case fallback,
  silent unit mismatch = bugs.
- **Real data, not just toys.** Toy-data benchmarks gate CI, but the tutorial
  and headline validation run against a real Allen Neuropixels NWB session.

## Locked decisions

| Item | Decision |
|---|---|
| Real-data dataset | `session_715093703.nwb` (Allen Brain Observatory Neuropixels Visual Coding) — `C:\Users\Sazgar\OneDrive\Desktop\Arman_Dinarvand code sample\neuropixel\NeuropixelVisCodingData_cache\session_715093703\session_715093703.nwb` |
| Tutorial cadence | One block per work session, with pause for absorption |
| Tutorial style | **Theory first, code second.** Motivation → math → primary refs → code walk → run on real data → exit ticket |
| Phase 4 reviewers | Three independent personas: A criticality neuroscientist, B statistician, C software engineer |
| Reviewer workflow | Independent reports first, then adjudication round |
| Phase 0 + 1 vs 6 | Phases 0 and 1 run in background between tutorial sessions; user shows up only for tutorial blocks |
| Real-data validation (Phase 7) | One headline published result reproduced end-to-end on the Allen session |

## Phase map

| # | Phase | Gate to next |
|---|---|---|
| 0 | Baseline & hygiene | All 12 CI matrix entries green; coverage ≥ 85 %; ruff + mypy clean on public API; lockfile |
| 1 | Invariant & property tests | Hypothesis suite passes; PR / m / α / γ / PID / TE / p-value invariants enforced |
| 2 | Mathematical correctness audit | Per-estimator table: formula ↔ code line ↔ primary ref ↔ toy expected ↔ observed |
| 3 | Numerical & reproducibility audit | RNG determinism across OS × py3.10–3.13; pickle round-trip; edge cases tested |
| 4 | Three-reviewer panel | A, B, C reports + adjudication doc + revision punch-list; all P0 issues closed |
| 5 | Figure pipeline audit (Cell/Nature) | Each `viz.figure_*` passes scientific + flexibility + compliance checklist; regression notebook diffs clean |
| 6 | Tutorial — theory-first walk through every module on the Allen session | All blocks completed with exit tickets passed |
| 7 | Headline real-data reproduction gate | One published number reproduced within stated tolerance |
| 8 | API freeze + docs + software paper draft | `__all__` audited; symbols tagged Stable/Experimental; CHANGELOG; JOSS/eLife draft |
| 9 | Release engineering | TestPyPI → PyPI → Zenodo DOI; GitHub release with auto-DOI integration |

---

## Phase 0 — Baseline & hygiene

- Confirm all 12 (`{ubuntu, macos, windows} × {3.10, 3.11, 3.12, 3.13}`) CI
  entries green on `main`.
- Coverage report; target ≥ 85 % line + branch.
- `ruff` clean; `mypy --strict` on the public API surface.
- Add `requirements-lock.txt` (or `uv.lock`) for reproducible CI installs.

## Phase 1 — Invariant & property tests

Build a Hypothesis-based suite that locks in the following invariants:

| Statistic | Invariant |
|---|---|
| `BranchingResult.m` | `≥ 0`; finite; matches single-step Pearson at `k_max = 1` |
| `CriticalityResult.alpha_s/alpha_t` | Both `> 1` for valid fits; `gamma_predicted = (alpha_t-1)/(alpha_s-1)` exactly |
| `DimensionalityResult.PR` | `1 ≤ PR ≤ n_units`; PR(identity covariance) = `n_units` |
| `LMCResult.C` | `≥ 0`; `C(uniform) = 0`; `C(delta) = 0` |
| `MSEResult.entropy[0]` | Equals sample entropy at scale 1 |
| `PIDResult` | All four atoms `≥ 0`; sum = `I(target; sources)` |
| `TransferEntropyResult.matrix` | Diagonal = 0; non-negative; row = source convention enforced |
| `InferenceResult.p_value` | `∈ (0, 1]` (Phipson-Smyth floor); FDR ≥ raw p elementwise |

Any failure here is a Phase 2 bug.

## Phase 2 — Mathematical correctness audit

Estimator-by-estimator audit table — formula in primary reference, line(s) in
our code, toy case with closed-form expected value, observed value.

Targets:

- **Wilting & Priesemann 2018** — multi-step regression branching ratio with
  subsampling-bias correction.
- **Schreiber 2000 TE** — history-length convention, log base (nats),
  Miller-Madow correction.
- **Williams & Beer 2010 PID** — I_min lattice; non-negativity; sum =
  MI(target; X1, X2); correct atoms on XOR/AND/COPY/RDN/UNQ; interrogate
  whether the 0.10-nat AND tolerance is tight enough.
- **Sethna 2001 crackling-noise** — `gamma_predicted = (alpha_t-1)/(alpha_s-1)`.
- **López-Ruiz, Mancini, Calbet 1995 LMC** — `C = H · D`; check population
  and trajectory modes separately.
- **Costa, Goldberger, Peng 2002 MSE** — coarse-graining (non-overlapping
  mean) + Richman-Moorman SampEn with `r = 0.15 * std`; check the `r`
  default.
- **Cunningham & Yu 2014 participation ratio** — `(Σλ)² / Σλ²` from sample
  covariance.
- **Phipson & Smyth 2010 +1 p-value** + **Benjamini-Hochberg 1995 FDR** —
  matrix-shape semantics, NaN handling, monotone-step adjustment.

Deliverable: audit notebook with toy cases that assert agreement to a
documented tolerance.

## Phase 3 — Numerical & reproducibility audit

- Determinism: seed → identical output across Linux / macOS / Windows ×
  py3.10–3.13, including joblib-parallel paths.
- Edge cases: empty bins, single-unit pop, all-zero spike train, single-event
  avalanche, `var = 0`, `log(0)`.
- dtype audit: `int64` for spike counts, `float64` for stats,
  `datetime64[ns, UTC]` for timestamps.
- Pickle + deepcopy round-trip for every Result dataclass.
- `MemoryAllocationWarning` triggers correctly at documented threshold.
- Order-independence: bootstrap CI invariant under unit reordering; TE matrix
  unchanged under permuted unit order with relabeled rows/cols.

## Phase 4 — Three-reviewer panel

The AI plays three independent reviewer personas, each writing without
seeing the others' reports, then a fourth pass adjudicates conflicts.

- **Reviewer A — senior computational neuroscientist (criticality and
  dynamical systems).** Focus: are the methods appropriate? Is the criticality
  framing defensible (avalanche definition, time-bin sensitivity, finite-size
  scaling)? Is the TE → "effective connectivity" framing defensible? Does the
  package over-claim?
- **Reviewer B — statistician + reproducible-research advocate.** Focus:
  surrogate-null choice (does `spike_dither` preserve enough to be a useful
  null for TE specifically?), bootstrap block-size justification, FDR family
  definition (per-matrix vs per-study), multiple-comparisons across
  populations.
- **Reviewer C — research-software engineer.** Focus: API surface, type
  hygiene, test coverage gaps, error messages, performance / memory, error
  recovery, dependency management.

Each writes per-function findings, per-doc findings, per-benchmark findings,
scored P0 / P1 / P2. Adjudication produces a single revision punch-list.

## Phase 5 — Figure pipeline audit (Cell/Nature)

For every `viz.figure_*`, three checks:

1. **Scientifically correct.** Avalanche size distribution → log-binned,
   log-log axes, fit-line over the fitted range, finite-size cutoff marked.
   TE network → arrows have direction, edge widths log-scaled if values span
   orders of magnitude, p < α filter explicit. PR → eigenspectrum panel
   includes cumulative curve. Etc.
2. **Flexible.** `palette`, `panel_label`, `ax`, `figsize`, plus per-figure
   knobs (log/linear, fit-range override, threshold override). All documented
   with numpydoc + examples.
3. **Cell/Nature compliant.** Invoke the `data-visualization-nature` skill
   once per figure type: panel-letter font/weight, mm-sized columns (single
   89 mm, 1.5 ~136 mm, double 183 mm), 5–7 pt text, sans-serif, ticks inward,
   no top/right spines unless meaningful, colourblind-safe palette default,
   no JPG-of-line-art.

Deliverable: regression notebook that renders every figure type from canned
fixtures and image-diffs against committed reference PNGs. Each figure also
gets a compliance checklist in its docstring.

## Phase 6 — Tutorial (theory first, on real Allen data)

Tutorial dataset: `session_715093703.nwb` (Allen Visual Coding).

Every block follows the same shape:

1. **Motivation (5 min)** — Why does extracellular electrophysiology need
   this? What real neuroscience question fails without it?
2. **Theory (15–25 min)** — Mathematical formalism, derivation of the
   estimator, what it assumes and what it discards. Whiteboard-style; no
   code.
3. **Primary references (5 min)** — The 1–3 canonical papers; what each
   contributed; which limits remain unresolved.
4. **Code walk (15 min)** — Read the module function by function, mapping
   each line back to the theory.
5. **Run on real data (15 min)** — Execute on the Allen session; interpret
   the numbers.
6. **Exit ticket (5 min)** — 3 comprehension questions. Wrong answer →
   revisit before moving on.

Block order:

| Block | Topic | Theory anchor |
|---|---|---|
| 0 | `SpikeRecording` + extracellular recording fundamentals | What an electrode measures; spike sorting → unit; why "good" units are a curation choice |
| 1 | I/O & curation | NWB schema; Kilosort → automated QC → human curation; why we re-validate at load |
| 2 | Criticality (avalanches, branching ratio, exponents, shape collapse) | Self-organized criticality; Beggs & Plenz 2003; Wilting-Priesemann subsampling fix; Sethna crackling-noise |
| 3 | Information flow (TE, PID, continuous signals) | Schreiber TE derivation; Williams-Beer PID lattice; why MI alone fails for triadic interactions |
| 4 | Geometry & complexity (PR, manifold, MSE, LMC) | Cunningham-Yu neural manifolds; Costa MSE; López-Ruiz LMC; the geometry-vs-complexity distinction |
| 5 | Autonomy & stationarity | Bertschinger autonomy index; why every analysis above is stationarity-sensitive |
| 6 | Inference (bootstrap, surrogates, FDR, alternatives) | Block bootstrap rationale; surrogate-data hierarchy; Phipson-Smyth +1 floor; BH-FDR |
| 7 | Visualisation | Tufte / Wilke / Cell-Nature display principles |
| 8 | CLI + benchmarks | Why benchmark suites matter; reproducibility envelope |

## Phase 7 — Headline reproduction gate

Pick one published result on the Allen Visual Coding cohort (e.g.,
near-critical `m̂ ≈ 0.98` in cortex, or a specific TE flow pattern) and
reproduce it within stated tolerance using the package. Document the delta
against the reference. Without this, "ready for the research community" is
unsupported.

## Phase 8 — API freeze, docs, software paper draft

- Audit every `__all__`; downgrade non-stable symbols to `_underscore`.
- Tag each public symbol `Stable` / `Experimental` in its docstring header.
- Write CHANGELOG entry for v1.2.0 / 2.0.0.
- Draft a ~1500-word software paper (JOSS or eLife): statement of need,
  summary, software components, demonstration on Allen data, validation
  against benchmarks.

## Phase 9 — Release engineering

- Version bump (1.2.0 if additive, 2.0.0 if any break).
- Build sdist + wheel, publish to TestPyPI, install in clean venv,
  smoke-test.
- Publish to PyPI; create GitHub release with Zenodo DOI integration.
- Update README badges (CI, codecov, PyPI, DOI).

---

## Tracking

Updates to this plan are tracked in git history. Any phase-scope changes
require updating this file before code changes begin.
