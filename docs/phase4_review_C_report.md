# Reviewer C — Research-software engineering review of `neurocomplexity` v1.1.0

## Summary recommendation

**Major Revision.** The package is unusually well-engineered for a pre-JOSS submission: immutable core data type with enforced invariants, lazy-import contract for heavy optional deps, a reproducible surrogate pool seeded via `SeedSequence`, a 12-entry OS × Python CI matrix, ~50 test modules including a Hypothesis invariant suite, a `py.typed` marker, an `MIT` LICENSE, `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, GitHub issue templates, a JOSS-style `paper/paper.md`, and a separate PyPI publish workflow. None of the findings below threaten the scientific claims; they threaten *adoption and reproducibility hygiene*. The blocking items are (1) public-facing format inconsistencies between the docs/CLI and `save_publication`, (2) stale `CITATION.cff` version, (3) no lockfile or pinned dependency snapshot, (4) no `ruff` / `mypy` tool configuration in `pyproject.toml` despite the publication plan promising "ruff clean; mypy --strict on the public API surface", and (5) several undocumented public re-exports. All are mechanical, low-risk fixes. Once those land, this package would be ready for JOSS review.

## Strengths

- Immutable `SpikeRecording` dataclass with `__post_init__` invariant enforcement (`neurocomplexity/core/recording.py:61-107`) — the right scientific-Python pattern; cheap to share between threads and trivially cacheable.
- Lazy-import contract for I/O loaders genuinely keeps `import neurocomplexity` cheap; `io/__init__.py:25-47` and `__init__.py:13-17` (`viz` ImportError guard) are both correct.
- Reproducible surrogate pool uses `SeedSequence.spawn` to derive per-draw child seeds (`inference/pool.py:81-82`), with a documented LRU cache. This is the right pattern for joblib-parallel determinism.
- `ProvenanceRecord` with BLAKE2b head+tail+filesize fingerprint (`core/provenance.py:83-98`) — sensible cheap fingerprint, attached to every recording and every result.
- Domain exception hierarchy rooted at `NeurocomplexityError` (`core/exceptions.py:10`) lets users `except NeurocomplexityError` cleanly.
- `MemoryAllocationWarning` is dynamic (25% of `psutil.virtual_memory().available`) rather than a static 1 GiB threshold (`analysis/_binning.py:35-54`) — more sensible than the docs claim.
- Full 12-cell CI matrix (`{ubuntu, macos, windows} × {3.10, 3.11, 3.12, 3.13}`) with `fail-fast: false` in `.github/workflows/test.yml:18-20`.
- JOSS-style `paper/paper.md` already drafted and renders to `paper/paper.pdf`; non-trivial advance over typical first-time JOSS submissions.
- Bin-allocation pre-flight (`estimate_bin_spikes_bytes`, `_binning.py:17`) exposed for the user to plan a session without triggering allocation.
- Domain-specific `Warning` subclasses (`_warnings.py:11-34`) cleanly re-exported under `neurocomplexity.warnings` so `warnings.filterwarnings(category=nc.warnings.X)` Just Works.

## Findings

### P0 — publication blockers

#### 1. Output-format contract is inconsistent across README, docs, CLI, and `save_publication`
- Where: `neurocomplexity/viz/_save.py:16` (`_VALID_FORMATS = {"svg", "tiff", "jpg"}`) vs `neurocomplexity/cli.py:367` (`default=["pdf", "svg", "png"]`) and `cli.py:407`, vs `README.md:122-124` ("PDF/SVG/PNG"), vs `docs/quickstart.md:131` ("PDF output is intentionally not supported").
- What is wrong: `save_publication` raises on `pdf` and rejects anything not in `{svg, tiff, jpg}`. But the CLI subcommand `analyze` defaults `--formats pdf svg png`, calls `viz.save_publication(..., formats=tuple(args.formats))`, and will therefore *crash on its default invocation* (line 239, 243, 247, 251, 255 of `cli.py`). The `figure` subcommand has the same bug (`cli.py:328`). `README.md:122-124` then advertises the broken default to new users.
- Why it matters: The headline CLI example in the README is broken. A new user `pip install`-ing today and running `neurocomplexity analyze ... -o results/` will hit a `ValueError` on the first per-analysis save. This is a load-bearing entry-point bug that will appear in every JOSS reviewer's smoke-test.
- Proposed fix: Reconcile to one supported set. Recommended: keep `save_publication`'s `{svg, tiff, jpg}` (it matches the Nature spec the package advertises) and change the CLI defaults + README + quickstart to match. Add a CLI-level validator that rejects unsupported formats with a one-line message *before* the analysis runs (failing late, after a 5-minute load, is unacceptable UX).

#### 2. `CITATION.cff` is stale (claims v1.0.0, package is v1.1.0)
- Where: `CITATION.cff:6` (`version: 1.0.0`, `date-released: "2026-05-16"`) vs `pyproject.toml:7` and `neurocomplexity/_version.py:1` (both `1.1.0`).
- What is wrong: Three sources of truth for the version (`pyproject.toml`, `_version.py`, `CITATION.cff`) drift independently. The CHANGELOG also has a `v1.1.0` entry.
- Why it matters: Zenodo + JOSS read `CITATION.cff` directly to mint DOIs. A v1.1.0 release with a v1.0.0 CITATION will publish a wrong-version DOI. JOSS reviewers run automated lints that compare versions.
- Proposed fix: Either (a) consolidate to a single source — read `__version__` from `_version.py` and have `pyproject.toml` use `[tool.setuptools.dynamic]` `version = {attr = "neurocomplexity._version.__version__"}`, and write a release script that bumps `CITATION.cff` alongside; or (b) at minimum, add a CI check that the three strings agree, and bump `CITATION.cff` to 1.1.0 now.

#### 3. No lockfile / reproducible install manifest
- Where: repo root — no `requirements-lock.txt`, no `uv.lock`, no `pip-tools` output. Confirmed by `ls` of the repo root.
- What is wrong: `pyproject.toml` floors every dependency (`numpy>=1.24`, `scipy>=1.10`, etc.) with no ceilings except on the docs extra. The CI `pip install -e ".[dev,nwb,viz]"` resolves whatever is on PyPI on the day the job runs.
- Why it matters: Phase 0 of the publication plan (`docs/publication_plan.md:37`) explicitly gates Phase 1 on "lockfile". The package's reproducibility claim is end-to-end: input file → result → figure, byte-stable across re-runs. That claim is undefended at the dependency layer. A future scipy / numpy minor bump can silently shift a `np.linalg.eigvalsh` result and quietly invalidate every benchmark in `results/benchmarks/v1.0.0.csv`. JOSS reviewers explicitly look for this.
- Proposed fix: Add `requirements-lock.txt` (or `uv.lock`) generated from the green CI matrix; add a `lock` CI job that re-installs from it and runs the smoke test. At minimum, pin numpy/scipy upper bounds (`numpy>=1.24,<3`, `scipy>=1.10,<2`) — the loose ceiling is a genuine breakage risk under numpy 2.x ABI changes already in production.

#### 4. No `ruff` / `mypy` configuration shipped despite plan promising both clean
- Where: `pyproject.toml` ends at line 53; no `[tool.ruff]`, no `[tool.mypy]`. Repo root has no `ruff.toml`, no `mypy.ini`, no `setup.cfg`. (Confirmed via `ls` of repo root.) `CONTRIBUTING.md` and `docs/publication_plan.md:55` both say "ruff clean; mypy --strict on the public API surface".
- What is wrong: There is no per-project ruff rule-set (which rules? line length? target-version?), no mypy configuration (strict on public surface only? what is the public surface?). The CI workflow runs neither `ruff` nor `mypy` (`test.yml:32-35`).
- Why it matters: `py.typed` is shipped as `package-data` (`pyproject.toml:46`), which advertises to downstream tools that this package is mypy-checkable. Without a mypy gate the contract is unenforced and downstream users hit `Any` / type errors that the maintainers never see. The plan explicitly makes this a Phase 0 gate.
- Proposed fix: Add `[tool.ruff]` (target 3.10, sensible default rule set) and `[tool.mypy]` (`python_version = "3.10"`, `strict = true` on `neurocomplexity` only — not tests), wire both into a `lint` CI job. If `mypy --strict` does not pass today, gate it on the `neurocomplexity` public package and document the gap.

#### 5. Several public re-exports are undocumented in `docs/`
- Where: `neurocomplexity/__init__.py:6-9` re-exports `ContinuousSignal`, `ProvenanceRecord`, `set_progress`, `estimate_bin_spikes_bytes`, `warnings`, `viz` at the top level. `docs/api/index.md` does not include the `ContinuousSignal`, `ProvenanceRecord`, `_warnings`, `_progress`, or `_binning` automodule directives. `docs/quickstart.md` mentions `nc.set_progress(True)` (line 159) and `nc.warnings.*` (line 144) by name but neither is documented in `api/index.md`.
- What is wrong: The public API surface is wider than what the docs commit to supporting. A user inspecting `dir(nc)` and pickling around a `ContinuousSignal` cannot find a reference page.
- Why it matters: The API-stability contract is undefined. The publication plan Phase 8 (`docs/publication_plan.md:204`) calls for tagging each public symbol Stable/Experimental — that audit cannot start until the documented surface matches `__all__`.
- Proposed fix: Either (a) add automodule pages for `core.continuous`, `core.provenance`, `_progress`, `_warnings`, and `analysis._binning` to `docs/api/index.md`; or (b) demote `estimate_bin_spikes_bytes` to `nc.analysis.estimate_bin_spikes_bytes` and prune the top-level re-export. Add a one-paragraph "API stability" section to `docs/index.md` stating the package follows SemVer and that anything underscore-prefixed or not in `docs/api/` is unsupported.

### P1 — must address

#### 6. CLI does not produce machine-parseable output for scripting
- Where: `cli.py:cmd_info` (line 93-110), `cmd_analyze`'s in-loop `print(f"  alpha_s={...}")` (lines 130-179).
- What is wrong: Every subcommand writes free-form `print(...)` to stdout with no `--json` flag. The `info` subcommand has no JSON mode at all. `analyze` writes `results.json` to disk but emits human-formatted progress to stdout that scripts must parse.
- Why it matters: A reproducible-research user wants `neurocomplexity info session.nwb --json | jq .` to drive a snakemake/nextflow pipeline. Today, that user has to scrape free-form text.
- Proposed fix: Add `--json` to `info` (dumps the same summary as a JSON dict to stdout). Send all human-progress output of `analyze` to stderr; reserve stdout for the JSON payload location.

#### 7. Warning dedup state is a process-global module-level `set` (race-prone, leaks across tests)
- Where: `neurocomplexity/_warnings.py:50` (`_seen: set[tuple[int, str]] = set()`) and `:83` (`_stationarity_seen`).
- What is wrong: Dedup keys use `id(rec)` which is a CPython address that gets recycled when a recording is garbage-collected, so a brand-new recording at the same address silently inherits the previous one's "already warned" state. Phase 1 audit flagged this as a py3.10 flake source. There is no thread-lock guarding mutation, so multi-thread inference (joblib threading backend) can race. There is no API to clear the set except a test-only `_reset_dedup()`.
- Why it matters: Silent-quietly-wrong; a user importing `neurocomplexity` from a long-running notebook will get warnings on some recordings and not others depending on which ids got recycled.
- Proposed fix: Switch from `id(rec)` to a stable hash that includes `rec.source.source_hash` + `analysis_name`. Document the dedup discipline. Add a thread lock if you keep the set. Expose a public `nc.warnings.reset()` for users running multi-session notebooks.

#### 8. `transfer_entropy` inner loop is O(P²) sequential with no parallel option
- Where: `analysis/transfer_entropy.py:150-152`.
- What is wrong: Every pair `(s, t)` is computed sequentially. With 8 populations that's 56 calls — fine. With 50 populations it's 2 450, with 200 populations 39 800. There is no `n_jobs=` kwarg, no chunked Numba/Cython kernel, no documentation that this is O(P²) so the user can plan.
- Why it matters: The package will be used on Allen Visual Coding sessions that easily reach 30+ recorded areas, and on aggregated multi-probe sessions with many sub-populations. A 30-population TE matrix today will spin on a single core.
- Proposed fix: Add `n_jobs: int = 1` and dispatch the pair loop via `joblib.Parallel(n_jobs=n_jobs)`; the binary-TE kernel has no shared mutable state. Document the complexity in the function docstring: "Wall time is O(P² · T) for P populations and T bins." Cap effective parallelism if the kernel ever takes locks.

#### 9. `_ResultEncoder` silently swallows exceptions (CLI JSON serialisation can be lossy)
- Where: `cli.py:36-50`. The `try / except Exception / pass` block on lines 47-49 silently falls through to `super().default(obj)` which itself raises — so the only visible effect is a less specific error, but the deeper problem is that `__dict__` fallback is brittle (any non-dataclass result with a non-serialisable internal state will mis-serialise without warning).
- What is wrong: Bare `except Exception` is the classic silent-bug shape that the project's own README explicitly forbids ("Fail loud, never quietly wrong" — publication plan principle).
- Proposed fix: Either implement `default` for each concrete result dataclass explicitly, or narrow the except to `(AttributeError, TypeError)` and re-raise everything else.

#### 10. Long-form `add_quality / add_anatomy / add_trials` not in `__all__` of `io/__init__.py`
- Where: `neurocomplexity/io/__init__.py:12-22` advertises them in `__all__` but `add_anatomy` and `add_trials` are reachable only via `__getattr__`. That is fine for runtime, but tools that introspect (Sphinx autodoc, IDE auto-completion that walks `__all__` statically) cannot see them.
- What is wrong: Mixed protocol — `add_quality` is eager (`from neurocomplexity.io._qc import add_quality`, line 10) while `add_anatomy/add_trials` are lazy. Inconsistent and unjustified.
- Proposed fix: Make all three either eager or lazy. Eager is cheap (none of them imports pynwb). Match `add_quality`'s pattern for the other two and drop them from `__getattr__`.

#### 11. README badge points at the wrong default branch
- Where: `README.md:4` — codecov badge uses `branch/master`. The plan and git log both say `main`.
- What is wrong: Cosmetic but visible. New visitor → broken badge → impression of dead project.
- Proposed fix: Switch to `branch/main`.

#### 12. README claims Python 3.10–3.12 while CI tests 3.10–3.13
- Where: `README.md:5` (`python-3.10-3.12`), `docs/installation.md:25` ("3.10, 3.11, 3.12") vs `pyproject.toml:10` (`>=3.10,<3.14`) and `test.yml:20` (`["3.10", "3.11", "3.12", "3.13"]`).
- What is wrong: The supported-Python band is internally inconsistent. A Python 3.13 user reading the README thinks they are unsupported.
- Proposed fix: Update README + docs/installation.md to 3.10–3.13.

#### 13. No CLI tests for the `analyze` and `figure` happy-path
- Where: `tests/test_benchmarks_cli.py` exists but no `tests/test_cli_analyze.py`, no `tests/test_cli_figure.py`.
- What is wrong: The CLI defaults are wrong (P0 #1) and no test caught it. The CLI is the most-likely entry point for a JOSS reviewer running the package for the first time.
- Proposed fix: Add a smoke test that invokes `main(["analyze", str(synth_nwb), "-o", str(tmp)])` and asserts the JSON + at least one figure file gets written. Use the existing `to_nwb` round-trip fixture if possible.

#### 14. No integration test that loads → analyses → saves figure end-to-end
- Where: tests are unit-shaped per estimator. No test asserts the README quickstart code runs verbatim.
- Why it matters: This is the single most useful test for a methods package — it catches CLI default bugs, encoder breakage, viz-format mismatches, and import surface gaps in one shot. Phase 6 of the plan ("Tutorial") effectively will do this manually; it should be automated.
- Proposed fix: One pytest module that runs the quickstart on the bundled synthetic fixture under `-W error::nc.warnings.QualityControlWarning` (to verify QC fires when expected).

### P2 — nice-to-have

#### 15. `__init__.py` puts `core, io, analysis, inference, warnings` into the same tuple as scalar re-exports
- Where: `__init__.py:9, 19-21`. Mixing module re-exports and scalar re-exports in one `__all__` is unidiomatic; users running `from neurocomplexity import *` get modules in their namespace.
- Proposed fix: Split into `_PUBLIC_SCALARS` and `_PUBLIC_MODULES`; combine for `__all__` but document the split.

#### 16. `_binning.py:46-47` warns at 25 % of *available* RAM rather than configurable threshold
- Where: `_binning.py:46`. Hard-coded `0.25 *`. The publication plan referenced "~1 GiB threshold"; the docs/quickstart says the same. The actual code is dynamic. Either the docs or the code is wrong, but more usefully this should be an env var (`NC_MEMORY_WARN_FRACTION`) so CI can mute it and HPC users with 1 TB nodes don't get noise.
- Proposed fix: Hoist the `0.25` to a module-level constant; allow override via env var or `nc.set_memory_warn_fraction(x)`.

#### 17. `MemoryAllocationWarning` silent on systems without `psutil`
- Where: `_binning.py:44` — `except ImportError: return`. `psutil` is not a declared dependency.
- What is wrong: A user installing the bare wheel gets no memory guard at all. The behaviour is invisible.
- Proposed fix: Add `psutil>=5.9` to base dependencies (it's tiny and Windows/macOS/Linux clean), or fall back to a static heuristic (e.g. warn above 1 GiB) when `psutil` is absent.

#### 18. `ContinuousSignal.__hash__` calls `.tobytes()` on a potentially-large array
- Where: `core/continuous.py:98`. For a 1 kHz pupil trace over a 60-minute session that's 28.8 MB hashed on every dict key / set membership check.
- Proposed fix: Cache the hash, or use a fingerprint (shape + dtype + first/last value + `xxhash` of bytes).

#### 19. No `responsibilities` / `code_owners` for community guidelines
- Where: `CONTRIBUTING.md` lacks PR review SLA, maintainer list, security disclosure policy.
- Proposed fix: Add `SECURITY.md` with a contact, and a maintainers list. JOSS expects community guidelines for support / reporting / contributing.

#### 20. Issue templates are bare-bones
- Where: `.github/ISSUE_TEMPLATE/bug_report.md` is the GitHub default with `**Describe the bug**` placeholders. No "version", "Python", "OS", "minimal reproducer" required fields wired in.
- Proposed fix: Convert to GitHub Issue Forms (`.yml`) with required fields; lowers triage friction.

#### 21. `BenchmarkResult.metadata: dict` is mutable default-factory'd but stored on a frozen dataclass
- Where: `benchmarks/runner.py:57`. The dataclass is `frozen=True` but `metadata: dict = field(default_factory=dict)` — mutable. Mutating `result.metadata[...] = ...` works (the field reference is frozen, the dict is not), which violates the "frozen" contract.
- Proposed fix: Use `MappingProxyType` or `types.MappingProxyType(dict(...))` after construction.

#### 22. CHANGELOG has no `[Unreleased]` section
- Where: `CHANGELOG.md:1-3`. Keep-a-Changelog convention requires it. Plan Phase 8 promises a v1.2.0 entry — easier to start it now.
- Proposed fix: Add `## [Unreleased]` at the top.

#### 23. `pyproject.toml` license format is legacy
- Where: `pyproject.toml:11` uses `license = { text = "MIT" }`. PEP 639 (now standard) prefers `license = "MIT"` plus `license-files = ["LICENSE"]`.
- Proposed fix: Update to PEP 639 form; modern build backends emit a deprecation warning on the old form.

## Specific responses to Phase-2 / Phase-3 carry-forwards

- **API symbols exposed but undocumented** (`ContinuousSignal`, `ProvenanceRecord`, `set_progress`, `estimate_bin_spikes_bytes`, `warnings`, `viz`): **Not acceptable as-is — P0.** The package needs a documented API surface before it can claim API stability, and that contract has to live in `docs/api/index.md`. Either document them all or demote four of them to module-qualified access. See finding #5.
- **No lockfile**: **Publication blocker — P0.** See finding #3. The plan itself gates Phase 0 on this. A reproducibility-positioned methods package without a pinned dependency snapshot is contradictory.
- **dateutil DeprecationWarning (upstream)**: **Acceptable as-is — P2.** Document it in the next CHANGELOG entry, file an issue against the upstream dependency, and add a `filterwarnings` entry in `pyproject.toml`'s `[tool.pytest.ini_options]` to keep CI green. Do not vendor a fix.

## JOSS / pyOpenSci-readiness statement

Not quite today. The substance is there — `paper/paper.md` is drafted, the LICENSE/CITATION/CODE_OF_CONDUCT/CONTRIBUTING quartet is present, a full OS × Python CI matrix is green, the test suite is large, and the package addresses a defensible gap in the methods ecosystem (the paper.md statement-of-need contrast against MR.Estimator, IDTxl, JIDT, MVGC, Elephant, SpikeInterface is the right framing). The blockers are mechanical: (a) fix the `pdf/png` vs `svg/tiff/jpg` contract so the README quickstart and CLI defaults actually run, (b) bump `CITATION.cff` to 1.1.0 and consolidate version sources, (c) ship a lockfile and add ruff+mypy CI gates (the plan promises both), (d) document `ContinuousSignal` / `ProvenanceRecord` / `set_progress` / `warnings` in `docs/api/index.md` or demote them, (e) add a CLI smoke test that catches future README-vs-code drift, and (f) align the README's Python-version badge with the CI matrix. With those six items closed — a 1-2 day fix-set — the package is JOSS-ready.

## What's missing

Standard research-software primitives the package does not yet provide. Not blockers; roadmap signal.

- **No type-checking CI job.** `py.typed` is shipped but never validated.
- **No coverage gate in CI.** `--cov-report=xml` is uploaded to codecov but there is no `--cov-fail-under=85` enforcement; the publication plan promises ≥ 85 %.
- **No benchmark-regression CI.** `results/benchmarks/v1.0.0.csv` is committed but no CI job diffs the new run against it; the schema-stability claim in the changelog is unenforced.
- **No `SECURITY.md`.**
- **No deprecation policy.** No `nc.deprecated` decorator, no `DeprecationWarning` machinery for future API renames.
- **No `pre-commit` config.** Easy onboarding wins for contributors.
- **No `tox.ini` / `nox.py`.** Useful for users reproducing the matrix locally.
- **`MappingProxyType` for `populations` / `signals` / `intervals` on `SpikeRecording`.** The class is `frozen=True` but those three fields are plain dicts — frozen at the attribute level only.
- **No `conftest.py`-level `filterwarnings` to convert `nc.warnings.*` into errors during tests** (or a marker to do so). Catches silently-broken QC paths in development.
- **No Python-level CLI completion** (`argcomplete`). One-line add, big UX win for terminal users.
- **No CI artifact for the rendered figures** (regression-diff against committed PNGs). Plan Phase 5 calls for this; not in place yet.
