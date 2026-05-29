# API stability contract

`neurocomplexity` follows [Semantic Versioning 2.0](https://semver.org).
The contract below tells you which surface is covered by the version bump
rules and which is not.

## What counts as the public API

**Stable surface** — covered by SemVer. Breaking changes here require a
major version bump and a deprecation cycle of at least one minor release.

Every symbol that is *both*:

1. exported from `neurocomplexity` or one of its declared subpackages
   (`neurocomplexity.core`, `neurocomplexity.io`, `neurocomplexity.analysis`,
   `neurocomplexity.inference`, `neurocomplexity.viz`,
   `neurocomplexity.benchmarks`, `neurocomplexity.warnings`),
2. *and* documented under [`docs/api/`](api/index.md),

is part of the stable surface. Concretely this is every result dataclass
(`SpikeRecording`, `BranchingResult`, `CriticalityResult`,
`ShapeCollapseResult`, `DimensionalityResult`, `TransferEntropyResult`,
`PIDResult`, `MSEResult`, `LMCResult`, `AutonomyResult`,
`StationarityResult`, `ManifoldResult`, `InferenceResult`,
`BenchmarkResult`), every loader (`nc.io.from_nwb`, `from_kilosort`,
`from_phy`, `from_spikeinterface`, `from_dict`, `to_nwb`, `add_quality`,
`add_anatomy`, `add_trials`), the `SpikeRecording.merge_probes` method,
every top-level analysis
(`nc.analysis.criticality`, `wilting_mr`, `shape_collapse`,
`dimensionality`, `manifold`, `multiscale_entropy`, `lmc_complexity`,
`transfer_entropy`, `partial_information`, `autonomy`, `stationarity`,
`extract_avalanches`, `fit_avalanche_exponents`), the inference
machinery (`nc.inference.test`, `bootstrap`, `pvalue_from_null`,
`SurrogatePool`), the viz functions (`nc.viz.figure_*`,
`save_publication`), the CLI subcommands
(`neurocomplexity info|analyze|figure|benchmark`), and the
`neurocomplexity.warnings` warning classes.

**Experimental surface** — best-effort. Symbols here may move, gain or
lose parameters, change default behaviour, or be removed in any minor
release. They are still importable; users are warned not to build
unversioned dependencies on them.

The following are explicitly experimental in 1.x:

| Symbol | Why experimental |
|---|---|
| `neurocomplexity.ContinuousSignal` | LFP / calcium dataclass is incomplete; the immutable-builder pattern is still being designed. |
| `neurocomplexity.ProvenanceRecord` | Schema may grow new fields; `for_memory` / `for_file` constructors are stable but new factory methods may appear. |
| `neurocomplexity.set_progress` | Global progress-bar hook; mechanism may switch to a per-call `progress=` parameter in 2.x. |
| `neurocomplexity.estimate_bin_spikes_bytes` | Memory pre-flight helper; signature may grow scale / overhead arguments. |
| `neurocomplexity.warnings` namespace as a whole | Warning class identities are stable; the `_deduplicator` private state is not. |
| `nc.analysis._continuous` | Internal — used by `transfer_entropy` and `multiscale_entropy` to share code with `ContinuousSignal`. |

**Private surface** — anything whose import path begins with an
underscore (`_progress`, `_warnings`, `analysis._binning` internals,
`analysis._continuous`, `inference._child_seeds`, `core._immutable`, …)
is private and may change at any release without notice. Do not import
from underscored modules in downstream code.

## Deprecation policy

When a stable symbol is removed or renamed, the previous name is kept for
one minor release with a `DeprecationWarning` issued on first use.
Examples that will be retired in 1.2:

- `CriticalityResult.kappa` — Beggs-Plenz κ-index is not Shew (2009) κ;
  retained for back-compat, deprecated in 1.2, removed in 2.0.
- `CriticalityResult.branching` — single-step Beggs ratio; use
  `wilting_mr.m` instead.

## What is *not* covered by SemVer

- The exact text of warning messages.
- The exact wording of error messages.
- The numerical values of benchmark scores (the bound is "still passes"
  not "exactly this score").
- The internal shape of arrays inside `bootstrap_distribution` /
  `null_distribution` (use the helpers, not raw indexing).
- The format of `results.json` written by the CLI: stable as a contract
  between `analyze` and `figure`, but downstream parsers should pin a
  package version.
- Anything imported from a path containing a leading underscore.

## Type annotations

The package ships a [PEP 561](https://peps.python.org/pep-0561/)
`py.typed` marker. Public functions and dataclasses have type hints; we
treat type changes that break downstream `mypy --strict` runs as breaking
changes on the same footing as runtime signature changes.

## See also

- [`docs/api/index.md`](api/index.md) — full automodule reference for the
  stable surface.
- `CHANGELOG.md` (repository root) — every deprecation and removal is
  recorded here.
