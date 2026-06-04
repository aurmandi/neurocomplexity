# Changelog

## Unreleased

### Added
- **Reviewer audit (June 2026) follow-ups** — ten new options / warnings
  prompted by an external multi-layer audit. Every default reproduces the
  v1.1.0 / paper Table 3 numbers bit-for-bit; the new behaviour is opt-in.
  - `nc.analysis.criticality(..., regression={"ols","rma","odr"})` — A1.
    Reduced Major Axis and Orthogonal Distance Regression alternatives to
    OLS for the `<S>(T)` slope that produces `gamma_fit`. ODR uses
    `scipy.odr` with a documented OLS fall-back. Recorded in
    `CriticalityResult.params['regression']`.
  - `nc.analysis.transfer_entropy(..., discretize={"binary","quantile","none"},
    n_quantile_bins=3)` — A5. Quantile and pre-discretised paths via a new
    `_schreiber_te_general` plug-in over arbitrary alphabet `K`. Matches
    the binary estimator bit-for-bit at `K=2`.
  - `nc.analysis.partial_information(..., redundancy={"imin","iccs"})` — A8.
    Ince (2017) pointwise common-change-in-surprisal redundancy alongside
    Williams & Beer `I_min`. Recorded in `PIDResult.params['redundancy']`.
  - `nc.analysis.multiscale_entropy(..., backend={"numpy","kdtree","auto"})`
    and `_sample_entropy(..., backend=...)` — A4. Optional
    `scipy.spatial.cKDTree.query_pairs` backend for sample entropy at
    `K > 2000`. Numerical agreement with the NumPy loop verified to
    `rtol=1e-9`.
  - `nc.analysis.bin_active_by_type(rec, populations, bin_size_seconds,
    cell_types, *, missing_label="unlabelled")` — A6. E/I-aware
    counterpart to `bin_all_active`; returns one count vector per cell-type
    label so downstream analyses can keep streams separated.
- New regression tests for every item above plus the items below.

### Fixed
- **A9 — Transfer-entropy bias correction reorder.** The Schreiber TE
  plug-in could drift slightly negative on finite samples; the previous
  order `te -= correction; return max(0, te)` under-corrected at the zero
  floor. New order clamps the raw plug-in to zero before subtracting the
  Miller-Madow correction and clamps again at the end. High-signal cases
  are unchanged; only the null Type-I regime can shift, and the
  five-seed benchmark sweep stays well within tolerance.
- **A11 — `_binning` int32 overflow guard.** The (T, P) count matrix is
  now promoted to `int64` when `T * P > 2**30` (e.g. > 10 h recordings at
  1 ms with thousands of populations). `estimate_bin_spikes_bytes` and
  the large-allocation warning track the promoted dtype.

### Deprecated
- **A7 — Naive branching in `CriticalityResult`.** The
  `<A_{t+1}/A_t>` field is biased by sub-sampling and external drive
  (Wilting & Priesemann 2018) and now emits a `DeprecationWarning`
  pointing at `nc.analysis.wilting_mr`. The field is retained for
  backwards-compatible diagnostics only.

### Changed (internal)
- **A12** — Renamed the internal `log_t` variable inside
  `fit_avalanche_exponents` / `criticality` to `log_t_bins` and added a
  units comment. Behaviour unchanged.
- **A13** — `multiscale_entropy` emits a single `RuntimeWarning` per call
  when any coarse-grained scale has fewer than `10**(m+1)` template
  windows (Richman & Moorman 2000 / Pincus 1991 reliability rule).
- **IO4** — `nc.io.from_spikeinterface` emits a `UserWarning` when the
  materialised spike array would exceed ~1e8 spikes (~800 MB float64),
  pointing the caller at `sorting.frame_slice(...)`.

### Added (legacy entries below were already pending in Unreleased)
- `nc.analysis.extract_avalanches` and `nc.analysis.fit_avalanche_exponents`
  are now re-exported from the `analysis` namespace. They were already part
  of the documented stable surface but were only reachable through the
  `analysis.criticality` submodule; the API-stability contract now matches
  the exports.
- `nc.inference.pvalue_from_null` is re-exported from the `inference`
  namespace (previously only on `inference.null_test`).
- `tests/test_api_stability.py` — freeze lock asserting every symbol named
  stable in `docs/api_stability.md` resolves at its advertised path.

### Fixed
- `docs/api_stability.md` listed `nc.io.merge_probes` (it is the
  `SpikeRecording.merge_probes` method) and a phantom `nc.viz.figure_panel`
  (never existed; `panel_label=` is a per-figure keyword). Contract wording
  corrected.
- `neurocomplexity.analysis` docstring no longer claims analysis functions
  are re-exported onto the package root; they are accessed via
  `nc.analysis.*`.
- Internally-built `SurrogatePool` in `nc.inference.test` now bounds its
  cache to the concurrent-worker count instead of the default 64, fixing a
  `MemoryError` on large (>20M-spike) recordings. Cache size is a pure
  performance parameter and never affects results.

## v1.1.0 — 2026-05-17

### Added
- `nc.io.from_phy(directory)` — load a Phy curation directory.
- `nc.io.from_kilosort(directory)` — load raw Kilosort output.
- `nc.io.from_spikeinterface(sorting, recording=None)` — bridge to any
  `spikeinterface.BaseSorting`.
- `[spikeinterface]` optional install extra.
- New `docs/io.md` loader reference page.

### Changed
- `neurocomplexity.io.__init__` now lazy-imports every heavy loader via
  `__getattr__`; importing the top-level package no longer touches
  pynwb or spikeinterface.

## v1.0.0 — 2026-05-16

Initial public release of `neurocomplexity`.

### Analyses

- Wilting–Priesemann multistep-regression branching ratio (`wilting_mr`),
  subsampling-robust per Wilting & Priesemann (2018).
- Avalanche size and lifetime exponents (`criticality`) with log-binned
  power-law fits and the scaling relation γ = (τ−1)/(α−1).
- Friedman et al. (2012) avalanche shape collapse (`shape_collapse`)
  with scale-invariant residual.
- Participation ratio (`dimensionality`) on the per-unit correlation
  matrix.
- Williams–Beer I_min partial information decomposition
  (`partial_information`) with quantile-equal multi-level discretisation
  and Miller–Madow bias correction.
- Schreiber transfer entropy (`transfer_entropy`) with binary symbols
  and Miller–Madow bias correction.
- VAR–Granger autonomy index (`autonomy`).

### Statistical inference

- Three neural-data-specific surrogate generators: spike dithering
  (Louis et al. 2010), per-unit ISI shuffle, and trial/interval
  shuffle.
- Surrogate-based null tests with FDR control and joblib parallelism.
- Per-analysis block-bootstrap CIs with Efron (1987) bias-corrected
  percentile method.
- Calibration tests for Type-I rate, power, and coverage.

### Benchmarks

- Public benchmark validation suite (`neurocomplexity.benchmarks`)
  with eleven ground-truth cases covering criticality (`m_hat`,
  `exponents`), information theory (`te_convergence`, `te_null`,
  `autonomy_calibration`), PID atoms (`atoms_{xor,and,copy,rdn,unq}`),
  and dimensionality (`pr_rank`).
- Four synthetic-data simulators: `branching_network` (with optional
  unsaturated Galton-Watson mode), `trial_based_avalanches`,
  `coupled_ar1` and `var1` (with analytic transfer-entropy ground
  truth), `pid_recording`, and `rank_r_population`.
- `run_case`, `run_all`, `list_cases` API returning tidy
  `pandas.DataFrame` output, plus a `neurocomplexity benchmark`
  CLI subcommand.
- Reference baseline at `results/benchmarks/v1.0.0.csv`
  (11/11 cases pass at `n_reps = 5`).

### Tooling

- NWB loader (`io.from_nwb`) for spike-sorted recordings.
- Matplotlib-based figure generators in `viz`.
- CLI: `info`, `analyze`, `figure`, `benchmark` subcommands.
- Sphinx-friendly docstrings throughout the public API.
