<p align="left">
  <img src="https://raw.githubusercontent.com/aurmandi/neurocomplexity/main/docs/_static/logo.svg" alt="neurocomplexity logo" width="260">
</p>

# neurocomplexity

[![tests](https://github.com/aurmandi/neurocomplexity/actions/workflows/test.yml/badge.svg)](https://github.com/aurmandi/neurocomplexity/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/aurmandi/neurocomplexity/branch/main/graph/badge.svg)](https://codecov.io/gh/aurmandi/neurocomplexity)
[![python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A Python package for measuring complexity, criticality and information flow in
spike-sorted recordings. It started as a private prototype for analysing
NetPyNE simulations and has been rewritten to run, instead, on real
experimental data: NWB/iloSort/Phy/SpikeInterface
output.

## What's actually in it

**Criticality & dynamics**

- `criticality` — avalanche size and lifetime exponents (`alpha_s`,
  `alpha_t`), the `kappa` index, the scaling exponent `gamma_fit` from the
  size–lifetime regression, and the theoretically-predicted
  `gamma_predicted = (alpha_t - 1) / (alpha_s - 1)` for a Sethna (2001)
  crackling-noise consistency test. The α_t exponent is fit directly to the
  lifetime distribution P(T), not derived from the regression slope.
- `wilting_mr` — Wilting & Priesemann (2018) multi-step regression branching
  ratio, robust to subsampling.
- `shape_collapse` — Friedman et al. (2012) avalanche shape collapse, with a
  scale-invariant residual and a bounded continuous optimiser for γ.

**Complexity & geometry**

- `dimensionality` — participation ratio on the per-unit correlation matrix.
- `manifold` — population-state embedding (PCA / UMAP / t-SNE) for the
  geometry of neural trajectories (Cunningham & Yu 2014, Gallego et al. 2018).
- `multiscale_entropy` — Costa, Goldberger, Peng (2002) MSE on
  population-rate series, built on Richman & Moorman (2000) sample entropy.
- `lmc_complexity` — López-Ruiz, Mancini, Calbet (1995) statistical
  complexity `C = H · D`. See
  [`docs/complexity_measures.md`](docs/complexity_measures.md) for when to
  use LMC vs MSE — they answer different questions.

**Information flow**

- `transfer_entropy` — binary Schreiber TE with Miller-Madow correction;
  significance via `inference.test` against jitter / ISI-shuffle nulls.
- `partial_information` — Williams–Beer I_min PID with quantile multi-level
  discretisation and Miller–Madow bias correction.
- `autonomy` — VAR-Granger self-predictability index.

**Inference**

Every analysis works with the unified `inference.test(result, rec,
surrogate=..., alternative=...)` API. Surrogates: jitter (uniform dithering
with optional refractory repair), ISI shuffle, and interval shuffle (now
validates non-overlap to prevent silent corruption). P-values use the
Phipson & Smyth (2010) +1 floor; the two-sided test is the conventional
`2 · min(p_greater, p_less)` clipped at 1 (robust to skewed null
distributions, unlike a mean-centred form). FDR via Benjamini–Hochberg.
Confidence intervals via percentile or BC bootstrap.

**Visualisation**

`viz` renders every result as a Nature-style figure (editable SVG/TIFF/JPG,
Arial 7 pt, no top/right spines, three named palettes). `figure_panel`
composes labelled multi-panel figures with automatic layout. A CLI wraps
the whole pipeline so you can run it from a terminal.

**Ingestion**

NWB, KiloSort/Phy, SpikeInterface. Helpers: `add_quality` (Bombcell
auto-detection, SpikeInterface metrics, threshold-based curation),
`add_anatomy` (Brainglobe lookups, CSV, SHARP-Track `.mat`), `add_trials`
(CSV/TSV/NWB), `merge_probes` (multi-probe sessions).

## Scope

Primary target: **spike-sorted extracellular recordings**. Continuous-signal
support (LFP / calcium) is provided via
`neurocomplexity.analysis._continuous` for cases where you want to apply the
same measure (TE, MSE, stationarity) to a non-spike trace from the same
recording — not as a full LFP toolbox. For that, use
[`mne-python`](https://mne.tools) or
[`elephant`](https://elephant.readthedocs.io).

## Installing

    pip install neurocomplexity

Optional extras: `pip install neurocomplexity[viz]` pulls in matplotlib,
`[dev]` pulls in pytest. There is also a `[sim]` extra for the legacy NetPyNE
simulation side, which most users will not need.

Python 3.10 or newer.

## A minimum-effort example

    import neurocomplexity as nc

    rec = (nc.io.from_nwb("session_715093703.nwb")
             .filter_units(quality=["good"])
             .with_populations(by="brain_area", on_unassigned="drop")
             .crop(3765.6, 4066.9))

    crit = nc.analysis.criticality(rec, populations=["VISp"])
    print(crit.alpha_s, crit.alpha_t, crit.r_squared)

The recording object is immutable and the builder methods return new
recordings, so it is safe to share between threads or reuse without worrying
about state.

## From the command line

    neurocomplexity analyze session.nwb \
        --start 3765.6 --end 4066.9 \
        --target VISp --sources LGd CA1 \
        -o results/session/

That writes a `results.json` and a set of SVG/TIFF/JPG figures (one composite
overview plus five per-analysis panels) into the output directory. If you
just want to re-render the figures from a cached run, `neurocomplexity figure
results.json -o new_figures/` does that without re-loading the NWB.

## Reproducibility

Every result carries a `ProvenanceRecord` with the source path, a BLAKE2b
fingerprint of the file (head + tail + filesize), the loader version, and the
package version. The CLI dumps all of that into `results.json`, so a figure
can always be traced back to a specific input.

## Tutorial

There is a worked tutorial in [`tutorial/`](tutorial/) that walks through
loading an Allen Institute NWB file, choosing a spontaneous epoch, and
running all five analyses with a paragraph of commentary at each step.

## What it's not

It's not a spike sorter. It expects sorted units. It's not a general-purpose
information-theory library either — the estimators in here are the ones the
package needs internally, with the bias corrections that matter at the sample
sizes a single recording gives you, and that's it.

## Citation

If this package contributes to a paper, please cite Wilting & Priesemann
(2018, Nature Communications) for the branching-ratio estimator, Friedman et
al. (2012, PRL) for the shape collapse, and Williams & Beer (2010) for the
PID decomposition. A citation file for the package itself will follow the
first archival release.

## License

MIT.
