# neurocomplexity

[![tests](https://github.com/aurmandi/neurocomplexity/actions/workflows/test.yml/badge.svg)](https://github.com/aurmandi/neurocomplexity/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/aurmandi/neurocomplexity/branch/master/graph/badge.svg)](https://codecov.io/gh/aurmandi/neurocomplexity)
[![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A Python package for measuring complexity, criticality and information flow in
spike-sorted recordings. It started as a private prototype for analysing
NetPyNE simulations and has been rewritten to run, instead, on real
experimental data — Allen Institute Neuropixels, KiloSort/SpikeInterface
output, anything that can be loaded into an NWB file.

The point of the package is to let you go from a `.nwb` file to a manuscript
figure in one command, without having to assemble five different ad-hoc
notebooks every time.

## What's actually in it

Five analyses, each implemented as a single function that returns a frozen
dataclass:

- `criticality` — avalanche size and lifetime exponents (alpha_s, alpha_t),
  the kappa index, and the bin size that maximises the goodness-of-fit of the
  size–lifetime scaling.
- `wilting_mr` — the Wilting & Priesemann (2018) multi-step regression
  branching ratio, robust to subsampling.
- `shape_collapse` — Friedman et al. (2012) avalanche shape collapse, with a
  scale-invariant residual and a bounded continuous optimiser for gamma.
- `dimensionality` — participation ratio on the per-unit correlation matrix.
- `partial_information` — Williams–Beer I_min PID with quantile multi-level
  discretisation and Miller–Madow bias correction.

Plus `transfer_entropy` (binary Schreiber + Miller-Madow), `autonomy`
(VAR-Granger), and `surrogate_test` (jitter and ISI-shuffle).

A small `viz` module renders each of these as a Nature-style figure (editable
PDF/SVG, Arial 7 pt, no top/right spines), and a CLI wraps the whole pipeline
so you can run it from a terminal.

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

That writes a `results.json` and a set of PDF/SVG/PNG figures (one composite
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
