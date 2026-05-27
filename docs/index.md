# neurocomplexity

A Python package for measuring complexity, criticality, and information flow
in spike-sorted neural recordings. Every analysis is validated against
synthetic ground truth (see [benchmarks](benchmarks.md)) and ships with
bootstrap CIs and surrogate-based null tests (see [inference](inference.md)).

## Scope

- **Primary target:** spike-sorted extracellular recordings (Neuropixels,
  Kilosort/SpikeInterface, Allen NWB, NetPyNE).
- **Secondary support:** continuous signals (LFP, calcium traces) via
  `neurocomplexity.analysis._continuous`. Provided for convenience when the
  same measure (transfer entropy, MSE, stationarity) is informative on a
  non-spike trace from the same recording. Not a general LFP toolbox — for
  full LFP analysis, see [`mne-python`](https://mne.tools) or
  [`elephant`](https://elephant.readthedocs.io).
- **Two scalar complexity measures** (`lmc_complexity` and
  `multiscale_entropy`) measure different things and are not redundant.
  See [complexity_measures](complexity_measures.md) for when to use which.

## Quick example

```python
import neurocomplexity as nc
rec = nc.io.from_nwb("session.nwb")
m = nc.wilting_mr(rec, populations=["all"], bin_size_ms=4)
print(f"branching ratio: {m.m:.3f}")
```

```{toctree}
:maxdepth: 2
:caption: User guide

installation
quickstart
io
examples/tutorial
benchmarks
inference
complexity_measures
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```

## Citation

If you use neurocomplexity in published work, please cite the Zenodo DOI; see
`CITATION.cff` in the repository.
