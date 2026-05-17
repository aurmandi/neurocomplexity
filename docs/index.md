# neurocomplexity

A Python package for measuring complexity, criticality, and information flow
in spike-sorted neural recordings. Every analysis is validated against
synthetic ground truth (see [benchmarks](benchmarks.md)) and ships with
bootstrap CIs and surrogate-based null tests (see [inference](inference.md)).

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
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```

## Citation

If you use neurocomplexity in published work, please cite the Zenodo DOI; see
`CITATION.cff` in the repository.
