# Quickstart

This page shows the minimum to go from a recording to a result. For a full
walk-through see the [tutorial notebook](examples/tutorial.ipynb).

## Load a recording

```python
import neurocomplexity as nc
rec = nc.io.from_nwb("path/to/session.nwb")
print(f"{rec.n_spikes} spikes from {rec.n_units} units")
```

NWB support requires the optional `nwb` extra (`pip install
"neurocomplexity[nwb]"`).

### From a Phy curation directory

```python
rec = nc.io.from_phy("path/to/phy_output/")
rec = rec.filter_units(quality=["good"])
```

### From raw Kilosort output (before Phy curation)

Raw Kilosort output has no curator-assigned quality, so attach an automated QC
table (Bombcell / ecephys_spike_sorting / SpikeInterface) and filter on it.
Without this step `neurocomplexity` emits a `QualityControlWarning` at analysis
time because uncurated sorter output is dominated by noise units and MUA.

```python
rec = nc.io.from_kilosort("path/to/kilosort_output/")
rec = nc.io.add_quality(rec, "path/to/bombcell/cluster_metrics.csv")
rec = rec.filter_units(quality="good")
```

### From any SpikeInterface sorter

```python
import spikeinterface.extractors as se
sorting = se.read_phy("path/to/phy_output/")
rec = nc.io.from_spikeinterface(sorting)
```

The SpikeInterface bridge is a soft dependency — install with
`pip install "neurocomplexity[spikeinterface]"`.

## Run an analysis

```python
m = nc.wilting_mr(rec, populations=["all"], bin_size_ms=4)
print(f"m_hat = {m.m:.3f}, R^2 = {m.r_squared:.3f}")
```

## Add inference

```python
from neurocomplexity.inference import bootstrap
ci = bootstrap(m, rec, n=200, block_seconds=5.0, seed=0)
print(f"95% CI: [{ci.ci_lower:.3f}, {ci.ci_upper:.3f}]")
```

## Run the benchmark suite against the published baseline

```bash
python -m neurocomplexity benchmark --reps 50 -o my_baseline.csv
```
