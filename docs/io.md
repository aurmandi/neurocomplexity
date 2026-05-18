# I/O loaders

`neurocomplexity.io` materialises a `SpikeRecording` from disk or
memory. All file/external loaders are lazy-imported so the package
stays light when you only need one format.

## Decision flow

```
have an NWB file?               →  from_nwb
have a Phy curation dir?        →  from_phy
have raw Kilosort output?       →  from_kilosort
have any SpikeInterface          →  from_spikeinterface
  sorter object?
have spike trains in memory?    →  from_dict
```

## `from_phy(directory, *, duration=None, populations=None)`

Reads `spike_times.npy`, `spike_clusters.npy`, `params.py`, and
`cluster_info.tsv` (falling back to `cluster_group.tsv`). Quality
labels come from the curated `group` column; no filtering is applied
at load time — call `rec.filter_units(quality=['good'])` downstream.

Columns normalized into `rec.units`: `id`, `quality`, `firing_rate`,
`peak_channel`, `depth`, `amplitude`, `contam_pct`, `n_spikes`. Any
other columns from `cluster_info.tsv` are passed through verbatim.

## `from_kilosort(directory, *, duration=None, populations=None)`

Same directory layout as `from_phy`, but quality labels come from
`cluster_KSLabel.tsv` (automatic) instead of `cluster_info.tsv`
(curated).

## `from_spikeinterface(sorting, *, recording=None, duration=None, populations=None)`

Accepts any `spikeinterface.BaseSorting`. Pull duration and channel
metadata from an optional paired `recording`. SpikeInterface is the
recommended path for formats not natively supported here (Open Ephys,
Blackrock, Plexon, MEArec, NEO-readable, …).

## Security note on `params.py`

Phy writes `params.py` as executable Python. `from_phy` and
`from_kilosort` execute it in an isolated namespace, matching the
behaviour of Phy itself and SpikeInterface. Treat sorter directories
the same way you treat any other code you would `python -m` against —
do not run `from_phy` on directories pulled from untrusted sources.

## Attaching lab artefacts

A `SpikeRecording` loaded with `from_kilosort` carries no curation. Real
analyses should attach quality, anatomy, and (optionally) behavioural trials
before running anything statistical.

### Kilosort + Bombcell example

```python
import neurocomplexity as nc

rec = nc.io.from_kilosort("path/to/kilosort_output/")
rec = nc.io.add_quality(rec, "path/to/bombcell/cluster_metrics.csv")
rec = rec.filter_units(quality="good")           # drop MUA + noise
rec = nc.io.add_anatomy(rec, "path/to/sharptrack_probe.mat")
rec = nc.io.add_trials(rec, "path/to/trials.csv", name="stim")

# Now safe to analyse
m = nc.analysis.branching_ratio(rec, bin_size=0.005, k_max=100)
```

If you skip `add_quality` / `filter_units`, the first analysis call will emit
a `QualityControlWarning` explaining what to do.

### Phy users

Phy already stores curator-assigned quality alongside the sort. `from_phy`
loads it directly; no `add_quality` step is needed.

### Multiple probes

Load each probe separately, then merge:

```python
rec_a = nc.io.from_kilosort("probeA/")
rec_b = nc.io.from_kilosort("probeB/")
rec = nc.SpikeRecording.merge_probes({"A": rec_a, "B": rec_b})
```

`merge_probes` re-codes every unit's `id` to a fresh sequential integer (so the
merged recording satisfies the int64 dtype invariant) and preserves the source
ids in a new `original_id` column. The `probe` column carries the source label.
Per-probe sub-populations (`probe_A`, `probe_B`) are added automatically; if
unit ids collide across probes, a `UserWarning` is emitted and `original_id`
records the colliding entries as `('probe', id)` strings.
