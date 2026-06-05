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

## `from_dict(spike_times_by_unit, duration, unit_metadata=None, sampling_rate=None, hint="dict")`

The in-memory loader. Use it when your data does not arrive as a sorter
directory — a `.mat` export, a custom binary, a NumPy array you already
have in a notebook. You supply the spike trains; `from_dict` assembles
the `SpikeRecording`.

- `spike_times_by_unit` — `{unit_id: spike_times_seconds}`. Each value
  is a 1-D array of spike times in **seconds**. Keys are the unit ids;
  if you pass `unit_metadata`, its `id` column must match these keys.
- `duration` — recording length in seconds. Set it explicitly: rate and
  avalanche statistics use it to define the final time bin, so inferring
  it from the last spike silently drops the trailing window.
- `unit_metadata` — optional per-unit `DataFrame`. Must carry an `id`
  column. Columns the package recognises (`quality`, `firing_rate`,
  `peak_channel`, `brain_area`) feed `filter_units` and
  `with_populations` directly; any other columns pass through verbatim.
- `sampling_rate` — source sampling rate in Hz, stored for provenance.
- `hint` — short label recorded in the recording's provenance for later
  identification.

```python
import numpy as np
import neurocomplexity as nc

spike_times_by_unit = {0: np.array([0.1, 0.4, 1.2]),
                       1: np.array([0.3, 0.9, 1.1, 1.8])}

rec = nc.io.from_dict(spike_times_by_unit, duration=2.0, hint="example")
```

For a worked end-to-end example — splitting a flattened multi-session
release into one recording, building the metadata table, and attaching
brain-area labels — see
`examples/npultra_waveforms_2024_tutorial.ipynb`.

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

### Anatomy formats

`add_anatomy` accepts four formats, auto-detected by column-name sniffing:

| `format=`       | Source                       | Detector hits |
|-----------------|------------------------------|---------------|
| `"brainglobe"`  | Brainglobe `probes.csv`      | `acronym`, `brain_region`, hierarchical region columns |
| `"pinpoint"`    | Pinpoint exporter            | `label` column |
| `"sharptrack"`  | SHARP-Track MATLAB `.mat`    | (loaded via `load_sharptrack`) |
| `"csv"`         | Two-column generic CSV       | `unit_id` + `brain_area` / `area` / `region` |

Pass `format=` explicitly if auto-detection fails or you want to hard-pin
the loader.

### Trial / interval tables

```python
rec = nc.io.add_trials(rec, "stim_intervals.csv", name="stim",
                       start_column="start_time", stop_column="stop_time")
```

Stored under `rec.intervals["stim"]`. Required columns: ``start_time``
and ``stop_time`` (or set ``start_column`` / ``stop_column`` to your
local names — they get renamed internally). Touching intervals are
allowed (``stop[i] == start[i+1]``); overlapping intervals are
rejected at the surrogate-generation step.

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
