# IO Extension Design — Phy, Kilosort, SpikeInterface loaders

**Status:** Approved 2026-05-17
**Target release:** v1.1.0

## Goal

Add three loaders to `neurocomplexity.io` so the package accepts the
file formats that real labs land on first — Phy curation directories,
raw Kilosort output, and any `SortingExtractor` from the SpikeInterface
ecosystem — without forcing users through an NWB export step.

## Scope

In scope:
- `from_phy(directory, *, duration=None, populations=None)`
- `from_kilosort(directory, *, duration=None, populations=None)`
- `from_spikeinterface(sorting, *, recording=None, duration=None, populations=None)`
- Shared internal helper `_load_sorter_output(directory, label_source)`
- Optional `[spikeinterface]` extra in `pyproject.toml`
- Lazy imports — none of `pynwb`, `spikeinterface`, etc., are touched
  unless the corresponding loader is called
- Tests with on-the-fly sorter-output fixtures
- Documentation updates (quickstart, installation, new `io.md` page)
- CHANGELOG entry

Out of scope (deferred):
- Probe geometry / channel maps as first-class types
- Writers (this is a read-only release)
- Auto-detect between Phy and Kilosort directories
- A dedicated `from_open_ephys`, `from_blackrock`, etc. (users go via
  `from_spikeinterface` for those)

## Module layout

```
neurocomplexity/io/
├── __init__.py            (lazy __getattr__ for all heavy loaders)
├── dict_loader.py         (unchanged)
├── nwb.py                 (unchanged)
├── _sorter_output.py      (new — shared helper)
├── phy.py                 (new)
├── kilosort.py            (new)
└── spikeinterface.py      (new)
```

Public surface:

```python
__all__ = ["from_nwb", "from_dict", "from_phy",
           "from_kilosort", "from_spikeinterface"]
```

All four file/external loaders are lazy-imported via `__getattr__`, so
`import neurocomplexity` does not pull pynwb, spikeinterface, or open
any sorter directory until called.

## Signatures

```python
def from_phy(
    directory: str | Path,
    *,
    duration: float | None = None,
    populations: Mapping[str, np.ndarray] | None = None,
) -> SpikeRecording: ...

def from_kilosort(
    directory: str | Path,
    *,
    duration: float | None = None,
    populations: Mapping[str, np.ndarray] | None = None,
) -> SpikeRecording: ...

def from_spikeinterface(
    sorting,                              # spikeinterface.BaseSorting
    *,
    recording=None,                       # optional spikeinterface.BaseRecording
    duration: float | None = None,
    populations: Mapping[str, np.ndarray] | None = None,
) -> SpikeRecording: ...
```

Per Q1: no `quality=` kwarg. Users filter downstream with
`rec.filter_units(quality=['good'])`.

## Shared helper: `_load_sorter_output`

```python
def _load_sorter_output(
    directory: Path,
    *,
    label_source: Literal["phy", "kilosort"],
    duration: float | None,
    populations: Mapping[str, np.ndarray] | None,
) -> SpikeRecording: ...
```

### File contract

| File | Required | Purpose |
|---|---|---|
| `spike_times.npy` | yes | int64 array of spike times in **samples** |
| `spike_clusters.npy` | yes (Phy) | int32 array of cluster id per spike (after Phy merges) |
| `spike_templates.npy` | yes (Kilosort fallback) | used only if `spike_clusters.npy` is missing |
| `params.py` | yes | Phy-format Python file holding `sample_rate`, `n_channels_dat`, `dtype`, `dat_path`, `offset`, `hp_filtered` |
| `cluster_info.tsv` | preferred (Phy) | full curation table |
| `cluster_group.tsv` | fallback (Phy) | minimal `cluster_id, group` table when no `cluster_info.tsv` |
| `cluster_KSLabel.tsv` | yes (Kilosort) | automatic labels `good`/`mua` |

### Algorithm

1. Validate `directory` exists and is a directory; else `FileNotFoundError`.
2. Parse `params.py` by executing it in an empty namespace (Phy convention,
   matches what SpikeInterface does); extract `sample_rate` as a float.
   Raise `RecordingValidationError` if `sample_rate` is missing or non-numeric.
   Document the trust assumption in the docstring.
3. Load `spike_times.npy` (samples → seconds via `/ sample_rate`).
4. Load `spike_clusters.npy` if present, else `spike_templates.npy`. Sort
   both arrays by spike time (stable sort).
5. Build the units DataFrame based on `label_source`:
   - **phy**: prefer `cluster_info.tsv`, fall back to `cluster_group.tsv`.
   - **kilosort**: read `cluster_KSLabel.tsv`.
   Normalize column names (see below).
6. Set `duration` to `max(spike_times) + 1.0` unless overridden.
7. Default `populations = {"all": np.ones(n_units, dtype=bool)}` unless
   overridden.
8. Build `ProvenanceRecord.for_file(directory, source_format=label_source)`
   — fingerprint hashes the `params.py` file, which is small and stable.
9. Construct and return `SpikeRecording`.

### Column normalization

The TSV is read with `pandas.read_csv(sep="\t")` and renamed:

| Source column | Normalized name |
|---|---|
| `cluster_id` | `id` |
| `group` (Phy) | `quality` |
| `KSLabel` (Kilosort) | `quality` |
| `fr` | `firing_rate` |
| `depth` | `depth` (kept) |
| `ch` | `peak_channel` |
| `n_spikes` | `n_spikes` (kept) |
| `Amplitude` | `amplitude` (lowercased) |
| `ContamPct` | `contam_pct` (lowercased + snake) |

All other columns pass through verbatim. If neither `group` nor
`KSLabel` is present, `quality` is filled with `"unsorted"`.

### Edge cases

| Case | Behaviour |
|---|---|
| `spike_clusters.npy` missing on Phy dir | Try `spike_templates.npy`; emit `UserWarning` noting Phy merges/splits will not be reflected |
| TSV missing for the requested `label_source` | `RecordingValidationError` with the missing path |
| `spike_times.npy` empty | `RecordingValidationError` (consistent with `from_nwb`) |
| Cluster id in `spike_clusters` not in TSV | Append a synthetic row with `quality="unsorted"` so `SpikeRecording.__post_init__` invariant holds |
| Negative spike times | `RecordingValidationError` (delegated to `SpikeRecording` validator) |

## `from_phy`

```python
def from_phy(directory, *, duration=None, populations=None):
    return _load_sorter_output(
        Path(directory), label_source="phy",
        duration=duration, populations=populations,
    )
```

## `from_kilosort`

```python
def from_kilosort(directory, *, duration=None, populations=None):
    return _load_sorter_output(
        Path(directory), label_source="kilosort",
        duration=duration, populations=populations,
    )
```

## `from_spikeinterface`

### Behaviour

1. Lazy `import spikeinterface as si`; on failure raise
   `ImportError("from_spikeinterface requires spikeinterface. "
   "Install with: pip install 'neurocomplexity[spikeinterface]'")`.
2. Accept either a `BaseSorting` directly or a tuple `(sorting,
   recording)`. The `recording` keyword argument is preferred.
3. Pull spike trains via `sorting.get_unit_spike_train(uid,
   return_times=True)` (returns seconds). Concatenate across units.
4. Build `units` DataFrame from `sorting.get_property_keys()` —
   pass every property through (`quality`, `firing_rate`,
   `amplitude`, anything custom users have annotated).
5. Sampling rate: `sorting.sampling_frequency`.
6. Duration: in priority order — `duration` kwarg →
   `recording.get_duration()` if recording given →
   `max(spike_times) + 1.0`.
7. Brain area: if `recording is not None` and the recording has a
   `brain_area` property per channel, propagate per-unit via
   `sorting.get_property("channel_ids")` if present.
8. Provenance: `ProvenanceRecord.for_memory("spikeinterface",
   hint=type(sorting).__name__)` — no file path because SI sortings
   are often constructed in memory.

### Why a soft dependency

`spikeinterface` pulls `probeinterface`, `neo`, `MEArec`, etc. — heavy
for a user who only needs Phy. It lives in a new optional extra:

```toml
spikeinterface = ["spikeinterface>=0.100"]
```

CI does **not** install this extra; the SI test uses
`pytest.importorskip("spikeinterface")`. A separate `si-ci` job can be
added later if SI breakage starts mattering.

## Tests

New files (all under `tests/`):

```
tests/
├── test_io_phy.py
├── test_io_kilosort.py
└── test_io_spikeinterface.py
```

### Fixture strategy

For Phy/Kilosort: write a synthetic sorter directory on the fly into
`tmp_path`. The fixture helper builds:

- `params.py` with `sample_rate = 30000.0` and a few siblings
- `spike_times.npy` (int64 samples)
- `spike_clusters.npy` (int32)
- `cluster_info.tsv` or `cluster_KSLabel.tsv` depending on the test

This keeps tests under 50 ms each and requires no large binary fixtures.

For SI: `pytest.importorskip("spikeinterface")` then construct an
in-memory `NumpySorting` from spike trains; assert the round-trip.

### Assertions (each loader)

- `n_units` matches the TSV row count (after the missing-cluster fix-up)
- `n_spikes` matches `spike_times.npy.size`
- `spike_times` are sorted and in seconds (max ≈ samples / sample_rate)
- `duration` ≥ `spike_times.max()`, equals override when given
- `units` has at least `id`, `quality`; column normalization works
- `populations["all"]` covers every unit
- `source.source_format` == `"phy"` / `"kilosort"` / `"spikeinterface"`
- Round-trip preservation: spikes per unit match the input dict

### Edge-case tests

- Phy dir with only `cluster_group.tsv` (no `cluster_info.tsv`)
- Phy dir missing `spike_clusters.npy` (falls back to templates + warns)
- Cluster id in spikes that doesn't appear in TSV
- `duration` override
- Empty `spike_times.npy` raises
- Missing `params.py` raises
- Missing `pynwb`/`spikeinterface` raises the right `ImportError`

## Documentation

1. **`docs/quickstart.md`** — replace the v1.1 placeholder block with
   working `from_phy` and `from_kilosort` snippets and a
   `filter_units(quality=['good'])` follow-up so users see the canonical
   filtering pattern.
2. **`docs/installation.md`** — add `[spikeinterface]` to the extras
   table.
3. **`docs/io.md`** — new page describing each loader, the `units`
   DataFrame columns produced, security note about `params.py`, and
   a decision flowchart ("you have an NWB → from_nwb; a Phy dir →
   from_phy; raw KS → from_kilosort; anything else → SpikeInterface →
   from_spikeinterface"). Linked from the toctree in `docs/index.md`.
4. **`CHANGELOG.md`** — v1.1.0 section with the three new loaders, the
   `[spikeinterface]` extra, and the new docs page.

## Packaging

`pyproject.toml`:

```toml
[project.optional-dependencies]
nwb = ["pynwb>=2.5", "h5py>=3.8"]
spikeinterface = ["spikeinterface>=0.100"]
viz = ["matplotlib>=3.7", "plotly>=5.15", "dash>=2.10"]
sim = ["netpyne", "mpi4py"]
dev = ["pytest>=7.4", "pytest-cov>=4.1", "ruff>=0.4"]
docs = [...]
```

CI continues to install `.[dev,nwb]`. SpikeInterface tests skip
cleanly via `importorskip`.

## Versioning

This is a feature release: bump `neurocomplexity/_version.py` to
`1.1.0`, add a v1.1.0 CHANGELOG section, regenerate
`results/benchmarks/v1.1.0.csv` from `--reps=200` for the published
baseline.
