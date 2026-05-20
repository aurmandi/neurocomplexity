# NWB Round-Trip Writer — Design Spec

**Date:** 2026-05-20
**Sub-project:** Round-trip writer (#2 from architectural critique)
**Status:** Approved by user 2026-05-20

## Goal

Enable `rec2 = nc.io.from_nwb(nc.io.to_nwb(rec, path))` to return a `SpikeRecording` that compares bitwise-equal to the original across every field, so collaborators can exchange curated/filtered/cropped recordings without re-running heavy curation.

## Non-Goals

- Streaming / partial reads of huge files. Single-session writes only.
- Compatibility with non-neurocomplexity NWB files lacking the extension (those continue to use existing `from_nwb` best-effort path).
- Append-to-existing-NWB. The writer creates a new file.

## Approach: Approach B — minimal extension

NWB already round-trips: `spike_times`, unit table columns (via `VectorData`), and `intervals` (via `TimeIntervals`). The vendored extension defines only what NWB cannot model natively.

## Vendored extension: `ndx-neurocomplexity`

Lives at `neurocomplexity/io/_ndx/`. Three new types:

### `NCPopulationMask` (extends `NWBDataInterface`)
- `name`: str (population label)
- `mask`: 1-D bool array, length = n_units, aligned to the `nc_unit_id` column of the Units table.

### `NCProvenance` (extends `NWBDataInterface`)
- `kind`: str (e.g. `"file"`, `"roundtrip"`)
- `path`: str (may be empty)
- `blake2b_head`: bytes (16)
- `blake2b_tail`: bytes (16)
- `size_bytes`: int64
- `created_at`: ISO-8601 str
- `chain_index`: int32 (preserves order in `rec.attachments`)
- `extras`: str (JSON blob for forward-compat fields)

### `NCFilteredFlag` (extends `NWBDataInterface`)
- `filtered`: bool — mirrors `SpikeRecording._filtered`.

These are attached to the `NWBFile.scratch` group under fixed names (`nc_populations/<label>`, `nc_provenance/<index>`, `nc_filtered_flag`).

## Writer algorithm (`nc.io.to_nwb`)

```python
def to_nwb(rec, path, *, session_description, identifier, session_start_time, overwrite=False):
    ...
```

1. Build `NWBFile` with required metadata. If `rec.attachments` carries a previous `session_description` / `identifier` / `session_start_time`, those serve as defaults.
2. Build `Units` table:
   - One row per `rec.unit_ids` entry, ordered as in `rec.unit_ids`.
   - `spike_times` per unit derived by indexing `rec.spike_times` with `rec.unit_ids == uid`. Sorted within each unit; original global order preserved separately (see step 3).
   - **Custom columns:** every column of `rec.units` becomes a `VectorData`. For each column, an attribute `nc_dtype` stores the original pandas dtype string for exact reconstruction.
   - **`nc_unit_id` column:** int64 column carrying the original ids (NWB's `id` is uint64; we ignore it on read).
3. Build `NCFlatSpikes` (extra scratch dataset): the flat `(spike_times: float64, unit_ids: int64)` pair, authoritative for reconstruction. Avoids any reordering ambiguity.
4. Write `intervals`: for each named DataFrame in `rec.intervals`, create a `TimeIntervals` named `nc_interval__<name>`. Extra columns get the same `nc_dtype` attribute treatment.
5. Write `NCPopulationMask` instances for every population.
6. Write `NCProvenance` entries with `chain_index = 0, 1, 2, …` matching the order in `rec.attachments`.
7. Write `NCFilteredFlag`.
8. Set `NWBFile.session_start_time`; set `Units.electrode_group` to None (we don't model electrodes here yet).
9. `NWBHDF5IO.write(file)`.

## Reader algorithm (extended `nc.io.from_nwb`)

1. Open file. Detect presence of `NCFlatSpikes` (scratch). If present → "neurocomplexity-authored" path; else → existing best-effort path with a `UserWarning("NWB file lacks neurocomplexity extension")`.
2. Authoritative path:
   - Read `NCFlatSpikes` → `spike_times`, `unit_ids` (preserves original global order, exact float64).
   - Read `Units` table rows in stored order; for each custom column, cast back using its `nc_dtype` attribute. Reconstruct `units` DataFrame with `nc_unit_id` as the id column (drop NWB `id`).
   - Read `intervals`: enumerate `nc_interval__<name>` groups; reconstruct DataFrames using `nc_dtype` attributes.
   - Read all `NCPopulationMask` → `populations` dict.
   - Read all `NCProvenance` ordered by `chain_index` → list of `ProvenanceRecord`.
   - Read `NCFilteredFlag.filtered` → `_filtered`.
3. Append a new `ProvenanceRecord(kind="roundtrip", path=str(path), ...)` to attachments.
4. Construct `SpikeRecording(...)` via the existing private constructor; do **not** re-run filter/curation logic.

## Risk register → tests

Each item is a behavioural test in `tests/io/test_nwb_roundtrip.py`:

| # | Risk | Test |
|---|------|------|
| 1 | `unit_ids` dtype lost to uint64 | `test_int64_unit_ids_preserved` — recording with `unit_ids = np.array([7, -3, 42], int64)` round-trips exactly. |
| 2 | `units` DataFrame dtype loss | `test_units_dtype_preservation` — columns of dtypes {category, Int64 nullable, bool, float32, object} all round-trip. |
| 3 | `spike_times` global order | `test_spike_times_global_order` — shuffle (spike_times, unit_ids) into unusual order before write; verify identical after read. |
| 4 | Population mask alignment | `test_population_mask_follows_units` — shuffle unit order on write; mask membership tracks the right unit ids on read. |
| 5 | Intervals extra columns | `test_intervals_extra_columns` — intervals DataFrame with extra `trial_type: category` + `score: float32` round-trip. |
| 6 | Attachments ordering | `test_attachments_chain_order` — three provenance records, append in order, read back in order. |
| 7 | `_filtered` flag | `test_filtered_flag_round_trips_true_and_false` — both states preserved. |

Plus:
- `test_full_equality` — fixture recording with everything populated, asserts `rec == rec2` (frozen-dataclass `__eq__`).
- `test_overwrite_flag` — second `to_nwb` to same path raises unless `overwrite=True`.
- `test_missing_extension_falls_back` — synthesize a plain NWB without our scratch entries, confirm `from_nwb` warns + best-effort reads.

## Public API additions

```python
nc.io.to_nwb(rec, path, *, session_description, identifier,
             session_start_time, overwrite=False) -> Path
```

`nc.io.from_nwb` signature unchanged; behavior extended.

## Dependencies

- `pynwb` (already in `nwb` extra).
- `hdmf` (transitive).
- **No new PyPI dependency** — `ndx-neurocomplexity` is vendored in-tree.

## File layout

```
neurocomplexity/io/_ndx/
    __init__.py         # registers types with pynwb's type map on import
    spec.py             # NWBGroupSpec / NWBDatasetSpec definitions
    types.py            # NCPopulationMask, NCProvenance, NCFilteredFlag, NCFlatSpikes classes
neurocomplexity/io/nwb.py    # to_nwb() added; from_nwb() extended
tests/io/test_nwb_roundtrip.py
```

## Out of scope

- Cross-version extension migration. Bump the extension namespace version and refuse mismatched files for now.
- Multi-session NWB files. One recording per file.
