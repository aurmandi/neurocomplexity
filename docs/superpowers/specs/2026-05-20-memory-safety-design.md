# Chunked bin_spikes + allocation warning — Design Spec

**Date:** 2026-05-20 (rewritten after design audit)
**Sub-project:** OOM safety (#1 from architectural critique, GPU deferred)
**Status:** Approved by user 2026-05-20 — leaner v2

## Goal

Stop `bin_spikes` from silently thrashing 8 GB laptops when a user passes
hundreds of unit-populations on a 30-minute window. No global state, no new
exception class, no budget API — the previous spec was overengineered.

## Non-Goals

- GPU acceleration.
- Out-of-core spike-time storage.
- A general "memory budget" framework. Earlier audit showed 95% of the OOM
  risk lives in one function (`bin_spikes`); the surrogate pool and
  bootstrap are already LRU-bounded.
- Making analyses faster.

## Approach

Three small additions, no global state:

### 1. `chunk_seconds=` on `bin_spikes`

```python
def bin_spikes(rec, populations, bin_size_s, *, chunk_seconds=None):
    """...
    chunk_seconds: float | None
        If given, iterate the recording in chunks of this many seconds and
        fill the output counts matrix in place. Saves nothing in total
        allocation but reduces transient peak memory by avoiding a full
        per-population copy of spike indices. Output is bit-identical to the
        un-chunked path.
    """
```

Behaviour:
- `chunk_seconds is None` → existing path, no behaviour change.
- `chunk_seconds > 0` → process `[0, duration]` in windows, filtering spikes
  with one mask per chunk, writing into `counts[t_lo:t_hi, :]`.
- `chunk_seconds <= 0` or `> duration` → `ValueError`.

### 2. Allocation warning when the buffer is large

At the top of `bin_spikes`, compute:

```python
T = int(np.floor(rec.duration / bin_size_s))
P = len(populations)
need = T * P * 8                          # output counts in float64
try:
    import psutil
    avail = psutil.virtual_memory().available
except ImportError:
    avail = None
if avail and need > 0.25 * avail and chunk_seconds is None:
    warnings.warn(
        f"bin_spikes will allocate ~{need/1e6:.0f} MB for the "
        f"({T} bins × {P} populations) counts matrix; "
        f"{avail/1e6:.0f} MB available. "
        f"Consider chunk_seconds=10.0 or rec.crop(...).",
        category=nc.warnings.MemoryAllocationWarning,
        stacklevel=2,
    )
```

`psutil` is optional — if not installed, warning is silenced.

### 3. `nc.estimate_bin_spikes_bytes`

```python
def estimate_bin_spikes_bytes(rec, populations, bin_size_ms) -> int:
    """Return the number of bytes bin_spikes would allocate for the output
    counts matrix. Pure-function, no side effects, no global state."""
    T = int(np.floor(rec.duration / (bin_size_ms / 1000.0)))
    P = len(populations) if not isinstance(populations, int) else populations
    return T * P * 8
```

Exported from `neurocomplexity.__init__` as `nc.estimate_bin_spikes_bytes`.

### Warning class

`MemoryAllocationWarning(UserWarning)` in `neurocomplexity/_warnings.py`,
re-exported via `neurocomplexity/warnings.py`. Same suppression pattern as
`QualityControlWarning` and `StationarityWarning`.

## Tests (`tests/test_memory_safety.py`)

| Test | Behaviour |
|------|-----------|
| `test_estimate_bytes_matches_formula` | `estimate_bin_spikes_bytes(rec, ['all'], 5.0)` == `floor(rec.duration / 0.005) * 1 * 8`. |
| `test_estimate_bytes_accepts_int_population_count` | Passing `populations=300` works equivalently. |
| `test_bin_spikes_chunked_matches_unchunked` | Same `(rec, populations, bs)` with and without `chunk_seconds=10.0` → arrays bitwise equal. |
| `test_bin_spikes_chunk_seconds_validation` | `chunk_seconds=0.0` and `chunk_seconds=-1.0` raise `ValueError`. |
| `test_bin_spikes_chunk_seconds_larger_than_duration_raises` | `chunk_seconds=999.0` on a 60 s recording raises `ValueError`. |
| `test_bin_spikes_allocation_warning_fires_for_huge_buffer` | Monkeypatch psutil to report tiny available memory; `bin_spikes` emits one `MemoryAllocationWarning`. |
| `test_bin_spikes_allocation_warning_silent_when_below_threshold` | Default-sized analysis on a typical fixture → no warning. |
| `test_bin_spikes_allocation_warning_silent_when_chunked` | `chunk_seconds=...` provided → warning suppressed even if the unchunked alloc would be huge. |
| `test_warning_class_exposed_on_nc_warnings` | `nc.warnings.MemoryAllocationWarning` is the same class as the internal one. |
| `test_warning_suppressible_via_filterwarnings` | Standard `warnings.filterwarnings("ignore", category=...)` silences it. |

## File layout

```
neurocomplexity/_warnings.py            # add MemoryAllocationWarning
neurocomplexity/warnings.py             # re-export
neurocomplexity/__init__.py             # export estimate_bin_spikes_bytes
neurocomplexity/core/binning.py         # chunk_seconds + warning + helper
tests/test_memory_safety.py             # new
```

## What was dropped (audit trail)

The original v1 spec proposed `nc.set_memory_budget(...)`, `MemoryBudgetError`,
and pre-flight estimators wired into `SurrogatePool` and `bootstrap._run`.
Dropped because:

1. The actual OOM exposure is ~all in `bin_spikes`.
2. SurrogatePool LRU-caches surrogates (default 8 entries) so it's self-bounded.
3. A global budget setting users never see is invisible until the error fires —
   the warning + opt-in `chunk_seconds` covers the same ground without state.
4. The budget numbers were wishful anyway (no Python overhead accounting).

If real-world OOMs emerge from surrogate pools later, add a
`SurrogatePool(max_cache_size=...)` knob then. YAGNI for v1.
