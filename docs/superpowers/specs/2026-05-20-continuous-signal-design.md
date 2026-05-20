# ContinuousSignal + binned-stream TE/PID — Design Spec

**Date:** 2026-05-20
**Sub-project:** ContinuousSignal (#3 from architectural critique)
**Status:** Approved by user 2026-05-20

## Goal

Let external continuous streams (pupil, running speed, stimulus contrast,
photometry) participate in `transfer_entropy` and `partial_information` as
first-class variables, alongside spike-derived population activity, so users
can ask "does pupil predict V1?" or PID-decompose `V1 from (LGN_spikes,
pupil)`.

## Non-Goals

- Irregularly-sampled signals (deferred — uniform only in v1).
- Continuous-valued estimators (KSG, Kraskov-Stögbauer-Grassberger).
  Discretisation only, matching the existing binary-Schreiber TE.
- Signal-only analyses (criticality, branching). Spike-derived only.
- Interpolation. Block-averaging only.

## Approach

### `ContinuousSignal` dataclass

```python
@dataclass(frozen=True)
class ContinuousSignal:
    values: np.ndarray            # 1-D float64, length n_samples
    sampling_rate: float          # Hz, > 0
    t_start: float = 0.0          # seconds, signal[i] is at t_start + i / sampling_rate
    label: str = ""               # optional human label
    units: str = ""               # optional physical units string (e.g. "px", "cm/s")
```

Frozen, validated in `__post_init__`:
- `values` is `np.ndarray`, ndim == 1, dtype castable to float64, no NaN.
- `sampling_rate > 0`.
- `t_start >= 0`.

`duration` property: `len(values) / sampling_rate`.

Lives at `neurocomplexity/core/continuous.py`. Public export
`nc.ContinuousSignal`.

### Extension to `SpikeRecording`

Add a new frozen-dataclass field:

```python
signals: Mapping[str, ContinuousSignal] = field(default_factory=dict)
```

Validated in `__post_init__`: each signal's `[t_start, t_start + duration]`
must be contained within `[0, rec.duration]` (else `RecordingValidationError`).

`SpikeRecording.with_signal(name, sig)` helper returns a new recording with the
signal added (matches the `with_populations` style). Existing constructors
default to `signals={}`.

### Round-trip

NWB writer (`io/_ndx/__init__.py`) already pickles the whole rec via
`nc_payload` scratch, so `signals` round-trip automatically with no extension
changes. The standard NWB representation gets one `TimeSeries` per signal,
written under `acquisition/`, named `nc_signal__<label>`, for cross-tool
visibility (writer adds this; round-trip authority stays in the pickle).

### Discretisation rule (binary median split)

Helper in `neurocomplexity/analysis/_continuous.py`:

```python
def bin_signal_binary(sig: ContinuousSignal, *, bin_size_s: float,
                      duration: float, t0: float = 0.0,
                      threshold: float | None = None) -> np.ndarray:
    """Return shape (T,) int array of {0,1}.

    1. Build T = floor(duration / bin_size_s) bins matching the spike grid.
    2. Block-average sig.values within each bin (skip bins outside signal coverage).
    3. Threshold: default = median of per-bin averages; user can override.
    4. Return (averaged >= threshold).astype(int).
    """
```

`bin_size_s` must be an integer multiple of `1 / sampling_rate` within
`1e-9` relative tolerance, else raise `ValueError` with a message naming
both numbers. (95% of common signals — pupil 60 Hz, speed 100 Hz, photometry
1 kHz — pair naturally with the package's default 5/10/20/40 ms bin grids.)

### TE/PID API extension

```python
transfer_entropy(rec, *, populations=None, signals=None,
                 bin_size_ms=5.0, delay_bins=1, estimator="binary",
                 ) -> TransferEntropyResult

partial_information(rec, *, target_pop, source_1, source_2,
                    bin_size_ms=5.0, ...) -> PIDResult
```

- `signals: Sequence[str] | None` — names from `rec.signals`. Each becomes a
  row/column in the TE matrix exactly like a population.
- The result matrix `populations` tuple is replaced by an ordered tuple of
  all names (populations first, signals second) so downstream visualisation
  can colour them differently.
- `signals` is also valid for `target_pop`, `source_1`, `source_2` in PID.

### Backwards compatibility

`signals=None` → no change in behaviour. Existing tests do not pass `signals`
so they remain green. The Units/TimeSeries NWB additions are opt-in via
`to_nwb`; nothing in `from_nwb` changes for files without signals.

## Tests (`tests/test_continuous_signal.py`, `tests/test_analysis_te_signals.py`)

| Test | Behaviour |
|------|-----------|
| `test_continuous_signal_validation` | Negative sampling_rate / 2-D values / NaN values raise. |
| `test_duration_property` | `len(values) / sampling_rate`. |
| `test_signal_attaches_via_with_signal` | Returns new rec with `signals['pupil']` present, original unchanged. |
| `test_signal_must_fit_within_recording_duration` | t_start + duration > rec.duration → RecordingValidationError. |
| `test_bin_signal_binary_uniform_pupil_60hz_at_50ms` | 50 ms = 3 samples × (1/60 s) so misaligned → ValueError mentioning both. |
| `test_bin_signal_binary_aligned_block_average` | 100 Hz signal, 10 ms bins → integer divide cleanly; output length T = floor(duration/0.01). |
| `test_bin_signal_binary_median_split` | Sinusoid → after binning, exactly half the bins are 1, half 0. |
| `test_bin_signal_binary_explicit_threshold_override` | threshold=value_high → all-zeros; threshold=value_low → all-ones. |
| `test_te_runs_with_signal_argument` | rec with pupil signal + 2 populations; `transfer_entropy(rec, populations=['e','i'], signals=['pupil'])` returns a 3×3 matrix and result.populations == ('e','i','pupil'). |
| `test_te_signal_only_recipient` | Stimulus signal that drives a population → TE(stim → pop) > TE(pop → stim). |
| `test_pid_with_signal_as_source` | PID(target=pop, source_1=pop2, source_2=signal) returns a valid PIDResult; redundancy + unique + synergy sum to MI(target; (s1,s2)). |
| `test_nwb_roundtrip_preserves_signals` | Add 2 signals to a rec, `to_nwb → from_nwb`, signals dict equal. |

## File layout

```
neurocomplexity/core/continuous.py            # new — ContinuousSignal
neurocomplexity/core/recording.py             # add signals field + validation
neurocomplexity/__init__.py                   # export ContinuousSignal
neurocomplexity/analysis/_continuous.py       # new — bin_signal_binary
neurocomplexity/analysis/transfer_entropy.py  # add signals= kwarg
neurocomplexity/analysis/pid.py               # accept signal names in source/target
neurocomplexity/io/_ndx/__init__.py           # write nc_signal__* TimeSeries
tests/test_continuous_signal.py               # new
tests/test_analysis_te_signals.py             # new
```

## Open question

Does `signals` need to participate in the round-trip provenance JSON? For v1
the pickle is authoritative and the JSON only carries metadata — so no.
Decision can be revisited if users need to inspect signal labels without
unpickling.
