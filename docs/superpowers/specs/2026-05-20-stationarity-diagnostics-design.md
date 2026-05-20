# Stationarity Diagnostics — Design Spec

**Date:** 2026-05-20
**Sub-project:** Stationarity diagnostics (#4 from architectural critique)
**Status:** Approved by user 2026-05-20 (design choices made without further clarifying questions per "proceed" instruction; revisable)

## Goal

Detect non-stationary firing-rate regimes before criticality / TE / branching analyses run on them, so users are warned when their recording violates a stationarity assumption that the downstream statistic silently depends on.

## Non-Goals

- Block analyses on non-stationary data. Diagnostics are **advisory warnings** only, like `QualityControlWarning`.
- Detrending or stationarisation transforms. Detection only; remediation is the user's call.
- Per-unit non-stationarity reports (population-level only in v1).

## Approach

New module `neurocomplexity/analysis/stationarity.py` exposing:

```python
@dataclass(frozen=True)
class StationarityResult:
    population_rate_cv: float          # CV of windowed population rate
    rate_drift_slope: float            # Hz / s, OLS slope of rate vs time
    rate_drift_pvalue: float           # two-sided slope-vs-zero p
    cv2_mean: float                    # mean CV2 across units (local ISI variability)
    rolling_var_ratio: float           # max(window var) / min(window var)
    n_windows: int
    window_s: float
    params: dict
    is_stationary: bool                # True if all diagnostics within thresholds
    flags: tuple[str, ...]             # human-readable reasons if not stationary

def stationarity(rec: SpikeRecording, *,
                 window_s: float = 30.0,
                 cv_threshold: float = 0.30,
                 slope_pvalue_threshold: float = 0.01,
                 var_ratio_threshold: float = 3.0,
                 cv2_threshold: float = 1.5,
                 ) -> StationarityResult: ...
```

### Diagnostics (each independently sourced)

| Metric | Formula | Threshold | Reference |
|--------|---------|-----------|-----------|
| `population_rate_cv` | `std(r) / mean(r)` over windows of `window_s`, where `r` = pop. rate (spikes/s) | > 0.30 → flag | Rate stationarity (Brody et al. 1999, *Neural Comput* 11:1527) |
| `rate_drift_slope` + p-value | OLS slope of `r` vs window-centre time | p < 0.01 → flag | Standard linear drift test |
| `cv2_mean` | mean over units of `mean( 2·|ISI_i - ISI_{i+1}| / (ISI_i + ISI_{i+1}) )` | > 1.5 → flag (extreme local variability) | Holt et al. 1996, *J Neurophysiol* 75:1806 (CV2) |
| `rolling_var_ratio` | `max(var(r_window)) / min(var(r_window))` | > 3.0 → flag | Heteroskedasticity proxy |

`is_stationary = len(flags) == 0`. Flags strings example: `"population_rate_cv=0.42 > 0.30"`, `"rate_drift p=0.003 < 0.01"`.

### Auto-warning hook

A new warning class in `neurocomplexity/_warnings.py`:

```python
class StationarityWarning(UserWarning):
    """Emitted when running a stationarity-sensitive analysis on a recording
    flagged as non-stationary by `analysis.stationarity`."""
```

Sensitive analyses (`criticality`, `wilting_mr`, `transfer_entropy`, `shape_collapse`) call a new helper:

```python
def _warn_if_nonstationary(rec, analysis_name: str) -> None:
    """Run stationarity() with defaults; warn if non-stationary. Deduplicated per (id(rec), analysis_name)."""
```

Mirrors `_warn_if_uncurated` exactly: same dedup set scheme, same suppression instructions in the message, stacklevel=3.

Default thresholds are deliberately permissive — we'd rather miss subtle drift than spam warnings on every recording. Users can call `stationarity(rec, ...)` directly for a richer report.

## Tests (`tests/analysis/test_stationarity.py`)

| Test | Behaviour |
|------|-----------|
| `test_homogeneous_poisson_is_stationary` | Generate 60s of homog. Poisson at 10 Hz, n_units=20 → `is_stationary=True`, no flags. |
| `test_linear_rate_drift_flagged` | Rate ramps 5→25 Hz over 60s → `rate_drift_pvalue < 0.01`, `flags` contains drift reason. |
| `test_high_population_cv_flagged` | Two regimes: 50% low-rate, 50% high-rate → `population_rate_cv > 0.30`. |
| `test_bursty_units_high_cv2` | Bursting ISI pattern (within-burst short, between-burst long) → `cv2_mean > 1.5`. |
| `test_window_count_correct` | duration=120s, window_s=30 → `n_windows == 4`. |
| `test_short_recording_uses_minimum_two_windows` | duration=10s with default window → uses 2 windows minimum, warns once via UserWarning. |
| `test_result_carries_params` | `result.params` round-trips all kwargs. |
| `test_warning_emitted_for_drift_recording_in_te` | drift recording + `transfer_entropy(rec)` → emits `StationarityWarning` once. |
| `test_warning_deduplicated_per_recording_and_analysis` | Two TE calls on same rec → exactly one warning. |
| `test_warning_silenced_for_stationary_recording` | Stationary rec + TE → no `StationarityWarning`. |
| `test_warning_suppressible_via_filterwarnings` | `warnings.filterwarnings("ignore", category=StationarityWarning)` suppresses. |

## Public exports

```python
from neurocomplexity.analysis.stationarity import stationarity, StationarityResult
# accessible as nc.analysis.stationarity / nc.StationarityResult
# warning class as nc.warnings.StationarityWarning
```

## File layout

```
neurocomplexity/analysis/stationarity.py     # new
neurocomplexity/_warnings.py                 # add StationarityWarning + _warn_if_nonstationary
neurocomplexity/analysis/__init__.py         # export stationarity
neurocomplexity/analysis/criticality.py      # call _warn_if_nonstationary
neurocomplexity/analysis/branching.py        # idem
neurocomplexity/analysis/transfer_entropy.py # idem
neurocomplexity/analysis/shape_collapse.py   # idem
tests/analysis/test_stationarity.py          # new
```
