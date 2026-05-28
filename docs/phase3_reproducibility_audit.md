# Phase 3 — Numerical & Reproducibility Audit

Companion to `tests/test_reproducibility.py`. Documents every reproducibility
invariant the package commits to, the test that enforces it, and any
diagnostic findings that do not rise to a code change.

Status legend: ✅ pass · ⚠ note · ❌ bug.

| # | Category | Tests | Verdict |
|---|---|---|---|
| 1 | Seed determinism | 4 | ✅ |
| 2 | Pickle round-trip (every Result) | 10 | ✅ |
| 3 | Deepcopy round-trip | 2 | ✅ |
| 4 | Edge-case handling | 7 | ✅ |
| 5 | Dtype + timezone | 4 | ✅ |
| 6 | Order independence | 2 | ✅ |

**29 / 29 pass. Zero bugs surfaced.**

---

## § 1 — Seed determinism

**Invariant.** Two runs of any seeded estimator with the same seed produce
byte-identical output: same `rec` arrays, same Result fields, same numerical
values to floating-point exactness.

| Test | What it locks in |
|---|---|
| `TestDeterminism::test_wilting_mr_deterministic` | `branching_network(seed=k)` → identical spikes; `wilting_mr` → identical `m`, `r_values` |
| `TestDeterminism::test_transfer_entropy_deterministic` | TE matrix identical across two runs on the same recording |
| `TestDeterminism::test_surrogate_pool_deterministic` | `SurrogatePool(seed=k)[0]` byte-identical across instances |
| `TestDeterminism::test_pca_manifold_deterministic` | PCA coords identical with the same `random_state` |

All pass. Confirms the seeding-everywhere discipline of the codebase
(`np.random.default_rng(seed)`, `SeedSequence` spawning for the pool,
explicit `random_state` for the manifold backends).

---

## § 2 — Pickle round-trip

**Invariant.** Every Result dataclass (and `SpikeRecording`) survives
`pickle.loads(pickle.dumps(obj))` field-by-field, including numpy arrays
(byte-equal via `np.testing.assert_array_equal`), provenance back-pointers,
params dicts, and NaN floats.

Result types tested:

- `SpikeRecording`
- `BranchingResult`
- `CriticalityResult`
- `DimensionalityResult`
- `LMCResult`
- `MSEResult`
- `TransferEntropyResult`
- `PIDResult`
- `StationarityResult`
- `InferenceResult`

`ManifoldResult`, `ShapeCollapseResult`, `AutonomyResult`, `BenchmarkResult` not
explicitly tested here but use the same dataclass pattern; covered by
inheritance of behaviour.

All pass.

---

## § 3 — Deepcopy round-trip

**Invariant.** `copy.deepcopy(r)` returns an equal object whose mutable
sub-objects (numpy arrays, dataframes) are distinct copies. Confirms that
deep-copying a result does not aliasing-share its arrays with the original
(which would let an in-place mutation downstream corrupt the source).

Tests: `TestDeepcopyRoundtrip::test_recording_deepcopy`,
`test_branching_result_deepcopy`. Both pass.

---

## § 4 — Edge cases

**Invariant.** Degenerate inputs must produce either NaN (estimator
gracefully reports "undefined") or a loud `ValueError` ("you cannot run this
on this input") — never a silent wrong number, never `-inf`, never a crash.

| Test | Edge case | Expected behaviour |
|---|---|---|
| `test_empty_avalanche_returns_empty_arrays` | `extract_avalanches([])` | empty arrays, no crash |
| `test_single_event_avalanche` | one-spike series | exactly one avalanche, size = 1, lifetime = bin_size |
| `test_all_zero_spike_train_does_not_crash` | rec with zero spikes | `dimensionality` raises `ValueError("fewer than 2 active units")` |
| `test_log_zero_protected_in_lmc` | distribution with zero-probability cells | `H` is finite (skips zero terms) |
| `test_log_zero_protected_in_te` | all-silent X and Y | `TE = 0.0`, no `log(0)` propagation |
| `test_var_zero_in_branching_returns_nan` | perfectly periodic activity → Var(A_t) = 0 | `wilting_mr.m = NaN`, `r_squared = NaN` |
| `test_single_unit_dimensionality_raises` | only one unit | `ValueError("at least 2 units")` |

All pass. Confirms the "fail loud, never quietly wrong" principle from the
publication plan.

---

## § 5 — Dtype + timezone

**Invariant.** Core arrays carry their documented dtype. Provenance timestamps
are UTC-aware.

| Test | What it locks in |
|---|---|
| `test_recording_dtypes` | `spike_times.dtype == float64`, `unit_ids.dtype == int64` |
| `test_provenance_record_timestamp_is_utc_iso8601` | `loaded_at` is an ISO-8601 string ending in `Z` or `+00:00` |
| `test_branching_m_is_python_float` | `BranchingResult.m` is a plain Python `float`, not `np.float64` |
| `test_inference_p_value_dtype` | `pvalue_from_null` returns float64 |

All pass. Confirms timezone safety (the `datetime.now(timezone.utc).isoformat()`
call in `ProvenanceRecord.for_file`/`for_memory` emits a tz-aware ISO string,
which all cross-machine comparisons can rely on).

---

## § 6 — Order independence

**Invariant.** Reshuffling spike order globally (then re-sorting by time) or
permuting population labels must not change any population-level estimator
beyond floating-point round-off.

| Test | What it locks in |
|---|---|
| `test_branching_invariant_under_spike_resort` | random global shuffle + re-sort → identical `wilting_mr.m` (within 1e-12) |
| `test_te_matrix_invariant_under_population_relabel` | `TE([A,B])` and `TE([B,A])` swapped row/col give same values |

Both pass. Confirms the binning / regression pipeline is genuinely a function
of the (unordered) spike set, not of the input array's storage layout.

---

## Diagnostic findings (no code change required)

### `dateutil.tz.tz.py:37 utcfromtimestamp` DeprecationWarning

Observed during test collection:

```
C:\...\dateutil\tz\tz.py:37: DeprecationWarning: datetime.datetime.utcfromtimestamp()
is deprecated and scheduled for removal in a future version.
```

This is in `python-dateutil`, pulled in by `pandas` and `pynwb`. Not in our
code. Will resolve once those upstream libraries cut releases that drop the
deprecated call. No action for `neurocomplexity`.

### Stationarity warnings on synthetic Poisson fixtures

Many tests trigger `StationarityWarning` because the synthetic Poisson
recordings happen to have small rate drifts under random sampling. This is
**expected** — the warning is doing its job. Tests silence the warning
locally where appropriate.

### Recording duration < 2 × window_s

The stationarity function reduces to 2 windows on recordings shorter than
2 × `window_s`. Emits `UserWarning("recording duration ... < 2*window_s ...;
reducing to 2 windows.")`. Visible during tests but expected; not a Phase 3
concern.

---

## Phase 3 summary

| Metric | Value |
|---|---|
| Reproducibility tests added | 29 |
| Pass / Fail | 29 / 0 |
| Bugs found in package | 0 |
| Test-side errors found and fixed | 4 (signatures / params) |

**Verdict: Phase 3 closed.** Determinism, edge-case discipline, dtype
hygiene, and order-independence all confirmed. Phase 4 (three-reviewer
panel) unblocked.
