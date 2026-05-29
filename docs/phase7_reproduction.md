# Phase 7 — Headline Real-Data Reproduction Gate

End-to-end execution of the entire `neurocomplexity` public surface
(analysis + inference + visualisation) on a real Allen Institute
Neuropixels recording. This is the Phase-7 acceptance gate: the package must
load, analyse, infer, and plot on genuine spike-sorted data — not just
synthetic fixtures — before the API can be frozen (Phase 8).

## Verdict

**PASS.** Every public analysis, inference, and figure entry point runs to
completion on the real recording. One real-data-only defect (a surrogate
null-test memory blow-up) was found, fixed, regression-tested, and
re-verified on the same recording.

| Run | Steps | Result |
|---|---|---|
| Initial full run | 31 | 27 PASS / 4 FAIL (`null_test` `MemoryError` → 3 dependent TE figures cascaded) |
| After cache fix (targeted re-verify) | 4 previously-failing | 4/4 PASS |
| **Effective post-fix** | **31** | **31 PASS / 0 FAIL** |

## Dataset

Allen Brain Observatory — Visual Coding Neuropixels.

| Property | Value |
|---|---|
| Session | `715093703` |
| File | `session_715093703.nwb` (HDF5, 2.7 GB) |
| Loader | `nc.io.from_nwb` (pynwb 3.1.3 backend) |
| Total units | 2 779 |
| Curated (`quality == 'good'`) | 2 110 |
| Total spikes | 131 242 535 |
| Session duration | 9 641.7 s |
| Analysis window | midpoint 30-min crop `[3921, 5721] s` (1 800 s, 21 887 530 spikes) |
| Populations | top-4 brain areas: `LP`, `CA1`, `VISrl`, `grey` |

## Headline results (30-min window)

These are smoke-gate values demonstrating each estimator runs and returns
finite, plausible output on real data — **not** a scientific claim about
this session (see caveat below).

| Measure | Value |
|---|---|
| Stationarity | `is_stationary=False` (rolling_var_ratio = 28.93, 60 windows) |
| Criticality αₛ (size) | 1.089 |
| Criticality α_t (duration, CSN MLE) | 1.124 |
| Branching ratio m (Wilting–Priesemann) | 0.9757 |
| Branching m bootstrap 95% CI | (0.971, 0.978) |
| Shape-collapse γ | 1.293 |
| Transfer entropy (4×4, max) | 0.0019 |
| Participation ratio | 849.07 |
| PID | R = 0.0130, U₁ = 0.0001, U₂ = 0.0000, S = 0.0107 |
| LMC entropy H per pop | [0.892, 0.806, 0.834, 0.811] |
| LMC complexity C per pop | [0.0046, 0.0056, 0.0100, 0.0083] |
| MSE SampEn matrix | shape (4, 8) |
| Manifold PCA explained var | [0.040, 0.020] |

### Scientific caveat (honesty flag)

The package's own stationarity diagnostic flagged this window as
**non-stationary** (`rolling_var_ratio = 28.93 ≫ 3.0`) and every
stationarity-sensitive estimator emitted a `StationarityWarning`
accordingly. Criticality exponents, branching ratio, shape-collapse γ, and
TE are therefore biased on this raw window and the values above must **not**
be interpreted as estimates of this session's true dynamics. The
appropriate workflow — restrict to a stationary epoch via `rec.crop(...)`
before estimating critical exponents — is exactly what the warning
instructs. The gate measures *that the pipeline runs and self-reports its
own assumption violations*, which it does correctly.

## Defect found and fixed

**`null_test` surrogate-pool OOM on large recordings** (real-data-only;
synthetic fixtures are too small to trigger it).

- *Symptom:* `inference.test(te, rec, surrogate="spike_dither", n=50)`
  raised `MemoryError` after 7 138 s at 2 976 MB RSS.
- *Cause:* the internally-built `SurrogatePool` used the default
  `cache_size=64`. The pool is consumed in a single forward pass and each
  surrogate is a full-size `SpikeRecording` copy, so on a 22M-spike window
  the 64-deep cache exhausted memory.
- *Fix:* bound the internal cache to the concurrent-worker count
  (`max(2, n_jobs)` when parallel, else 1), independent of `n`. Surrogate
  generation is deterministic per index (fixed child seeds), so cache size
  is a pure performance/memory parameter and never affects results.
- *Re-verification:* on the same session, `null_test` now completes at
  **153 MB** peak Python allocation, and the three dependent TE figures
  render.
- *Regression lock:* `tests/test_inference_null_test.py::test_internal_pool_cache_is_bounded`.
- *Commit:* `713d277`.

## Figures produced (12/12)

`outputs/integration_session_715093703/` (git-ignored):
`criticality`, `branching`, `branching_bootstrap`, `dimensionality`,
`shape_collapse`, `pid`, `lmc_complexity`, `mse`, `manifold`,
`te_significance_matrix`, `te_volcano`, `te_network`.

## Reproduction

```
python examples/integration_session_715093703.py
```

The script loads the NWB session, curates to good units, crops to the
midpoint 30-min window, builds top-4 brain-area populations, then runs every
analysis → inference → figure path with per-step PASS/FAIL + timing + RSS,
ending in a summary line. Environment: pynwb 3.1.3, h5py 3.13.0, Python 3.12.

## Disposition

Real-data gate cleared; the only blocking defect is fixed and locked. **Phase
7 closes. Phase 8 (API freeze + docs + software-paper draft) is unblocked**,
and the software paper may now cite genuine Allen session 715093703 results.
