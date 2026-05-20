# Spec A — Complexity Figures (LMC + MSE) Design

**Date:** 2026-05-21
**Status:** Approved (brainstorm-condensed). Spec B (flow figures) tracked separately.

## Goal

Add two new analysis + figure pairs to `neurocomplexity`:

1. **LMC statistical complexity** (López-Ruiz, Mancini, Calbet 1995) — "Information Complexity vs. Randomness" plane. Two-panel: per-population snapshot with surrogate cloud, plus sliding-window trajectory.
2. **Multi-Scale Entropy (MSE) profile** (Costa, Goldberger, Peng 2002) — SampEn vs scale, one curve per population, with surrogate envelope.

Both analyses operate on a single `SpikeRecording`, integrate with the existing inference pipeline (`null_test`, `bootstrap`, `_adapters`), and ship per-analysis viz functions matching the existing per-analysis viz convention (`figure_branching`, `figure_criticality`, ...).

## Non-Goals

- Multi-recording batch comparison (no batch infrastructure exists; out of scope).
- Bin-size sweep variant of LMC (semantically wrong — bin size is a meta-parameter, not "randomness").
- Tononi-Sporns-Edelman neural complexity C_N (deferred; combinatorial cost).
- Spec B figures (effective connectivity network, attractor manifold) — separate spec.

## Architecture

Mirror existing per-analysis convention from the project rationale:

```
neurocomplexity/
  analysis/
    complexity.py       # NEW — lmc_complexity, LMCResult
    mse.py              # NEW — multiscale_entropy, MSEResult
  viz/
    complexity.py       # NEW — figure_lmc_complexity
    mse.py              # NEW — figure_mse
  inference/
    _adapters.py        # MODIFY — register LMCResult + MSEResult
tests/
  test_analysis_complexity.py     # NEW
  test_analysis_mse.py            # NEW
  test_inference_adapters_complexity.py   # NEW (covers both)
  viz/
    test_complexity.py            # NEW
    test_mse.py                   # NEW
```

Module-layout choice: **Option A — two separate analysis modules + two separate viz modules.** Matches the existing convention exactly (one analysis file = one analytic concept). No shared private helpers across LMC/MSE; each module is self-contained.

## Math

### LMC statistical complexity

For a discrete probability distribution `p = (p_1, ..., p_N)` with `N` accessible states:

- **Normalized Shannon entropy:** `H = -sum_i p_i log p_i / log N`, so `H ∈ [0, 1]`.
- **Disequilibrium (LMC 1995):** `D = sum_i (p_i - 1/N)^2`. `D ∈ [0, (N-1)/N]`.
- **Complexity:** `C = H · D`. Dimensionless. Peaks at intermediate `H`.

**State distribution for population dynamics:** bin the recording with `bin_size_s` to get `(T, P)` int counts. For each population `p`, take the column (`T,` count series), compute the empirical histogram of count values (states are the distinct integer counts observed up to `max_count + 1`), and compute `H, D, C` from that distribution.

Rationale: this captures the diversity of activity states the population visits over time, which matches the LMC interpretation ("a system is complex when it visits a structured-but-non-trivial mixture of states"). It's cheap and stable.

### MSE (Costa 2002)

For a 1-D series `x_1, ..., x_T`:

- **Coarse-graining at scale τ:** `y_j^(τ) = (1/τ) sum_{i=(j-1)τ+1}^{jτ} x_i`, length `floor(T/τ)`.
- **SampEn(m, r):** standard sample entropy (Richman & Moorman 2000) with template length `m` and tolerance `r`. `r = r_factor · SD(x_original)` — tolerance is fixed on the original series so scales are comparable.
- Plot `SampEn(τ)` vs `τ` for `τ = 1..τ_max`. One curve per population.

Coarse-graining input series: the per-population binned rate (a column of `bin_spikes` output).

### Surrogates

Reuse `SurrogatePool` from `inference/pool.py`. Default method = `"spike_dither"` (Louis-Gerstein-Grün 2010). Both analyses go through `inference._adapters` so users can call `nc.inference.test(result)` and `nc.inference.bootstrap(result)` uniformly.

For figure surrogate overlays, the viz functions take an optional `null_result=` kwarg containing the `NullTestResult` produced by `nc.inference.test(...)`. Viz never runs surrogates itself.

## Public API

### `neurocomplexity.analysis.complexity`

```python
@dataclass(frozen=True)
class LMCResult:
    populations: tuple[str, ...]
    kind: str                          # "population" | "trajectory" | "both"
    # population mode (always populated):
    H_per_pop: np.ndarray              # shape (P,), normalized in [0, 1]
    D_per_pop: np.ndarray              # shape (P,)
    C_per_pop: np.ndarray              # shape (P,)
    # trajectory mode (populated when kind in {"trajectory","both"}, else None):
    H_traj: np.ndarray | None          # shape (W, P), one row per window
    D_traj: np.ndarray | None          # shape (W, P)
    C_traj: np.ndarray | None          # shape (W, P)
    window_centers_s: np.ndarray | None  # shape (W,)
    # bookkeeping:
    bin_size_seconds: float
    window_seconds: float | None
    step_seconds: float | None
    n_states_per_pop: np.ndarray       # shape (P,), number of distinct count states observed
    source: object
    params: dict

def lmc_complexity(rec: SpikeRecording,
                   populations: Sequence[str] | None = None,
                   *,
                   bin_size_s: float = 0.05,
                   kind: str = "both",
                   window_seconds: float = 1.0,
                   step_seconds: float = 0.5,
                   ) -> LMCResult: ...
```

Defaults: `bin_size_s=0.05` (50 ms — standard for population-rate work), `window_seconds=1.0`, `step_seconds=0.5`. `kind="both"` is the default so the figure works without further config.

Validation:
- `populations=None` → use all `rec.populations`.
- `kind not in {"population","trajectory","both"}` → `ValueError`.
- For trajectory: `window_seconds > 0`, `step_seconds > 0`, and recording must fit at least one window.
- Each window must contain at least 2 bins (`window_seconds > bin_size_s`); else `ValueError`.

`alternative` metadata in `params`: defaults to `"greater"` (high C is the directional claim — populations more complex than dithered spike trains).

### `neurocomplexity.analysis.mse`

```python
@dataclass(frozen=True)
class MSEResult:
    populations: tuple[str, ...]
    scales: np.ndarray                 # shape (S,) int, e.g. arange(1, 21)
    sampen: np.ndarray                 # shape (P, S); NaN where series too short
    bin_size_seconds: float
    m: int                             # template length
    r_factor: float                    # tolerance factor; r = r_factor * SD(series)
    r_per_pop: np.ndarray              # shape (P,), the absolute r used per pop
    source: object
    params: dict

def multiscale_entropy(rec: SpikeRecording,
                       populations: Sequence[str] | None = None,
                       *,
                       bin_size_s: float = 0.05,
                       scale_max: int = 20,
                       m: int = 2,
                       r_factor: float = 0.2,
                       ) -> MSEResult: ...
```

Defaults follow Costa 2002: `m=2`, `r_factor=0.2`. `scale_max=20` is a sensible upper bound for typical recordings.

Validation:
- `scale_max >= 2`, `m >= 1`, `r_factor > 0`.
- For each (pop, scale), if `T/scale < m + 2` the entry is `NaN` (too few points for SampEn).
- `alternative` metadata in `params`: defaults to `"greater"`.

### `neurocomplexity.inference._adapters`

Register two new entries:

- `LMCResult` → returns `C_per_pop` vector (shape `(P,)`). This is the directional stat — surrogates dither spikes, real C should sit above the null.
- `MSEResult` → returns `sampen` matrix (shape `(P, S)`). Cell-wise null test.

`observed_statistic` extended to match.

### Viz API

```python
# neurocomplexity/viz/complexity.py
def figure_lmc_complexity(result: LMCResult,
                           *,
                           kind: str | None = None,       # override result.kind
                           null_result=None,              # NullTestResult for cloud
                           ax=None,                       # if single-panel; else creates 1x2
                           palette: str | None = None,
                           figsize=None,
                           ) -> matplotlib.figure.Figure: ...

# neurocomplexity/viz/mse.py
def figure_mse(result: MSEResult,
                *,
                null_result=None,
                ax=None,
                palette: str | None = None,
                show_envelope: bool = True,
                figsize=None,
                ) -> matplotlib.figure.Figure: ...
```

LMC viz layout:
- `kind="population"` → single axes, x=H, y=C, one dot per population, optional grey surrogate cloud (from `null_result`).
- `kind="trajectory"` → single axes, x=H, y=C, scatter with line connecting time-ordered windows per pop (color = pop; alpha varies with time so the user sees evolution).
- `kind="both"` → 1x2 panel, left = population, right = trajectory.

MSE viz layout: x=scale, y=SampEn, one line per pop (color from palette). If `null_result` given and `show_envelope`, draw `[mean ± 2 SD]` grey band per pop across surrogate matrix.

Both viz functions follow the existing `_palettes.py` rules (force text color black, return `Figure`, accept `ax=` for embedding).

## Cross-cutting Concerns

### Warnings

Both analyses call (top of fn):
- `_warn_if_uncurated(rec, "lmc_complexity")` / `("multiscale_entropy")`
- `_warn_if_nonstationary(rec, "lmc_complexity")` / `("multiscale_entropy")`

Matches existing convention from `branching.py`, `criticality.py`, etc.

### Progress bars

- LMC: trajectory loop wrapped with `progress_iter(...)` (length = W * P).
- MSE: scale loop wrapped with `progress_iter(...)` (length = S * P).
- Surrogate machinery already wraps null_test/bootstrap loops; no extra work.

### Memory

LMC/MSE consume `bin_spikes` output which already emits `MemoryAllocationWarning` when the (T,P) matrix exceeds 25% of RAM. Trajectory mode never rebins — it slices the same (T,P) matrix per window. No new memory hooks needed.

### Stationarity

`stationarity` already auto-warns from `criticality`, `wilting_mr`, `transfer_entropy`, `shape_collapse`. Add `complexity.lmc_complexity` and `mse.multiscale_entropy` to the dedup set used by `_warn_if_nonstationary`.

## Testing Strategy

Following the existing TDD pattern from the five prior fixes. Each analysis file gets a test file with:

1. **Smoke test:** synthetic Poisson rec → fn runs, returns Result with expected shapes.
2. **Determinism:** same rec → identical result (no internal RNG; surrogates are external).
3. **Math sanity:** uniform-rate Poisson → high H, low C; bursty rec → lower H, higher D, higher C.
4. **Validation:** bad params → `ValueError`.
5. **Edge cases:** silent population (all zeros) → handled gracefully (H=0, D=max, C=0; or NaN with reason in params — pick H=0 path).
6. **Warning hooks:** uncurated rec → `QualityWarning`; nonstationary rec → `StationarityWarning`.

Adapter tests: round-trip through `adapter_for(result)(rec)` and `observed_statistic(result)` produce the same array.

Viz tests: smoke (figure renders without exception), palette respected, panel count matches `kind`, optional `null_result` overlay draws extra artists.

Target: ~24 tests across the five new test files.

## Risks

| Risk | Mitigation |
|---|---|
| State-space size N varies per pop → H normalization comparison messy | Normalize by `log N` where N = observed distinct count values + 1 (sentinel for unseen). Document in docstring. |
| MSE on count series (discrete, low-rate) → SampEn often NaN | Default `bin_size_s=0.05` keeps counts dense; document that very sparse pops give NaN entries; viz skips NaN cleanly. |
| Trajectory mode + surrogate cloud combined could be very expensive | Trajectory in figures only — null_test runs against `C_per_pop` (population mode), not trajectory. Document explicitly. |
| `kind` proliferation in viz signature | One enum-like string, validated at top; covered by tests. |

## Out-of-Scope / Deferred

- Tononi neural complexity C_N (combinatorial; future spec).
- Multi-recording batch viz (no infra).
- MSE on `ContinuousSignal` inputs (trivially addable once analysis is settled; defer).
- Permutation-entropy-based complexity (Rosso 2007 plane; defer).

## Spec B Preview (separate spec, not in scope here)

For tracking only — separate brainstorm + design doc forthcoming:

- **Effective Connectivity & Information Flow Network** — graph viz over existing `TransferEntropyResult` + `NullTestResult`. New: `viz/network.py` with `figure_te_network(te_result, null_result, ...)`. No new analysis.
- **Low-Dim Attractor Manifold** — PCA (and optional UMAP) of `(T, P)` bin matrix → 2D/3D trajectory. New: `analysis/manifold.py` + `viz/manifold.py`.
