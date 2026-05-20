# Complexity Figures (LMC + MSE) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LMC statistical complexity and Multi-Scale Entropy analyses with two-panel and surrogate-envelope figures, fully integrated with the existing inference adapter pipeline.

**Architecture:** Two independent analysis modules (`analysis/complexity.py`, `analysis/mse.py`) each returning a frozen dataclass `Result`. Two viz modules (`viz/complexity.py`, `viz/mse.py`) following the existing per-analysis-figure convention. Both `Result` types register in `inference/_adapters.py` so `nc.inference.test(...)` and `nc.inference.bootstrap(...)` work without further wiring.

**Tech Stack:** Python 3.11+, NumPy, SciPy (linregress only, optional), matplotlib (viz, optional), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-21-complexity-figures-design.md`.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `neurocomplexity/analysis/complexity.py` | CREATE | `LMCResult`, `lmc_complexity`, private entropy helpers |
| `neurocomplexity/analysis/mse.py` | CREATE | `MSEResult`, `multiscale_entropy`, private `_sample_entropy`/`_coarse_grain` helpers |
| `neurocomplexity/analysis/__init__.py` | MODIFY | Export new names |
| `neurocomplexity/viz/complexity.py` | CREATE | `figure_lmc_complexity` |
| `neurocomplexity/viz/mse.py` | CREATE | `figure_mse` |
| `neurocomplexity/viz/__init__.py` | MODIFY | Export new figure functions |
| `neurocomplexity/inference/_adapters.py` | MODIFY | Register `LMCResult` and `MSEResult` |
| `tests/test_analysis_complexity.py` | CREATE | LMC unit tests |
| `tests/test_analysis_mse.py` | CREATE | MSE unit tests |
| `tests/test_inference_adapters_complexity.py` | CREATE | Adapter round-trip + null_test integration |
| `tests/viz/test_complexity.py` | CREATE | LMC viz smoke + layout |
| `tests/viz/test_mse.py` | CREATE | MSE viz smoke + envelope |

---

## Conventions (apply to every task)

- All new public dataclasses `@dataclass(frozen=True)`.
- `bin_size_s` is in **seconds** (matches package convention).
- Every analysis function starts with:
  ```python
  from neurocomplexity._warnings import _warn_if_uncurated, _warn_if_nonstationary
  _warn_if_uncurated(rec, "<analysis_name>")
  _warn_if_nonstationary(rec, "<analysis_name>")
  ```
- `params` dict in every `Result` stores **every kwarg** needed by the adapter to recompute the statistic on a surrogate recording. The adapter uses `**result.params` directly.
- Long loops in analyses wrap with `from neurocomplexity._progress import progress_iter`.
- Tests use `pytest -v`; run from repo root.
- Commit after each task with the message shown in the final step.

---

## Task 1: Skeleton `analysis/complexity.py` + Shannon entropy helper + `LMCResult`

**Files:**
- Create: `neurocomplexity/analysis/complexity.py`
- Create: `tests/test_analysis_complexity.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis_complexity.py
import numpy as np
import pytest

from neurocomplexity.analysis.complexity import (
    _shannon_entropy_counts,
    _lmc_disequilibrium,
    LMCResult,
)


def test_shannon_uniform_distribution_returns_one():
    # 4 states each with probability 0.25 -> normalized H = 1.0
    counts = np.array([10, 10, 10, 10])
    H = _shannon_entropy_counts(counts)
    assert H == pytest.approx(1.0)


def test_shannon_single_state_returns_zero():
    counts = np.array([100, 0, 0, 0])
    H = _shannon_entropy_counts(counts)
    assert H == pytest.approx(0.0)


def test_disequilibrium_uniform_is_zero():
    counts = np.array([10, 10, 10, 10])
    D = _lmc_disequilibrium(counts)
    assert D == pytest.approx(0.0)


def test_disequilibrium_delta_is_max():
    counts = np.array([100, 0, 0, 0])
    D = _lmc_disequilibrium(counts)
    N = 4
    expected = (1 - 1 / N) ** 2 + (N - 1) * (1 / N) ** 2
    assert D == pytest.approx(expected)


def test_lmcresult_is_frozen_dataclass():
    import dataclasses
    assert dataclasses.is_dataclass(LMCResult)
    fields = {f.name for f in dataclasses.fields(LMCResult)}
    assert {"populations", "kind", "H_per_pop", "D_per_pop", "C_per_pop",
            "H_traj", "D_traj", "C_traj", "window_centers_s",
            "bin_size_seconds", "window_seconds", "step_seconds",
            "n_states_per_pop", "source", "params"}.issubset(fields)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis_complexity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'neurocomplexity.analysis.complexity'`.

- [ ] **Step 3: Write minimal implementation**

```python
# neurocomplexity/analysis/complexity.py
"""LMC statistical complexity (López-Ruiz, Mancini, Calbet 1995).

For a discrete distribution p over N states:
    H = -sum p_i log p_i / log N    (normalized Shannon entropy in [0, 1])
    D = sum (p_i - 1/N)^2           (LMC disequilibrium)
    C = H * D                       (statistical complexity)

C peaks at intermediate H, i.e. structured-but-non-trivial activity.

Reference:
    López-Ruiz, Mancini, Calbet. Phys. Lett. A 209 (1995) 321-326.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LMCResult:
    populations: tuple[str, ...]
    kind: str
    H_per_pop: np.ndarray
    D_per_pop: np.ndarray
    C_per_pop: np.ndarray
    H_traj: np.ndarray | None
    D_traj: np.ndarray | None
    C_traj: np.ndarray | None
    window_centers_s: np.ndarray | None
    bin_size_seconds: float
    window_seconds: float | None
    step_seconds: float | None
    n_states_per_pop: np.ndarray
    source: object
    params: dict = field(default_factory=dict)


def _shannon_entropy_counts(counts: np.ndarray) -> float:
    """Normalized Shannon entropy in [0, 1] of an integer count vector.

    Returns H / log(N) where N = len(counts). Empty bins (zero count) are
    skipped in the sum.
    """
    counts = np.asarray(counts, dtype=np.float64)
    N = counts.size
    if N <= 1:
        return 0.0
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    nz = p > 0
    H = -np.sum(p[nz] * np.log(p[nz]))
    return float(H / np.log(N))


def _lmc_disequilibrium(counts: np.ndarray) -> float:
    """LMC disequilibrium D = sum (p_i - 1/N)^2."""
    counts = np.asarray(counts, dtype=np.float64)
    N = counts.size
    if N <= 1:
        return 0.0
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    return float(np.sum((p - 1.0 / N) ** 2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis_complexity.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add neurocomplexity/analysis/complexity.py tests/test_analysis_complexity.py
git commit -m "feat(analysis): skeleton complexity module with H, D helpers and LMCResult"
```

---

## Task 2: `lmc_complexity` population mode

**Files:**
- Modify: `neurocomplexity/analysis/complexity.py`
- Modify: `tests/test_analysis_complexity.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_analysis_complexity.py`:

```python
from neurocomplexity.analysis.complexity import lmc_complexity
from neurocomplexity.core.recording import SpikeRecording


def _poisson_rec(rate_hz: float, duration_s: float, n_units: int = 30,
                  seed: int = 0, populations: dict | None = None) -> SpikeRecording:
    rng = np.random.default_rng(seed)
    spike_times = []
    unit_ids = []
    for u in range(n_units):
        n = rng.poisson(rate_hz * duration_s)
        ts = np.sort(rng.uniform(0, duration_s, n))
        spike_times.append(ts)
        unit_ids.append(np.full(ts.size, u, dtype=np.int64))
    spike_times = np.concatenate(spike_times) if spike_times else np.array([])
    unit_ids = np.concatenate(unit_ids) if unit_ids else np.array([], dtype=np.int64)
    order = np.argsort(spike_times, kind="stable")
    spike_times = spike_times.astype(np.float64)[order]
    unit_ids = unit_ids[order]
    import pandas as pd
    units = pd.DataFrame({"id": np.arange(n_units, dtype=np.int64),
                          "quality": ["good"] * n_units})
    pops = populations or {"all": np.ones(n_units, dtype=bool)}
    return SpikeRecording(
        spike_times=spike_times, unit_ids=unit_ids, units=units,
        populations=pops, duration=float(duration_s), sampling_rate=30000.0,
        source="synthetic", _filtered=True,
    )


def test_lmc_population_mode_returns_per_pop_arrays():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0, n_units=20)
    result = lmc_complexity(rec, kind="population", bin_size_s=0.05)
    assert result.kind == "population"
    assert result.populations == ("all",)
    assert result.H_per_pop.shape == (1,)
    assert result.D_per_pop.shape == (1,)
    assert result.C_per_pop.shape == (1,)
    assert result.H_traj is None
    assert 0.0 <= result.H_per_pop[0] <= 1.0
    assert result.D_per_pop[0] >= 0.0
    assert result.C_per_pop[0] == pytest.approx(
        result.H_per_pop[0] * result.D_per_pop[0])


def test_lmc_two_populations_returns_two_dots():
    n = 40
    mask_a = np.zeros(n, dtype=bool); mask_a[:20] = True
    mask_b = ~mask_a
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0, n_units=n,
                       populations={"a": mask_a, "b": mask_b})
    result = lmc_complexity(rec, kind="population", bin_size_s=0.05)
    assert result.populations == ("a", "b")
    assert result.H_per_pop.shape == (2,)


def test_lmc_invalid_kind_raises():
    rec = _poisson_rec(rate_hz=20.0, duration_s=5.0)
    with pytest.raises(ValueError, match="kind"):
        lmc_complexity(rec, kind="banana", bin_size_s=0.05)


def test_lmc_params_dict_is_recompute_complete():
    rec = _poisson_rec(rate_hz=20.0, duration_s=5.0)
    result = lmc_complexity(rec, kind="population", bin_size_s=0.05)
    # Adapter must be able to call lmc_complexity(rec, **params) on a surrogate.
    redo = lmc_complexity(rec, **result.params)
    assert np.allclose(redo.C_per_pop, result.C_per_pop)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis_complexity.py -v`
Expected: 4 new tests FAIL with `ImportError` or `AttributeError`.

- [ ] **Step 3: Implement `lmc_complexity` (population mode)**

Append to `neurocomplexity/analysis/complexity.py`:

```python
from neurocomplexity._warnings import _warn_if_uncurated, _warn_if_nonstationary
from neurocomplexity.analysis._binning import bin_spikes
from neurocomplexity.core.recording import SpikeRecording

_ALLOWED_KIND = ("population", "trajectory", "both")


def _hdc_from_count_series(series: np.ndarray) -> tuple[float, float, float, int]:
    """Compute (H, D, C, N_states) for a 1-D integer count series."""
    series = np.asarray(series, dtype=np.int64)
    if series.size == 0:
        return 0.0, 0.0, 0.0, 0
    max_count = int(series.max())
    # State space = {0, 1, ..., max_count}; size = max_count + 1.
    edges = np.arange(max_count + 2)
    counts, _ = np.histogram(series, bins=edges)
    H = _shannon_entropy_counts(counts)
    D = _lmc_disequilibrium(counts)
    return H, D, float(H * D), int(counts.size)


def lmc_complexity(rec: SpikeRecording,
                    populations: Sequence[str] | None = None,
                    *,
                    bin_size_s: float = 0.05,
                    kind: str = "both",
                    window_seconds: float = 1.0,
                    step_seconds: float = 0.5,
                    ) -> LMCResult:
    """LMC statistical complexity for spike populations.

    See module docstring for the math. ``kind`` selects:
      - ``"population"``: one (H, C) point per population from the full recording.
      - ``"trajectory"``: sliding-window (H, C) over time; one row per window.
      - ``"both"``: both, returned in a single result.
    """
    _warn_if_uncurated(rec, "lmc_complexity")
    _warn_if_nonstationary(rec, "lmc_complexity")
    if kind not in _ALLOWED_KIND:
        raise ValueError(f"kind must be one of {_ALLOWED_KIND}; got {kind!r}")
    if bin_size_s <= 0:
        raise ValueError("bin_size_s must be > 0")
    if populations is None:
        populations = list(rec.populations.keys())
    populations = list(populations)

    params = {"populations": list(populations), "bin_size_s": float(bin_size_s),
              "kind": kind, "window_seconds": float(window_seconds),
              "step_seconds": float(step_seconds)}

    counts = bin_spikes(rec, populations, bin_size_s)  # (T, P) int32
    T, P = counts.shape

    H_pop = np.zeros(P, dtype=np.float64)
    D_pop = np.zeros(P, dtype=np.float64)
    C_pop = np.zeros(P, dtype=np.float64)
    Nstates = np.zeros(P, dtype=np.int64)
    for p in range(P):
        H, D, C, n = _hdc_from_count_series(counts[:, p])
        H_pop[p] = H; D_pop[p] = D; C_pop[p] = C; Nstates[p] = n

    H_traj = D_traj = C_traj = win_centers = None
    if kind in ("trajectory", "both"):
        H_traj, D_traj, C_traj, win_centers = _trajectory(
            counts, bin_size_s, window_seconds, step_seconds)

    return LMCResult(
        populations=tuple(populations), kind=kind,
        H_per_pop=H_pop, D_per_pop=D_pop, C_per_pop=C_pop,
        H_traj=H_traj, D_traj=D_traj, C_traj=C_traj,
        window_centers_s=win_centers,
        bin_size_seconds=float(bin_size_s),
        window_seconds=float(window_seconds) if kind != "population" else None,
        step_seconds=float(step_seconds) if kind != "population" else None,
        n_states_per_pop=Nstates,
        source=rec.source, params=params,
    )


def _trajectory(counts, bin_size_s, window_seconds, step_seconds):
    """Placeholder; real implementation lands in Task 3."""
    return None, None, None, None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis_complexity.py -v`
Expected: 9 PASS.

- [ ] **Step 5: Commit**

```bash
git add neurocomplexity/analysis/complexity.py tests/test_analysis_complexity.py
git commit -m "feat(analysis): lmc_complexity population mode with H, D, C per population"
```

---

## Task 3: `lmc_complexity` trajectory mode + `both`

**Files:**
- Modify: `neurocomplexity/analysis/complexity.py`
- Modify: `tests/test_analysis_complexity.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_analysis_complexity.py`:

```python
def test_lmc_trajectory_mode_shapes():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    result = lmc_complexity(rec, kind="trajectory", bin_size_s=0.05,
                             window_seconds=1.0, step_seconds=0.5)
    assert result.kind == "trajectory"
    # windows: floor((10 - 1)/0.5) + 1 = 19
    assert result.H_traj.shape == (19, 1)
    assert result.D_traj.shape == (19, 1)
    assert result.C_traj.shape == (19, 1)
    assert result.window_centers_s.shape == (19,)
    # population fields also populated even in trajectory mode
    assert result.H_per_pop.shape == (1,)


def test_lmc_both_mode_populates_everything():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    result = lmc_complexity(rec, kind="both")
    assert result.H_per_pop.shape == (1,)
    assert result.H_traj is not None
    assert result.H_traj.shape[1] == 1


def test_lmc_trajectory_window_smaller_than_bin_raises():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    with pytest.raises(ValueError, match="window_seconds"):
        lmc_complexity(rec, kind="trajectory",
                        bin_size_s=0.1, window_seconds=0.05)


def test_lmc_trajectory_values_in_range():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    r = lmc_complexity(rec, kind="trajectory", bin_size_s=0.05,
                       window_seconds=1.0, step_seconds=0.5)
    assert np.all((r.H_traj >= 0) & (r.H_traj <= 1))
    assert np.all(r.D_traj >= 0)
    assert np.allclose(r.C_traj, r.H_traj * r.D_traj)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis_complexity.py -v -k trajectory or both_mode`
Expected: 4 new tests FAIL (`AssertionError` or `AttributeError` since `_trajectory` returns `None`s).

- [ ] **Step 3: Replace the `_trajectory` stub with the real implementation**

Replace the placeholder `_trajectory` function in `neurocomplexity/analysis/complexity.py` with:

```python
def _trajectory(counts: np.ndarray, bin_size_s: float,
                 window_seconds: float, step_seconds: float):
    """Sliding-window (H, D, C) on per-population binned counts.

    Returns (H, D, C, centers) where H, D, C have shape (W, P) and centers
    has shape (W,) in seconds (window midpoints).
    """
    from neurocomplexity._progress import progress_iter
    T, P = counts.shape
    win_bins = int(round(window_seconds / bin_size_s))
    step_bins = int(round(step_seconds / bin_size_s))
    if win_bins < 2:
        raise ValueError(
            f"window_seconds ({window_seconds}) must yield >= 2 bins at "
            f"bin_size_s={bin_size_s}")
    if step_bins < 1:
        raise ValueError(
            f"step_seconds ({step_seconds}) must yield >= 1 bin at "
            f"bin_size_s={bin_size_s}")
    if T < win_bins:
        raise ValueError(
            f"recording has {T} bins; need >= window_bins={win_bins}")
    starts = np.arange(0, T - win_bins + 1, step_bins, dtype=np.int64)
    W = starts.size
    H = np.zeros((W, P), dtype=np.float64)
    D = np.zeros((W, P), dtype=np.float64)
    C = np.zeros((W, P), dtype=np.float64)
    for wi, s in enumerate(progress_iter(starts, total=W, desc="lmc-traj")):
        window = counts[s:s + win_bins, :]
        for p in range(P):
            h, d, c, _ = _hdc_from_count_series(window[:, p])
            H[wi, p] = h; D[wi, p] = d; C[wi, p] = c
    centers = (starts + win_bins / 2.0) * bin_size_s
    return H, D, C, centers
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis_complexity.py -v`
Expected: all 13 PASS.

- [ ] **Step 5: Commit**

```bash
git add neurocomplexity/analysis/complexity.py tests/test_analysis_complexity.py
git commit -m "feat(analysis): lmc_complexity trajectory and both modes"
```

---

## Task 4: LMC inference adapter

**Files:**
- Modify: `neurocomplexity/inference/_adapters.py`
- Create: `tests/test_inference_adapters_complexity.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_adapters_complexity.py
import numpy as np
import pytest

from neurocomplexity.analysis.complexity import lmc_complexity, LMCResult
from neurocomplexity.inference._adapters import adapter_for, observed_statistic
from tests.test_analysis_complexity import _poisson_rec


def test_lmc_adapter_returns_c_per_pop_vector():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    result = lmc_complexity(rec, kind="population", bin_size_s=0.05)
    fn = adapter_for(result)
    stat = fn(rec)
    assert isinstance(stat, np.ndarray)
    assert stat.shape == result.C_per_pop.shape
    assert np.allclose(stat, result.C_per_pop)


def test_lmc_observed_statistic_matches_adapter():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    result = lmc_complexity(rec, kind="population", bin_size_s=0.05)
    fn = adapter_for(result)
    assert np.allclose(observed_statistic(result), fn(rec))


def test_lmc_adapter_works_for_both_mode():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    result = lmc_complexity(rec, kind="both")
    fn = adapter_for(result)
    stat = fn(rec)
    assert stat.shape == result.C_per_pop.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inference_adapters_complexity.py -v`
Expected: FAIL with `AdapterError: no inference adapter for LMCResult`.

- [ ] **Step 3: Register the adapter**

Modify `neurocomplexity/inference/_adapters.py`:

Add import at the top with the other analysis imports:

```python
from neurocomplexity.analysis.complexity import lmc_complexity, LMCResult
```

Add the adapter function next to the others:

```python
def _lmc_adapter(result: LMCResult):
    kw = dict(result.params)
    def f(rec):
        return np.asarray(lmc_complexity(rec, **kw).C_per_pop, dtype=float)
    return f
```

Add to `_REGISTRY`:

```python
    LMCResult: _lmc_adapter,
```

Extend `observed_statistic` with:

```python
    if isinstance(result, LMCResult):
        return np.asarray(result.C_per_pop, dtype=float)
```

(Insert before the final `raise AdapterError(...)`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_inference_adapters_complexity.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add neurocomplexity/inference/_adapters.py tests/test_inference_adapters_complexity.py
git commit -m "feat(inference): register LMCResult adapter for null_test/bootstrap"
```

---

## Task 5: `figure_lmc_complexity` viz

**Files:**
- Create: `neurocomplexity/viz/complexity.py`
- Create: `tests/viz/test_complexity.py`

- [ ] **Step 1: Write failing test**

```python
# tests/viz/test_complexity.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurocomplexity.analysis.complexity import lmc_complexity
from neurocomplexity.viz.complexity import figure_lmc_complexity
from tests.test_analysis_complexity import _poisson_rec


def test_figure_lmc_population_renders():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    r = lmc_complexity(rec, kind="population", bin_size_s=0.05)
    fig = figure_lmc_complexity(r)
    assert len(fig.axes) == 1
    # one scatter point per population
    pts = fig.axes[0].collections[0]
    assert pts.get_offsets().shape == (1, 2)
    plt.close(fig)


def test_figure_lmc_both_renders_two_panels():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    r = lmc_complexity(rec, kind="both")
    fig = figure_lmc_complexity(r)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_figure_lmc_accepts_ax_for_single_kind():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    r = lmc_complexity(rec, kind="population")
    fig, ax = plt.subplots()
    out = figure_lmc_complexity(r, ax=ax)
    assert out is fig
    plt.close(fig)


def test_figure_lmc_kind_override():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    r = lmc_complexity(rec, kind="both")
    fig = figure_lmc_complexity(r, kind="population")
    assert len(fig.axes) == 1
    plt.close(fig)


def test_figure_lmc_axis_labels_present():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    r = lmc_complexity(rec, kind="population")
    fig = figure_lmc_complexity(r)
    xl = fig.axes[0].get_xlabel().lower()
    yl = fig.axes[0].get_ylabel().lower()
    assert "h" in xl or "entropy" in xl
    assert "c" in yl or "complexity" in yl
    plt.close(fig)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/viz/test_complexity.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement viz**

```python
# neurocomplexity/viz/complexity.py
"""Figures for LMC statistical complexity."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from neurocomplexity.analysis.complexity import LMCResult
from neurocomplexity.viz._palettes import get_palette, DEFAULT_PALETTE


def _pop_colors(palette_name: str, n: int) -> list[str]:
    p = get_palette(palette_name)
    cat = p["categorical"]
    # cycle if more populations than colors
    return [cat[i % len(cat)] for i in range(n)]


def figure_lmc_complexity(result: LMCResult, *,
                            kind: str | None = None,
                            null_result=None,
                            ax=None,
                            palette: str = DEFAULT_PALETTE,
                            figsize: tuple[float, float] | None = None):
    """Plot the LMC C-vs-H plane.

    Parameters
    ----------
    result
        ``LMCResult`` from ``lmc_complexity``.
    kind
        Override ``result.kind``. One of ``"population"``, ``"trajectory"``,
        ``"both"``. If ``None``, uses ``result.kind``.
    null_result
        Optional ``NullTestResult`` containing surrogate C vectors; drawn as
        a grey cloud behind the real points (population panel only).
    ax
        Axes to draw into when ``kind`` is single-panel. Ignored for ``"both"``.
    palette
        Palette name (``forest`` / ``wine`` / ``sage``).
    figsize
        Figure size override.
    """
    k = kind or result.kind
    if k not in ("population", "trajectory", "both"):
        raise ValueError(f"kind must be one of population/trajectory/both; got {k!r}")
    if k == "trajectory" and result.H_traj is None:
        raise ValueError("result has no trajectory data; recompute with kind='trajectory' or 'both'")

    colors = _pop_colors(palette, len(result.populations))
    p = get_palette(palette)

    if k == "both":
        fig, axes = plt.subplots(1, 2, figsize=figsize or (8.5, 4.0))
        _draw_population(axes[0], result, colors, null_result, p)
        _draw_trajectory(axes[1], result, colors, p)
        fig.tight_layout()
        return fig

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (4.5, 4.0))
    else:
        fig = ax.figure

    if k == "population":
        _draw_population(ax, result, colors, null_result, p)
    else:
        _draw_trajectory(ax, result, colors, p)
    fig.tight_layout()
    return fig


def _draw_population(ax, result: LMCResult, colors, null_result, p):
    if null_result is not None:
        cloud = np.asarray(null_result.null_distribution)
        if cloud.ndim == 2 and cloud.shape[1] == result.C_per_pop.size:
            # cloud[s, p] = C of population p under surrogate s
            for pi in range(cloud.shape[1]):
                ax.scatter(np.full(cloud.shape[0], result.H_per_pop[pi]),
                           cloud[:, pi], s=8, color=p["muted"], alpha=0.4, zorder=1)
    for pi, name in enumerate(result.populations):
        ax.scatter([result.H_per_pop[pi]], [result.C_per_pop[pi]],
                   s=60, color=colors[pi], label=name, zorder=3,
                   edgecolor=p["text"], linewidth=0.5)
    ax.set_xlabel("H (normalized Shannon entropy)", color=p["text"])
    ax.set_ylabel("C (LMC complexity)", color=p["text"])
    ax.set_xlim(-0.02, 1.02)
    if len(result.populations) > 1:
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.18),
                  ncol=min(4, len(result.populations)))


def _draw_trajectory(ax, result: LMCResult, colors, p):
    H = result.H_traj; C = result.C_traj
    for pi, name in enumerate(result.populations):
        # Color = pop; alpha encodes time (early dim, late opaque).
        n = H.shape[0]
        for i in range(n - 1):
            a = 0.2 + 0.8 * (i / max(1, n - 1))
            ax.plot(H[i:i+2, pi], C[i:i+2, pi], color=colors[pi], alpha=a, lw=1.0)
        ax.scatter(H[:, pi], C[:, pi], s=12, color=colors[pi],
                   edgecolor=p["text"], linewidth=0.3, label=name)
    ax.set_xlabel("H (normalized Shannon entropy)", color=p["text"])
    ax.set_ylabel("C (LMC complexity)", color=p["text"])
    ax.set_xlim(-0.02, 1.02)
    if len(result.populations) > 1:
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.18),
                  ncol=min(4, len(result.populations)))
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/viz/test_complexity.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add neurocomplexity/viz/complexity.py tests/viz/test_complexity.py
git commit -m "feat(viz): figure_lmc_complexity with population and trajectory panels"
```

---

## Task 6: Skeleton `analysis/mse.py` + helpers + `MSEResult`

**Files:**
- Create: `neurocomplexity/analysis/mse.py`
- Create: `tests/test_analysis_mse.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_analysis_mse.py
import numpy as np
import pytest

from neurocomplexity.analysis.mse import (
    _coarse_grain,
    _sample_entropy,
    MSEResult,
)


def test_coarse_grain_scale_1_is_identity():
    x = np.arange(10, dtype=np.float64)
    assert np.allclose(_coarse_grain(x, 1), x)


def test_coarse_grain_scale_2_averages_pairs():
    x = np.array([0., 2., 4., 6., 8., 10.])
    cg = _coarse_grain(x, 2)
    assert np.allclose(cg, [1., 5., 9.])


def test_coarse_grain_drops_trailing_partial_window():
    x = np.array([1., 2., 3., 4., 5.])
    cg = _coarse_grain(x, 2)
    assert cg.shape == (2,)


def test_sample_entropy_constant_series_is_zero_or_nan():
    x = np.zeros(200, dtype=np.float64)
    val = _sample_entropy(x, m=2, r=0.1)
    assert np.isnan(val) or val == 0.0


def test_sample_entropy_random_series_finite_positive():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    val = _sample_entropy(x, m=2, r=0.2 * x.std())
    assert np.isfinite(val) and val > 0.0


def test_mseresult_is_frozen_dataclass():
    import dataclasses
    assert dataclasses.is_dataclass(MSEResult)
    fields = {f.name for f in dataclasses.fields(MSEResult)}
    assert {"populations", "scales", "sampen", "bin_size_seconds",
            "m", "r_factor", "r_per_pop", "source", "params"}.issubset(fields)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_analysis_mse.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement helpers**

```python
# neurocomplexity/analysis/mse.py
"""Multi-Scale Entropy (Costa, Goldberger, Peng 2002).

Coarse-grain a 1-D series at integer scales tau = 1..tau_max and compute the
sample entropy (Richman & Moorman 2000) of each coarse-grained series with a
fixed tolerance r = r_factor * SD(original). One curve per population.

References:
    Costa, Goldberger, Peng. Phys. Rev. Lett. 89 (2002) 068102.
    Richman & Moorman. Am. J. Physiol. Heart Circ. Physiol. 278 (2000) H2039.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MSEResult:
    populations: tuple[str, ...]
    scales: np.ndarray
    sampen: np.ndarray
    bin_size_seconds: float
    m: int
    r_factor: float
    r_per_pop: np.ndarray
    source: object
    params: dict = field(default_factory=dict)


def _coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    """Costa coarse-graining: non-overlapping mean of ``scale`` consecutive samples."""
    if scale < 1:
        raise ValueError("scale must be >= 1")
    x = np.asarray(x, dtype=np.float64)
    n = x.size // scale
    if n == 0:
        return np.empty(0, dtype=np.float64)
    return x[:n * scale].reshape(n, scale).mean(axis=1)


def _sample_entropy(x: np.ndarray, m: int, r: float) -> float:
    """Richman & Moorman sample entropy with template length ``m`` and tolerance ``r``.

    Returns NaN if either A or B is zero (insufficient matches).
    """
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    if N < m + 2:
        return float("nan")
    if r <= 0:
        return float("nan")

    def _count_matches(length: int) -> int:
        # Templates of length ``length`` starting at indices 0..N-length.
        K = N - length + 1
        # Build (K, length) view via stride tricks.
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(x, length)  # (K, length)
        count = 0
        for i in range(K - 1):
            d = np.max(np.abs(windows[i + 1:] - windows[i]), axis=1)
            count += int(np.count_nonzero(d <= r))
        return count

    B = _count_matches(m)
    A = _count_matches(m + 1)
    if B == 0 or A == 0:
        return float("nan")
    return float(-np.log(A / B))
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_analysis_mse.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add neurocomplexity/analysis/mse.py tests/test_analysis_mse.py
git commit -m "feat(analysis): skeleton mse module with coarse-graining and sample entropy"
```

---

## Task 7: `multiscale_entropy` function

**Files:**
- Modify: `neurocomplexity/analysis/mse.py`
- Modify: `tests/test_analysis_mse.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_analysis_mse.py`:

```python
from neurocomplexity.analysis.mse import multiscale_entropy
from tests.test_analysis_complexity import _poisson_rec


def test_mse_shapes():
    rec = _poisson_rec(rate_hz=30.0, duration_s=30.0)
    r = multiscale_entropy(rec, bin_size_s=0.05, scale_max=10)
    assert r.populations == ("all",)
    assert r.scales.shape == (10,)
    assert r.sampen.shape == (1, 10)
    assert r.r_per_pop.shape == (1,)


def test_mse_scale_1_finite_for_dense_signal():
    rec = _poisson_rec(rate_hz=50.0, duration_s=30.0)
    r = multiscale_entropy(rec, bin_size_s=0.05, scale_max=5)
    assert np.isfinite(r.sampen[0, 0])


def test_mse_invalid_params_raise():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    with pytest.raises(ValueError):
        multiscale_entropy(rec, scale_max=1)
    with pytest.raises(ValueError):
        multiscale_entropy(rec, m=0)
    with pytest.raises(ValueError):
        multiscale_entropy(rec, r_factor=0)


def test_mse_params_dict_is_recompute_complete():
    rec = _poisson_rec(rate_hz=30.0, duration_s=15.0)
    r = multiscale_entropy(rec, bin_size_s=0.05, scale_max=6)
    redo = multiscale_entropy(rec, **r.params)
    # NaN-equal compare.
    a = np.where(np.isnan(r.sampen), -1, r.sampen)
    b = np.where(np.isnan(redo.sampen), -1, redo.sampen)
    assert np.allclose(a, b)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_analysis_mse.py -v -k "mse_"`
Expected: 4 new tests FAIL.

- [ ] **Step 3: Implement `multiscale_entropy`**

Append to `neurocomplexity/analysis/mse.py`:

```python
from neurocomplexity._warnings import _warn_if_uncurated, _warn_if_nonstationary
from neurocomplexity._progress import progress_iter
from neurocomplexity.analysis._binning import bin_spikes
from neurocomplexity.core.recording import SpikeRecording


def multiscale_entropy(rec: SpikeRecording,
                        populations: Sequence[str] | None = None,
                        *,
                        bin_size_s: float = 0.05,
                        scale_max: int = 20,
                        m: int = 2,
                        r_factor: float = 0.2,
                        ) -> MSEResult:
    """Multi-Scale Entropy profile per population."""
    _warn_if_uncurated(rec, "multiscale_entropy")
    _warn_if_nonstationary(rec, "multiscale_entropy")
    if bin_size_s <= 0:
        raise ValueError("bin_size_s must be > 0")
    if scale_max < 2:
        raise ValueError("scale_max must be >= 2")
    if m < 1:
        raise ValueError("m must be >= 1")
    if r_factor <= 0:
        raise ValueError("r_factor must be > 0")

    if populations is None:
        populations = list(rec.populations.keys())
    populations = list(populations)

    params = {"populations": list(populations), "bin_size_s": float(bin_size_s),
              "scale_max": int(scale_max), "m": int(m),
              "r_factor": float(r_factor)}

    counts = bin_spikes(rec, populations, bin_size_s).astype(np.float64)  # (T, P)
    T, P = counts.shape
    scales = np.arange(1, scale_max + 1, dtype=np.int64)
    S = scales.size
    sampen = np.full((P, S), np.nan, dtype=np.float64)
    r_per_pop = np.zeros(P, dtype=np.float64)

    total = P * S
    pbar = progress_iter(range(total), total=total, desc="mse")
    it = iter(pbar)
    for p in range(P):
        series = counts[:, p]
        r = r_factor * float(series.std(ddof=0))
        r_per_pop[p] = r
        if r <= 0:
            for _ in scales:
                next(it, None)
            continue
        for si, tau in enumerate(scales):
            cg = _coarse_grain(series, int(tau))
            sampen[p, si] = _sample_entropy(cg, m=m, r=r)
            next(it, None)

    return MSEResult(populations=tuple(populations), scales=scales,
                     sampen=sampen, bin_size_seconds=float(bin_size_s),
                     m=int(m), r_factor=float(r_factor),
                     r_per_pop=r_per_pop, source=rec.source, params=params)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_analysis_mse.py -v`
Expected: 10 PASS.

- [ ] **Step 5: Commit**

```bash
git add neurocomplexity/analysis/mse.py tests/test_analysis_mse.py
git commit -m "feat(analysis): multiscale_entropy per-population SampEn profile"
```

---

## Task 8: MSE inference adapter

**Files:**
- Modify: `neurocomplexity/inference/_adapters.py`
- Modify: `tests/test_inference_adapters_complexity.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_inference_adapters_complexity.py`:

```python
from neurocomplexity.analysis.mse import multiscale_entropy, MSEResult


def test_mse_adapter_returns_sampen_matrix():
    rec = _poisson_rec(rate_hz=30.0, duration_s=20.0)
    result = multiscale_entropy(rec, bin_size_s=0.05, scale_max=6)
    fn = adapter_for(result)
    stat = fn(rec)
    assert isinstance(stat, np.ndarray)
    assert stat.shape == result.sampen.shape


def test_mse_observed_statistic_matches_adapter():
    rec = _poisson_rec(rate_hz=30.0, duration_s=20.0)
    result = multiscale_entropy(rec, bin_size_s=0.05, scale_max=6)
    fn = adapter_for(result)
    a = np.where(np.isnan(observed_statistic(result)), -1, observed_statistic(result))
    b = np.where(np.isnan(fn(rec)), -1, fn(rec))
    assert np.allclose(a, b)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_inference_adapters_complexity.py -v -k mse`
Expected: 2 new tests FAIL with `AdapterError`.

- [ ] **Step 3: Register the adapter**

Modify `neurocomplexity/inference/_adapters.py`:

Add import:

```python
from neurocomplexity.analysis.mse import multiscale_entropy, MSEResult
```

Add adapter:

```python
def _mse_adapter(result: MSEResult):
    kw = dict(result.params)
    def f(rec):
        return np.asarray(multiscale_entropy(rec, **kw).sampen, dtype=float)
    return f
```

Add to `_REGISTRY`:

```python
    MSEResult: _mse_adapter,
```

Extend `observed_statistic`:

```python
    if isinstance(result, MSEResult):
        return np.asarray(result.sampen, dtype=float)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_inference_adapters_complexity.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add neurocomplexity/inference/_adapters.py tests/test_inference_adapters_complexity.py
git commit -m "feat(inference): register MSEResult adapter for null_test/bootstrap"
```

---

## Task 9: `figure_mse` viz with surrogate envelope

**Files:**
- Create: `neurocomplexity/viz/mse.py`
- Create: `tests/viz/test_mse.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/viz/test_mse.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurocomplexity.analysis.mse import multiscale_entropy
from neurocomplexity.viz.mse import figure_mse
from tests.test_analysis_complexity import _poisson_rec


def test_figure_mse_renders_one_line_per_pop():
    rec = _poisson_rec(rate_hz=30.0, duration_s=20.0)
    r = multiscale_entropy(rec, bin_size_s=0.05, scale_max=8)
    fig = figure_mse(r)
    ax = fig.axes[0]
    # one Line2D per population
    assert len(ax.lines) >= 1
    plt.close(fig)


def test_figure_mse_axis_labels():
    rec = _poisson_rec(rate_hz=30.0, duration_s=20.0)
    r = multiscale_entropy(rec, bin_size_s=0.05, scale_max=8)
    fig = figure_mse(r)
    xl = fig.axes[0].get_xlabel().lower()
    yl = fig.axes[0].get_ylabel().lower()
    assert "scale" in xl
    assert "sampen" in yl or "entropy" in yl
    plt.close(fig)


def test_figure_mse_accepts_ax():
    rec = _poisson_rec(rate_hz=30.0, duration_s=20.0)
    r = multiscale_entropy(rec, bin_size_s=0.05, scale_max=8)
    fig, ax = plt.subplots()
    out = figure_mse(r, ax=ax)
    assert out is fig
    plt.close(fig)


def test_figure_mse_envelope_off_when_no_null_result():
    rec = _poisson_rec(rate_hz=30.0, duration_s=20.0)
    r = multiscale_entropy(rec, bin_size_s=0.05, scale_max=8)
    fig = figure_mse(r, show_envelope=True)
    # no fill_between collections without null_result
    assert len(fig.axes[0].collections) == 0
    plt.close(fig)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/viz/test_mse.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement viz**

```python
# neurocomplexity/viz/mse.py
"""Multi-scale entropy profile figure."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from neurocomplexity.analysis.mse import MSEResult
from neurocomplexity.viz._palettes import get_palette, DEFAULT_PALETTE


def _pop_colors(palette_name: str, n: int) -> list[str]:
    p = get_palette(palette_name)
    cat = p["categorical"]
    return [cat[i % len(cat)] for i in range(n)]


def figure_mse(result: MSEResult, *,
                null_result=None,
                ax=None,
                palette: str = DEFAULT_PALETTE,
                show_envelope: bool = True,
                figsize: tuple[float, float] | None = None):
    """Plot SampEn vs scale, one line per population.

    With ``null_result`` provided and ``show_envelope=True``, draw a grey
    [mean ± 2 SD] band per population from the surrogate sampen matrices.
    """
    colors = _pop_colors(palette, len(result.populations))
    p = get_palette(palette)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (5.0, 4.0))
    else:
        fig = ax.figure

    scales = result.scales
    sampen = result.sampen  # (P, S)

    if null_result is not None and show_envelope:
        null_arr = np.asarray(null_result.null_distribution)
        # null_arr shape expected (N_surr, P, S)
        if null_arr.ndim == 3 and null_arr.shape[1:] == sampen.shape:
            mean = np.nanmean(null_arr, axis=0)  # (P, S)
            sd = np.nanstd(null_arr, axis=0)
            for pi in range(sampen.shape[0]):
                ax.fill_between(scales, mean[pi] - 2 * sd[pi], mean[pi] + 2 * sd[pi],
                                color=p["muted"], alpha=0.3, linewidth=0)

    for pi, name in enumerate(result.populations):
        ax.plot(scales, sampen[pi], color=colors[pi], marker="o",
                label=name, lw=1.2, markersize=4)
    ax.set_xlabel("Scale (tau)", color=p["text"])
    ax.set_ylabel("SampEn", color=p["text"])
    if len(result.populations) > 1:
        ax.legend(frameon=False, loc="upper left",
                  bbox_to_anchor=(0.0, 1.18),
                  ncol=min(4, len(result.populations)))
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/viz/test_mse.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add neurocomplexity/viz/mse.py tests/viz/test_mse.py
git commit -m "feat(viz): figure_mse with optional surrogate envelope"
```

---

## Task 10: Top-level exports + full-suite green

**Files:**
- Modify: `neurocomplexity/analysis/__init__.py`
- Modify: `neurocomplexity/viz/__init__.py`

- [ ] **Step 1: Add failing import test**

Create `tests/test_complexity_top_level_exports.py`:

```python
import numpy as np


def test_top_level_complexity_exports():
    import neurocomplexity as nc
    assert hasattr(nc.analysis, "lmc_complexity")
    assert hasattr(nc.analysis, "LMCResult")
    assert hasattr(nc.analysis, "multiscale_entropy")
    assert hasattr(nc.analysis, "MSEResult")


def test_top_level_viz_exports():
    import neurocomplexity as nc
    assert hasattr(nc.viz, "figure_lmc_complexity")
    assert hasattr(nc.viz, "figure_mse")
```

Run: `pytest tests/test_complexity_top_level_exports.py -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 2: Modify `neurocomplexity/analysis/__init__.py`**

Add imports:

```python
from neurocomplexity.analysis.complexity import lmc_complexity, LMCResult
from neurocomplexity.analysis.mse import multiscale_entropy, MSEResult
```

Extend `__all__`:

```python
    "lmc_complexity", "LMCResult",
    "multiscale_entropy", "MSEResult",
```

- [ ] **Step 3: Modify `neurocomplexity/viz/__init__.py`**

Add imports and `__all__` entries:

```python
from neurocomplexity.viz.complexity import figure_lmc_complexity
from neurocomplexity.viz.mse import figure_mse
```

And add `"figure_lmc_complexity"`, `"figure_mse"` to `__all__`.

- [ ] **Step 4: Run the full suite to confirm no regressions**

Run: `pytest tests/test_complexity_top_level_exports.py tests/test_analysis_complexity.py tests/test_analysis_mse.py tests/test_inference_adapters_complexity.py tests/viz/test_complexity.py tests/viz/test_mse.py -v`
Expected: all new tests PASS (~26 total).

Then run the full suite to check for regressions:

Run: `pytest -x`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add neurocomplexity/analysis/__init__.py neurocomplexity/viz/__init__.py tests/test_complexity_top_level_exports.py
git commit -m "feat: expose LMC complexity + MSE at neurocomplexity.analysis and .viz"
```

---

## Final State

- Two new analyses: `lmc_complexity`, `multiscale_entropy`.
- Two new figures: `figure_lmc_complexity`, `figure_mse`.
- Full inference integration via `_adapters` — `nc.inference.test(result)` and `nc.inference.bootstrap(result)` work without further wiring for both.
- ~26 new tests, full suite green.
- Spec B (effective connectivity network + attractor manifold) remains open as a separate spec to brainstorm next.
