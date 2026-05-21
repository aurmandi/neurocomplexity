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
    from neurocomplexity._warnings import _warn_if_uncurated, _warn_if_nonstationary
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
