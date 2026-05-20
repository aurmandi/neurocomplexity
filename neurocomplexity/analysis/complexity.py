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
