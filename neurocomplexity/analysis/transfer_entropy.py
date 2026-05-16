"""Pairwise transfer entropy across populations.

Estimator: Schreiber (2000) on binary-thresholded spike counts, with
Miller-Madow bias correction. Robust on integer counts where Kraskov k-NN
degenerates (zero distances).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from neurocomplexity.analysis._binning import bin_spikes
from neurocomplexity.core.recording import SpikeRecording


@dataclass(frozen=True)
class TransferEntropyResult:
    matrix: np.ndarray       # (P, P) TE in nats; row=source, col=target
    populations: tuple[str, ...]
    bin_size_seconds: float
    delay_bins: int
    source: object
    params: dict = field(default_factory=dict)


def _binary_schreiber_te(source_ts: np.ndarray, target_ts: np.ndarray,
                          delay: int = 1) -> float:
    """TE(source -> target) in nats."""
    y = (np.asarray(target_ts) > 0).astype(np.int8)
    x = (np.asarray(source_ts) > 0).astype(np.int8)
    T = len(y)
    if T <= delay + 1:
        return 0.0

    y_future = y[delay:]
    y_past = y[:-delay]
    x_past = x[:-delay]
    N = len(y_future)

    idx = 4 * y_future + 2 * y_past + x_past
    counts = np.bincount(idx, minlength=8).astype(np.float64)
    p_joint = counts / N

    te = 0.0
    for yf in (0, 1):
        for yp in (0, 1):
            for xp in (0, 1):
                i = 4 * yf + 2 * yp + xp
                if counts[i] == 0:
                    continue
                p_yyx = p_joint[i]
                p_yx = p_joint[4 * 0 + 2 * yp + xp] + p_joint[4 * 1 + 2 * yp + xp]
                p_yy = p_joint[4 * yf + 2 * yp + 0] + p_joint[4 * yf + 2 * yp + 1]
                p_y = sum(p_joint[4 * yfp + 2 * yp + xpp]
                          for yfp in (0, 1) for xpp in (0, 1))
                if p_yx == 0 or p_yy == 0 or p_y == 0:
                    continue
                cond_full = p_yyx / p_yx
                cond_red = p_yy / p_y
                if cond_full == 0 or cond_red == 0:
                    continue
                te += p_yyx * np.log(cond_full / cond_red)

    m = int(np.sum(counts > 0))
    te -= (m - 1) / (2.0 * N)
    return max(0.0, float(te))


def transfer_entropy(rec: SpikeRecording,
                     populations: Sequence[str] | None = None,
                     bin_size_ms: float = 5.0,
                     delay_bins: int = 1,
                     estimator: str = "binary",
                     ) -> TransferEntropyResult:
    """Pairwise TE matrix across the given populations."""
    if estimator != "binary":
        raise ValueError(f"only estimator='binary' is implemented in v0.1; got {estimator!r}")
    if populations is None:
        populations = list(rec.populations.keys())
    if len(populations) < 2:
        raise ValueError("need at least 2 populations for TE")

    bs = float(bin_size_ms) / 1000.0
    counts = bin_spikes(rec, populations, bs)  # (T, P)
    T, P = counts.shape
    mat = np.zeros((P, P), dtype=np.float64)
    for s in range(P):
        for t in range(P):
            if s == t:
                continue
            mat[s, t] = _binary_schreiber_te(counts[:, s], counts[:, t], delay=delay_bins)

    return TransferEntropyResult(
        matrix=mat,
        populations=tuple(populations),
        bin_size_seconds=bs,
        delay_bins=delay_bins,
        source=rec.source,
        params={"populations": list(populations),
                "bin_size_ms": float(bin_size_ms),
                "delay_bins": int(delay_bins),
                "estimator": estimator},
    )
