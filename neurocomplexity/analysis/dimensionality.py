"""Participation ratio of pairwise-correlation eigenvalues.

PR = (sum lambda_i)^2 / sum(lambda_i^2)
Bounded in [1, N]. For an isotropic Gaussian PR -> N. For one dominant mode PR -> 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from neurocomplexity.analysis._binning import bin_spikes
from neurocomplexity.core.recording import SpikeRecording


@dataclass(frozen=True)
class DimensionalityResult:
    participation_ratio: float
    eigenvalues: np.ndarray   # descending
    n_units: int
    bin_size_seconds: float
    populations: tuple[str, ...]
    source: object
    params: dict = field(default_factory=dict)


def _participation_ratio(eig: np.ndarray) -> float:
    eig = np.asarray(eig, dtype=np.float64)
    eig = eig[eig > 0]
    if eig.size == 0:
        return float("nan")
    s = eig.sum()
    return float((s * s) / np.sum(eig * eig))


def dimensionality(rec: SpikeRecording,
                   populations: Sequence[str] | None = None,
                   bin_size_ms: float = 10.0,
                   ) -> DimensionalityResult:
    """Participation ratio computed on the per-UNIT correlation matrix.

    Units belonging to the requested populations are pooled. Bin counts are
    z-scored per unit; constant units are dropped (their variance is 0).
    """
    from neurocomplexity._warnings import _warn_if_uncurated
    _warn_if_uncurated(rec, "dimensionality")
    if populations is None:
        populations = list(rec.populations.keys())
    bs = float(bin_size_ms) / 1000.0

    # Need per-unit time-series, not per-population. Build it directly.
    keep_mask = np.zeros(len(rec.units), dtype=bool)
    for name in populations:
        keep_mask |= rec.populations[name]
    unit_ids = rec.units.loc[keep_mask, "id"].to_numpy(dtype=np.int64)
    if unit_ids.size < 2:
        raise ValueError("need at least 2 units for participation ratio")

    T = int(np.floor(rec.duration / bs))
    if T < 10:
        raise ValueError(f"duration {rec.duration}s too short for bin {bs}s")

    # Group spike_times by unit_id, bin each separately.
    # Use np.searchsorted: build (T, N) count matrix.
    N = unit_ids.size
    X = np.zeros((T, N), dtype=np.float32)
    sel = np.isin(rec.unit_ids, unit_ids)
    spike_times = rec.spike_times[sel]
    spike_owners = rec.unit_ids[sel]
    id_to_col = {int(u): i for i, u in enumerate(unit_ids)}
    # Vectorise the column index lookup
    cols = np.fromiter((id_to_col[int(u)] for u in spike_owners),
                      dtype=np.int64, count=spike_owners.size)
    rows = np.floor(spike_times / bs).astype(np.int64)
    valid = (rows >= 0) & (rows < T)
    rows = rows[valid]; cols = cols[valid]
    np.add.at(X, (rows, cols), 1.0)

    # z-score per unit, drop constant units
    sd = X.std(axis=0)
    keep = sd > 0
    if keep.sum() < 2:
        raise ValueError("fewer than 2 active units after binning")
    Z = (X[:, keep] - X[:, keep].mean(axis=0)) / sd[keep]
    C = np.cov(Z.T)
    eig = np.linalg.eigvalsh(C)[::-1]   # descending
    pr = _participation_ratio(eig)
    return DimensionalityResult(
        participation_ratio=pr,
        eigenvalues=eig,
        n_units=int(keep.sum()),
        bin_size_seconds=bs,
        populations=tuple(populations),
        source=rec.source,
        params={"populations": list(populations),
                "bin_size_ms": float(bin_size_ms)},
    )
