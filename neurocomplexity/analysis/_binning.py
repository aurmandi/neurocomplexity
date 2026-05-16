"""Shared binning. All inputs in seconds; the entire package uses seconds-only."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from neurocomplexity.core.exceptions import PopulationError
from neurocomplexity.core.recording import SpikeRecording


def bin_spikes(rec: SpikeRecording, populations: Sequence[str],
               bin_size_seconds: float) -> np.ndarray:
    """Return an (T, P) int32 array of spike counts per bin per population.

    T = floor(duration / bin_size_seconds).
    """
    if bin_size_seconds <= 0:
        raise ValueError("bin_size_seconds must be > 0")
    T = int(np.floor(rec.duration / bin_size_seconds))
    if T <= 0:
        raise ValueError(f"duration {rec.duration}s is shorter than bin {bin_size_seconds}s")
    P = len(populations)
    out = np.zeros((T, P), dtype=np.int32)

    for p_idx, name in enumerate(populations):
        if name not in rec.populations:
            raise PopulationError(f"unknown population {name!r}")
        mask = rec.populations[name]
        keep_ids = rec.units.loc[mask, "id"].to_numpy(dtype=np.int64)
        if keep_ids.size == 0:
            continue
        sel = np.isin(rec.unit_ids, keep_ids, assume_unique=False)
        times = rec.spike_times[sel]
        if times.size == 0:
            continue
        idx = np.floor(times / bin_size_seconds).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < T)]
        np.add.at(out[:, p_idx], idx, 1)
    return out


def bin_all_active(rec: SpikeRecording, populations: Sequence[str],
                   bin_size_seconds: float) -> np.ndarray:
    """Like bin_spikes, but summed across given populations → (T,) total counts."""
    return bin_spikes(rec, populations, bin_size_seconds).sum(axis=1)
