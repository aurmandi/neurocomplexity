"""Shared binning. All inputs in seconds; the entire package uses seconds-only."""
from __future__ import annotations

import warnings as _pywarnings
from collections.abc import Sequence

import numpy as np

from neurocomplexity._warnings import MemoryAllocationWarning
from neurocomplexity.core.exceptions import PopulationError
from neurocomplexity.core.recording import SpikeRecording

_COUNTS_DTYPE = np.int32
_COUNTS_BYTES = 4


def estimate_bin_spikes_bytes(rec: SpikeRecording,
                              populations,
                              bin_size_ms: float) -> int:
    """Bytes that ``bin_spikes`` would allocate for the (T, P) counts matrix.

    ``populations`` may be a sequence of names or an integer count.
    Pure function; no side effects.
    """
    if bin_size_ms <= 0:
        raise ValueError("bin_size_ms must be > 0")
    T = int(np.floor(rec.duration / (bin_size_ms / 1000.0)))
    if isinstance(populations, (int, np.integer)):
        P = int(populations)
    else:
        P = len(populations)
    return int(T * P * _COUNTS_BYTES)


def _maybe_warn_large_allocation(T: int, P: int) -> None:
    """Warn if (T*P*4) exceeds 25% of available RAM.

    Silent if psutil is not installed.
    """
    need = T * P * _COUNTS_BYTES
    try:
        import psutil
        avail = int(psutil.virtual_memory().available)
    except ImportError:
        return
    if need > 0.25 * avail:
        _pywarnings.warn(
            f"bin_spikes will allocate ~{need/1e6:.0f} MB for the "
            f"({T} bins x {P} populations) counts matrix; "
            f"{avail/1e6:.0f} MB available. "
            f"Consider chunk_seconds=10.0 or rec.crop(...) on small-RAM systems.",
            category=MemoryAllocationWarning,
            stacklevel=3,
        )


def bin_spikes(rec: SpikeRecording, populations: Sequence[str],
               bin_size_seconds: float, *,
               chunk_seconds: float | None = None) -> np.ndarray:
    """Return an (T, P) int32 array of spike counts per bin per population.

    T = floor(duration / bin_size_seconds).

    Parameters
    ----------
    chunk_seconds
        If given, process the recording in chunks of this many seconds. The
        output array is identical to the unchunked path; only the transient
        per-population mask + indexed-spike-times buffers shrink. Useful when
        the recording has many spikes and ``P`` is large.
    """
    if bin_size_seconds <= 0:
        raise ValueError("bin_size_seconds must be > 0")
    T = int(np.floor(rec.duration / bin_size_seconds))
    if T <= 0:
        raise ValueError(f"duration {rec.duration}s is shorter than bin {bin_size_seconds}s")
    P = len(populations)

    if chunk_seconds is not None:
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be > 0")
        if chunk_seconds > rec.duration:
            raise ValueError(
                f"chunk_seconds ({chunk_seconds}) must be <= rec.duration ({rec.duration})"
            )

    if chunk_seconds is None:
        _maybe_warn_large_allocation(T, P)

    out = np.zeros((T, P), dtype=_COUNTS_DTYPE)

    pop_ids: list[np.ndarray] = []
    for name in populations:
        if name not in rec.populations:
            raise PopulationError(f"unknown population {name!r}")
        mask = rec.populations[name]
        pop_ids.append(rec.units.loc[mask, "id"].to_numpy(dtype=np.int64))

    if chunk_seconds is None:
        for p_idx, keep_ids in enumerate(pop_ids):
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

    # Chunked path: iterate time windows, filter once per chunk.
    n_chunks = int(np.ceil(rec.duration / chunk_seconds))
    for c in range(n_chunks):
        t_lo = c * chunk_seconds
        t_hi = min((c + 1) * chunk_seconds, rec.duration)
        time_mask = (rec.spike_times >= t_lo) & (rec.spike_times < t_hi)
        if not np.any(time_mask):
            continue
        chunk_times = rec.spike_times[time_mask]
        chunk_uids = rec.unit_ids[time_mask]
        for p_idx, keep_ids in enumerate(pop_ids):
            if keep_ids.size == 0:
                continue
            sel = np.isin(chunk_uids, keep_ids, assume_unique=False)
            if not np.any(sel):
                continue
            tt = chunk_times[sel]
            idx = np.floor(tt / bin_size_seconds).astype(np.int64)
            idx = idx[(idx >= 0) & (idx < T)]
            np.add.at(out[:, p_idx], idx, 1)
    return out


def bin_all_active(rec: SpikeRecording, populations: Sequence[str],
                   bin_size_seconds: float) -> np.ndarray:
    """Like bin_spikes, but summed across given populations → (T,) total counts."""
    return bin_spikes(rec, populations, bin_size_seconds).sum(axis=1)
