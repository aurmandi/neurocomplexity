"""Shared binning. All inputs in seconds; the entire package uses seconds-only."""
from __future__ import annotations

import warnings as _pywarnings
from collections.abc import Mapping, Sequence

import numpy as np

from neurocomplexity._warnings import MemoryAllocationWarning
from neurocomplexity.core.exceptions import PopulationError
from neurocomplexity.core.recording import SpikeRecording

_COUNTS_DTYPE = np.int32
_COUNTS_BYTES = 4
# Promote to int64 when the (T, P) count matrix is large enough that a
# single column's cumulative count could overflow int32 (max ~2.1e9). The
# heuristic threshold T * P > 2**30 is conservative: in practice a single
# bin rarely accumulates anywhere near 2^31 spikes, but at very long
# recordings (>~10 h Neuropixels @ 1 ms with thousands of units) the cost
# of doubling memory is preferable to a silent overflow.
_PROMOTE_THRESHOLD = 1 << 30


def _counts_dtype(T: int, P: int) -> np.dtype:
    """Return ``np.int64`` when ``T * P`` exceeds the promotion threshold."""
    return np.int64 if (int(T) * int(P)) > _PROMOTE_THRESHOLD else _COUNTS_DTYPE


def _counts_bytes(T: int, P: int) -> int:
    """Per-element size of the dtype ``bin_spikes`` would allocate at (T, P)."""
    return 8 if (int(T) * int(P)) > _PROMOTE_THRESHOLD else _COUNTS_BYTES


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
    return int(T * P * _counts_bytes(T, P))


def _maybe_warn_large_allocation(T: int, P: int) -> None:
    """Warn if (T*P*4) exceeds 25% of available RAM.

    Silent if psutil is not installed.
    """
    need = T * P * _counts_bytes(T, P)
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

    out = np.zeros((T, P), dtype=_counts_dtype(T, P))

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
    """Like bin_spikes, but summed across given populations → (T,) total counts.

    .. warning::
        This sums all spike counts regardless of cell type. For
        E/I-balanced dynamics where excitatory and inhibitory cascades
        carry different dynamical signatures, use
        :func:`bin_active_by_type` and analyse each stream separately.
        Avalanches extracted from a pooled E + I count series can confuse
        a sub-critical balanced state with a critical excitatory cascade
        (Beggs & Plenz 2003; Plenz 2014).
    """
    return bin_spikes(rec, populations, bin_size_seconds).sum(axis=1)


def bin_active_by_type(
    rec: SpikeRecording,
    populations: Sequence[str],
    bin_size_seconds: float,
    cell_types: Mapping[str, str],
    *,
    missing_label: str = "unlabelled",
) -> dict[str, np.ndarray]:
    """Bin spikes once and return per-cell-type total-count streams.

    Parameters
    ----------
    rec
        Spike recording.
    populations
        Population keys to bin (each becomes one column of the underlying
        (T, P) count matrix).
    bin_size_seconds
        Bin width in seconds.
    cell_types
        Mapping ``population -> label`` (e.g. ``"E"`` / ``"I"`` /
        ``"PV"`` / ``"SST"``). Populations not present in this mapping
        are placed under ``missing_label`` so the caller cannot silently
        drop streams.
    missing_label
        Bucket for populations absent from ``cell_types``. Default
        ``"unlabelled"``.

    Returns
    -------
    streams : dict[str, np.ndarray]
        One ``(T,)`` count vector per distinct label that appears among
        the requested populations. Iteration order is insertion order of
        the first population assigned to each label.

    Notes
    -----
    Use this when the downstream analysis depends on cell-type identity —
    most criticality / avalanche studies, balanced-network diagnostics,
    and E/I-aware Transfer Entropy. For the legacy pooled "total
    activity" series use :func:`bin_all_active`.
    """
    counts = bin_spikes(rec, populations, bin_size_seconds)  # (T, P)
    streams: dict[str, np.ndarray] = {}
    pop_list = list(populations)
    for p_idx, name in enumerate(pop_list):
        label = str(cell_types.get(name, missing_label))
        col = counts[:, p_idx]
        if label in streams:
            streams[label] = streams[label] + col
        else:
            streams[label] = col.copy()
    return streams
