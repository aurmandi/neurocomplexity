"""Neural-data-specific surrogate generators.

References:
  * Louis S, Gerstein GL, Grun S (2010). Generation and selection of surrogate
    methods for correlation analysis. In: Grun S, Rotter S (eds), Analysis of
    Parallel Spike Trains, Springer, pp. 359-382.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from neurocomplexity.core.recording import SpikeRecording


def spike_dither(
    rec: SpikeRecording,
    *,
    delta_ms: float = 5.0,
    seed: int | None = None,
    repair_refractory_ms: float | None = 1.0,
    max_repair_iter: int = 10,
) -> SpikeRecording:
    """Louis-Gerstein-Grun (2010) uniform spike dithering.

    Each spike is displaced by an i.i.d. uniform jitter on [-delta, +delta].
    If `repair_refractory_ms` is set, spikes that violate the per-unit
    refractory bound are re-drawn (up to `max_repair_iter` rounds) and any
    remaining violators are dropped.
    """
    rng = np.random.default_rng(seed)
    delta = float(delta_ms) / 1000.0
    new_st = rec.spike_times + rng.uniform(-delta, delta, size=rec.spike_times.shape)
    new_st = np.clip(new_st, 0.0, rec.duration - 1e-9)

    if repair_refractory_ms is not None and repair_refractory_ms > 0:
        refr = float(repair_refractory_ms) / 1000.0
        owners = rec.unit_ids
        for _ in range(max_repair_iter):
            order = np.lexsort((new_st, owners))
            sorted_st = new_st[order]
            sorted_uid = owners[order]
            same_unit = np.diff(sorted_uid) == 0
            violates = np.zeros_like(sorted_st, dtype=bool)
            violates[1:] = same_unit & (np.diff(sorted_st) < refr)
            if not violates.any():
                break
            idx = order[violates]
            new_st[idx] = rec.spike_times[idx] + rng.uniform(-delta, delta, size=idx.size)
            new_st = np.clip(new_st, 0.0, rec.duration - 1e-9)
        order = np.lexsort((new_st, owners))
        sorted_st = new_st[order]
        sorted_uid = owners[order]
        same_unit = np.diff(sorted_uid) == 0
        keep = np.ones_like(sorted_st, dtype=bool)
        keep[1:] = ~(same_unit & (np.diff(sorted_st) < refr))
        new_st = sorted_st[keep]
        new_owners = sorted_uid[keep]
    else:
        order = np.argsort(new_st, kind="stable")
        new_st = new_st[order]
        new_owners = rec.unit_ids[order]

    order = np.argsort(new_st, kind="stable")
    return replace(rec, spike_times=new_st[order], unit_ids=new_owners[order])


def isi_shuffle(rec: SpikeRecording, *, seed: int | None = None) -> SpikeRecording:
    """Independently shuffle each unit's inter-spike intervals.

    Preserves per-unit ISI distribution exactly, destroys cross-unit timing.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(rec.unit_ids, kind="stable")
    sorted_uids = rec.unit_ids[order]
    sorted_times = rec.spike_times[order]
    if sorted_uids.size == 0:
        return replace(rec)

    splits = np.flatnonzero(np.diff(sorted_uids)) + 1
    time_groups = np.split(sorted_times, splits)
    uid_groups = np.split(sorted_uids, splits)

    new_times = np.empty_like(rec.spike_times)
    new_owners = np.empty_like(rec.unit_ids)
    cursor = 0
    for times, owners in zip(time_groups, uid_groups):
        n = times.size
        if n <= 1:
            new_times[cursor:cursor + n] = times
        else:
            t_sorted = np.sort(times)
            isis = np.diff(t_sorted)
            rng.shuffle(isis)
            shuffled = np.empty(n, dtype=np.float64)
            shuffled[0] = t_sorted[0]
            shuffled[1:] = t_sorted[0] + np.cumsum(isis)
            shuffled = np.clip(shuffled, 0.0, rec.duration - 1e-9)
            new_times[cursor:cursor + n] = shuffled
        new_owners[cursor:cursor + n] = owners
        cursor += n

    order2 = np.argsort(new_times, kind="stable")
    return replace(rec, spike_times=new_times[order2], unit_ids=new_owners[order2])


def interval_shuffle(
    rec: SpikeRecording,
    intervals_name: str,
    *,
    seed: int | None = None,
) -> SpikeRecording:
    """Shuffle which trial each (unit, trial) spike-block came from.

    For each unit independently and each equal-duration group of trials, the
    spike-time offsets inside each trial are re-assigned to a permuted ordering
    of the trials. Inter-trial spikes are kept in place.
    """
    if intervals_name not in rec.intervals:
        raise KeyError(f"intervals table {intervals_name!r} not present")
    rng = np.random.default_rng(seed)
    trials = rec.intervals[intervals_name].sort_values("start_time").reset_index(drop=True)
    starts = trials["start_time"].to_numpy()
    stops = trials["stop_time"].to_numpy()
    # Reject overlapping intervals: a spike inside two windows would otherwise
    # be reassigned twice, silently corrupting the surrogate. Tolerance of 1 us
    # to absorb float noise from NWB timestamps.
    if starts.size > 1:
        overlap = stops[:-1] - starts[1:]
        worst = float(overlap.max()) if overlap.size else 0.0
        if worst > 1e-6:
            bad = int(np.argmax(overlap))
            raise ValueError(
                f"interval_shuffle requires non-overlapping intervals; "
                f"interval {bad} stops at {stops[bad]:.6f}s but interval "
                f"{bad+1} starts at {starts[bad+1]:.6f}s "
                f"(overlap = {worst*1e3:.3f} ms)"
            )
    durs_round = np.round(stops - starts, 6)

    groups: dict[float, np.ndarray] = {
        float(d): np.flatnonzero(durs_round == d) for d in np.unique(durs_round)
    }

    new_times = rec.spike_times.copy()
    for uid in np.unique(rec.unit_ids):
        unit_mask = rec.unit_ids == uid
        for d, idx in groups.items():
            if idx.size < 2:
                continue
            perm = rng.permutation(idx)
            offsets: list[np.ndarray] = []
            masks: list[np.ndarray] = []
            for i in idx:
                m = unit_mask & (rec.spike_times >= starts[i]) & (rec.spike_times < stops[i])
                offsets.append(rec.spike_times[m] - starts[i])
                masks.append(m)
            pos_of = {int(i): p for p, i in enumerate(idx)}
            for src, dst in zip(idx, perm):
                p = pos_of[int(src)]
                new_times[masks[p]] = starts[dst] + offsets[p]

    order = np.argsort(new_times, kind="stable")
    return replace(rec, spike_times=new_times[order], unit_ids=rec.unit_ids[order])
