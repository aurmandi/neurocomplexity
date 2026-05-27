"""SpikeRecording.merge_probes implementation."""
from __future__ import annotations

import warnings as _warnings
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording


AlignLiteral = Literal["max", "min", "strict"]


def merge_probes_impl(
    recordings: Mapping[str, SpikeRecording],
    *,
    align_durations: AlignLiteral = "max",
) -> SpikeRecording:
    """Combine multiple single-probe recordings into one multi-probe recording.

    Used internally by
    :meth:`SpikeRecording.merge_probes <neurocomplexity.core.recording.SpikeRecording.merge_probes>`.
    Prefer the method form on the recording object.

    Parameters
    ----------
    recordings
        Mapping ``{probe_label: SpikeRecording}``. The label is prepended
        to populations and used to disambiguate colliding unit ids.
    align_durations
        How to reconcile different per-probe ``duration`` values:

        * ``"max"`` (default) — pad shorter probes; merged duration is the
          longest.
        * ``"min"`` — trim longer probes; merged duration is the shortest.
        * ``"strict"`` — raise if any probe disagrees by more than 1 ms.

    Returns
    -------
    :class:`~neurocomplexity.core.recording.SpikeRecording`
        Merged recording. Populations from each probe are prefixed with
        ``"{probe}::"``. Colliding unit ids are replaced with tuple-strings
        ``"('probe_label', orig_id)"`` (a :class:`UserWarning` is emitted).

    Raises
    ------
    ValueError
        If ``recordings`` is empty, or ``align_durations="strict"`` and the
        durations disagree.
    """
    if not recordings:
        raise ValueError("recordings dict is empty")
    labels = list(recordings.keys())

    # Duration handling
    durs = {lbl: r.duration for lbl, r in recordings.items()}
    if align_durations == "strict":
        d0 = next(iter(durs.values()))
        for lbl, d in durs.items():
            if abs(d - d0) > 1e-3:
                raise ValueError(
                    f"durations differ across probes (strict mode): {durs}"
                )
        merged_duration = d0
    elif align_durations == "min":
        merged_duration = min(durs.values())
    else:
        merged_duration = max(durs.values())

    # Detect ID collisions
    id_seen: dict[int, list[str]] = {}
    for lbl, r in recordings.items():
        for uid in r.units["id"].tolist():
            id_seen.setdefault(int(uid), []).append(lbl)
    colliding = {uid for uid, owners in id_seen.items() if len(owners) > 1}
    if colliding:
        _warnings.warn(
            f"{len(colliding)} unit-id collision(s) across probes; "
            f"colliding ids replaced with tuple-strings like \"('probe', id)\"",
            UserWarning, stacklevel=3,
        )

    # Build merged units, spikes, populations
    units_frames = []
    all_spike_times = []
    all_unit_ids_obj = []  # object dtype temporarily; remapped to int codes later
    merged_pops: dict[str, np.ndarray] = {}
    n_total = sum(len(r.units) for r in recordings.values())
    offset = 0

    def _new_id(label: str, uid: int):
        return f"('{label}', {uid})" if uid in colliding else uid

    for lbl, r in recordings.items():
        st = r.spike_times
        uid_arr = r.unit_ids
        if align_durations == "min" and r.duration > merged_duration:
            mask = st < merged_duration
            st = st[mask]
            uid_arr = uid_arr[mask]

        u = r.units.copy()
        u["probe"] = lbl
        new_ids_for_units = [_new_id(lbl, int(x)) for x in u["id"].tolist()]
        id_map = dict(zip(u["id"].tolist(), new_ids_for_units))
        u["id"] = new_ids_for_units
        units_frames.append(u)

        remapped_uid = np.array([id_map[int(x)] for x in uid_arr.tolist()], dtype=object)
        all_spike_times.append(st)
        all_unit_ids_obj.append(remapped_uid)

        probe_mask = np.zeros(n_total, dtype=bool)
        probe_mask[offset:offset + len(u)] = True
        merged_pops[f"probe_{lbl}"] = probe_mask

        for pname, pmask in r.populations.items():
            if pname == "all":
                continue
            full = np.zeros(n_total, dtype=bool)
            full[offset:offset + len(u)] = pmask
            merged_pops[f"probe_{lbl}_{pname}"] = full

        offset += len(u)

    merged_units = pd.concat(units_frames, ignore_index=True)

    # Recode id to int64; preserve human-readable form in original_id.
    final_ids = merged_units["id"].tolist()
    code_map = {uid: i for i, uid in enumerate(final_ids)}
    merged_units["original_id"] = merged_units["id"]
    merged_units["id"] = np.arange(len(merged_units), dtype=np.int64)

    spike_times_arr = np.concatenate(all_spike_times) if all_spike_times else np.zeros(0, dtype=np.float64)
    spike_uids_obj = np.concatenate(all_unit_ids_obj) if all_unit_ids_obj else np.zeros(0, dtype=object)
    spike_uids_int = np.array([code_map[x] for x in spike_uids_obj.tolist()], dtype=np.int64)

    # Merge intervals (collision raises)
    merged_intervals: dict[str, pd.DataFrame] = {}
    for lbl, r in recordings.items():
        for name, df in r.intervals.items():
            if name in merged_intervals:
                raise KeyError(
                    f"interval table {name!r} present on multiple probes; "
                    f"rename before merging"
                )
            merged_intervals[name] = df

    merge_record = ProvenanceRecord.for_memory(
        "merge_probes", hint=",".join(labels)
    )
    merged_attachments = tuple(r.source for r in recordings.values())
    for r in recordings.values():
        merged_attachments = merged_attachments + tuple(r.attachments)

    rates = {r.sampling_rate for r in recordings.values()}
    sr = rates.pop() if len(rates) == 1 else None

    return SpikeRecording(
        spike_times=spike_times_arr,
        unit_ids=spike_uids_int,
        units=merged_units,
        populations=merged_pops,
        duration=merged_duration,
        sampling_rate=sr,
        source=merge_record,
        intervals=merged_intervals,
        attachments=merged_attachments,
        _filtered=all(r._filtered for r in recordings.values()),
    )
