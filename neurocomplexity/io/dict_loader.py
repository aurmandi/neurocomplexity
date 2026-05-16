"""In-memory loader, primarily for tests and quick exploration."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording


def from_dict(spike_times_by_unit: Mapping[int, np.ndarray],
              duration: float,
              unit_metadata: pd.DataFrame | None = None,
              sampling_rate: float | None = None,
              hint: str = "dict") -> SpikeRecording:
    """Build a SpikeRecording from a {unit_id: spike_times_seconds} mapping."""
    ids: list[int] = []
    times_chunks: list[np.ndarray] = []
    for uid, st in spike_times_by_unit.items():
        st = np.asarray(st, dtype=np.float64)
        ids.append(int(uid))
        times_chunks.append(st)

    if unit_metadata is None:
        unit_metadata = pd.DataFrame({"id": ids, "quality": ["good"] * len(ids),
                                       "firing_rate": [len(t) / max(duration, 1e-9)
                                                       for t in times_chunks]})
    else:
        if "id" not in unit_metadata.columns:
            raise ValueError("unit_metadata must have an 'id' column")
        unit_metadata = unit_metadata.reset_index(drop=True)

    flat_times: list[float] = []
    flat_uids: list[int] = []
    for uid, st in zip(ids, times_chunks):
        flat_times.extend(st.tolist())
        flat_uids.extend([uid] * len(st))
    spike_times = np.asarray(flat_times, dtype=np.float64)
    unit_ids_arr = np.asarray(flat_uids, dtype=np.int64)
    order = np.argsort(spike_times, kind="stable")

    return SpikeRecording(
        spike_times=spike_times[order],
        unit_ids=unit_ids_arr[order],
        units=unit_metadata,
        populations={"all": np.ones(len(unit_metadata), dtype=bool)},
        duration=float(duration),
        sampling_rate=sampling_rate,
        source=ProvenanceRecord.for_memory("dict", hint=hint),
    )
