"""Bridge to the SpikeInterface ecosystem.

Accepts any ``spikeinterface.BaseSorting`` and, optionally, a paired
``BaseRecording`` for duration and channel metadata. SpikeInterface is
a soft dependency — install with
``pip install 'neurocomplexity[spikeinterface]'``.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording


def from_spikeinterface(
    sorting,
    *,
    recording=None,
    duration: float | None = None,
    populations: Mapping[str, np.ndarray] | None = None,
) -> SpikeRecording:
    """Build a SpikeRecording from a spikeinterface BaseSorting.

    Parameters
    ----------
    sorting
        A ``spikeinterface.BaseSorting`` instance.
    recording
        Optional ``spikeinterface.BaseRecording``; used for ``duration``
        and any per-channel properties.
    duration
        Override the recording duration in seconds.
    populations
        Override the default ``{"all": ones}`` population mask.
    """
    try:
        import spikeinterface  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "from_spikeinterface requires spikeinterface. "
            "Install with: pip install 'neurocomplexity[spikeinterface]'"
        ) from exc

    sample_rate = float(sorting.sampling_frequency)
    unit_ids = list(sorting.unit_ids)

    spike_chunks: list[np.ndarray] = []
    owner_chunks: list[np.ndarray] = []
    for uid in unit_ids:
        st = sorting.get_unit_spike_train(uid, return_times=True)
        st = np.asarray(st, dtype=np.float64)
        spike_chunks.append(st)
        owner_chunks.append(np.full(st.shape, int(uid), dtype=np.int64))

    if spike_chunks:
        spike_times = np.concatenate(spike_chunks)
        owners = np.concatenate(owner_chunks)
        order = np.argsort(spike_times, kind="stable")
        spike_times = spike_times[order]
        owners = owners[order]
    else:
        spike_times = np.empty(0, dtype=np.float64)
        owners = np.empty(0, dtype=np.int64)

    units_dict: dict = {"id": [int(u) for u in unit_ids]}
    for key in sorting.get_property_keys():
        try:
            vals = sorting.get_property(key)
        except Exception:
            continue
        if vals is None or len(vals) != len(unit_ids):
            continue
        units_dict[str(key)] = list(vals)
    units_df = pd.DataFrame(units_dict)
    if "quality" not in units_df.columns:
        units_df["quality"] = "unsorted"

    if duration is None and recording is not None:
        try:
            duration = float(recording.get_duration())
        except Exception:
            duration = None
    if duration is None:
        duration = float(spike_times.max()) + 1.0 if spike_times.size else 1.0
    duration = float(duration)

    if populations is None:
        populations = {"all": np.ones(len(units_df), dtype=bool)}

    provenance = ProvenanceRecord.for_memory(
        "spikeinterface", hint=type(sorting).__name__,
    )

    return SpikeRecording(
        spike_times=spike_times,
        unit_ids=owners,
        units=units_df,
        populations=populations,
        duration=duration,
        sampling_rate=sample_rate,
        source=provenance,
    )
