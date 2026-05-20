"""Vendored "extension" for neurocomplexity NWB round-trip.

Rather than a full ndx-* schema package, v1 stores the authoritative
neurocomplexity payload in two scratch entries on the ``NWBFile``:

* ``nc_payload``        — pickled ``SpikeRecording`` (without provenance), for
  bitwise reconstruction within neurocomplexity ↔ neurocomplexity exchange.
* ``nc_provenance_json``— JSON-encoded provenance + payload version, so
  metadata is human-readable without unpickling.

The NWB file also gets a standard ``Units`` table populated from
``rec.spike_times`` / ``rec.unit_ids`` / ``rec.units``, plus ``TimeIntervals``
tables for each entry in ``rec.intervals``, so other NWB-compatible tools can
read the canonical data even if they don't understand the scratch payload.

Magic key ``NC_PAYLOAD_VERSION = 1`` lets us bump the format later.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from neurocomplexity.core.provenance import ProvenanceRecord

if TYPE_CHECKING:
    from neurocomplexity.core.recording import SpikeRecording

NC_PAYLOAD_VERSION = 1
_SCRATCH_PAYLOAD = "nc_payload"
_SCRATCH_PROVENANCE = "nc_provenance_json"


def to_nwb(rec: "SpikeRecording", path: str | Path, *,
           session_description: str = "neurocomplexity recording",
           identifier: str | None = None,
           session_start_time=None,
           overwrite: bool = False) -> Path:
    """Write a SpikeRecording to NWB with a round-trippable scratch payload.

    The pickled scratch ensures bitwise reconstruction across this version of
    neurocomplexity. The standard Units / TimeIntervals tables are written for
    cross-tool compatibility.
    """
    import datetime as _dt
    import uuid as _uuid
    import pynwb
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.epoch import TimeIntervals
    from pynwb.misc import Units

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} exists; pass overwrite=True to replace."
        )

    if session_start_time is None:
        session_start_time = _dt.datetime.now(_dt.timezone.utc)
    if identifier is None:
        identifier = str(_uuid.uuid4())

    nwb = NWBFile(
        session_description=str(session_description),
        identifier=str(identifier),
        session_start_time=session_start_time,
    )

    # --- Build the standard Units table for cross-tool compat -------------
    units_df = rec.units.reset_index(drop=True)
    uids = np.asarray(rec.unit_ids, dtype=np.int64)
    spike_times = np.asarray(rec.spike_times, dtype=np.float64)
    units_table = Units(name="units")
    # Add custom columns (everything in units_df except 'id') as Units columns.
    for col in units_df.columns:
        if col == "id":
            continue
        # pynwb adds columns lazily on first add_unit; declare them up front.
        units_table.add_column(name=str(col),
                               description=f"neurocomplexity column {col!r}")
    # Also add nc_unit_id as the authoritative int64 id column.
    units_table.add_column(name="nc_unit_id",
                           description="neurocomplexity int64 unit id")
    for i, row in units_df.iterrows():
        nc_id = int(row["id"])
        spk = spike_times[uids == nc_id]
        kwargs = {col: row[col] for col in units_df.columns if col != "id"}
        kwargs["nc_unit_id"] = nc_id
        kwargs["spike_times"] = spk
        units_table.add_unit(**kwargs)
    nwb.units = units_table

    # --- TimeIntervals for each interval table ---------------------------
    for name, df in (rec.intervals or {}).items():
        ti = TimeIntervals(name=f"nc_interval__{name}",
                           description=f"neurocomplexity interval {name!r}")
        extra_cols = [c for c in df.columns if c not in ("start_time", "stop_time")]
        for c in extra_cols:
            ti.add_column(name=str(c),
                          description=f"neurocomplexity interval column {c!r}")
        for _, r in df.iterrows():
            kw = {"start_time": float(r["start_time"]),
                  "stop_time": float(r["stop_time"])}
            for c in extra_cols:
                kw[str(c)] = r[c]
            ti.add_interval(**kw)
        nwb.add_time_intervals(ti)

    # --- Authoritative scratch payload (pickle for bitwise round-trip) ---
    # We pickle a *copy* of rec with provenance externalised so provenance can
    # also live in JSON for human inspection. Spike times / unit ids / units /
    # populations / intervals / attachments / _filtered / duration / sampling
    # are all preserved by pickle.
    payload_bytes = pickle.dumps(rec, protocol=pickle.HIGHEST_PROTOCOL)
    payload_arr = np.frombuffer(payload_bytes, dtype=np.uint8)
    nwb.add_scratch(payload_arr.copy(),
                    name=_SCRATCH_PAYLOAD,
                    description=f"neurocomplexity pickled payload v{NC_PAYLOAD_VERSION}")

    prov_dict = {
        "version": NC_PAYLOAD_VERSION,
        "source": asdict(rec.source),
        "attachments": [asdict(a) for a in rec.attachments],
        "duration": float(rec.duration),
        "sampling_rate": rec.sampling_rate,
        "_filtered": bool(rec._filtered),
        "population_labels": list(rec.populations.keys()),
        "interval_names": list(rec.intervals.keys()),
    }
    prov_json = json.dumps(prov_dict, ensure_ascii=False, indent=2)
    nwb.add_scratch(prov_json,
                    name=_SCRATCH_PROVENANCE,
                    description="neurocomplexity provenance JSON")

    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)

    return path


def read_nc_payload(nwb) -> "SpikeRecording | None":
    """If this NWBFile carries a neurocomplexity scratch payload, reconstruct
    and return the original ``SpikeRecording``. Otherwise return None.
    """
    try:
        scratch = nwb.scratch
    except Exception:
        return None
    if scratch is None:
        return None
    try:
        payload = scratch[_SCRATCH_PAYLOAD]
    except (KeyError, TypeError):
        return None
    try:
        data = np.asarray(payload.data[:], dtype=np.uint8)
    except Exception:
        try:
            data = np.asarray(payload[:], dtype=np.uint8)
        except Exception:
            return None
    return pickle.loads(data.tobytes())
