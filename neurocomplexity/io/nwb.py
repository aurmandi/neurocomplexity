"""NWB loader. Targets the Allen Visual Coding Neuropixels schema but tolerates
any standard NWB Units table.

Conventions:
  * Spike times are stored in NWB in seconds → no conversion needed.
  * Anatomical location is joined from electrodes[location] via
    units.peak_channel_id → electrodes.id when available; the resulting
    column is exposed as ``brain_area``.
  * Duration is the session length: max(spike_times) + 1.0s as a safe
    fallback (NWB ``session_start_time`` is absolute, not duration).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neurocomplexity.core.exceptions import NWBSchemaError
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording

log = logging.getLogger(__name__)


def from_nwb(path: str | Path) -> SpikeRecording:
    try:
        import pynwb  # noqa: F401  (heavy optional dep)
    except ImportError as exc:
        raise ImportError(
            "from_nwb requires pynwb. Install with: pip install 'neurocomplexity[nwb]'"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with pynwb.NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        # Fast path: neurocomplexity-authored scratch payload guarantees a
        # bitwise round-trip. Fall back to the best-effort schema reader if
        # the payload is absent.
        from neurocomplexity.io._ndx import read_nc_payload
        nc_payload = read_nc_payload(nwb)
        if nc_payload is not None:
            return nc_payload
        if nwb.units is None or len(nwb.units.id[:]) == 0:
            raise NWBSchemaError(
                f"{path.name} has no Units table. "
                "If this is ophys/calcium data, neurocomplexity v0.1 does not support it."
            )
        units_ddt = nwb.units
        unit_ids = np.asarray(units_ddt.id[:], dtype=np.int64)
        cols = list(units_ddt.colnames)

        # Build per-unit metadata DataFrame from all scalar columns.
        meta = {"id": unit_ids}
        for c in cols:
            if c in ("spike_times", "spike_amplitudes", "waveform_mean"):
                continue
            try:
                vals = units_ddt[c][:]
            except Exception:
                continue
            if isinstance(vals, np.ndarray) and vals.ndim == 1 and len(vals) == len(unit_ids):
                meta[c] = vals
        units_df = pd.DataFrame(meta)

        # Anatomical location from electrodes table.
        if "peak_channel_id" in units_df.columns and nwb.electrodes is not None:
            elec = nwb.electrodes.to_dataframe()
            if "location" in elec.columns:
                # electrodes table index is the electrode id
                loc_map = elec["location"].to_dict()
                units_df["brain_area"] = units_df["peak_channel_id"].map(loc_map)
            else:
                log.warning("electrodes table has no 'location' column")

        # Spike times: ragged via VectorIndex.
        n_units = len(unit_ids)
        all_times: list[np.ndarray] = []
        all_owners: list[np.ndarray] = []
        for i in range(n_units):
            st = np.asarray(units_ddt["spike_times"][i], dtype=np.float64)
            if st.size == 0:
                continue
            if (st < 0).any():
                raise NWBSchemaError(
                    f"unit index {i} (id={unit_ids[i]}) has negative spike times"
                )
            if not np.all(np.diff(st) >= 0):
                st = np.sort(st)
            all_times.append(st)
            all_owners.append(np.full(st.shape, unit_ids[i], dtype=np.int64))

        if not all_times:
            raise NWBSchemaError(f"{path.name} has units but no spikes")

        spike_times = np.concatenate(all_times)
        owners = np.concatenate(all_owners)
        order = np.argsort(spike_times, kind="stable")
        spike_times = spike_times[order]
        owners = owners[order]

        duration = float(spike_times.max()) + 1.0
        sampling_rate: float | None = None

        populations = {"all": np.ones(len(units_df), dtype=bool)}

        # Pull any interval / epoch tables (stimulus presentations, epochs,
        # invalid_times, ...). NWB stores them in ``nwb.intervals``; ``epochs``
        # also lives at the top level.
        interval_tables: dict[str, pd.DataFrame] = {}
        try:
            for name, tbl in (nwb.intervals or {}).items():
                try:
                    df = tbl.to_dataframe()
                except Exception:
                    continue
                if {"start_time", "stop_time"}.issubset(df.columns):
                    interval_tables[str(name)] = df.reset_index(drop=True)
        except Exception:
            pass
        if getattr(nwb, "epochs", None) is not None:
            try:
                df = nwb.epochs.to_dataframe()
                if {"start_time", "stop_time"}.issubset(df.columns):
                    interval_tables.setdefault("epochs", df.reset_index(drop=True))
            except Exception:
                pass

        provenance = ProvenanceRecord.for_file(path, source_format="nwb")

        return SpikeRecording(
            spike_times=spike_times,
            unit_ids=owners,
            units=units_df,
            populations=populations,
            duration=duration,
            sampling_rate=sampling_rate,
            source=provenance,
            intervals=interval_tables,
        )
