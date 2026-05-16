from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from neurocomplexity.core.exceptions import (
    PopulationError,
    RecordingValidationError,
)
from neurocomplexity.core.provenance import ProvenanceRecord


@dataclass(frozen=True)
class SpikeRecording:
    """Immutable spike-train recording.

    Invariants enforced at construction:
      * spike_times are float64 seconds, monotonically nondecreasing
      * spike_times and unit_ids have identical length
      * unit_ids in spike_times all appear in units['id']
      * populations[name] is a bool mask of length len(units)
      * duration > 0
    """

    spike_times: np.ndarray
    unit_ids: np.ndarray
    units: pd.DataFrame
    populations: Mapping[str, np.ndarray]
    duration: float
    sampling_rate: float | None
    source: ProvenanceRecord
    # Optional named interval tables carried alongside the recording (e.g.
    # stimulus presentation tables from an NWB file). Each value is a
    # DataFrame with at least ``start_time`` and ``stop_time`` columns.
    intervals: Mapping[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        st = np.asarray(self.spike_times, dtype=np.float64)
        uid = np.asarray(self.unit_ids, dtype=np.int64)
        if st.shape != uid.shape:
            raise RecordingValidationError(
                f"spike_times {st.shape} and unit_ids {uid.shape} must match"
            )
        if st.size and not np.all(np.diff(st) >= 0):
            order = np.argsort(st, kind="stable")
            st = st[order]
            uid = uid[order]
        if st.size and st[0] < 0:
            raise RecordingValidationError(f"negative spike time encountered: {st[0]}")
        if self.duration <= 0:
            raise RecordingValidationError(f"duration must be > 0, got {self.duration}")
        if not isinstance(self.units, pd.DataFrame):
            raise RecordingValidationError("units must be a pandas DataFrame")
        if "id" not in self.units.columns:
            raise RecordingValidationError("units DataFrame must have an 'id' column")

        n_units = len(self.units)
        clean_pops: dict[str, np.ndarray] = {}
        for name, mask in self.populations.items():
            m = np.asarray(mask, dtype=bool)
            if m.shape != (n_units,):
                raise PopulationError(
                    f"population {name!r} mask has length {m.shape}, expected ({n_units},)"
                )
            clean_pops[name] = m

        object.__setattr__(self, "spike_times", st)
        object.__setattr__(self, "unit_ids", uid)
        object.__setattr__(self, "populations", clean_pops)

    # ---- builder pattern (immutable) ----

    def filter_units(self, query: str | None = None,
                     **criteria) -> "SpikeRecording":
        """Keep only units whose metadata matches the given conditions.

        Two forms are supported (and may be combined):

        * a pandas-style ``query`` string evaluated against ``self.units``::

              rec.filter_units("quality == 'good' and isi_violations < 0.5 "
                                "and amplitude_cutoff < 0.1")

        * keyword arguments ``column=value`` (scalar, iterable, or callable)::

              rec.filter_units(quality=["good"], firing_rate=lambda x: x > 0.1)
        """
        keep = np.ones(len(self.units), dtype=bool)
        if query is not None:
            try:
                kept_idx = self.units.query(query).index
            except Exception as e:
                raise PopulationError(f"filter_units query failed: {e}") from e
            qmask = self.units.index.isin(kept_idx)
            keep &= qmask
        for col, val in criteria.items():
            if col not in self.units.columns:
                raise PopulationError(f"unit column {col!r} not found")
            series = self.units[col]
            if callable(val):
                mask = series.map(val).to_numpy(dtype=bool)
            elif isinstance(val, (list, tuple, set, np.ndarray, pd.Series)):
                mask = series.isin(list(val)).to_numpy()
            else:
                mask = (series == val).to_numpy()
            keep &= mask

        new_units = self.units.loc[keep].reset_index(drop=True)
        keep_ids = new_units["id"].to_numpy(dtype=np.int64)
        spike_mask = np.isin(self.unit_ids, keep_ids)
        new_st = self.spike_times[spike_mask]
        new_uid = self.unit_ids[spike_mask]

        # rebuild population masks against the new units order
        new_pops: dict[str, np.ndarray] = {}
        old_index = {uid: i for i, uid in enumerate(self.units["id"].to_numpy())}
        keep_pos = np.array([old_index[u] for u in new_units["id"].to_numpy()], dtype=int)
        for name, mask in self.populations.items():
            new_pops[name] = mask[keep_pos]

        return replace(self, spike_times=new_st, unit_ids=new_uid,
                       units=new_units, populations=new_pops)

    def with_populations(self, definition=None, *, by: str | None = None,
                         on_unassigned: str = "error") -> "SpikeRecording":
        """Define populations either by a units-metadata column (`by=`) or by an explicit dict."""
        if by is not None and definition is not None:
            raise PopulationError("pass either `definition` OR `by=`, not both")
        if by is not None:
            if by not in self.units.columns:
                raise PopulationError(f"units has no column {by!r}")
            values = self.units[by].fillna("__none__").to_numpy()
            pops: dict[str, np.ndarray] = {}
            for v in pd.unique(values):
                if v == "__none__":
                    continue
                pops[str(v)] = (values == v)
        elif isinstance(definition, Mapping):
            pops = {str(k): np.asarray(v, dtype=bool) for k, v in definition.items()}
        else:
            raise PopulationError("provide either `by=` or a mapping of name->mask")

        assigned = np.zeros(len(self.units), dtype=bool)
        for m in pops.values():
            assigned |= m
        unassigned = ~assigned
        if unassigned.any():
            if on_unassigned == "error":
                raise PopulationError(
                    f"{unassigned.sum()} units are unassigned. "
                    "Pass on_unassigned='drop' or 'other' to handle them."
                )
            elif on_unassigned == "other":
                pops["Other"] = unassigned
            elif on_unassigned == "drop":
                keep_ids = self.units.loc[~unassigned, "id"].to_numpy(dtype=np.int64)
                spike_mask = np.isin(self.unit_ids, keep_ids)
                new_units = self.units.loc[~unassigned].reset_index(drop=True)
                new_pops = {k: v[~unassigned] for k, v in pops.items()}
                return replace(self,
                               spike_times=self.spike_times[spike_mask],
                               unit_ids=self.unit_ids[spike_mask],
                               units=new_units,
                               populations=new_pops)
            else:
                raise PopulationError(
                    f"on_unassigned must be 'error'|'drop'|'other', got {on_unassigned!r}"
                )

        return replace(self, populations=pops)

    def crop(self, start: float, end: float) -> "SpikeRecording":
        if not (0 <= start < end):
            raise RecordingValidationError(f"invalid crop window [{start}, {end})")
        end = min(end, self.duration)
        mask = (self.spike_times >= start) & (self.spike_times < end)
        new_st = self.spike_times[mask] - start
        new_uid = self.unit_ids[mask]
        return replace(self, spike_times=new_st, unit_ids=new_uid,
                       duration=end - start)

    def crop_to_intervals(self, intervals) -> "SpikeRecording":
        """Concatenate only the time spans listed in ``intervals``.

        ``intervals`` is either the *name* of a table in ``self.intervals``
        (e.g. ``"spontaneous_presentations"``) or a DataFrame / array with
        ``start_time`` and ``stop_time`` columns (or two-column array of
        ``[start, stop]`` pairs).

        Returns a new recording whose timeline is the gap-free concatenation
        of the requested spans. Spike timestamps are remapped onto that
        contiguous timeline; the original absolute times are discarded.

        This is the standard way to restrict an Allen-style NWB recording to
        a single stimulus condition before running scale-free analyses.
        """
        if isinstance(intervals, str):
            if intervals not in self.intervals:
                raise PopulationError(
                    f"no interval table {intervals!r} on this recording; "
                    f"available: {list(self.intervals)}"
                )
            df = self.intervals[intervals]
        else:
            df = intervals
        if isinstance(df, pd.DataFrame):
            if not {"start_time", "stop_time"}.issubset(df.columns):
                raise RecordingValidationError(
                    "intervals DataFrame must have 'start_time' and 'stop_time' columns"
                )
            spans = df[["start_time", "stop_time"]].to_numpy(dtype=np.float64)
        else:
            spans = np.asarray(df, dtype=np.float64).reshape(-1, 2)
        if spans.size == 0:
            raise RecordingValidationError("no intervals to crop to")

        # Sort, clamp to [0, duration), and merge overlaps so the offset
        # arithmetic below stays simple.
        spans = spans[np.argsort(spans[:, 0])]
        spans[:, 0] = np.clip(spans[:, 0], 0.0, self.duration)
        spans[:, 1] = np.clip(spans[:, 1], 0.0, self.duration)
        merged: list[list[float]] = []
        for s, e in spans:
            if e <= s:
                continue
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([float(s), float(e)])
        if not merged:
            raise RecordingValidationError("all intervals collapsed to empty")

        new_st_chunks: list[np.ndarray] = []
        new_uid_chunks: list[np.ndarray] = []
        cumulative = 0.0
        for s, e in merged:
            mask = (self.spike_times >= s) & (self.spike_times < e)
            new_st_chunks.append(self.spike_times[mask] - s + cumulative)
            new_uid_chunks.append(self.unit_ids[mask])
            cumulative += e - s
        new_st = (np.concatenate(new_st_chunks) if new_st_chunks
                  else np.empty(0, dtype=np.float64))
        new_uid = (np.concatenate(new_uid_chunks) if new_uid_chunks
                   else np.empty(0, dtype=np.int64))
        order = np.argsort(new_st, kind="stable")
        new_st = new_st[order]; new_uid = new_uid[order]

        # The intervals collection is invalidated by the remap — drop it.
        return replace(self, spike_times=new_st, unit_ids=new_uid,
                       duration=cumulative, intervals={})

    def classify_cell_type(self, method: str = "waveform_duration",
                            *, threshold_ms: float = 0.4,
                            column: str | None = None) -> "SpikeRecording":
        """Add an ``ei_class`` column to ``self.units``.

        method="waveform_duration" (the only one implemented in v1.0):
            splits units into ``"narrow"`` (putative fast-spiking, inhibitory)
            and ``"broad"`` (putative regular-spiking, excitatory) based on
            the spike-width column ``column`` (Allen NWB exposes it as
            ``"duration"``, in ms). Units with no waveform data become
            ``"unknown"``.

        Once classified you can ``rec.filter_units("ei_class == 'narrow'")``
        or ``rec.with_populations(by="ei_class")``.
        """
        if method != "waveform_duration":
            raise ValueError(f"unknown classification method {method!r}")
        if column is None:
            # Auto-detect: NWB exposes 'waveform_duration', the Allen CSV
            # cache uses 'duration'. Prefer the NWB spelling.
            for candidate in ("waveform_duration", "duration"):
                if candidate in self.units.columns:
                    column = candidate
                    break
            else:
                raise PopulationError(
                    "no waveform-duration column found "
                    "(tried 'waveform_duration' and 'duration')"
                )
        if column not in self.units.columns:
            raise PopulationError(
                f"units DataFrame has no {column!r} column for waveform "
                f"classification"
            )
        new_units = self.units.copy()
        dur = pd.to_numeric(new_units[column], errors="coerce")
        labels = np.where(dur.isna(), "unknown",
                          np.where(dur < threshold_ms, "narrow", "broad"))
        new_units["ei_class"] = labels
        return replace(self, units=new_units)

    # ---- views ----

    def population_unit_ids(self, name: str) -> np.ndarray:
        if name not in self.populations:
            raise PopulationError(f"unknown population {name!r}")
        mask = self.populations[name]
        return self.units.loc[mask, "id"].to_numpy(dtype=np.int64)

    def spike_times_in(self, name: str) -> np.ndarray:
        ids = self.population_unit_ids(name)
        mask = np.isin(self.unit_ids, ids)
        return self.spike_times[mask]

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def n_spikes(self) -> int:
        return int(self.spike_times.size)

    def __repr__(self) -> str:
        pops = ", ".join(f"{k}({int(v.sum())})" for k, v in self.populations.items()) or "—"
        return (f"SpikeRecording(units={self.n_units}, spikes={self.n_spikes:,}, "
                f"duration={self.duration:.1f}s, populations=[{pops}])")
