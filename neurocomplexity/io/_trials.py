"""add_trials: attach behavioural trial tables to a SpikeRecording."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording


FormatLiteral = Literal["auto", "csv", "tsv", "nwb"]


def _load_trials_table(path: Path, format: FormatLiteral, name: str) -> tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()
    if format == "auto":
        if suffix == ".csv":
            format = "csv"
        elif suffix == ".tsv":
            format = "tsv"
        elif suffix in {".nwb", ".h5"}:
            format = "nwb"
        else:
            raise ValueError(f"could not auto-detect trials format from suffix {suffix!r}; pass format=")

    if format == "csv":
        return pd.read_csv(path), "csv"
    if format == "tsv":
        return pd.read_csv(path, sep="\t"), "tsv"
    if format == "nwb":
        from pynwb import NWBHDF5IO
        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            nwb = io.read()
            if name in nwb.intervals:
                table = nwb.intervals[name].to_dataframe()
            elif name == "trials" and nwb.trials is not None:
                table = nwb.trials.to_dataframe()
            else:
                avail = list(nwb.intervals)
                raise KeyError(f"no interval table {name!r} in NWB; available: {avail}")
        return table, "nwb"
    raise ValueError(f"unknown trials format: {format!r}")


def add_trials(
    rec: SpikeRecording,
    path: str | os.PathLike,
    *,
    name: str,
    format: FormatLiteral = "auto",
    start_column: str = "start_time",
    stop_column: str = "stop_time",
) -> SpikeRecording:
    """Attach a behavioural / experimental interval table to a recording.

    Reads a CSV / TSV / NWB file with one row per interval and joins it
    onto ``rec.intervals`` under ``name``. Returns a new immutable
    recording.

    Parameters
    ----------
    rec
        Recording to enrich.
    path
        Path to ``.csv``, ``.tsv`` or ``.nwb`` source file.
    name
        Key under which to store the interval table on ``rec.intervals``.
        Must not already exist.
    format
        ``"auto"`` (default; uses the file extension), or one of
        ``"csv"`` / ``"tsv"`` / ``"nwb"``. For NWB, the function reads
        ``nwbfile.intervals[name]`` if present, else
        ``nwbfile.trials`` if ``name == "trials"``.
    start_column, stop_column
        Names of the start / stop time columns in the source file. Both
        get renamed to ``start_time`` / ``stop_time`` internally so that
        :func:`~neurocomplexity.core.recording.SpikeRecording.crop_to` and
        :func:`~neurocomplexity.inference.surrogates.interval_shuffle` can
        find them.

    Returns
    -------
    :class:`~neurocomplexity.core.recording.SpikeRecording`

    Raises
    ------
    KeyError
        If ``name`` already exists on ``rec.intervals``.
    ValueError
        If a row has ``start_time >= stop_time``, or any interval falls
        outside ``[0, rec.duration]``, or required columns are missing.

    Notes
    -----
    Intervals MAY touch (``stop[i] == start[i+1]``) but must not overlap.
    :func:`~neurocomplexity.inference.surrogates.interval_shuffle`
    enforces this with a hard error.
    """
    if name in rec.intervals:
        raise KeyError(
            f"interval table {name!r} already exists on this recording; "
            f"existing names: {list(rec.intervals)}"
        )

    path = Path(path)
    raw, detected = _load_trials_table(path, format, name)

    if start_column not in raw.columns:
        raise ValueError(f"start column {start_column!r} not found in {path.name}")
    if stop_column not in raw.columns:
        raise ValueError(f"stop column {stop_column!r} not found in {path.name}")

    table = raw.rename(columns={start_column: "start_time", stop_column: "stop_time"}).copy()
    other_cols = [c for c in table.columns if c not in ("start_time", "stop_time")]
    table = table[["start_time", "stop_time"] + other_cols]

    starts = table["start_time"].to_numpy(dtype=float)
    stops = table["stop_time"].to_numpy(dtype=float)
    for i, (s, e) in enumerate(zip(starts, stops)):
        if not (s < e):
            raise ValueError(f"trials row {i}: start_time={s} must be < stop_time={e}")
        if s < 0 or e > rec.duration:
            raise ValueError(
                f"trials row {i}: window [{s}, {e}] outside recording duration [0, {rec.duration}]"
            )

    new_intervals = dict(rec.intervals)
    new_intervals[name] = table.reset_index(drop=True)

    attachment = ProvenanceRecord.for_file(path, source_format=f"trials:{detected}")
    return replace(rec, intervals=new_intervals,
                   attachments=rec.attachments + (attachment,))
