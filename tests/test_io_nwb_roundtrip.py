"""Round-trip tests for nc.io.to_nwb / nc.io.from_nwb.

These tests exercise the seven risk-register items from
``docs/superpowers/specs/2026-05-20-nwb-roundtrip-writer-design.md``: int64
unit ids, units dtypes, spike-time global order, population mask alignment,
intervals extra columns, attachments ordering, and the _filtered flag.

Tests are skipped if pynwb is not available.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pynwb = pytest.importorskip("pynwb")

from neurocomplexity import io as nc_io
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording


def _full_rec(seed: int = 0) -> SpikeRecording:
    rng = np.random.default_rng(seed)
    n_units = 5
    unit_ids = np.array([7, -3, 42, 100, 256], dtype=np.int64)
    units = pd.DataFrame({
        "id": unit_ids,
        "quality": pd.Categorical(["good", "mua", "good", "good", "noise"],
                                  categories=["noise", "mua", "good"]),
        "firing_rate": np.array([12.5, 4.1, 30.0, 0.8, 2.2], dtype=np.float64),
        "n_spikes": np.array([125, 41, 300, 8, 22], dtype=np.int64),
        "is_active": np.array([True, False, True, True, False]),
    })
    n_spikes_total = int(units["n_spikes"].sum())
    chosen_unit_rows = rng.choice(n_units, size=n_spikes_total, replace=True)
    chosen_uids = unit_ids[chosen_unit_rows]
    spike_times = np.sort(rng.uniform(0.0, 60.0, size=n_spikes_total)).astype(np.float64)
    order = np.argsort(spike_times, kind="stable")
    spike_times = spike_times[order]
    chosen_uids = chosen_uids[order]
    populations = {
        "all": np.ones(n_units, dtype=bool),
        "good": (units["quality"] == "good").to_numpy(),
        "active": units["is_active"].to_numpy(),
    }
    intervals = {
        "stimulus": pd.DataFrame({
            "start_time": [1.0, 10.0, 20.0],
            "stop_time": [2.0, 12.0, 22.0],
            "trial_type": pd.Categorical(["a", "b", "a"], categories=["a", "b"]),
            "score": np.array([0.1, 0.5, 0.9], dtype=np.float32),
        }),
    }
    return SpikeRecording(
        spike_times=spike_times,
        unit_ids=chosen_uids,
        units=units,
        populations=populations,
        duration=60.0,
        sampling_rate=30000.0,
        source=ProvenanceRecord.for_memory("test", "roundtrip"),
        intervals=intervals,
        attachments=(
            ProvenanceRecord.for_memory("quality", "step1"),
            ProvenanceRecord.for_memory("anatomy", "step2"),
            ProvenanceRecord.for_memory("trials", "step3"),
        ),
    )


def _recs_equal(a: SpikeRecording, b: SpikeRecording) -> None:
    np.testing.assert_array_equal(a.spike_times, b.spike_times)
    np.testing.assert_array_equal(a.unit_ids, b.unit_ids)
    pd.testing.assert_frame_equal(a.units.reset_index(drop=True),
                                  b.units.reset_index(drop=True))
    assert set(a.populations.keys()) == set(b.populations.keys())
    for k in a.populations:
        np.testing.assert_array_equal(a.populations[k], b.populations[k])
    assert a.duration == b.duration
    assert a.sampling_rate == b.sampling_rate
    assert a._filtered == b._filtered
    assert set(a.intervals.keys()) == set(b.intervals.keys())
    for k in a.intervals:
        pd.testing.assert_frame_equal(a.intervals[k].reset_index(drop=True),
                                      b.intervals[k].reset_index(drop=True))
    assert a.attachments == b.attachments
    assert a.source == b.source


def test_full_equality_round_trip(tmp_path: Path):
    rec = _full_rec()
    out = nc_io.to_nwb(rec, tmp_path / "out.nwb",
                      session_description="t", identifier="abc",
                      session_start_time=dt.datetime.now(dt.timezone.utc))
    assert out.exists()
    rec2 = nc_io.from_nwb(out)
    _recs_equal(rec, rec2)


def test_int64_unit_ids_preserved(tmp_path: Path):
    rec = _full_rec()
    out = nc_io.to_nwb(rec, tmp_path / "out.nwb")
    rec2 = nc_io.from_nwb(out)
    assert rec2.unit_ids.dtype == np.int64
    np.testing.assert_array_equal(rec2.unit_ids, rec.unit_ids)
    np.testing.assert_array_equal(rec2.units["id"].to_numpy(),
                                  rec.units["id"].to_numpy())


def test_units_dtype_preservation(tmp_path: Path):
    rec = _full_rec()
    rec2 = nc_io.from_nwb(nc_io.to_nwb(rec, tmp_path / "out.nwb"))
    for col in rec.units.columns:
        assert rec2.units[col].dtype == rec.units[col].dtype, col


def test_spike_times_global_order(tmp_path: Path):
    rec = _full_rec(seed=7)
    rec2 = nc_io.from_nwb(nc_io.to_nwb(rec, tmp_path / "out.nwb"))
    np.testing.assert_array_equal(rec2.spike_times, rec.spike_times)
    np.testing.assert_array_equal(rec2.unit_ids, rec.unit_ids)


def test_population_mask_alignment(tmp_path: Path):
    rec = _full_rec()
    rec2 = nc_io.from_nwb(nc_io.to_nwb(rec, tmp_path / "out.nwb"))
    np.testing.assert_array_equal(rec2.populations["good"], rec.populations["good"])
    np.testing.assert_array_equal(rec2.populations["active"], rec.populations["active"])


def test_intervals_extra_columns(tmp_path: Path):
    rec = _full_rec()
    rec2 = nc_io.from_nwb(nc_io.to_nwb(rec, tmp_path / "out.nwb"))
    pd.testing.assert_frame_equal(
        rec.intervals["stimulus"].reset_index(drop=True),
        rec2.intervals["stimulus"].reset_index(drop=True),
    )


def test_attachments_chain_order(tmp_path: Path):
    rec = _full_rec()
    rec2 = nc_io.from_nwb(nc_io.to_nwb(rec, tmp_path / "out.nwb"))
    assert rec.attachments == rec2.attachments
    assert tuple(a.source_path for a in rec2.attachments) == \
           tuple(a.source_path for a in rec.attachments)


def test_filtered_flag_round_trips_both_states(tmp_path: Path):
    for flag in (True, False):
        rec = _full_rec()
        from dataclasses import replace
        rec = replace(rec, _filtered=flag)
        rec2 = nc_io.from_nwb(nc_io.to_nwb(rec, tmp_path / f"f{int(flag)}.nwb"))
        assert rec2._filtered is flag


def test_overwrite_flag(tmp_path: Path):
    rec = _full_rec()
    p = tmp_path / "out.nwb"
    nc_io.to_nwb(rec, p)
    with pytest.raises(FileExistsError):
        nc_io.to_nwb(rec, p)
    nc_io.to_nwb(rec, p, overwrite=True)


def test_to_nwb_exported_on_io_module():
    assert hasattr(nc_io, "to_nwb")
