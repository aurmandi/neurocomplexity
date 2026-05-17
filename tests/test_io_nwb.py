"""Roundtrip test for ``neurocomplexity.io.from_nwb``.

Writes a minimal NWB file on the fly (3 units, 5 s, hand-built spike trains),
reads it back through the public loader, and asserts the contract:
``SpikeRecording`` invariants, spike-count preservation, and per-unit spike
preservation. Also asserts the import-error path when ``pynwb`` is missing,
via a brief sys.modules patch.

Skipped if the ``[nwb]`` extra is not installed.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pynwb = pytest.importorskip("pynwb")
from pynwb import NWBFile, NWBHDF5IO  # noqa: E402

import neurocomplexity as nc  # noqa: E402


def _write_minimal_nwb(path: Path) -> dict[int, np.ndarray]:
    """Write a 3-unit, 5-second NWB and return the ground-truth spike trains."""
    spikes = {
        0: np.array([0.10, 0.42, 1.07, 2.55, 3.91], dtype=np.float64),
        1: np.array([0.05, 0.93, 2.10, 4.80], dtype=np.float64),
        2: np.array([0.30, 0.70, 1.50, 2.20, 3.00, 4.10], dtype=np.float64),
    }
    nwbfile = NWBFile(
        session_description="neurocomplexity test fixture",
        identifier="neurocomplexity-test",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    for uid, st in spikes.items():
        nwbfile.add_unit(id=int(uid), spike_times=st)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    return spikes


def test_from_nwb_roundtrip(tmp_path):
    truth = _write_minimal_nwb(tmp_path / "session.nwb")
    rec = nc.io.from_nwb(tmp_path / "session.nwb")

    # Contract: invariants enforced by SpikeRecording.__post_init__.
    assert rec.n_units == len(truth)
    assert rec.n_spikes == sum(len(v) for v in truth.values())
    assert np.all(np.diff(rec.spike_times) >= 0), "spike_times must be sorted"
    assert rec.duration > rec.spike_times.max()

    # Per-unit preservation: every original spike round-trips into the
    # combined arrays with the right owner id.
    for uid, st in truth.items():
        recovered = np.sort(rec.spike_times[rec.unit_ids == uid])
        np.testing.assert_allclose(recovered, st, atol=1e-9)

    # Default population is the all-units mask.
    assert "all" in rec.populations
    assert rec.populations["all"].sum() == len(truth)

    # Provenance is attached.
    assert rec.source is not None
    assert rec.source.source_format == "nwb"


def test_from_nwb_missing_pynwb_raises(tmp_path, monkeypatch):
    """If pynwb is unavailable the loader raises a clear ImportError."""
    # Force `import pynwb` to fail inside the loader without uninstalling it.
    monkeypatch.setitem(sys.modules, "pynwb", None)
    with pytest.raises(ImportError, match=r"neurocomplexity\[nwb\]"):
        nc.io.from_nwb(tmp_path / "anything.nwb")
