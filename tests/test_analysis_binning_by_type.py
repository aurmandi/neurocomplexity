"""E/I-aware binning helper (A6).

``bin_all_active`` sums all populations regardless of cell type and is
appropriate for "total activity" diagnostics. For E/I-balanced dynamics
the new :func:`bin_active_by_type` returns one count vector per
cell-type label so downstream analyses can keep streams separated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurocomplexity.analysis._binning import (
    bin_active_by_type,
    bin_all_active,
    bin_spikes,
)
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording


def _rec(n_units: int = 4, duration: float = 5.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    # 4 single-unit populations P0..P3
    times_per_unit = [
        np.sort(rng.uniform(0.0, duration, size=int(rng.poisson(40))))
        for _ in range(n_units)
    ]
    times = np.concatenate(times_per_unit)
    owners = np.concatenate([
        np.full(len(s), i, dtype=np.int64) for i, s in enumerate(times_per_unit)
    ])
    order = np.argsort(times, kind="stable")
    populations = {f"P{i}": (np.arange(n_units) == i) for i in range(n_units)}
    return SpikeRecording(
        spike_times=times[order].astype(np.float64),
        unit_ids=owners[order],
        units=pd.DataFrame({"id": np.arange(n_units, dtype=np.int64)}),
        populations=populations,
        duration=float(duration),
        sampling_rate=None,
        source=ProvenanceRecord.for_memory("test"),
    )


def test_by_type_separates_e_and_i():
    """Two E and two I populations produce exactly two streams."""
    rec = _rec(n_units=4)
    pops = ["P0", "P1", "P2", "P3"]
    types = {"P0": "E", "P1": "E", "P2": "I", "P3": "I"}
    streams = bin_active_by_type(rec, pops, 0.010, types)
    assert set(streams) == {"E", "I"}
    raw = bin_spikes(rec, pops, 0.010)  # (T, 4)
    assert np.array_equal(streams["E"], raw[:, 0] + raw[:, 1])
    assert np.array_equal(streams["I"], raw[:, 2] + raw[:, 3])


def test_by_type_handles_missing_label():
    """Populations without a label land in ``missing_label``."""
    rec = _rec(n_units=4)
    pops = ["P0", "P1", "P2", "P3"]
    types = {"P0": "E", "P2": "I"}  # P1 and P3 unlabelled
    streams = bin_active_by_type(rec, pops, 0.010, types)
    assert set(streams) == {"E", "I", "unlabelled"}
    raw = bin_spikes(rec, pops, 0.010)
    assert np.array_equal(streams["unlabelled"], raw[:, 1] + raw[:, 3])


def test_by_type_custom_missing_label():
    rec = _rec(n_units=2)
    streams = bin_active_by_type(rec, ["P0", "P1"], 0.010,
                                 cell_types={}, missing_label="other")
    assert set(streams) == {"other"}


def test_by_type_sum_equals_bin_all_active():
    """Sum of per-type streams equals ``bin_all_active`` total."""
    rec = _rec(n_units=4)
    pops = ["P0", "P1", "P2", "P3"]
    types = {"P0": "E", "P1": "E", "P2": "I", "P3": "I"}
    streams = bin_active_by_type(rec, pops, 0.010, types)
    total_from_types = sum(streams.values())
    assert np.array_equal(total_from_types, bin_all_active(rec, pops, 0.010))
