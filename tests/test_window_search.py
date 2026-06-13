"""Tests for datasets/window_search.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import neurocomplexity as nc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "datasets"))
from window_search import scan_joint_stationary, top_units_in_band  # noqa: E402


def _toy_rec_multiarea(duration=600.0, units_per_area=15, rate=8.0, seed=0):
    """Build one SpikeRecording with two area populations A, B."""
    rng = np.random.default_rng(seed)
    units: dict[int, np.ndarray] = {}
    area_for: dict[int, str] = {}
    next_id = 0
    for area in ("A", "B"):
        for _ in range(units_per_area):
            n = rng.poisson(rate * duration)
            units[next_id] = np.sort(rng.uniform(0.0, duration, size=n))
            area_for[next_id] = area
            next_id += 1
    rec = nc.io.from_dict(units, duration)
    uids = np.unique(rec.unit_ids)
    masks = {
        area: np.array([area_for[int(u)] == area for u in uids], dtype=bool)
        for area in ("A", "B")
    }
    return rec.with_populations(masks)


def test_scan_returns_window_passing_all_areas():
    rec = _toy_rec_multiarea(seed=1)
    hits = scan_joint_stationary(
        rec, areas=("A", "B"),
        window_s=200.0, step_s=50.0,
    )
    assert hits, "stationary Poisson rec must yield at least one joint window"
    win, reports = hits[0]
    assert win[1] - win[0] == pytest.approx(200.0)
    assert all(getattr(r, "is_stationary", False) for r in reports.values())
    assert set(reports) == {"A", "B"}


def test_scan_rejects_unknown_area():
    rec = _toy_rec_multiarea(seed=2)
    with pytest.raises(KeyError):
        scan_joint_stationary(rec, areas=("A", "NOPE"),
                              window_s=200.0, step_s=50.0)


def test_top_units_filters_rate_band_and_caps_K():
    rng = np.random.default_rng(7)
    duration = 600.0
    units: dict[int, np.ndarray] = {}
    rates = [0.2, 0.5, 0.8] + [3.0, 5.0, 8.0, 12.0, 18.0] * 5 + [40.0, 80.0]
    for u, r in enumerate(rates):
        n = rng.poisson(r * duration)
        units[u] = np.sort(rng.uniform(0.0, duration, size=n))
    rec = nc.io.from_dict(units, duration)
    chosen = top_units_in_band(rec, K=5, rate_min=1.0, rate_max=25.0)
    assert len(chosen) == 5
    rates_out = [len(units[u]) / duration for u in chosen]
    assert all(1.0 <= r <= 25.0 for r in rates_out)
