"""Tests for neurocomplexity.analysis.selection."""
from __future__ import annotations

import numpy as np
import pytest

import neurocomplexity as nc
from neurocomplexity.analysis.selection import (
    rate_matched_subsample,
    top_firing_band,
)


def _toy_rec(rates, duration=600.0, seed=0):
    rng = np.random.default_rng(seed)
    units = {}
    for u, r in enumerate(rates):
        n = rng.poisson(r * duration)
        units[u] = np.sort(rng.uniform(0.0, duration, size=n))
    return nc.io.from_dict(units, duration)


def test_top_firing_band_returns_K_highest_in_band():
    rates = [0.2, 0.5, 0.8] + [3.0, 5.0, 8.0, 12.0, 18.0] + [40.0, 80.0]
    rec = _toy_rec(rates, seed=1)
    chosen = top_firing_band(rec, K=3, rate_min=1.0, rate_max=25.0)
    assert len(chosen) == 3
    counts = {int(u): int((rec.unit_ids == u).sum()) for u in np.unique(rec.unit_ids)}
    chosen_rates = [counts[u] / rec.duration for u in chosen]
    assert chosen_rates == sorted(chosen_rates, reverse=True)
    assert all(1.0 <= r <= 25.0 for r in chosen_rates)


def test_top_firing_band_area_filter():
    rates = [3.0] * 5 + [10.0] * 5
    rec = _toy_rec(rates, seed=2)
    uids = np.unique(rec.unit_ids)
    masks = {
        "A": np.array([int(u) < 5 for u in uids], dtype=bool),
        "B": np.array([int(u) >= 5 for u in uids], dtype=bool),
    }
    rec_pop = rec.with_populations(masks)
    chosen_b = top_firing_band(rec_pop, area="B", K=3, rate_min=1.0, rate_max=25.0)
    assert all(int(u) >= 5 for u in chosen_b)


def test_rate_matched_subsample_returns_K_with_seed_determinism():
    rates = list(np.linspace(0.5, 30.0, 20))
    rec = _toy_rec(rates, seed=3)
    a = rate_matched_subsample(rec, K=4, rate_min=1.0, rate_max=25.0, seed=0)
    b = rate_matched_subsample(rec, K=4, rate_min=1.0, rate_max=25.0, seed=0)
    assert a == b
    assert len(a) == 4


def test_top_firing_band_rejects_bad_args():
    rec = _toy_rec([5.0] * 5)
    with pytest.raises(ValueError):
        top_firing_band(rec, K=0, rate_min=1.0, rate_max=25.0)
    with pytest.raises(ValueError):
        top_firing_band(rec, K=2, rate_min=25.0, rate_max=1.0)
