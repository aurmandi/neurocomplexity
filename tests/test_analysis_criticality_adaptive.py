"""Adaptive binning: per-population IEI bin; default 4 ms unchanged."""
import numpy as np

from neurocomplexity.analysis.criticality import criticality
from neurocomplexity.benchmarks.simulators.branching_network import (
    trial_based_avalanches,
)


def test_adaptive_bin_runs_and_records_bin():
    rec = trial_based_avalanches(n_units=40, n_trials=2000, m=1.0,
                                 bin_ms=4.0, seed=0)
    res = criticality(rec, populations=["all"], bin_size="adaptive")
    assert np.isfinite(res.optimal_bin)
    assert res.params["bin_selection"] == "adaptive"
    assert res.optimal_bin > 0.0


def test_default_bin_is_still_four_ms():
    rec = trial_based_avalanches(n_units=40, n_trials=2000, m=1.0,
                                 bin_ms=4.0, seed=0)
    res = criticality(rec, populations=["all"])
    assert abs(res.optimal_bin - 4.0) < 1e-9
    assert res.params["bin_selection"] == "single"


def test_adaptive_bin_value_matches_mean_iei():
    rec = trial_based_avalanches(n_units=40, n_trials=2000, m=1.0,
                                 bin_ms=4.0, seed=1)
    res = criticality(rec, populations=["all"], bin_size="adaptive")
    n_spikes = rec.spike_times.size
    expected_ms = (rec.duration / n_spikes) * 1000.0
    assert abs(res.optimal_bin - expected_ms) < 1e-6
