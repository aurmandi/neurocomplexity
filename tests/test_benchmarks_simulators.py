import numpy as np
from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.benchmarks.simulators.branching_network import branching_network


def test_branching_network_returns_recording():
    rec = branching_network(n_units=30, m=0.9, duration_s=20.0, bin_ms=4.0, seed=0)
    assert isinstance(rec, SpikeRecording)
    assert rec.spike_times.size > 0
    assert rec.duration >= 19.9


def test_branching_network_reproducible():
    r1 = branching_network(n_units=20, m=0.9, duration_s=10.0, seed=42)
    r2 = branching_network(n_units=20, m=0.9, duration_s=10.0, seed=42)
    assert np.array_equal(r1.spike_times, r2.spike_times)


def test_branching_network_higher_m_more_spikes():
    r_low = branching_network(n_units=20, m=0.5, duration_s=20.0, seed=0)
    r_high = branching_network(n_units=20, m=0.95, duration_s=20.0, seed=0)
    assert r_high.spike_times.size > r_low.spike_times.size


from neurocomplexity.benchmarks.simulators.ar_processes import coupled_ar1, var1


def test_coupled_ar1_returns_xy_and_te():
    x, y, te = coupled_ar1(c=0.3, a=0.5, sigma=1.0, n_samples=10_000, seed=0)
    assert x.shape == (10_000,)
    assert y.shape == (10_000,)
    assert te > 0
    x0, y0, te0 = coupled_ar1(c=0.0, a=0.5, sigma=1.0, n_samples=10_000, seed=0)
    assert abs(te0) < 1e-9


def test_coupled_ar1_te_increases_with_c():
    _, _, te_low = coupled_ar1(c=0.1, a=0.5, sigma=1.0, n_samples=1000, seed=0)
    _, _, te_high = coupled_ar1(c=0.5, a=0.5, sigma=1.0, n_samples=1000, seed=0)
    assert te_high > te_low


def test_var1_shape_and_stationarity():
    A = np.array([[0.5, 0.2], [0.0, 0.5]])
    Sigma = np.eye(2)
    X = var1(A=A, Sigma=Sigma, n_samples=5000, seed=0)
    assert X.shape == (5000, 2)
    assert np.all(np.std(X, axis=0) < 10)


def test_var1_rejects_unstable():
    import pytest
    A = np.array([[1.5, 0.0], [0.0, 0.5]])
    with pytest.raises(ValueError):
        var1(A=A, Sigma=np.eye(2), n_samples=100, seed=0)


from neurocomplexity.benchmarks.simulators.pid_distributions import pid_recording


def test_pid_recording_has_three_populations():
    rec = pid_recording("xor", n_bins=1000, seed=0)
    assert set(rec.populations.keys()) >= {"source_1", "source_2", "target"}


def test_pid_recording_xor_target_consistent():
    rec = pid_recording("xor", n_bins=2000, seed=0)
    n_target_bins = (rec.unit_ids == 2).sum()
    assert 800 < n_target_bins < 1200


def test_pid_recording_unknown_distribution_raises():
    import pytest
    with pytest.raises(ValueError):
        pid_recording("nope", n_bins=100, seed=0)


from neurocomplexity.benchmarks.simulators.structured_covariance import rank_r_population


def test_rank_r_population_returns_recording():
    rec = rank_r_population(n_units=20, rank=3, n_bins=2000, seed=0)
    assert rec.spike_times.size > 0


def test_rank_r_population_reproducible():
    r1 = rank_r_population(n_units=10, rank=2, n_bins=1000, seed=7)
    r2 = rank_r_population(n_units=10, rank=2, n_bins=1000, seed=7)
    assert np.array_equal(r1.spike_times, r2.spike_times)
