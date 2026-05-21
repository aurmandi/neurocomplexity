import numpy as np
import pytest

from neurocomplexity.analysis.complexity import (
    _shannon_entropy_counts,
    _lmc_disequilibrium,
    LMCResult,
)


def test_shannon_uniform_distribution_returns_one():
    # 4 states each with probability 0.25 -> normalized H = 1.0
    counts = np.array([10, 10, 10, 10])
    H = _shannon_entropy_counts(counts)
    assert H == pytest.approx(1.0)


def test_shannon_single_state_returns_zero():
    counts = np.array([100, 0, 0, 0])
    H = _shannon_entropy_counts(counts)
    assert H == pytest.approx(0.0)


def test_disequilibrium_uniform_is_zero():
    counts = np.array([10, 10, 10, 10])
    D = _lmc_disequilibrium(counts)
    assert D == pytest.approx(0.0)


def test_disequilibrium_delta_is_max():
    counts = np.array([100, 0, 0, 0])
    D = _lmc_disequilibrium(counts)
    N = 4
    expected = (1 - 1 / N) ** 2 + (N - 1) * (1 / N) ** 2
    assert D == pytest.approx(expected)


def test_lmcresult_is_frozen_dataclass():
    import dataclasses
    assert dataclasses.is_dataclass(LMCResult)
    fields = {f.name for f in dataclasses.fields(LMCResult)}
    assert {"populations", "kind", "H_per_pop", "D_per_pop", "C_per_pop",
            "H_traj", "D_traj", "C_traj", "window_centers_s",
            "bin_size_seconds", "window_seconds", "step_seconds",
            "n_states_per_pop", "source", "params"}.issubset(fields)


from neurocomplexity.analysis.complexity import lmc_complexity
from neurocomplexity.core.recording import SpikeRecording


def _poisson_rec(rate_hz: float, duration_s: float, n_units: int = 30,
                  seed: int = 0, populations: dict | None = None) -> SpikeRecording:
    rng = np.random.default_rng(seed)
    spike_times = []
    unit_ids = []
    for u in range(n_units):
        n = rng.poisson(rate_hz * duration_s)
        ts = np.sort(rng.uniform(0, duration_s, n))
        spike_times.append(ts)
        unit_ids.append(np.full(ts.size, u, dtype=np.int64))
    spike_times = np.concatenate(spike_times) if spike_times else np.array([])
    unit_ids = np.concatenate(unit_ids) if unit_ids else np.array([], dtype=np.int64)
    order = np.argsort(spike_times, kind="stable")
    spike_times = spike_times.astype(np.float64)[order]
    unit_ids = unit_ids[order]
    import pandas as pd
    units = pd.DataFrame({"id": np.arange(n_units, dtype=np.int64),
                          "quality": ["good"] * n_units})
    pops = populations or {"all": np.ones(n_units, dtype=bool)}
    return SpikeRecording(
        spike_times=spike_times, unit_ids=unit_ids, units=units,
        populations=pops, duration=float(duration_s), sampling_rate=30000.0,
        source="synthetic", _filtered=True,
    )


def test_lmc_population_mode_returns_per_pop_arrays():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0, n_units=20)
    result = lmc_complexity(rec, kind="population", bin_size_s=0.05)
    assert result.kind == "population"
    assert result.populations == ("all",)
    assert result.H_per_pop.shape == (1,)
    assert result.D_per_pop.shape == (1,)
    assert result.C_per_pop.shape == (1,)
    assert result.H_traj is None
    assert 0.0 <= result.H_per_pop[0] <= 1.0
    assert result.D_per_pop[0] >= 0.0
    assert result.C_per_pop[0] == pytest.approx(
        result.H_per_pop[0] * result.D_per_pop[0])


def test_lmc_two_populations_returns_two_dots():
    n = 40
    mask_a = np.zeros(n, dtype=bool); mask_a[:20] = True
    mask_b = ~mask_a
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0, n_units=n,
                       populations={"a": mask_a, "b": mask_b})
    result = lmc_complexity(rec, kind="population", bin_size_s=0.05)
    assert result.populations == ("a", "b")
    assert result.H_per_pop.shape == (2,)


def test_lmc_invalid_kind_raises():
    rec = _poisson_rec(rate_hz=20.0, duration_s=5.0)
    with pytest.raises(ValueError, match="kind"):
        lmc_complexity(rec, kind="banana", bin_size_s=0.05)


def test_lmc_params_dict_is_recompute_complete():
    rec = _poisson_rec(rate_hz=20.0, duration_s=5.0)
    result = lmc_complexity(rec, kind="population", bin_size_s=0.05)
    # Adapter must be able to call lmc_complexity(rec, **params) on a surrogate.
    redo = lmc_complexity(rec, **result.params)
    assert np.allclose(redo.C_per_pop, result.C_per_pop)


def test_lmc_trajectory_mode_shapes():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    result = lmc_complexity(rec, kind="trajectory", bin_size_s=0.05,
                             window_seconds=1.0, step_seconds=0.5)
    assert result.kind == "trajectory"
    # windows: floor((10 - 1)/0.5) + 1 = 19
    assert result.H_traj.shape == (19, 1)
    assert result.D_traj.shape == (19, 1)
    assert result.C_traj.shape == (19, 1)
    assert result.window_centers_s.shape == (19,)
    # population fields also populated even in trajectory mode
    assert result.H_per_pop.shape == (1,)


def test_lmc_both_mode_populates_everything():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    result = lmc_complexity(rec, kind="both")
    assert result.H_per_pop.shape == (1,)
    assert result.H_traj is not None
    assert result.H_traj.shape[1] == 1


def test_lmc_trajectory_window_smaller_than_bin_raises():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    with pytest.raises(ValueError, match="window_seconds"):
        lmc_complexity(rec, kind="trajectory",
                        bin_size_s=0.1, window_seconds=0.05)


def test_lmc_trajectory_values_in_range():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    r = lmc_complexity(rec, kind="trajectory", bin_size_s=0.05,
                       window_seconds=1.0, step_seconds=0.5)
    assert np.all((r.H_traj >= 0) & (r.H_traj <= 1))
    assert np.all(r.D_traj >= 0)
    assert np.allclose(r.C_traj, r.H_traj * r.D_traj)
