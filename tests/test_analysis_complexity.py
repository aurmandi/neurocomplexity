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
