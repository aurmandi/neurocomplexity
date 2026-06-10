"""Autonomy F-test is calibrated: Type-I near nominal on independent VAR(1)."""
import numpy as np
import pytest

from neurocomplexity.analysis.autonomy import autonomy
from neurocomplexity.benchmarks.simulators.ar_processes import var1
from neurocomplexity.benchmarks.cases.info_theory import _ar_to_recording


def _per_population_type_i(n_reps=300, seed=0, significance="analytic"):
    rng = np.random.default_rng(seed)
    A_null = np.array([[0.5, 0.0], [0.0, 0.5]])
    Sigma = np.eye(2)
    below = 0
    total = 0
    for _ in range(n_reps):
        s = int(rng.integers(2 ** 31))
        X = var1(A=A_null, Sigma=Sigma, n_samples=2000, seed=s)
        rec = _ar_to_recording(X[:, 0], X[:, 1], base_rate_hz=80.0,
                               modulation=0.9, units_per_pop=8, seed=s)
        a = autonomy(rec, populations=["X", "Y"], bin_size_ms=10.0,
                     significance=significance, seed=s)
        for v in a.values.values():
            total += 1
            if v < 0.05:
                below += 1
    return below / total


def test_analytic_type_i_near_nominal():
    rate = _per_population_type_i(n_reps=300, significance="analytic")
    # 95% binomial band around 0.05 at n=600 population-tests: ~[0.033, 0.067].
    assert 0.02 <= rate <= 0.09, f"per-population Type-I = {rate:.3f}"


@pytest.mark.slow
def test_permutation_type_i_near_nominal():
    rate = _per_population_type_i(n_reps=150, significance="permutation")
    assert 0.01 <= rate <= 0.11, f"permutation per-population Type-I = {rate:.3f}"


def test_permutation_path_runs_and_bounds_p():
    import numpy as np
    from neurocomplexity.analysis.autonomy import autonomy
    from neurocomplexity.benchmarks.simulators.ar_processes import var1
    from neurocomplexity.benchmarks.cases.info_theory import _ar_to_recording
    X = var1(A=np.array([[0.5, 0.0], [0.3, 0.5]]), Sigma=np.eye(2),
             n_samples=2000, seed=1)
    rec = _ar_to_recording(X[:, 0], X[:, 1], base_rate_hz=80.0,
                           modulation=0.9, units_per_pop=8, seed=1)
    a = autonomy(rec, populations=["X", "Y"], bin_size_ms=10.0,
                 significance="permutation", n_perm=200, seed=1)
    for v in a.values.values():
        assert 0.0 < v <= 1.0  # Phipson-Smyth floor keeps p strictly > 0
