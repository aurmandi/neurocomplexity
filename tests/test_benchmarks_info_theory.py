import pytest
from neurocomplexity.benchmarks.cases.info_theory import (
    bench_te_convergence, bench_te_null, bench_autonomy_calibration,
)


@pytest.mark.slow
def test_bench_te_convergence_passes():
    res = bench_te_convergence(n_reps=5, seed=0)
    assert res.name == "info_theory.te_convergence"
    assert res.passed, f"te_convergence benchmark failed: {res}"


@pytest.mark.slow
def test_bench_te_null_passes():
    res = bench_te_null(n_reps=10, seed=0)
    assert res.name == "info_theory.te_null"
    assert res.passed, f"te_null benchmark failed: {res}"


@pytest.mark.slow
def test_bench_autonomy_calibration_passes():
    res = bench_autonomy_calibration(n_reps=30, seed=0)
    assert res.name == "info_theory.autonomy_calibration"
    assert res.passed, f"autonomy_calibration benchmark failed: {res}"
