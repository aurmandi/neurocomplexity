import pytest
from neurocomplexity.benchmarks.cases.criticality import (
    bench_m_hat, bench_avalanche_exponents,
)


@pytest.mark.slow
def test_bench_m_hat_passes_at_reduced_reps():
    res = bench_m_hat(n_reps=5, seed=0)
    assert res.name == "criticality.m_hat"
    assert res.passed, f"m_hat benchmark failed: {res}"


@pytest.mark.slow
def test_bench_avalanche_exponents_passes_at_reduced_reps():
    res = bench_avalanche_exponents(n_reps=3, seed=0)
    assert res.name == "criticality.exponents"
    assert res.passed, f"exponents benchmark failed: {res}"
