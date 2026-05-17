import pytest
import pandas as pd
from neurocomplexity.benchmarks import BenchmarkResult, list_cases, run_case, run_all


def test_benchmark_result_is_frozen():
    r = BenchmarkResult(
        name="x.y", observed=1.0, expected=1.0,
        tolerance=0.1, passed=True, runtime_s=0.01, n_reps=1, metadata={},
    )
    with pytest.raises(Exception):
        r.observed = 2.0


def test_list_cases_returns_list():
    names = list_cases()
    assert isinstance(names, list)


def test_list_cases_includes_all_eleven():
    names = list_cases()
    expected = {
        "criticality.m_hat", "criticality.exponents",
        "info_theory.te_convergence", "info_theory.te_null",
        "info_theory.autonomy_calibration",
        "pid.atoms_xor", "pid.atoms_and", "pid.atoms_copy",
        "pid.atoms_rdn", "pid.atoms_unq",
        "dimensionality.pr_rank",
    }
    assert set(names) == expected


def test_run_case_returns_benchmark_result():
    # XOR is the cheapest case (~2 s at n_reps=2).
    res = run_case("pid.atoms_xor", n_reps=2, seed=0)
    assert res.name == "pid.atoms_xor"
    assert res.n_reps == 2


def test_run_all_returns_dataframe():
    df = run_all(cases=["pid.atoms_xor"], n_reps=2, seed=0, verbose=False)
    assert isinstance(df, pd.DataFrame)
    assert "passed" in df.columns
    assert len(df) == 1


def test_run_case_unknown_raises():
    with pytest.raises(KeyError):
        run_case("does.not.exist", n_reps=1)
