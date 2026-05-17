import pytest
from neurocomplexity.benchmarks.cases.pid import (
    bench_atoms_xor, bench_atoms_and, bench_atoms_copy,
    bench_atoms_rdn, bench_atoms_unq,
)


@pytest.mark.slow
@pytest.mark.parametrize("fn,name", [
    (bench_atoms_xor, "pid.atoms_xor"),
    (bench_atoms_and, "pid.atoms_and"),
    (bench_atoms_copy, "pid.atoms_copy"),
    (bench_atoms_rdn, "pid.atoms_rdn"),
    (bench_atoms_unq, "pid.atoms_unq"),
])
def test_pid_atom_benchmark_passes(fn, name):
    res = fn(n_reps=3, seed=0)
    assert res.name == name
    assert res.passed, f"{name} failed: {res.metadata}"
