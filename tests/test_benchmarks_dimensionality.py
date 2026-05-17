import pytest
from neurocomplexity.benchmarks.cases.dimensionality import bench_pr_rank


@pytest.mark.slow
def test_bench_pr_rank_passes():
    res = bench_pr_rank(n_reps=5, seed=0)
    assert res.name == "dimensionality.pr_rank"
    assert res.passed, f"pr_rank failed: {res.metadata}"
