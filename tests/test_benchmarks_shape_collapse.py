"""Shape-collapse benchmark case recovers the mean-field gamma."""
import pytest

from neurocomplexity.benchmarks.cases.shape_collapse import bench_gamma_collapse
from neurocomplexity.benchmarks.runner import list_cases


def test_shape_collapse_case_registered():
    assert "shape_collapse.gamma" in list_cases()


@pytest.mark.slow
def test_shape_collapse_case_passes_small():
    res = bench_gamma_collapse(n_reps=5, seed=0)
    assert res.name == "shape_collapse.gamma"
    assert res.metadata["n_used"] > 0, "all replicates were skipped"
    assert res.passed, (
        f"observed={res.observed} tol={res.tolerance} "
        f"gamma_mean={res.metadata.get('gamma_mean')}"
    )
