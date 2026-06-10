"""Shape-collapse benchmark case recovers the mean-field gamma."""
from neurocomplexity.benchmarks.runner import list_cases, run_case


def test_shape_collapse_case_registered():
    assert "shape_collapse.collapse" in list_cases()


def test_shape_collapse_case_passes_small():
    res = run_case("shape_collapse.collapse", n_reps=3, seed=0)
    assert res.name == "shape_collapse.collapse"
    assert res.passed, (
        f"observed={res.observed} tol={res.tolerance} "
        f"gamma_mean={res.metadata.get('gamma_mean')}"
    )
