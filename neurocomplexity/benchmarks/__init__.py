"""Benchmark validation suite for neurocomplexity analyses.

Provides synthetic-data simulators with closed-form or simulator-derived
ground truth and a set of benchmark cases that compare each analysis in
the package against that truth. See ``docs/benchmarks.md``.
"""
from neurocomplexity.benchmarks.runner import (
    BenchmarkResult,
    list_cases,
    register,
    run_all,
    run_case,
)

__all__ = [
    "BenchmarkResult", "list_cases", "register", "run_case", "run_all",
]
