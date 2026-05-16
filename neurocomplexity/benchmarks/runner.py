"""Benchmark runner: BenchmarkResult dataclass, case registry, run_case / run_all.

The registry is populated as a side effect of importing the case modules
under ``neurocomplexity.benchmarks.cases``. Case functions are decorated
with ``@register("group.name")`` and accept ``n_reps`` / ``seed`` keyword
arguments (small for CI gates, large for published benchmarking). They
return a :class:`BenchmarkResult`.

The :func:`run_all` entry point invokes every registered case (or a
user-specified subset) and returns a tidy :class:`pandas.DataFrame` with
the columns ``name, observed, expected, tolerance, passed, runtime_s,
n_reps``. The full per-case ``metadata`` dict is dropped from the tabular
form; users wanting the inner detail should call :func:`run_case`.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import time
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    observed: float
    expected: float
    tolerance: float
    passed: bool
    runtime_s: float
    n_reps: int
    metadata: dict = field(default_factory=dict)


_REGISTRY: dict[str, Callable[..., BenchmarkResult]] = {}


def register(name: str):
    """Decorator: register a benchmark case under the dotted name."""
    def deco(fn: Callable[..., BenchmarkResult]):
        _REGISTRY[name] = fn
        return fn
    return deco


def _ensure_cases_loaded() -> None:
    """Import case modules so their @register decorators fire."""
    from neurocomplexity.benchmarks.cases import (  # noqa: F401
        criticality, info_theory, pid, dimensionality,
    )


def list_cases() -> list[str]:
    """Return the sorted names of all registered benchmark cases."""
    _ensure_cases_loaded()
    return sorted(_REGISTRY.keys())


def run_case(name: str, *, n_reps: int = 200, seed: int = 0) -> BenchmarkResult:
    """Run a single benchmark case by name."""
    _ensure_cases_loaded()
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown benchmark case: {name!r}. Known: {list_cases()}"
        )
    return _REGISTRY[name](n_reps=n_reps, seed=seed)


def run_all(
    cases: list[str] | None = None,
    *,
    n_reps: int = 200,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every (or a subset of) registered cases and return a tidy DataFrame.

    Parameters
    ----------
    cases : list of names, optional
        Subset to run; defaults to all registered cases.
    n_reps : int
        Replicate count passed to each case. Increase for the published
        benchmark baseline; CI gates run with small reps.
    seed : int
        Master seed, fanned out per case via the case's own RNG seeding.
    verbose : bool
        Whether to print live progress (one line per case).
    """
    _ensure_cases_loaded()
    names = cases if cases is not None else list_cases()
    rows: list[dict] = []
    for nm in names:
        if verbose:
            print(f"[benchmarks] running {nm} (n_reps={n_reps}) ...", flush=True)
        t0 = time.time()
        res = run_case(nm, n_reps=n_reps, seed=seed)
        if verbose:
            ok = "PASS" if res.passed else "FAIL"
            print(
                f"  -> {ok}  observed={res.observed:.4f} "
                f"tol={res.tolerance:.4f}  ({time.time() - t0:.1f}s)",
                flush=True,
            )
        d = asdict(res)
        d.pop("metadata", None)
        rows.append(d)
    return pd.DataFrame(rows, columns=[
        "name", "observed", "expected", "tolerance",
        "passed", "runtime_s", "n_reps",
    ])
