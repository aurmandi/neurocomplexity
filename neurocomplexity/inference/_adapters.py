"""Recompute-statistic adapters.

Given an analysis Result, produce a callable f(rec) -> float | np.ndarray
that re-runs the analysis on `rec` using the same parameters. Used by the
null-test machinery to evaluate the statistic on surrogate recordings.
"""
from __future__ import annotations
from typing import Callable, Any

import numpy as np

from neurocomplexity.analysis.transfer_entropy import (
    transfer_entropy, TransferEntropyResult,
)
from neurocomplexity.analysis.pid import partial_information, PIDResult
from neurocomplexity.analysis.branching import wilting_mr, BranchingResult
from neurocomplexity.analysis.criticality import criticality, CriticalityResult
from neurocomplexity.analysis.shape_collapse import shape_collapse, ShapeCollapseResult
from neurocomplexity.analysis.dimensionality import (
    dimensionality, DimensionalityResult,
)
from neurocomplexity.analysis.autonomy import autonomy, AutonomyResult


class AdapterError(TypeError):
    """Raised when no adapter is registered for a result type."""


def _te_adapter(result: TransferEntropyResult) -> Callable[[Any], np.ndarray]:
    kw = dict(result.params)
    def f(rec):
        return np.asarray(transfer_entropy(rec, **kw).matrix, dtype=float)
    return f


def _pid_adapter(result: PIDResult) -> Callable[[Any], np.ndarray]:
    kw = dict(result.params)
    def f(rec):
        r = partial_information(rec, **kw)
        return np.array([r.redundancy, r.unique_1, r.unique_2, r.synergy])
    return f


def _branching_adapter(result: BranchingResult) -> Callable[[Any], float]:
    kw = dict(result.params)
    def f(rec):
        return float(wilting_mr(rec, **kw).m)
    return f


def _crit_adapter(result: CriticalityResult) -> Callable[[Any], np.ndarray]:
    kw = dict(result.params)
    def f(rec):
        r = criticality(rec, **kw)
        return np.array([r.alpha_s, r.alpha_t])
    return f


def _collapse_adapter(result: ShapeCollapseResult) -> Callable[[Any], float]:
    kw = dict(result.params)
    def f(rec):
        return float(shape_collapse(rec, **kw).gamma)
    return f


def _pr_adapter(result: DimensionalityResult) -> Callable[[Any], float]:
    kw = dict(result.params)
    def f(rec):
        return float(dimensionality(rec, **kw).participation_ratio)
    return f


def _autonomy_adapter(result: AutonomyResult) -> Callable[[Any], np.ndarray]:
    kw = dict(result.params)
    def f(rec):
        r = autonomy(rec, **kw)
        keys = sorted(r.values.keys())
        return np.array([r.values[k] for k in keys])
    return f


_REGISTRY = {
    TransferEntropyResult: _te_adapter,
    PIDResult: _pid_adapter,
    BranchingResult: _branching_adapter,
    CriticalityResult: _crit_adapter,
    ShapeCollapseResult: _collapse_adapter,
    DimensionalityResult: _pr_adapter,
    AutonomyResult: _autonomy_adapter,
}


def adapter_for(result) -> Callable[[Any], Any]:
    fn = _REGISTRY.get(type(result))
    if fn is None:
        raise AdapterError(f"no inference adapter for {type(result).__name__}")
    return fn(result)


def observed_statistic(result) -> Any:
    """Extract the same scalar/array that the adapter returns, from `result`."""
    if isinstance(result, TransferEntropyResult):
        return np.asarray(result.matrix, dtype=float)
    if isinstance(result, PIDResult):
        return np.array([result.redundancy, result.unique_1, result.unique_2, result.synergy])
    if isinstance(result, BranchingResult):
        return float(result.m)
    if isinstance(result, CriticalityResult):
        return np.array([result.alpha_s, result.alpha_t])
    if isinstance(result, ShapeCollapseResult):
        return float(result.gamma)
    if isinstance(result, DimensionalityResult):
        return float(result.participation_ratio)
    if isinstance(result, AutonomyResult):
        keys = sorted(result.values.keys())
        return np.array([result.values[k] for k in keys])
    raise AdapterError(f"no observed_statistic for {type(result).__name__}")
