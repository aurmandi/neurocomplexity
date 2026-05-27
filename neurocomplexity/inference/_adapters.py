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
from neurocomplexity.analysis.complexity import lmc_complexity, LMCResult
from neurocomplexity.analysis.mse import multiscale_entropy, MSEResult


class AdapterError(TypeError):
    """Raised when no inference adapter is registered for a result type.

    A subclass of :class:`TypeError` so existing code that catches the
    latter continues to work.
    """


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


def _lmc_adapter(result: LMCResult) -> Callable[[Any], np.ndarray]:
    kw = dict(result.params)
    def f(rec):
        return np.asarray(lmc_complexity(rec, **kw).C_per_pop, dtype=float)
    return f


def _mse_adapter(result: MSEResult) -> Callable[[Any], np.ndarray]:
    kw = dict(result.params)
    def f(rec):
        return np.asarray(multiscale_entropy(rec, **kw).sampen, dtype=float)
    return f


_REGISTRY = {
    TransferEntropyResult: _te_adapter,
    LMCResult: _lmc_adapter,
    MSEResult: _mse_adapter,
    PIDResult: _pid_adapter,
    BranchingResult: _branching_adapter,
    CriticalityResult: _crit_adapter,
    ShapeCollapseResult: _collapse_adapter,
    DimensionalityResult: _pr_adapter,
    AutonomyResult: _autonomy_adapter,
}


def adapter_for(result) -> Callable[[Any], Any]:
    """Return a callable ``f(rec) -> array_or_float`` that recomputes the
    statistic carried by ``result`` on any surrogate recording.

    The returned function captures the original ``params`` so the surrogate
    statistic is evaluated under identical conditions.

    Raises
    ------
    AdapterError
        If no adapter is registered for ``type(result)``.
    """
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
    if isinstance(result, LMCResult):
        return np.asarray(result.C_per_pop, dtype=float)
    if isinstance(result, MSEResult):
        return np.asarray(result.sampen, dtype=float)
    raise AdapterError(f"no observed_statistic for {type(result).__name__}")
