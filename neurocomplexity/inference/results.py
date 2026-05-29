"""Container for the output of inference procedures (null tests, bootstrap)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InferenceResult:
    """Unified output of :func:`~neurocomplexity.inference.test` and
    :func:`~neurocomplexity.inference.bootstrap`.

    Carries the observed statistic, the empirical null and / or bootstrap
    distributions, p-values (raw and FDR-adjusted), effect size, and a
    confidence interval — alongside enough metadata to reproduce the run.

    Attributes
    ----------
    statistic_name
        Human-readable name of the statistic under test (e.g. ``"TE"``,
        ``"alpha_s"``, ``"PR"``).
    observed
        Point estimate from the real data. Scalar for scalar analyses
        (branching ratio, PR), ``ndarray`` for matrix / vector analyses
        (TE matrix, PID atoms).
    null_distribution
        Surrogate-test null distribution, shape ``(n_resamples, *obs.shape)``
        or ``None`` for a pure-bootstrap call.
    bootstrap_distribution
        Bootstrap distribution of the statistic, same shape convention as
        ``null_distribution``. ``None`` for a pure-null-test call.
    p_value
        Phipson-Smyth-floored permutation p-value(s). Shape matches
        ``observed``. ``None`` when only a bootstrap CI was computed.
    p_value_fdr
        Benjamini-Hochberg adjusted p-values. ``None`` when ``observed`` is
        scalar or ``fdr=False`` was passed.
    effect_size
        ``(observed - mean(null)) / std(null)``. Scalar or ndarray matching
        ``observed``. ``None`` when no null distribution is available.
    ci_lower, ci_upper
        Lower / upper edges of the bootstrap confidence interval. ``None``
        when only a null test was run.
    ci_level
        Nominal confidence level of the interval (e.g. ``0.95``).
    method
        Surrogate or bootstrap method label (e.g. ``"spike_dither"``).
    n_resamples
        Number of surrogate / bootstrap draws used.
    seed
        Master RNG seed for reproducibility. ``None`` if not specified.
    metadata
        Free-form dictionary carrying e.g. the alternative hypothesis,
        surrogate kwargs, and the pool's provenance.

    Methods
    -------
    to_dict()
        Return a plain-Python dictionary suitable for JSON serialisation
        (ndarrays become lists).
    """

    statistic_name: str
    observed: float | np.ndarray
    null_distribution: np.ndarray | None
    bootstrap_distribution: np.ndarray | None
    p_value: float | np.ndarray | None
    p_value_fdr: float | np.ndarray | None
    effect_size: float | np.ndarray | None
    ci_lower: float | np.ndarray | None
    ci_upper: float | np.ndarray | None
    ci_level: float
    method: str
    n_resamples: int
    seed: int | None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python dict; numpy arrays become lists for JSON-friendliness."""
        out: dict[str, Any] = {}
        for k, v in asdict(self).items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out
