from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class InferenceResult:
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
