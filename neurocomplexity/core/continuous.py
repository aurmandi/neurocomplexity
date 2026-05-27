"""Uniformly-sampled continuous signal carried alongside a SpikeRecording.

External streams (pupil diameter, running speed, stimulus contrast,
photometry) can be wrapped as ``ContinuousSignal`` and attached to a
``SpikeRecording`` via ``rec.with_signal(name, sig)``. Analyses
(``transfer_entropy``, ``partial_information``) consume them through the
``signals=`` kwarg and discretise via binary median split.

Only uniformly-sampled signals are supported in v1: pass ``values``,
``sampling_rate``, and an optional ``t_start``. The implicit time of sample
``i`` is ``t_start + i / sampling_rate`` (seconds).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContinuousSignal:
    """Uniformly-sampled 1-D continuous signal.

    Attach to a recording with
    :meth:`SpikeRecording.with_signal <neurocomplexity.core.recording.SpikeRecording.with_signal>`,
    then pass via ``signals=...`` to
    :func:`~neurocomplexity.analysis.transfer_entropy` or
    :func:`~neurocomplexity.analysis.partial_information`. They are
    discretised via binary median split for the binary-Schreiber TE and
    via quantile bins for PID.

    Attributes
    ----------
    values
        1-D ``float64`` sample vector. Must be finite and non-empty.
    sampling_rate
        Samples per second. Must be strictly positive.
    t_start
        Time of sample 0 in seconds (default 0).
    label
        Human-readable label (e.g. ``"pupil_diameter"``).
    units
        Physical units (e.g. ``"mm"``, ``"deg/s"``).

    Notes
    -----
    Sample ``i`` occurs at time ``t_start + i / sampling_rate`` seconds.
    For non-uniformly-sampled streams (eye-tracking with dropped frames,
    photometry with variable exposure), resample to a fixed rate first;
    only uniform sampling is supported in v1.
    """

    values: np.ndarray
    sampling_rate: float
    t_start: float = 0.0
    label: str = ""
    units: str = ""

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=np.float64)
        if v.ndim != 1:
            raise ValueError(
                f"ContinuousSignal.values must be 1-D; got shape {v.shape}"
            )
        if v.size == 0:
            raise ValueError("ContinuousSignal.values must be non-empty")
        if not np.all(np.isfinite(v)):
            raise ValueError("ContinuousSignal.values must be finite (no NaN/Inf)")
        if not (self.sampling_rate > 0):
            raise ValueError(
                f"sampling_rate must be > 0; got {self.sampling_rate!r}"
            )
        if self.t_start < 0:
            raise ValueError(f"t_start must be >= 0; got {self.t_start}")
        object.__setattr__(self, "values", v)
        object.__setattr__(self, "sampling_rate", float(self.sampling_rate))
        object.__setattr__(self, "t_start", float(self.t_start))

    @property
    def duration(self) -> float:
        """Seconds spanned by the samples."""
        return float(len(self.values)) / float(self.sampling_rate)

    @property
    def t_end(self) -> float:
        return self.t_start + self.duration

    def __eq__(self, other) -> bool:
        if not isinstance(other, ContinuousSignal):
            return NotImplemented
        return (np.array_equal(self.values, other.values)
                and self.sampling_rate == other.sampling_rate
                and self.t_start == other.t_start
                and self.label == other.label
                and self.units == other.units)

    def __hash__(self) -> int:
        return hash((self.values.tobytes(), self.sampling_rate,
                     self.t_start, self.label, self.units))
