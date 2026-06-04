"""Materialisation warning for the SpikeInterface bridge (IO4).

``from_spikeinterface`` eagerly concatenates per-unit spike trains, which
defeats the point of a memory-mapped Sorting. We now emit a UserWarning
when the materialised array would exceed ~1e8 spikes (~800 MB float64).
"""
from __future__ import annotations

import warnings as _warnings

import pytest

from neurocomplexity.io.spikeinterface import _maybe_warn_si_materialise


def test_warn_above_threshold():
    """Above 1e8 spikes the helper emits a UserWarning."""
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _maybe_warn_si_materialise(150_000_000)
    msgs = [w for w in caught
            if issubclass(w.category, UserWarning)
            and "Materialising" in str(w.message)
            or "materialising" in str(w.message)]
    assert msgs, [str(w.message) for w in caught]


def test_no_warn_below_threshold():
    """Below 1e8 spikes the helper is silent."""
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _maybe_warn_si_materialise(1_000_000)
    msgs = [w for w in caught
            if issubclass(w.category, UserWarning)
            and "materialising" in str(w.message).lower()]
    assert not msgs, [str(w.message) for w in caught]


def test_custom_threshold_overrides_default():
    """The threshold is overridable so callers can opt into a tighter warning."""
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _maybe_warn_si_materialise(10_000, threshold=5_000)
    msgs = [w for w in caught
            if issubclass(w.category, UserWarning)
            and "materialising" in str(w.message).lower()]
    assert msgs, [str(w.message) for w in caught]
