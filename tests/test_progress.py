"""Tests for the global progress-bar toggle (nc.set_progress)."""
from __future__ import annotations

import pytest

import neurocomplexity as nc
from neurocomplexity import _progress


@pytest.fixture(autouse=True)
def _reset_progress_state():
    """Ensure each test starts with progress disabled."""
    _progress.set_progress(False)
    yield
    _progress.set_progress(False)


def test_default_disabled_after_reset():
    assert _progress._enabled is False
    src = [1, 2, 3]
    assert _progress.progress_iter(src) is src


def test_set_progress_enables_tqdm_wrap():
    nc.set_progress(True)
    wrapped = _progress.progress_iter([1, 2, 3], desc="x", total=3)
    assert hasattr(wrapped, "update")
    assert hasattr(wrapped, "close")
    wrapped.close()


def test_set_progress_disable_restores_passthrough():
    nc.set_progress(True)
    nc.set_progress(False)
    src = [1, 2, 3]
    assert _progress.progress_iter(src) is src


def test_iteration_values_intact_when_enabled():
    nc.set_progress(True)
    out = list(_progress.progress_iter(range(5), desc="t", total=5))
    assert out == [0, 1, 2, 3, 4]


def test_set_progress_idempotent():
    nc.set_progress(True)
    nc.set_progress(True)
    assert _progress._enabled is True
    nc.set_progress(False)
    nc.set_progress(False)
    assert _progress._enabled is False


def test_set_progress_exported_from_top_level():
    assert hasattr(nc, "set_progress")
    assert nc.set_progress is _progress.set_progress
