import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurocomplexity.analysis.mse import multiscale_entropy
from neurocomplexity.viz.mse import figure_mse
from tests.test_analysis_complexity import _poisson_rec


def test_figure_mse_renders_one_line_per_pop():
    rec = _poisson_rec(rate_hz=30.0, duration_s=20.0)
    r = multiscale_entropy(rec, bin_size_s=0.05, scale_max=8)
    fig = figure_mse(r)
    ax = fig.axes[0]
    assert len(ax.lines) >= 1
    plt.close(fig)


def test_figure_mse_axis_labels():
    rec = _poisson_rec(rate_hz=30.0, duration_s=20.0)
    r = multiscale_entropy(rec, bin_size_s=0.05, scale_max=8)
    fig = figure_mse(r)
    xl = fig.axes[0].get_xlabel().lower()
    yl = fig.axes[0].get_ylabel().lower()
    assert "scale" in xl
    assert "sampen" in yl or "entropy" in yl
    plt.close(fig)


def test_figure_mse_accepts_ax():
    rec = _poisson_rec(rate_hz=30.0, duration_s=20.0)
    r = multiscale_entropy(rec, bin_size_s=0.05, scale_max=8)
    fig, ax = plt.subplots()
    out = figure_mse(r, ax=ax)
    assert out is fig
    plt.close(fig)


def test_figure_mse_envelope_off_when_no_null_result():
    rec = _poisson_rec(rate_hz=30.0, duration_s=20.0)
    r = multiscale_entropy(rec, bin_size_s=0.05, scale_max=8)
    fig = figure_mse(r, show_envelope=True)
    assert len(fig.axes[0].collections) == 0
    plt.close(fig)
