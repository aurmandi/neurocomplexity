import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurocomplexity.analysis.complexity import lmc_complexity
from neurocomplexity.viz.complexity import figure_lmc_complexity
from tests.test_analysis_complexity import _poisson_rec


def test_figure_lmc_population_renders():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    r = lmc_complexity(rec, kind="population", bin_size_s=0.05)
    fig = figure_lmc_complexity(r)
    assert len(fig.axes) == 1
    pts = fig.axes[0].collections[0]
    assert pts.get_offsets().shape == (1, 2)
    plt.close(fig)


def test_figure_lmc_both_renders_two_panels():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    r = lmc_complexity(rec, kind="both")
    fig = figure_lmc_complexity(r)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_figure_lmc_accepts_ax_for_single_kind():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    r = lmc_complexity(rec, kind="population")
    fig, ax = plt.subplots()
    out = figure_lmc_complexity(r, ax=ax)
    assert out is fig
    plt.close(fig)


def test_figure_lmc_kind_override():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    r = lmc_complexity(rec, kind="both")
    fig = figure_lmc_complexity(r, kind="population")
    assert len(fig.axes) == 1
    plt.close(fig)


def test_figure_lmc_axis_labels_present():
    rec = _poisson_rec(rate_hz=20.0, duration_s=10.0)
    r = lmc_complexity(rec, kind="population")
    fig = figure_lmc_complexity(r)
    xl = fig.axes[0].get_xlabel().lower()
    yl = fig.axes[0].get_ylabel().lower()
    assert "h" in xl or "entropy" in xl
    assert "c" in yl or "complexity" in yl
    plt.close(fig)
