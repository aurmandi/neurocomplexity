"""Tests for viz.manifold (figure_manifold)."""
import matplotlib
matplotlib.use("Agg")
import warnings as _warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurocomplexity.analysis.manifold import manifold
from neurocomplexity.viz.manifold import figure_manifold
from tests.test_analysis_manifold import _poisson_rec


def _r(**kw):
    rec = _poisson_rec(n_units=8, duration_s=20.0, seed=10)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        return rec, manifold(rec, method="pca", dims=2, bin_size_s=0.1, **kw)


def test_figure_manifold_renders_2d():
    rec, r = _r()
    fig = figure_manifold(r)
    # Main axes + colorbar axes (time gradient legend).
    assert len(fig.axes) >= 1
    ax = fig.axes[0]
    assert len(ax.collections) >= 1 or len(ax.lines) >= 1
    plt.close(fig)


def test_figure_manifold_renders_3d():
    rec = _poisson_rec(n_units=8, duration_s=20.0, seed=11)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        r = manifold(rec, method="pca", dims=3, bin_size_s=0.1)
    fig = figure_manifold(r)
    from mpl_toolkits.mplot3d import Axes3D
    assert isinstance(fig.axes[0], Axes3D)
    plt.close(fig)


def test_figure_manifold_accepts_ax_for_2d():
    rec, r = _r()
    fig, ax = plt.subplots()
    out = figure_manifold(r, ax=ax)
    assert out is fig
    plt.close(fig)


def test_figure_manifold_color_by_time_uses_colormap():
    rec, r = _r()
    fig = figure_manifold(r, color_by="time")
    fig.canvas.draw()  # materialize colormapped facecolors
    ax = fig.axes[0]
    sc = ax.collections[0]
    fc = sc.get_facecolors()
    # scatter w/ cmap should produce one RGBA per point after draw
    assert fc.shape[0] >= 2
    assert not np.allclose(fc[0], fc[-1])
    plt.close(fig)


def test_figure_manifold_pca_shows_explained_variance():
    rec, r = _r()
    fig = figure_manifold(r)
    title = fig.axes[0].get_title(loc="left")
    assert "%" in title
    plt.close(fig)


def test_figure_manifold_invalid_color_by_raises():
    rec, r = _r()
    with pytest.raises(ValueError, match="color_by"):
        figure_manifold(r, color_by="bogus")
    plt.close("all")
