import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurocomplexity.viz._scale_bar import add_scale_bar


def test_adds_x_scale_bar_with_label():
    fig, ax = plt.subplots()
    ax.plot([0, 10], [0, 1])
    add_scale_bar(ax, x_length=2.0, x_label="2 s", location="lower right")
    lines = [ln for ln in ax.lines if abs(np.ptp(ln.get_xdata())) > 0]
    assert any(abs(np.ptp(ln.get_xdata()) - 2.0) < 1e-9 for ln in lines)
    texts = [t.get_text() for t in ax.texts]
    assert "2 s" in texts
    plt.close(fig)


def test_adds_y_scale_bar_strips_y_axis():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 100])
    add_scale_bar(ax, y_length=50.0, y_label="50 mV", location="lower left")
    assert not ax.spines["left"].get_visible()
    plt.close(fig)


def test_both_axes_scale_bars_strip_both_axes():
    fig, ax = plt.subplots()
    ax.plot([0, 5], [0, 5])
    add_scale_bar(ax, x_length=1.0, x_label="1 s",
                  y_length=1.0, y_label="1 unit")
    assert not ax.spines["bottom"].get_visible()
    assert not ax.spines["left"].get_visible()
    plt.close(fig)


def test_color_falls_back_to_palette_text():
    from neurocomplexity.viz._style import current_palette
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    add_scale_bar(ax, x_length=0.5, x_label="0.5")
    expected = current_palette()["text"].lower()
    line = [ln for ln in ax.lines if abs(np.ptp(ln.get_xdata()) - 0.5) < 1e-9][0]
    assert line.get_color().lower() == expected
    plt.close(fig)


def test_invalid_location_raises():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with pytest.raises(ValueError, match="location"):
        add_scale_bar(ax, x_length=0.5, x_label="0.5", location="middle")
    plt.close(fig)
