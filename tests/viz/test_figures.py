import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


@pytest.fixture
def crit_result():
    from neurocomplexity.analysis.criticality import CriticalityResult
    rng = np.random.default_rng(0)
    sizes = rng.pareto(1.4, size=2000) + 1
    lifetimes = rng.pareto(1.8, size=2000) + 1
    return CriticalityResult(
        alpha_s=1.5, alpha_t=2.0, r_squared=0.95,
        sizes=sizes, lifetimes=lifetimes,
        optimal_bin_seconds=0.004,
        branching=0.95,
        kappa=1.03,
        populations=("all",),
        source=None,
        params={},
    )


@pytest.fixture
def branch_result():
    from neurocomplexity.analysis.branching import BranchingResult
    ks = np.arange(1, 21)
    return BranchingResult(
        m=0.95, r_values=0.7 * 0.95 ** ks, k_lags=ks,
        r_squared=0.99, n_bins=10000, bin_size_seconds=0.004,
        populations=("all",), source=None, params={},
    )


def test_figure_criticality_default_palette_returns_figure(crit_result):
    from neurocomplexity.viz.criticality import figure_criticality
    fig = figure_criticality(crit_result)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2
    plt.close(fig)


@pytest.mark.parametrize("palette", ["forest", "wine", "sage"])
def test_figure_criticality_signal_colour_matches_palette(crit_result, palette):
    from neurocomplexity.viz.criticality import figure_criticality
    from neurocomplexity.viz._palettes import get_palette
    fig = figure_criticality(crit_result, palette=palette)
    expected = get_palette(palette)["signal"].lower()
    line = fig.axes[0].lines[0]
    assert line.get_color().lower() == expected
    plt.close(fig)


def test_figure_branching_returns_one_axes(branch_result):
    from neurocomplexity.viz.branching import figure_branching
    fig = figure_branching(branch_result)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_figure_branching_into_provided_axes(branch_result):
    from neurocomplexity.viz.branching import figure_branching
    fig, ax = plt.subplots()
    out = figure_branching(branch_result, ax=ax)
    assert out is fig
    assert ax.has_data()
    plt.close(fig)


def test_panel_label_renders_letter(branch_result):
    from neurocomplexity.viz.branching import figure_branching
    fig = figure_branching(branch_result, panel_label="b")
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "b" in texts
    plt.close(fig)


def test_figsize_honoured(branch_result):
    from neurocomplexity.viz.branching import figure_branching
    fig = figure_branching(branch_result, figsize=(5.0, 3.0))
    w, h = fig.get_size_inches()
    assert abs(w - 5.0) < 1e-6 and abs(h - 3.0) < 1e-6
    plt.close(fig)
