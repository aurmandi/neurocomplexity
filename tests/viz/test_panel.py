import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


def _crit_result():
    from neurocomplexity.analysis.criticality import CriticalityResult
    import inspect
    rng = np.random.default_rng(0)
    # Inspect actual fields and provide all required ones
    fields = [f for f in inspect.signature(CriticalityResult.__init__).parameters
              if f != "self"]
    args = {}
    if "alpha_s" in fields: args["alpha_s"] = 1.5
    if "alpha_t" in fields: args["alpha_t"] = 2.0
    if "r_squared" in fields: args["r_squared"] = 0.95
    if "sizes" in fields: args["sizes"] = rng.pareto(1.4, 500) + 1
    if "lifetimes" in fields: args["lifetimes"] = rng.pareto(1.8, 500) + 1
    if "optimal_bin_seconds" in fields: args["optimal_bin_seconds"] = 0.004
    if "branching" in fields: args["branching"] = 0.95
    if "kappa" in fields: args["kappa"] = 1.0
    if "populations" in fields: args["populations"] = ("all",)
    if "source" in fields: args["source"] = None
    if "n_avalanches" in fields: args["n_avalanches"] = 500
    if "params" in fields: args["params"] = {}
    return CriticalityResult(**args)


def _branch_result():
    from neurocomplexity.analysis.branching import BranchingResult
    ks = np.arange(1, 21)
    return BranchingResult(
        m=0.95, r_values=0.7 * 0.95 ** ks, k_lags=ks,
        r_squared=0.99, n_bins=5000, bin_size_seconds=0.004,
        populations=("all",), source=None, params={},
    )


def _pid_result():
    from neurocomplexity.analysis.pid import PIDResult
    import inspect
    fields = [f for f in inspect.signature(PIDResult.__init__).parameters
              if f != "self"]
    args = {}
    if "sources" in fields: args["sources"] = ("A", "B")
    if "target" in fields: args["target"] = "C"
    if "redundancy" in fields: args["redundancy"] = 0.1
    if "unique_1" in fields: args["unique_1"] = 0.2
    if "unique_2" in fields: args["unique_2"] = 0.15
    if "synergy" in fields: args["synergy"] = 0.05
    if "total_mi" in fields: args["total_mi"] = 0.5
    if "bin_size_seconds" in fields: args["bin_size_seconds"] = 0.005
    if "n_levels" in fields: args["n_levels"] = 3
    if "source" in fields: args["source"] = None
    if "params" in fields: args["params"] = {}
    return PIDResult(**args)


def test_panel_two_results_auto_layout():
    from neurocomplexity.viz._panel import figure_panel
    fig = figure_panel(_branch_result(), _pid_result())
    visible = [a for a in fig.axes if a.has_data() or a.patches or a.lines]
    assert len(visible) >= 2
    plt.close(fig)


def test_panel_three_results_get_letters_a_b_c():
    from neurocomplexity.viz._panel import figure_panel
    fig = figure_panel(_branch_result(), _pid_result(), _branch_result(),
                       panel_labels=True)
    letters_in_figure = []
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text() in string.ascii_lowercase[:5]:
                letters_in_figure.append(t.get_text())
    for letter in "abc":
        assert letter in letters_in_figure, f"{letter} missing"
    plt.close(fig)


def test_panel_explicit_labels():
    from neurocomplexity.viz._panel import figure_panel
    fig = figure_panel(_branch_result(), _pid_result(),
                       panel_labels=["i", "ii"])
    letters_in_figure = []
    for ax in fig.axes:
        for t in ax.texts:
            letters_in_figure.append(t.get_text())
    assert "i" in letters_in_figure
    assert "ii" in letters_in_figure
    plt.close(fig)


def test_panel_explicit_label_length_mismatch_raises():
    from neurocomplexity.viz._panel import figure_panel
    with pytest.raises(ValueError, match="length"):
        figure_panel(_branch_result(), _pid_result(),
                     panel_labels=["i"])
    plt.close("all")


def test_panel_unknown_result_type_raises():
    from neurocomplexity.viz._panel import figure_panel
    with pytest.raises(TypeError, match="result type|no figure function"):
        figure_panel(_branch_result(), "not_a_result")
    plt.close("all")


def test_panel_explicit_layout():
    from neurocomplexity.viz._panel import figure_panel
    fig = figure_panel(_branch_result(), _pid_result(), layout=(1, 2))
    visible = [a for a in fig.axes if a.has_data() or a.patches or a.lines]
    assert len(visible) >= 2
    plt.close(fig)
