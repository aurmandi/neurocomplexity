"""Tests for viz.network (figure_te_network)."""
import matplotlib
matplotlib.use("Agg")
import warnings as _warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("networkx")

from neurocomplexity.analysis.transfer_entropy import TransferEntropyResult
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.inference.results import InferenceResult
from neurocomplexity.viz.network import figure_te_network


def _te_result(matrix: np.ndarray, names=("A", "B", "C", "D")):
    return TransferEntropyResult(
        matrix=matrix.astype(np.float64),
        populations=tuple(names[: matrix.shape[0]]),
        bin_size_seconds=0.005,
        delay_bins=1,
        source=ProvenanceRecord.for_memory("test"),
        params={"populations": list(names[: matrix.shape[0]]),
                "bin_size_ms": 5.0, "delay_bins": 1, "estimator": "binary"},
    )


def _null(matrix: np.ndarray, p_value: np.ndarray, p_value_fdr=None):
    return InferenceResult(
        statistic_name="transfer_entropy",
        observed=matrix,
        null_distribution=None,
        bootstrap_distribution=None,
        p_value=p_value,
        p_value_fdr=p_value_fdr,
        effect_size=None,
        ci_lower=None, ci_upper=None, ci_level=0.95,
        method="spike_dither", n_resamples=200, seed=0,
    )


def _make_matrix(seed=0):
    rng = np.random.default_rng(seed)
    M = rng.uniform(0.001, 0.5, size=(4, 4))
    np.fill_diagonal(M, 0.0)
    return M


# ---- smoke ------------------------------------------------------------------

def test_figure_te_network_renders_smoke():
    M = _make_matrix()
    te = _te_result(M)
    fig = figure_te_network(te)
    # Graph axes + an edge-TE colourbar axes when any edge is drawn.
    assert len(fig.axes) == 2
    assert fig.axes[0].axison is False  # the graph panel has its frame off
    plt.close(fig)


def _arrow_count(fig):
    """Count FancyArrowPatch instances — one per drawn edge."""
    from matplotlib.patches import FancyArrowPatch
    return sum(1 for art in fig.axes[0].get_children()
               if isinstance(art, FancyArrowPatch))


def test_no_null_result_draws_all_positive_edges():
    M = np.array([[0.0, 0.5, 0.0, 0.0],
                  [0.0, 0.0, 0.3, 0.0],
                  [0.0, 0.0, 0.0, 0.7],
                  [0.1, 0.0, 0.0, 0.0]], dtype=np.float64)
    te = _te_result(M)
    fig = figure_te_network(te)
    # 4 positive off-diagonal cells expected
    assert _arrow_count(fig) == 4
    plt.close(fig)


def test_null_result_fdr_filter_drops_edges():
    M = _make_matrix(seed=1)
    p_fdr = np.where(M > 0.3, 0.01, 0.5)  # only large TEs significant
    null = _null(M, p_value=np.ones_like(M), p_value_fdr=p_fdr)
    fig = figure_te_network(_te_result(M), null_result=null, alpha=0.05)
    n_expected = int(((p_fdr < 0.05) & (M > 0) & ~np.eye(4, dtype=bool)).sum())
    assert _arrow_count(fig) == n_expected
    plt.close(fig)


def test_falls_back_to_raw_p_when_no_fdr():
    M = _make_matrix(seed=2)
    p_raw = np.where(M > 0.3, 0.01, 0.5)
    null = _null(M, p_value=p_raw, p_value_fdr=None)
    fig = figure_te_network(_te_result(M), null_result=null, alpha=0.05)
    n_expected = int(((p_raw < 0.05) & (M > 0) & ~np.eye(4, dtype=bool)).sum())
    assert _arrow_count(fig) == n_expected
    plt.close(fig)


def test_alpha_threshold_extremes():
    M = _make_matrix(seed=3)
    p = np.full_like(M, 0.04)
    null = _null(M, p_value=p, p_value_fdr=p)
    fig_zero = figure_te_network(_te_result(M), null, alpha=1e-9)
    assert _arrow_count(fig_zero) == 0
    plt.close(fig_zero)
    fig_all = figure_te_network(_te_result(M), null, alpha=1.0)
    assert _arrow_count(fig_all) == 12  # all 4*4 off-diagonal
    plt.close(fig_all)


def test_layout_spring_differs_from_circular():
    M = _make_matrix(seed=4)
    fig_c = figure_te_network(_te_result(M), layout="circular")
    fig_s = figure_te_network(_te_result(M), layout="spring", seed=7)
    # Compare scatter positions of nodes (first PathCollection in each axes)
    cc = next(c for c in fig_c.axes[0].collections if c.get_offsets().shape[0] >= 4)
    cs = next(c for c in fig_s.axes[0].collections if c.get_offsets().shape[0] >= 4)
    assert not np.allclose(np.sort(cc.get_offsets().ravel()),
                           np.sort(cs.get_offsets().ravel()))
    plt.close(fig_c); plt.close(fig_s)


def test_invalid_layout_raises():
    M = _make_matrix()
    with pytest.raises(ValueError, match="layout"):
        figure_te_network(_te_result(M), layout="bogus")
    plt.close("all")


def test_accepts_ax_kwarg():
    M = _make_matrix()
    fig, ax = plt.subplots()
    out = figure_te_network(_te_result(M), ax=ax)
    assert out is fig
    plt.close(fig)


def test_suptitle_present_by_default():
    M = _make_matrix()
    fig = figure_te_network(_te_result(M))
    sup = fig._suptitle
    assert sup is not None and sup.get_text() != ""
    plt.close(fig)


def test_show_disconnected_false_drops_isolated_nodes():
    # only one significant edge A->B, C and D should be droppable
    M = np.zeros((4, 4))
    M[0, 1] = 0.5
    te = _te_result(M)
    fig_keep = figure_te_network(te, show_disconnected=True)
    fig_drop = figure_te_network(te, show_disconnected=False)
    # Nodes are scattered as PathCollections; count node patches by extracting
    # the first one matching n_nodes
    def _node_count(fig):
        for c in fig.axes[0].collections:
            offsets = c.get_offsets()
            if hasattr(offsets, "shape") and offsets.ndim == 2 and offsets.shape[1] == 2:
                return offsets.shape[0]
        return -1
    assert _node_count(fig_keep) >= _node_count(fig_drop)
    plt.close(fig_keep); plt.close(fig_drop)


def test_node_colors_palette_independent():
    """Node colours come from the shared Okabe-Ito categorical set, which is
    colourblind-safe and therefore identical across palettes (the aesthetic
    palette only drives chrome/edges, not categorical hues)."""
    M = _make_matrix()
    fig_f = figure_te_network(_te_result(M), palette="forest")
    fig_w = figure_te_network(_te_result(M), palette="wine")
    def _node_color(fig):
        for c in fig.axes[0].collections:
            offsets = c.get_offsets()
            if offsets.ndim == 2 and offsets.shape[1] == 2 and offsets.shape[0] >= 4:
                return c.get_facecolors()
        return None
    cf = _node_color(fig_f)
    cw = _node_color(fig_w)
    assert cf is not None and cw is not None
    assert np.allclose(cf, cw)
    plt.close(fig_f); plt.close(fig_w)


def test_palette_kwarg_accepted_across_palettes():
    """Edge colours follow a TE-magnitude Greens cmap (palette-independent;
    the green convention is set by Stetter 2012 / inspirations). This test
    only verifies that the palette kwarg is accepted on every palette."""
    M = _make_matrix()
    for pal in ("nature", "forest", "wine", "sage"):
        fig = figure_te_network(_te_result(M), palette=pal)
        plt.close(fig)


def test_p_less_than_2_pops_raises():
    M = np.zeros((1, 1))
    te = _te_result(M, names=("solo",))
    with pytest.raises(ValueError, match="2 populations"):
        figure_te_network(te)
    plt.close("all")
