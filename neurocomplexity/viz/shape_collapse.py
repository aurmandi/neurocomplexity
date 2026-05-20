"""Avalanche shape collapse: raw shape family + collapsed onto F(t/T)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from neurocomplexity.viz._palettes import get_palette, DEFAULT_PALETTE
from neurocomplexity.viz._style import _apply_panel_label


# Viridis is the matplotlib default perceptually-uniform sequential colormap
# (Smith & van der Walt 2015) and the de-facto standard in the avalanche shape-
# collapse literature (Fontenele et al. 2019; Ponce-Alvarez et al. 2018). It is
# colour-blind safe, monotonic in luminance, and prints to greyscale faithfully
# — properties no hand-tuned palette can match. Decoupled from the figure
# palette by design.
SHAPE_COLLAPSE_CMAP = "viridis"


def _sequential_cmap(p):  # kept for backward compat with callers / tests
    return plt.get_cmap(SHAPE_COLLAPSE_CMAP)


def _draw_collapsed(ax, result, *, p, cmap, colors, show_annot=True):
    u = result.rescaled_x
    for y, c in zip(result.rescaled_y, colors):
        ax.plot(u, y, lw=0.9, color=c, alpha=0.85)
    ax.set_xlabel(r"$t/T$")
    ax.set_ylabel(r"$\langle a\rangle / T^{\gamma-1}$")
    if show_annot:
        ax.text(0.98, 0.97,
                fr"$\gamma={result.gamma:.3f}$" + "\n" +
                f"resid$={result.residual:.2g}$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6, color=p["text"])


def figure_shape_collapse(
    result,
    *,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
):
    p = get_palette(palette)
    durations = np.asarray(result.durations_used)
    n = len(durations)
    cmap = plt.get_cmap(SHAPE_COLLAPSE_CMAP)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    composite = ax is not None

    if composite:
        fig = ax.figure
        _draw_collapsed(ax, result, p=p, cmap=cmap, colors=colors)
        _apply_panel_label(ax, panel_label)
        return fig

    size = figsize if figsize is not None else (5.8, 2.6)
    fig, (ax_raw, ax_col) = plt.subplots(1, 2, figsize=size)

    bs = result.bin_size_seconds
    for T, shape, c in zip(durations, result.mean_shapes, colors):
        t = np.arange(T) * bs * 1e3
        ax_raw.plot(t, shape, lw=0.9, color=c)
    ax_raw.set_xlabel("Time within avalanche (ms)")
    ax_raw.set_ylabel("Mean activity")

    _draw_collapsed(ax_col, result, p=p, cmap=cmap, colors=colors)

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=durations.min(), vmax=durations.max()),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_col, fraction=0.045, pad=0.04)
    cb.set_label("Duration $T$ (bins)", fontsize=6)
    cb.ax.tick_params(labelsize=6)

    _apply_panel_label(ax_raw, panel_label)
    return fig
