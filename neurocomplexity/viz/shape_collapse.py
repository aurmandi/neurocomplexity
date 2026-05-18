"""Avalanche shape collapse: raw shape family + collapsed onto F(t/T)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from neurocomplexity.viz._palettes import get_palette, DEFAULT_PALETTE
from neurocomplexity.viz._style import _apply_panel_label


def figure_shape_collapse(
    result,
    *,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
):
    p = get_palette(palette)
    if ax is None:
        size = figsize if figsize is not None else (4.6, 2.2)
        fig, axes = plt.subplots(1, 2, figsize=size)
    else:
        fig = ax.figure
        ax.set_axis_off()
        gs = ax.get_subplotspec().subgridspec(1, 2)
        axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    ax_raw, ax_col = axes

    durations = np.asarray(result.durations_used)
    n = len(durations)
    cmap = LinearSegmentedColormap.from_list("nc_seq", [p["fill"], p["signal"]])
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    bs = result.bin_size_seconds
    for T, shape, c in zip(durations, result.mean_shapes, colors):
        t = np.arange(T) * bs * 1e3
        ax_raw.plot(t, shape, lw=0.8, color=c)
    ax_raw.set_xlabel("Time within avalanche (ms)")
    ax_raw.set_ylabel("Mean activity")

    u = result.rescaled_x
    for y, c in zip(result.rescaled_y, colors):
        ax_col.plot(u, y, lw=0.8, color=c, alpha=0.85)
    ax_col.set_xlabel(r"$t/T$")
    ax_col.set_ylabel(r"$\langle a\rangle / T^{\gamma-1}$")
    ax_col.text(0.98, 0.97,
                fr"$\gamma={result.gamma:.3f}$" + "\n" +
                f"resid$={result.residual:.2g}$",
                transform=ax_col.transAxes, ha="right", va="top",
                fontsize=6, color=p["muted"])

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=durations.min(), vmax=durations.max()),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_col, fraction=0.04, pad=0.03)
    cb.set_label("Duration T (bins)", fontsize=6)
    cb.ax.tick_params(labelsize=6)

    _apply_panel_label(ax_raw, panel_label)
    fig.tight_layout()
    return fig
