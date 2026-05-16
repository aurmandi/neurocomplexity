"""Avalanche shape collapse figure.

Left: raw mean shapes coloured by duration (sequential family).
Right: shapes after dividing by T^(gamma-1) — successful collapse onto one
universal function F(t/T).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from neurocomplexity.viz._style import PALETTE, panel_label


def figure_shape_collapse(result, *, fig=None):
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.2))
    else:
        axes = fig.subplots(1, 2)
    ax_raw, ax_col = axes

    durations = np.asarray(result.durations_used)
    # Sequential colormap for ordered durations
    cmap = cm.get_cmap("viridis")
    n = len(durations)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    # Raw shapes: native time axis (bins → seconds)
    bs = result.bin_size_seconds
    for T, shape, c in zip(durations, result.mean_shapes, colors):
        t = np.arange(T) * bs * 1e3   # ms
        ax_raw.plot(t, shape, lw=0.8, color=c)
    ax_raw.set_xlabel("Time within avalanche (ms)")
    ax_raw.set_ylabel("Mean activity")
    ax_raw.text(0.02, 0.98,
                f"{n} duration classes\nT in [{durations.min()},{durations.max()}] bins",
                transform=ax_raw.transAxes, va="top", fontsize=6,
                color=PALETTE["muted"])

    # Collapsed shapes on u = t/T
    u = result.rescaled_x
    for y, c in zip(result.rescaled_y, colors):
        ax_col.plot(u, y, lw=0.8, color=c, alpha=0.85)
    ax_col.set_xlabel(r"$t/T$")
    ax_col.set_ylabel(r"$\langle a\rangle\,/\,T^{\gamma-1}$")
    ax_col.text(0.98, 0.97,
                fr"$\gamma={result.gamma:.3f}$" + "\n" +
                f"resid$={result.residual:.2g}$",
                transform=ax_col.transAxes, ha="right", va="top", fontsize=6,
                color=PALETTE["muted"])

    # Colorbar for duration ordering
    sm = cm.ScalarMappable(cmap=cmap,
                            norm=plt.Normalize(vmin=durations.min(),
                                                vmax=durations.max()))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_col, fraction=0.04, pad=0.03)
    cb.set_label("Duration T (bins)", fontsize=6)
    cb.ax.tick_params(labelsize=6)

    panel_label(ax_raw, "a")
    panel_label(ax_col, "b")
    fig.tight_layout()
    return fig
