"""Avalanche shape collapse: raw shape family + collapsed onto F(t/T)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from neurocomplexity.viz._palettes import DEFAULT_PALETTE, get_palette
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
    ys = np.asarray(result.rescaled_y)  # shape (n_durations, n_interp)
    for y, c in zip(ys, colors):
        ax.plot(u, y, lw=0.7, color=c, alpha=0.55)
    # Median collapse curve + IQR band — what readers actually judge quality
    # against (Friedman et al. 2012; Fontenele et al. 2019).
    if ys.size:
        med = np.nanmedian(ys, axis=0)
        q1 = np.nanpercentile(ys, 25, axis=0)
        q3 = np.nanpercentile(ys, 75, axis=0)
        ax.fill_between(u, q1, q3, color=p["accent"], alpha=0.25, linewidth=0,
                        zorder=3, label="IQR")
        ax.plot(u, med, "-", color=p["accent"], lw=1.6, zorder=4,
                label="median")
        ax.legend(loc="lower right", fontsize=6, handlelength=1.6,
                  frameon=False)
        # Robust y-limits: percentile clip so outlier shapes don't dominate.
        finite = ys[np.isfinite(ys)]
        if finite.size:
            lo = float(np.percentile(finite, 1))
            hi = float(np.percentile(finite, 99))
            if hi > lo:
                ax.set_ylim(lo, hi)
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
    """Render mean avalanche shapes before / after rescaling by ``gamma``.

    Two sub-panels: raw mean shapes per duration class, and collapsed
    curves at the optimal ``gamma``. Residual is annotated.

    See :func:`~neurocomplexity.viz.branching.figure_branching` for shared
    keyword arguments.
    """
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
    # Track raw values for robust ylim clipping.
    all_vals: list[float] = []
    for T, shape, c in zip(durations, result.mean_shapes, colors):
        t = np.arange(T) * bs * 1e3
        ax_raw.plot(t, shape, lw=0.7, color=c, alpha=0.7)
        all_vals.extend(np.asarray(shape, dtype=float).ravel().tolist())
    ax_raw.set_xlabel("Time within avalanche (ms)")
    ax_raw.set_ylabel("Mean activity")
    # Robust y-limits suppress single-shape spikes (e.g. one avalanche with a
    # giant burst) that otherwise compress the bulk of curves.
    if all_vals:
        arr = np.array(all_vals)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            lo, hi = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
            if hi > lo:
                ax_raw.set_ylim(lo, hi)

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
