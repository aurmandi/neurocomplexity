"""Figures for LMC statistical complexity."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from neurocomplexity.analysis.complexity import LMCResult
from neurocomplexity.viz._palettes import DEFAULT_PALETTE, get_palette, series_styles
from neurocomplexity.viz._style import stats_box


def figure_lmc_complexity(result: LMCResult, *,
                          kind: str | None = None,
                          null_result=None,
                          ax=None,
                          palette: str = DEFAULT_PALETTE,
                          figsize: tuple[float, float] | None = None):
    """Plot the LMC complexity plane ``C = H · D`` (or a trajectory).

    Renders one of three views of an :class:`~neurocomplexity.analysis.LMCResult`:

    * ``kind="population"`` — scatter of ``(H, D)`` per population in the
      complexity plane, with ``C`` indicated by point size.
    * ``kind="trajectory"`` — time-coloured trajectory of ``(H(t), D(t))``
      from the sliding-window analysis. Requires ``result.H_traj`` to be
      populated.
    * ``kind="both"`` — both panels stacked.

    Parameters
    ----------
    result
        :class:`~neurocomplexity.analysis.LMCResult`.
    kind
        Override ``result.kind``. ``None`` → use ``result.kind``.
    null_result
        Optional :class:`~neurocomplexity.inference.InferenceResult` to
        overlay a null-distribution envelope around each population's
        complexity.
    ax
        Existing ``Axes`` to draw into.
    palette
        Palette name.
    figsize
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``kind`` is not a valid view, or ``kind="trajectory"`` is
        requested but the result has no trajectory.
    """
    k = kind or result.kind
    if k not in ("population", "trajectory", "both"):
        raise ValueError(f"kind must be one of population/trajectory/both; got {k!r}")
    if k == "trajectory" and result.H_traj is None:
        raise ValueError("result has no trajectory data; recompute with kind='trajectory' or 'both'")
    if k == "both" and result.H_traj is None:
        raise ValueError("result has no trajectory data for kind='both'; recompute with kind='both'")

    styles = series_styles(len(result.populations), palette)
    p = get_palette(palette)

    if k == "both":
        fig, axes = plt.subplots(1, 2, figsize=figsize or (8.5, 4.0))
        _draw_population(axes[0], result, styles, null_result, p,
                          show_legend=True)
        _draw_trajectory(axes[1], result, styles, p, show_legend=False)
        # Snapshot panel: points scatter mid-axes → top-right empty.
        box = f"bin = {result.bin_size_seconds*1e3:.0f} ms"
        if result.window_seconds:
            box += (f"\nwindow = {result.window_seconds:.0f} s"
                    f"\nstep = {result.step_seconds:.0f} s")
        stats_box(axes[0], box, corner="tr")
        # Trajectory panel: alpha-encoded time, dim→opaque (caption-level info)
        stats_box(axes[1], box, corner="tr")
        return fig

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (4.5, 4.0))
    else:
        fig = ax.figure

    if k == "population":
        _draw_population(ax, result, styles, null_result, p, show_legend=True)
    else:
        _draw_trajectory(ax, result, styles, p, show_legend=True)
    stats_box(ax, f"bin = {result.bin_size_seconds*1e3:.0f} ms", corner="tr")
    return fig


def _zoom_lims(values: np.ndarray, lo_default: float = 0.0,
               hi_default: float = 1.0, pad: float = 0.05) -> tuple[float, float]:
    """Auto-zoom that pads observed data range but stays inside [0, 1]."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (lo_default, hi_default)
    lo = max(lo_default, float(v.min()) - pad)
    hi = min(hi_default, float(v.max()) + pad)
    if hi - lo < 0.1:
        c = 0.5 * (lo + hi)
        lo, hi = max(lo_default, c - 0.05), min(hi_default, c + 0.05)
    return lo, hi


def _draw_population(ax, result: LMCResult, styles, null_result, p,
                     show_legend: bool = True):
    if null_result is not None:
        cloud = np.asarray(null_result.null_distribution)
        if cloud.ndim == 2 and cloud.shape[1] == result.C_per_pop.size:
            for pi in range(cloud.shape[1]):
                ax.scatter(np.full(cloud.shape[0], result.H_per_pop[pi]),
                           cloud[:, pi], s=8, color=p["muted"], alpha=0.4, zorder=1)
    for pi, name in enumerate(result.populations):
        ax.scatter([result.H_per_pop[pi]], [result.C_per_pop[pi]],
                   s=60, color=styles[pi]["color"], marker=styles[pi]["marker"],
                   label=name, zorder=3,
                   edgecolor=p["text"], linewidth=0.5)
    ax.set_xlabel("H (normalized Shannon entropy)", color=p["text"])
    ax.set_ylabel("C (LMC complexity)", color=p["text"])
    ax.set_xlim(*_zoom_lims(result.H_per_pop))
    if show_legend and len(result.populations) > 1:
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.18),
                  ncol=min(4, len(result.populations)))


def _draw_trajectory(ax, result: LMCResult, styles, p,
                     show_legend: bool = True):
    H = result.H_traj
    C = result.C_traj
    for pi, name in enumerate(result.populations):
        st = styles[pi]
        n = H.shape[0]
        for i in range(n - 1):
            a = 0.2 + 0.8 * (i / max(1, n - 1))
            ax.plot(H[i:i+2, pi], C[i:i+2, pi], color=st["color"],
                    linestyle=st["linestyle"], alpha=a, lw=1.0)
        ax.scatter(H[:, pi], C[:, pi], s=12, color=st["color"],
                   marker=st["marker"],
                   edgecolor=p["text"], linewidth=0.3, label=name)
    ax.set_xlabel("H (normalized Shannon entropy)", color=p["text"])
    ax.set_ylabel("C (LMC complexity)", color=p["text"])
    ax.set_xlim(*_zoom_lims(H))
    if show_legend and len(result.populations) > 1:
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.18),
                  ncol=min(4, len(result.populations)))
