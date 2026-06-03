"""PID atoms: stacked bar in canonical order R / U_X / U_Y / S."""
from __future__ import annotations

import numpy as np

from neurocomplexity.viz._palettes import DEFAULT_PALETTE
from neurocomplexity.viz._style import (
    _apply_panel_label, _resolve_palette_and_axes, stats_box,
)


def figure_pid(
    result,
    *,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
):
    """Render Williams-Beer PID atoms as a labelled bar.

    Bars in canonical order: redundancy, unique source 1, unique source 2,
    synergy. Source names are read from ``result.sources``.

    See :func:`~neurocomplexity.viz.branching.figure_branching` for shared
    keyword arguments.
    """
    p, fig, ax = _resolve_palette_and_axes(
        palette=palette, ax=ax, figsize=figsize, default_size=(3.6, 2.7),
    )

    s1, s2 = result.sources
    labels = ["Redundancy", f"Unique\n({s1})", f"Unique\n({s2})", "Synergy"]
    vals = np.array([result.redundancy, result.unique_1,
                     result.unique_2, result.synergy])
    bar_colors = [p["muted"], p["signal"], p["accent"], p["categorical"][-1]]
    x = np.arange(len(vals))
    ax.bar(x, vals, color=bar_colors, edgecolor="none", width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Information about {result.target} (nats)")
    ymax = max(vals.max(), 1e-12)
    # Headroom for value labels at the top + the corner stats annotation.
    ax.set_ylim(0, ymax * 1.4)
    for xi, v in zip(x, vals):
        # Print "≈ 0" for atoms that round to zero so the reader knows the
        # bar isn't missing — it's empty by computation.
        txt = "≈ 0" if abs(v) < ymax * 1e-3 else f"{v:.4f}"
        ax.text(xi, v + ymax * 0.04, txt,
                ha="center", va="bottom", fontsize=6,
                color=p["text"])
    # Bars at left/centre, value labels at top → top-RIGHT clear.
    stats_box(
        ax,
        f"total MI = {result.total_mi:.4f}\n"
        f"bin = {result.bin_size_seconds * 1e3:.1f} ms",
        corner="tr",
    )
    _apply_panel_label(ax, panel_label)
    return fig
