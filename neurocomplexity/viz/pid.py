"""PID atoms: labelled bar in canonical order R / U_X / U_Y / S.

Colour mapping follows the convention used in the PID literature
(Wibral, Lizier & Priesemann 2014; Pope et al. 2021; Luppi et al. 2022):
redundancy is blue (shared by both sources), the two unique atoms are
green and orange (private to each source), and synergy is vermilion
(emergent — present only when sources are observed jointly). The
Okabe-Ito set provides the four colourblind-safe hues.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from neurocomplexity.viz._palettes import DEFAULT_PALETTE, get_palette
from neurocomplexity.viz._style import _apply_panel_label

# Atom colours (Okabe-Ito; Wibral 2014 / Luppi 2022 mapping).
_PID_COLORS = {
    "redundancy": "#0072B2",   # blue   — shared information
    "unique_1":   "#009E73",   # green  — private to source 1
    "unique_2":   "#E69F00",   # orange — private to source 2
    "synergy":    "#D55E00",   # vermilion — emergent / synergistic
}


def figure_pid(
    result,
    *,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = "Partial Information Decomposition",
    ax=None,
):
    """Render Williams-Beer PID atoms as a labelled bar chart.

    Bars in canonical order: redundancy, unique source 1, unique source 2,
    synergy. Source names are read from ``result.sources``. Atom colours
    follow the PID convention (R=blue, U=green/orange, S=vermilion).

    Parameters
    ----------
    title
        Bold sans-serif suptitle above the panel. ``None`` suppresses it.

    See :func:`~neurocomplexity.viz.branching.figure_branching` for other
    shared keyword arguments.
    """
    p = get_palette(palette)
    if ax is None:
        size = figsize if figsize is not None else (3.8, 2.8)
        fig, ax = plt.subplots(figsize=size)
    else:
        fig = ax.figure

    s1, s2 = result.sources
    labels = ["Redundancy", f"Unique\n({s1})", f"Unique\n({s2})", "Synergy"]
    vals = np.array([result.redundancy, result.unique_1,
                     result.unique_2, result.synergy])
    bar_colors = [_PID_COLORS["redundancy"], _PID_COLORS["unique_1"],
                  _PID_COLORS["unique_2"], _PID_COLORS["synergy"]]
    x = np.arange(len(vals))
    ax.bar(x, vals, color=bar_colors, edgecolor="none", width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Information about {result.target} (nats)")
    ymax = max(vals.max(), 1e-12)
    # Headroom for per-bar value labels.
    ax.set_ylim(0, ymax * 1.25)
    for xi, v in zip(x, vals):
        # Print "≈ 0" for atoms that round to zero so the reader knows the
        # bar isn't missing — it's empty by computation.
        txt = "≈ 0" if abs(v) < ymax * 1e-3 else f"{v:.4f}"
        ax.text(xi, v + ymax * 0.03, txt,
                ha="center", va="bottom", fontsize=6,
                color=p["text"])
    if title:
        fig.suptitle(title, fontweight="bold", fontsize=9)
    _apply_panel_label(ax, panel_label)
    return fig
