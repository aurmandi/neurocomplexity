"""PID atoms: stacked bar in canonical order R / U_X / U_Y / S."""
from __future__ import annotations

import numpy as np

from neurocomplexity.viz._palettes import get_palette, DEFAULT_PALETTE
from neurocomplexity.viz._style import _resolve_palette_and_axes, _apply_panel_label


def figure_pid(
    result,
    *,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
):
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
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel(f"Information about {result.target} (nats)")
    ymax = max(vals.max(), 1e-12)
    ax.set_ylim(0, ymax * 1.25)
    for xi, v in zip(x, vals):
        ax.text(xi, v + ymax * 0.03, f"{v:.4f}",
                ha="center", va="bottom", fontsize=5.5,
                color=p["text"])
    ax.text(0.98, 0.97,
            f"total MI = {result.total_mi:.4f}\n"
            f"bin = {result.bin_size_seconds * 1e3:.1f} ms",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6, color=p["text"])
    _apply_panel_label(ax, panel_label)
    return fig
