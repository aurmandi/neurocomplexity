"""PID atom bar chart: redundancy / unique_1 / unique_2 / synergy."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from neurocomplexity.viz._style import PALETTE, panel_label


def figure_pid(result, *, fig=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(3.0, 2.2))
    else:
        ax = fig.subplots(1, 1)
    s1, s2 = result.sources
    labels = ["Redundancy", f"Unique\n({s1})", f"Unique\n({s2})", "Synergy"]
    vals = np.array([result.redundancy, result.unique_1,
                     result.unique_2, result.synergy])
    colors = [PALETTE["muted"], PALETTE["signal"],
              PALETTE["ok"], PALETTE["accent"]]
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, edgecolor="none", width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel(f"Information about {result.target} (nats)")
    ymax = max(vals.max(), 1e-12)
    ax.set_ylim(0, ymax * 1.25)
    for xi, v in zip(x, vals):
        ax.text(xi, v + ymax * 0.03, f"{v:.4f}",
                ha="center", va="bottom", fontsize=5.5,
                color=PALETTE["neutral"])
    ax.text(0.98, 0.97,
            f"total MI = {result.total_mi:.4f}\nbin = {result.bin_size_seconds*1e3:.1f} ms",
            transform=ax.transAxes, ha="right", va="top", fontsize=6,
            color=PALETTE["muted"])
    panel_label(ax, "a", x=-0.22)
    fig.tight_layout()
    return fig
