"""Nature-style matplotlib defaults & save helpers.

Typography: Arial 7 pt body, 8 pt panel titles. Lines 0.8 pt, ticks inward.
Top/right spines off. SVG text editable, PDF uses TrueType (fonttype 42).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


# Restrained palette: one neutral, one signal, one accent + 4 categorical extras.
PALETTE = {
    "neutral":   "#2b2b2b",
    "muted":     "#7f7f7f",
    "signal":    "#1f6feb",  # blue
    "accent":    "#d6604d",  # warm red
    "ok":        "#2ca25f",  # green (gain/critical region)
    "warn":      "#e6a700",  # amber
    "fill":      "#cfe1f7",  # signal fill
    "categorical": ["#1f6feb", "#d6604d", "#2ca25f", "#e6a700",
                    "#7f3eb8", "#0fb5ae", "#b8336a"],
}


def apply_style() -> None:
    """Set the global rcParams to Nature-style defaults."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "figure.titlesize": 9,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save_publication(fig, path, *, formats: Iterable[str] = ("pdf", "svg", "png"),
                     dpi: int = 600) -> list[Path]:
    """Save ``fig`` to ``path`` in multiple formats. Returns list of paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        p = path.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=dpi)
        out.append(p)
    return out


def panel_label(ax, text: str, *, x: float = -0.18, y: float = 1.08) -> None:
    """Add bold panel label (a, b, c, ...) at the canonical top-left position."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")
