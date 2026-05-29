"""Publication-quality matplotlib rcParams + palette switching.

Provides ``apply_style(palette)``, ``set_palette(name)``, ``current_palette()``,
and re-exports ``PALETTES``. Calling ``apply_style`` propagates the palette's
``text`` colour to every axis-chrome rcParam so the palette extends to
spines and tick labels, not just data colours.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from neurocomplexity.viz._palettes import (
    DEFAULT_PALETTE,
    get_palette,
)

_CURRENT = {"name": DEFAULT_PALETTE}


def current_palette() -> dict:
    """Return the role-keyed dict of the currently-active palette.

    See :func:`~neurocomplexity.viz._palettes.get_palette` for the schema.
    """
    return get_palette(_CURRENT["name"])


def apply_style(palette: str = DEFAULT_PALETTE) -> None:
    """Set matplotlib rcParams to publication defaults using ``palette``."""
    p = get_palette(palette)
    _CURRENT["name"] = palette
    text = p["text"]
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
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "figure.dpi": 120,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.04,
        "figure.constrained_layout.w_pad": 0.04,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "text.color": text,
        "axes.labelcolor": text,
        "axes.edgecolor": text,
        "xtick.color": text,
        "ytick.color": text,
    })


def set_palette(name: str) -> None:
    """Switch the active palette and re-apply publication ``rcParams``.

    Equivalent to ``apply_style(palette=name)``. Subsequent figure
    functions that pass ``palette=...`` will still override on a per-call
    basis.
    """
    apply_style(palette=name)


class _LegacyPaletteAccessor(dict):
    """Backward-compat shim: maps legacy flat-dict keys to new palette roles."""
    def __getitem__(self, key):
        p = current_palette()
        if key in p:
            return p[key]
        mapping = {"ok": p["accent"], "warn": p["accent"], "neutral": p["text"]}
        if key in mapping:
            return mapping[key]
        raise KeyError(key)


PALETTE = _LegacyPaletteAccessor()


apply_style(palette=DEFAULT_PALETTE)


def panel_label(ax, letter: str, *, x: float = -0.07, y: float = 1.02) -> None:
    """Add a bold panel letter (a, b, c, ...) to the upper-left of an axes."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold",
            ha="right", va="bottom",
            color=current_palette()["text"])


def _resolve_palette_and_axes(*, palette, ax, figsize, default_size):
    """Common entry-point boilerplate shared by every figure_X function."""
    p = get_palette(palette)
    if ax is None:
        size = figsize if figsize is not None else default_size
        fig, ax = plt.subplots(figsize=size)
    else:
        fig = ax.figure
    return p, fig, ax


def _apply_panel_label(ax, label):
    if label is None:
        return
    panel_label(ax, label)
