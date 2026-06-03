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
        # Journal-standard sans-serif stack. Arial / Helvetica is the
        # de-facto Nature / Cell / Science figure typeface. We list common
        # near-clones first so the fallback chain lands on a Helvetica-class
        # face wherever the figure is rendered.
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Segoe UI", "Helvetica Neue", "Helvetica", "Arial",
            "Liberation Sans", "Nimbus Sans", "TeX Gyre Heros",
            "DejaVu Sans", "sans-serif",
        ],
        "font.weight": "regular",
        # Sans-serif math so Greek / italic variables blend with body text
        # (Nature figure convention). 'regular' default keeps math glyphs at
        # the body weight rather than bold.
        "mathtext.fontset": "stixsans",
        "mathtext.default": "regular",
        # Sizing hierarchy — Nature minimum is 7 pt. We use 7 / 7.5 / 8.5.
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.titlepad": 6.0,
        "axes.labelsize": 7,
        "axes.labelweight": "regular",
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "legend.title_fontsize": 6.5,
        # Editor-friendly outputs: keep text as text in SVG / PDF.
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Tufte chrome: no top / right spines, hairline axes, outward ticks.
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        # No grid by default — Tufte's data-ink ratio. Opt-in via ax.grid.
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.color": "#EEEEEE",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderpad": 0.3,
        "legend.columnspacing": 1.4,
        "figure.dpi": 120,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.04,
        "figure.constrained_layout.w_pad": 0.04,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
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


def stats_box(ax, text: str, *, corner: str = "tr",
              fontsize: float = 9.0) -> None:
    """In-panel parameters box.

    Renders a small white rounded-rect at one corner of the axes carrying
    fit parameters / bin sizes / counts — never the analysis name (panel
    label + caption already say what the panel is). Text is always
    left-justified inside the box so ``param = value`` columns read left
    to right, regardless of which corner the box anchors to.

    Parameters
    ----------
    ax
        Target axes.
    text
        Multi-line annotation. Math symbols (``$\\tau_{max}$`` etc.) render
        via mathtext; plain body text uses the current sans-serif family.
    corner
        ``"tr"`` (default), ``"tl"``, ``"br"``, ``"bl"`` — which axes
        corner the box sits in. Pick the emptiest corner for the panel
        type (decay/scatter -> ``"tr"``; rising/cumulative -> ``"tl"``).
    fontsize
        Body font size in points. Default 9.0 (per package convention).
    """
    anchor = {
        "tr": (0.97, 0.97, "right", "top"),
        "tl": (0.03, 0.97, "left",  "top"),
        "br": (0.97, 0.03, "right", "bottom"),
        "bl": (0.03, 0.03, "left",  "bottom"),
    }[corner]
    x, y, ha, va = anchor
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha=ha, va=va,
        multialignment="left",      # left-to-right inside the box
        fontsize=fontsize,
        color=current_palette()["text"],
        linespacing=1.35,
        zorder=10,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white", edgecolor="#cccccc",
            linewidth=0.5, alpha=0.85,
        ),
    )


def top_strip(ax, text: str, *, pad: float = 6.0) -> None:
    """Deprecated alias for :func:`stats_box` (kept for back-compat).

    Use :func:`stats_box` directly in new code. The previous top-left
    title-strip placement is replaced by an in-panel box because the
    strip ate vertical space and collided with panel letters / suptitles.
    """
    stats_box(ax, text, corner="tr")


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
