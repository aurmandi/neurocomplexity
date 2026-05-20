"""add_scale_bar: replace explicit axis labels with a scale bar."""
from __future__ import annotations

from typing import Literal

import matplotlib.axes
import matplotlib.lines as mlines


_LOCATIONS = {
    "lower right": (0.95, 0.05),
    "lower left":  (0.05, 0.05),
    "upper right": (0.95, 0.95),
    "upper left":  (0.05, 0.95),
}


def _strip_axis(ax: matplotlib.axes.Axes, which: Literal["x", "y"]) -> None:
    if which == "x":
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        ax.set_xlabel("")
    else:
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.set_ylabel("")


def add_scale_bar(
    ax: matplotlib.axes.Axes,
    *,
    x_length: float | None = None,
    x_label: str | None = None,
    y_length: float | None = None,
    y_label: str | None = None,
    location: str = "lower right",
    color: str | None = None,
    linewidth: float = 1.4,
    pad: float = 0.04,
) -> None:
    """Add a scale bar to ``ax`` and strip the corresponding axis chrome."""
    if x_length is None and y_length is None:
        raise ValueError("provide x_length and/or y_length")
    if location not in _LOCATIONS:
        raise ValueError(
            f"unknown location {location!r}; choose from {sorted(_LOCATIONS)}"
        )

    if color is None:
        from neurocomplexity.viz._style import current_palette
        color = current_palette()["text"]

    x0_frac, y0_frac = _LOCATIONS[location]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Use signed spans so inverted axes (e.g. imshow with origin='upper') still
    # place anchors and label offsets on the correct side of the data.
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    y_inverted = ax.yaxis_inverted()
    # For inverted y, "lower" in display coords corresponds to ylim[0] (the
    # numerically larger value), and the bar must grow toward smaller y values.
    y_sign = -1.0 if y_inverted else 1.0

    anchor_x = xlim[0] + x0_frac * xspan
    anchor_y = ylim[0] + y0_frac * yspan

    if x_length is not None:
        _strip_axis(ax, "x")
        if "right" in location:
            x_start, x_end = anchor_x - x_length, anchor_x
        else:
            x_start, x_end = anchor_x, anchor_x + x_length
        bar = mlines.Line2D(
            [x_start, x_end], [anchor_y, anchor_y],
            color=color, linewidth=linewidth, solid_capstyle="butt",
            clip_on=False,
        )
        ax.add_line(bar)
        if x_label is not None:
            label_y = anchor_y - pad * yspan
            label_x = 0.5 * (x_start + x_end)
            ax.text(label_x, label_y, x_label,
                    color=color, ha="center", va="top", fontsize=6.5)

    if y_length is not None:
        _strip_axis(ax, "y")
        # On an inverted y-axis (imshow), "upper" in display means smaller data
        # y; the y_sign flip keeps the bar growing toward the data interior.
        signed_len = y_sign * y_length
        if "upper" in location:
            y_start, y_end = anchor_y - signed_len, anchor_y
        else:
            y_start, y_end = anchor_y, anchor_y + signed_len
        bar = mlines.Line2D(
            [anchor_x, anchor_x], [y_start, y_end],
            color=color, linewidth=linewidth, solid_capstyle="butt",
            clip_on=False,
        )
        ax.add_line(bar)
        if y_label is not None:
            label_x = anchor_x - pad * xspan
            label_y = 0.5 * (y_start + y_end)
            ax.text(label_x, label_y, y_label,
                    color=color, ha="right", va="center", fontsize=6.5,
                    rotation=90)
