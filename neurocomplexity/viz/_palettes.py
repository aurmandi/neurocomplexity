"""Three named palettes with semantic role assignments.

The palettes follow the colour blocks specified in the v1.1 spec
(``docs/specs/2026-05-18-visualization-rewrite.md``):

* forest — Ebony / Space Indigo / Palm Leaf / Lilac Ash
* wine   — Slate / Wine / Tan
* sage   — Slate Grey / Wine Plum / Dry Sage

Each palette assigns six semantic roles so all figure functions can ask the
palette for ``text``/``signal``/``accent``/``muted``/``fill``/``categorical``
without caring which scheme is active.
"""
from __future__ import annotations

import colorsys


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    )


def _lighten(hex_color: str, target_l: float = 0.75) -> str:
    """Push the HLS lightness of ``hex_color`` to ``target_l`` (0..1)."""
    r, g, b = _hex_to_rgb(hex_color)
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(h, target_l, s)
    return _rgb_to_hex(r2, g2, b2)


PALETTES: dict[str, dict] = {
    # Text is always black for maximum print legibility, regardless of palette.
    "forest": {
        "text":   "#000000",
        "signal": "#2C2A4A",
        "accent": "#A6A867",
        "muted":  "#9D91A3",
        "fill":   _lighten("#9D91A3", target_l=0.80),
        "categorical": ["#2C2A4A", "#A6A867", "#9D91A3", "#51513D"],
    },
    "wine": {
        "text":   "#000000",
        "signal": "#66232A",
        "accent": "#C39B60",
        "muted":  "#60566B",
        "fill":   _lighten("#C39B60", target_l=0.80),
        "categorical": ["#66232A", "#C39B60", "#60566B"],
    },
    "sage": {
        "text":   "#000000",
        "signal": "#723D46",
        "accent": "#C9CBA3",
        "muted":  "#76818E",
        "fill":   _lighten("#C9CBA3", target_l=0.85),
        "categorical": ["#723D46", "#C9CBA3", "#76818E"],
    },
}


DEFAULT_PALETTE = "forest"


def get_palette(name: str) -> dict:
    if name not in PALETTES:
        raise KeyError(
            f"unknown palette {name!r}; choose from {sorted(PALETTES)}"
        )
    return PALETTES[name]


def diverging_cmap(name: str = DEFAULT_PALETTE):
    """Return a LinearSegmentedColormap for population heatmaps."""
    from matplotlib.colors import LinearSegmentedColormap
    p = get_palette(name)
    return LinearSegmentedColormap.from_list(
        f"nc_{name}_div", [p["fill"], "#FFFFFF", p["signal"]]
    )
