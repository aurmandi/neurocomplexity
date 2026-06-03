"""Named palettes with semantic role assignments.

* nature (default) — Nature/Cell visual identity: black chrome, Okabe-Ito
  blue signal, Okabe-Ito vermilion accent, neutral greys
* forest — Ebony / Space Indigo / Palm Leaf / Lilac Ash
* wine   — Slate / Wine / Tan
* sage   — Slate Grey / Wine Plum / Dry Sage

Each palette assigns six semantic roles so all figure functions can ask the
palette for ``text``/``signal``/``accent``/``muted``/``fill``/``categorical``
without caring which scheme is active. The categorical cycle is the
colourblind-safe Okabe-Ito set across every palette.
"""
from __future__ import annotations

import colorsys


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return f"#{int(round(r * 255)):02X}{int(round(g * 255)):02X}{int(round(b * 255)):02X}"


def _lighten(hex_color: str, target_l: float = 0.75) -> str:
    """Push the HLS lightness of ``hex_color`` to ``target_l`` (0..1)."""
    r, g, b = _hex_to_rgb(hex_color)
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(h, target_l, s)
    return _rgb_to_hex(r2, g2, b2)


# Colorblind-safe categorical colours (Okabe & Ito 2008,
# https://jfly.uni-koeln.de/color/). Deuteranopia/protanopia/tritanopia
# safe. Pure black is omitted (reserved for text/axes) and the low-contrast
# yellow #F0E442 is placed last so it is only reached by figures with >6
# populations. All three palettes share this categorical set: colorblind
# safety is independent of the aesthetic role colours.
OKABE_ITO: list[str] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermilion
    "#F0E442",  # yellow (low contrast on white; last)
]

# Redundant non-colour channels so series stay distinguishable under
# colourblindness AND greyscale print (Wilke, "Fundamentals of Data Viz",
# ch. 19). Cycle lengths (7 / 8 / 4) are mutually offset so the joint
# (colour, marker, linestyle) combination is unique well beyond any realistic
# population count.
CATEGORICAL_MARKERS: list[str] = ["o", "s", "^", "D", "v", "P", "X", "*"]
CATEGORICAL_LINESTYLES: list[str] = ["-", "--", "-.", ":"]


PALETTES: dict[str, dict] = {
    # Default. Mirrors the visual identity of the Nature / Cell figure style
    # (Tufte/Wilke/Healy): black axis chrome, an Okabe-Ito blue for the primary
    # data series, an Okabe-Ito vermilion for fits / highlights (a colourblind-
    # safe complement to the blue), and neutral greys for reference lines and
    # fills. The categorical cycle is the full Okabe-Ito set.
    "nature": {
        "text":   "#000000",
        "signal": "#0072B2",   # Okabe-Ito blue  — primary data
        "accent": "#D55E00",   # Okabe-Ito vermilion — fits / highlights
        "muted":  "#999999",   # neutral grey — reference lines
        "fill":   "#CDE3F0",   # pale blue — CI / IQR bands
        "categorical": list(OKABE_ITO),
    },
    # Text is always black for maximum print legibility, regardless of palette.
    "forest": {
        "text":   "#000000",
        "signal": "#2C2A4A",
        "accent": "#A6A867",
        "muted":  "#9D91A3",
        "fill":   _lighten("#9D91A3", target_l=0.80),
        "categorical": list(OKABE_ITO),
    },
    "wine": {
        "text":   "#000000",
        "signal": "#66232A",
        "accent": "#C39B60",
        "muted":  "#60566B",
        "fill":   _lighten("#C39B60", target_l=0.80),
        "categorical": list(OKABE_ITO),
    },
    "sage": {
        "text":   "#000000",
        "signal": "#723D46",
        "accent": "#C9CBA3",
        "muted":  "#76818E",
        "fill":   _lighten("#C9CBA3", target_l=0.85),
        "categorical": list(OKABE_ITO),
    },
}


DEFAULT_PALETTE = "nature"


def get_palette(name: str) -> dict:
    """Return the role-keyed colour dict for a named palette.

    Keys: ``text``, ``signal``, ``accent``, ``muted``, ``fill``,
    ``categorical`` (list).

    Parameters
    ----------
    name
        ``"forest"``, ``"wine"`` or ``"sage"``.

    Raises
    ------
    KeyError
        If ``name`` is not a registered palette.
    """
    if name not in PALETTES:
        raise KeyError(
            f"unknown palette {name!r}; choose from {sorted(PALETTES)}"
        )
    return PALETTES[name]


def series_styles(n: int, palette: str = DEFAULT_PALETTE) -> list[dict]:
    """Return ``n`` distinct series styles for multi-population line/scatter.

    Each entry is a dict with keys ``color``, ``marker``, ``linestyle``
    cycling jointly over the colourblind-safe Okabe-Ito colours and the
    redundant marker / linestyle channels. The three channels have
    coprime-ish cycle lengths (7 / 8 / 4) so the combined style is unique for
    any realistic ``n``; even when the colour repeats (``n > 7``) the marker
    and linestyle still separate the series for colourblind and greyscale
    readers (Wilke 2019, ch. 19; Okabe & Ito 2008).

    Parameters
    ----------
    n
        Number of series.
    palette
        Palette name (only its ``categorical`` list is consulted).

    Returns
    -------
    list[dict]
        ``[{"color": ..., "marker": ..., "linestyle": ...}, ...]`` length ``n``.
    """
    cat = get_palette(palette)["categorical"]
    return [
        {
            "color": cat[i % len(cat)],
            "marker": CATEGORICAL_MARKERS[i % len(CATEGORICAL_MARKERS)],
            "linestyle": CATEGORICAL_LINESTYLES[i % len(CATEGORICAL_LINESTYLES)],
        }
        for i in range(n)
    ]


def diverging_cmap(name: str = DEFAULT_PALETTE):
    """Build a divergent matplotlib colormap from a named palette.

    The colormap goes ``fill → white → signal``, intended for
    population-rate heatmaps and any other use where zero should appear
    white.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """
    from matplotlib.colors import LinearSegmentedColormap
    p = get_palette(name)
    return LinearSegmentedColormap.from_list(
        f"nc_{name}_div", [p["fill"], "#FFFFFF", p["signal"]]
    )
