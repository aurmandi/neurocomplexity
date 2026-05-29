"""save_publication: write a figure to SVG + TIFF + JPG.

Explicitly no PDF support per project policy. SVG preserves vector text
(``svg.fonttype='none'``); TIFF uses LZW compression at the user-picked dpi;
JPG is always 600 dpi at quality 95.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import matplotlib.figure

_VALID_FORMATS = {"svg", "tiff", "jpg"}


def save_publication(
    fig: matplotlib.figure.Figure,
    path: str | os.PathLike,
    *,
    tiff_dpi: Literal[600, 1200] = 600,
    formats: tuple[str, ...] = ("svg", "tiff", "jpg"),
) -> dict[str, Path]:
    """Save ``fig`` under the path stem in each requested format.

    Parameters
    ----------
    fig
        The matplotlib Figure to write.
    path
        Path stem; extensions are appended automatically (e.g. passing
        ``"figs/branching"`` writes ``figs/branching.svg``,
        ``figs/branching.tiff``, ``figs/branching.jpg``).
    tiff_dpi
        Resolution for the TIFF in dots per inch; must be 600 or 1200.
    formats
        Subset of ``("svg", "tiff", "jpg")`` to write. Passing "pdf" raises.

    Returns
    -------
    dict[str, Path]
        Mapping from format name to the file written.
    """
    if tiff_dpi not in (600, 1200):
        raise ValueError(f"tiff_dpi must be 600 or 1200, got {tiff_dpi!r}")
    for f in formats:
        if f == "pdf":
            raise ValueError("pdf output is not supported by save_publication")
        if f not in _VALID_FORMATS:
            raise ValueError(
                f"unknown format {f!r}; must be one of {sorted(_VALID_FORMATS)}"
            )

    stem = Path(path)
    out: dict[str, Path] = {}

    if "svg" in formats:
        p = stem.with_suffix(".svg")
        fig.savefig(p, format="svg", bbox_inches="tight", pad_inches=0.02)
        out["svg"] = p.resolve()

    if "tiff" in formats:
        p = stem.with_suffix(".tiff")
        fig.savefig(
            p, format="tiff", dpi=tiff_dpi,
            bbox_inches="tight", pad_inches=0.02,
            pil_kwargs={"compression": "tiff_lzw"},
        )
        out["tiff"] = p.resolve()

    if "jpg" in formats:
        p = stem.with_suffix(".jpg")
        fig.savefig(
            p, format="jpg", dpi=600,
            bbox_inches="tight", pad_inches=0.02,
            pil_kwargs={"quality": 95},
        )
        out["jpg"] = p.resolve()

    return out
