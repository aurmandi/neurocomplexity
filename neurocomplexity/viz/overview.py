"""Composite overview figure — paper-style multi-panel summary.

Thin wrapper around :func:`neurocomplexity.viz._panel.figure_panel`. Accepts
a dict ``{name: result}`` (legacy signature) or positional args.
"""
from __future__ import annotations

from neurocomplexity.viz._panel import figure_panel
from neurocomplexity.viz._palettes import DEFAULT_PALETTE


def figure_overview(
    results,
    *,
    palette: str = DEFAULT_PALETTE,
    figsize: tuple[float, float] | None = None,
    panel_labels=True,
):
    """Composite figure of any combination of analysis results.

    ``results`` may be:
      * a dict ``{"criticality": ..., "branching": ..., ...}`` (legacy form),
      * a list/tuple of result objects, or
      * a single result.

    None values in a dict/list are skipped.
    """
    if isinstance(results, dict):
        items = [v for v in results.values() if v is not None]
    elif isinstance(results, (list, tuple)):
        items = [v for v in results if v is not None]
    else:
        items = [results]

    if not items:
        raise ValueError("figure_overview requires at least one non-None result")

    return figure_panel(
        *items, palette=palette, figsize=figsize, panel_labels=panel_labels,
    )
