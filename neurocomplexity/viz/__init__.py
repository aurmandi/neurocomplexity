"""Nature-style figures for neurocomplexity results.

All figure functions take the corresponding analysis result dataclass and
return a matplotlib Figure. Saving is the caller's job (``fig.savefig(...)``)
or use ``save_publication(fig, path)`` for the standard editable-PDF +
editable-SVG + 600 dpi-TIFF triplet.

Importing this module configures global matplotlib rcParams for Nature-style
output (Arial 7 pt, editable text, no top/right spines, 0.8 pt lines).
"""
from neurocomplexity.viz._style import apply_style, save_publication, PALETTE
from neurocomplexity.viz.criticality import figure_criticality
from neurocomplexity.viz.branching import figure_branching
from neurocomplexity.viz.shape_collapse import figure_shape_collapse
from neurocomplexity.viz.dimensionality import figure_dimensionality
from neurocomplexity.viz.pid import figure_pid
from neurocomplexity.viz.overview import figure_overview

apply_style()

__all__ = [
    "apply_style", "save_publication", "PALETTE",
    "figure_criticality", "figure_branching", "figure_shape_collapse",
    "figure_dimensionality", "figure_pid", "figure_overview",
]
