"""Publication-quality figures for neurocomplexity results.

Per-result figure functions take a result dataclass and return a
matplotlib Figure. All accept ``palette=``, ``panel_label=``, ``figsize=``,
and ``ax=`` keyword arguments. For the standard SVG + TIFF + JPG triplet
use :func:`save_publication`.

Importing this module applies the default ``"forest"`` palette to global
matplotlib rcParams. Switch palettes with :func:`set_palette` or the
``palette=`` kwarg on any figure call.
"""
from neurocomplexity.viz._palettes import (
    DEFAULT_PALETTE,
    PALETTES,
    diverging_cmap,
    get_palette,
)
from neurocomplexity.viz._save import save_publication
from neurocomplexity.viz._scale_bar import add_scale_bar
from neurocomplexity.viz._style import (
    PALETTE,
    apply_style,
    current_palette,
    set_palette,
)
from neurocomplexity.viz.branching import figure_branching
from neurocomplexity.viz.complexity import figure_lmc_complexity
from neurocomplexity.viz.criticality import figure_criticality
from neurocomplexity.viz.dimensionality import figure_dimensionality
from neurocomplexity.viz.inference import (
    figure_bootstrap,
    figure_null_test,
    figure_significance_matrix,
    figure_volcano,
)
from neurocomplexity.viz.manifold import figure_manifold
from neurocomplexity.viz.mse import figure_mse
from neurocomplexity.viz.network import figure_te_network
from neurocomplexity.viz.pid import figure_pid
from neurocomplexity.viz.shape_collapse import figure_shape_collapse

__all__ = [
    "PALETTES", "DEFAULT_PALETTE", "get_palette", "diverging_cmap",
    "apply_style", "set_palette", "current_palette", "PALETTE",
    "save_publication", "add_scale_bar",
    "figure_criticality", "figure_branching", "figure_shape_collapse",
    "figure_dimensionality", "figure_pid",
    "figure_bootstrap", "figure_null_test",
    "figure_significance_matrix", "figure_volcano",
    "figure_lmc_complexity", "figure_mse",
    "figure_manifold",
    "figure_te_network",
]
