"""figure_panel: composite multi-panel figure for paper submissions."""
from __future__ import annotations

import string
from typing import Callable

import matplotlib.pyplot as plt

from neurocomplexity.viz._palettes import DEFAULT_PALETTE


_REGISTRY: dict[type, str] = {}


def register_figure(result_type: type, figure_fn_path: str) -> None:
    _REGISTRY[result_type] = figure_fn_path


def _resolve(fn_path: str) -> Callable:
    mod_name, fn_name = fn_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def _bootstrap_registry() -> None:
    if _REGISTRY:
        return
    from neurocomplexity.analysis.criticality import CriticalityResult
    from neurocomplexity.analysis.branching import BranchingResult
    from neurocomplexity.analysis.shape_collapse import ShapeCollapseResult
    from neurocomplexity.analysis.dimensionality import DimensionalityResult
    from neurocomplexity.analysis.pid import PIDResult
    from neurocomplexity.core.recording import SpikeRecording

    register_figure(CriticalityResult,    "neurocomplexity.viz.criticality.figure_criticality")
    register_figure(BranchingResult,      "neurocomplexity.viz.branching.figure_branching")
    register_figure(ShapeCollapseResult,  "neurocomplexity.viz.shape_collapse.figure_shape_collapse")
    register_figure(DimensionalityResult, "neurocomplexity.viz.dimensionality.figure_dimensionality")
    register_figure(PIDResult,            "neurocomplexity.viz.pid.figure_pid")
    register_figure(SpikeRecording,       "neurocomplexity.viz.population.figure_population_heatmap")


def _auto_layout(n: int) -> tuple[int, int]:
    if n <= 1: return (1, 1)
    if n <= 2: return (1, 2)
    if n <= 4: return (2, 2)
    if n <= 6: return (2, 3)
    if n <= 9: return (3, 3)
    return (4, 3)


def figure_panel(
    *items,
    layout="auto",
    palette: str = DEFAULT_PALETTE,
    figsize: tuple[float, float] | None = None,
    panel_labels=True,
):
    """Lay out per-result figures into one composite figure."""
    _bootstrap_registry()
    n = len(items)
    if n == 0:
        raise ValueError("figure_panel requires at least one result")

    # Validate all result types BEFORE creating any figure
    fn_paths: list[str] = []
    for item in items:
        match = None
        for cls, path in _REGISTRY.items():
            if isinstance(item, cls):
                match = path
                break
        if match is None:
            raise TypeError(
                f"no figure function registered for result type "
                f"{type(item).__name__}"
            )
        fn_paths.append(match)

    # Validate panel_labels length before creating any figure
    if isinstance(panel_labels, list):
        if len(panel_labels) != n:
            raise ValueError(
                f"panel_labels length {len(panel_labels)} != number of items {n}"
            )
        labels = panel_labels
    elif panel_labels:
        labels = list(string.ascii_lowercase[:n])
    else:
        labels = [None] * n

    rows, cols = _auto_layout(n) if layout == "auto" else layout
    size = figsize if figsize is not None else (cols * 3.0, rows * 2.4)
    fig, axarr = plt.subplots(rows, cols, figsize=size, squeeze=False)
    axes_flat = axarr.flatten().tolist()

    for item, fn_path, ax, label in zip(items, fn_paths, axes_flat[:n], labels):
        fn = _resolve(fn_path)
        fn(item, palette=palette, panel_label=label, ax=ax)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig
