"""Effective-connectivity network figure for TransferEntropyResult.

Renders the TE matrix as a directed graph: nodes = populations, edges = TE
arrows i -> j with width proportional to TE magnitude. Edges are filtered by
significance when a ``NullTestResult`` (``InferenceResult``) is supplied,
preferring the FDR-corrected p-values.

References:
  * Schreiber T (2000). Measuring information transfer. PRL 85, 461.
  * Bullmore E, Sporns O (2009). Complex brain networks. Nat Rev Neurosci 10, 186.
  * Krzywinski M et al. (2009). Circos: an information aesthetic for
    comparative genomics. Genome Res 19, 1639.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from neurocomplexity.analysis.transfer_entropy import TransferEntropyResult
from neurocomplexity.viz._palettes import DEFAULT_PALETTE, get_palette


def _categorical_colors(palette_name: str, n: int) -> list[str]:
    p = get_palette(palette_name)
    cat = p["categorical"]
    return [cat[i % len(cat)] for i in range(n)]


def figure_te_network(te_result: TransferEntropyResult,
                      null_result=None,
                      *,
                      alpha: float = 0.05,
                      layout: str = "circular",
                      palette: str = DEFAULT_PALETTE,
                      ax=None,
                      figsize: tuple[float, float] | None = None,
                      seed: int = 0,
                      show_disconnected: bool = True,
                      width_scale: float = 4.0,
                      ):
    """Render a directed effective-connectivity network from a TE matrix.

    Nodes are populations; edges ``i → j`` are drawn with width proportional
    to ``TE[i, j]`` (clipped to non-zero entries). When ``null_result`` is
    provided, only edges with ``p_fdr < alpha`` (or ``p < alpha`` if no FDR
    is available) are drawn.

    Parameters
    ----------
    te_result
        :class:`~neurocomplexity.analysis.TransferEntropyResult`.
    null_result
        Optional :class:`~neurocomplexity.inference.InferenceResult` from
        ``inference.test(te_result, ...)``. Used to filter edges.
    alpha
        Significance threshold (default 0.05). Compared to ``p_value_fdr``
        if present, else ``p_value``.
    layout
        ``"circular"`` (default) lays nodes on a unit circle in declaration
        order; ``"spring"`` runs a force-directed layout via
        :func:`networkx.spring_layout` with ``seed=seed``.
    palette
        Palette name (used for node fill and edge colour gradient).
    ax
        Existing ``Axes`` to draw into.
    figsize
        Figure size in inches.
    seed
        Seed for the spring layout (reproducible node placement).
    show_disconnected
        If ``False``, hide nodes that have no significant in- or out-edges.
    width_scale
        Multiplier from TE magnitude to edge linewidth (default 4).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ImportError
        If ``networkx`` is not installed.
    ValueError
        If fewer than 2 populations, mismatched matrix shape, or unknown
        ``layout``.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "figure_te_network requires networkx. Install with: pip install networkx"
        ) from exc

    if layout not in ("circular", "spring"):
        raise ValueError(f"layout must be one of circular/spring; got {layout!r}")

    M = np.asarray(te_result.matrix, dtype=np.float64)
    populations = list(te_result.populations)
    P = len(populations)
    if P < 2:
        raise ValueError("need at least 2 populations to draw a network")
    if M.shape != (P, P):
        raise ValueError(f"te_result.matrix shape {M.shape} != ({P}, {P})")
    diag_mask = np.eye(P, dtype=bool)
    M_safe = M.copy()
    M_safe[diag_mask] = 0.0

    # Significance mask.
    if null_result is None:
        mask = M_safe > 0
    else:
        p_fdr = getattr(null_result, "p_value_fdr", None)
        if p_fdr is not None:
            pmat = np.asarray(p_fdr, dtype=np.float64)
        else:
            pmat = np.asarray(null_result.p_value, dtype=np.float64)
        if pmat.shape != M.shape:
            raise ValueError(f"null p_value shape {pmat.shape} != {M.shape}")
        mask = (pmat < alpha) & (M_safe > 0)

    mask[diag_mask] = False
    n_edges = int(mask.sum())
    n_total = P * (P - 1)

    # Build graph.
    G = nx.DiGraph()
    G.add_nodes_from(populations)
    max_w = float(M_safe[mask].max()) if n_edges > 0 else 1.0
    for i in range(P):
        for j in range(P):
            if mask[i, j]:
                G.add_edge(populations[i], populations[j], weight=float(M_safe[i, j]))

    # Drop isolates if requested.
    if not show_disconnected:
        isolates = [n for n in list(G.nodes()) if G.in_degree(n) == 0 and G.out_degree(n) == 0]
        G.remove_nodes_from(isolates)

    # Layout.
    if layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)

    p = get_palette(palette)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (5.5, 5.0))
    else:
        fig = ax.figure

    node_colors = _categorical_colors(palette, len(G.nodes()))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600,
                           node_color=node_colors, edgecolors=p["text"],
                           linewidths=0.8)

    # Edge widths.
    if n_edges > 0:
        widths = [width_scale * np.sqrt(G[u][v]["weight"] / max(max_w, 1e-12))
                  for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=widths,
            edge_color=p["signal"],
            arrows=True, arrowstyle="-|>", arrowsize=14,
            connectionstyle="arc3,rad=0.15",
            node_size=600,
        )

    # (Skip nx default labels — we draw our own offset labels below.)
    # Labels offset OUTSIDE the node circles so dark text on dark fill is
    # never an issue, and the node label cannot be clipped by the figure
    # margin. networkx places labels at pos; we shift each label radially
    # away from layout centre.
    if pos:
        cx = float(np.mean([v[0] for v in pos.values()]))
        cy = float(np.mean([v[1] for v in pos.values()]))
        for node, (x, y) in pos.items():
            dx, dy = x - cx, y - cy
            norm = float(np.hypot(dx, dy)) or 1.0
            offset = 0.18
            tx = x + (dx / norm) * offset
            ty = y + (dy / norm) * offset
            ax.text(tx, ty, str(node), color=p["text"], fontsize=9,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="white", alpha=0.8,
                              edgecolor="none"))

    ax.set_axis_off()
    # Expand axis box so offset labels are not clipped.
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")

    title = f"TE network   alpha={alpha}   edges={n_edges}/{n_total}"
    if n_edges == 0:
        title += "   (no significant edges)"
        ax.text(0.5, 0.02,
                "no FDR-significant edges at this α",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7, color=p["muted"], style="italic")
    ax.set_title(title, color=p["text"], loc="left")
    return fig
