"""Effective-connectivity network figure for TransferEntropyResult.

Renders the TE matrix as a directed graph: nodes = populations, edges =
TE arrows ``i → j`` whose colour and width encode TE magnitude on a green
sequential scale (Stetter et al. 2012, PLoS Comp Biol Fig 7 convention).
Node fill encodes out-strength (sum of outgoing TE) on a red sequential
scale so the dominant senders read directly off the panel. Edges are
filtered by significance when a ``NullTestResult`` (``InferenceResult``)
is supplied, preferring the FDR-corrected p-values.

References:
  * Schreiber T (2000). Measuring information transfer. PRL 85, 461.
  * Bullmore E, Sporns O (2009). Complex brain networks. Nat Rev Neurosci 10, 186.
  * Stetter O et al. (2012). Model-free reconstruction of excitatory neuronal
    connectivity from calcium imaging signals. PLoS Comp Biol 8, e1002653.
  * Krzywinski M et al. (2009). Circos: an information aesthetic for
    comparative genomics. Genome Res 19, 1639.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from neurocomplexity.analysis.transfer_entropy import TransferEntropyResult
from neurocomplexity.viz._palettes import DEFAULT_PALETTE, get_palette

# Sequential cmaps: edges go pale → deep green (TE magnitude);
# node fills go pale → deep red (out-strength). Matches the Stetter 2012
# FIG 7 dMI/TE network convention adopted across the spike-train literature.
_EDGE_CMAP = LinearSegmentedColormap.from_list(
    "te_edge_green", ["#9CCC9C", "#13420F"])
_NODE_CMAP = LinearSegmentedColormap.from_list(
    "te_node_red", ["#FDECEA", "#B71C1C"])


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
                      node_gap: float = 1.0,
                      title: str | None = "Effective Connectivity (Transfer Entropy)",
                      ):
    """Render a directed effective-connectivity network from a TE matrix.

    Nodes are populations. Edges ``i → j`` are drawn with width and colour
    proportional to ``TE[i, j]`` (clipped to non-zero entries). Node fill
    intensity encodes out-strength (sum of outgoing TE). When
    ``null_result`` is provided, only edges with ``p_fdr < alpha`` (or
    ``p < alpha`` if no FDR is available) are drawn.

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
        Palette name (drives text/outline colour).
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
    title
        Bold sans-serif suptitle above the panel. ``None`` suppresses it.

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

    # Layout. ``node_gap`` scales the circle radius outward so node centres
    # sit farther apart (markers keep their point size → larger visible gaps)
    # while staying inside the padded frame.
    if layout == "circular":
        pos = nx.circular_layout(G, scale=node_gap)
    else:
        pos = nx.spring_layout(G, seed=seed)

    p = get_palette(palette)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (5.5, 5.0))
    else:
        fig = ax.figure

    # Node size scales DOWN as node count grows so labels offset radially
    # outside the marker do not crash into adjacent node circles.
    n_nodes = max(len(G.nodes()), 1)
    if n_nodes <= 6:
        node_px, label_pt, label_offset = 520, 8.5, 0.20
    elif n_nodes <= 12:
        node_px, label_pt, label_offset = 320, 7.5, 0.24
    else:
        node_px, label_pt, label_offset = 180, 6.5, 0.30

    # Per-node out-strength → node fill on the red sequential cmap. Nodes
    # that emit no significant edges stay near white (pale red), strong
    # senders go deep red.
    node_list = list(G.nodes())
    out_strength = np.array([
        sum(G[u][v]["weight"] for v in G.successors(u)) for u in node_list
    ], dtype=float) if n_edges > 0 else np.zeros(n_nodes)
    if out_strength.max() > 0:
        node_norm = Normalize(vmin=0.0, vmax=float(out_strength.max()))
        node_colors = [_NODE_CMAP(node_norm(s)) for s in out_strength]
        # Node area also grows with out-strength so the dominant senders read
        # as large red hubs (FIG 7 convention); a floor keeps silent nodes
        # visible. Both channels (size + colour) encode the same quantity.
        smax = float(out_strength.max())
        node_sizes = [node_px * (0.45 + 0.85 * s / smax) for s in out_strength]
    else:
        node_colors = ["#FFFFFF"] * n_nodes
        node_sizes = [node_px] * n_nodes

    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list,
                           node_size=node_sizes,
                           node_color=node_colors, edgecolors=p["text"],
                           linewidths=0.7)

    # Edge widths AND edge colours encode TE magnitude on the green cmap.
    # Normalise from the smallest *significant* edge (not 0) so even the
    # weakest drawn edge takes a visible mid-green rather than washing out.
    if n_edges > 0:
        w_all = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)
        wmin = float(w_all.min())
        edge_norm = Normalize(vmin=wmin, vmax=max_w if max_w > wmin else wmin + 1e-12)
        widths = []
        ecolors = []
        for w in w_all:
            widths.append(0.6 + width_scale * np.sqrt(w / max(max_w, 1e-12)))
            ecolors.append(_EDGE_CMAP(edge_norm(w)))
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=widths,
            edge_color=ecolors,
            arrows=True, arrowstyle="-|>", arrowsize=9,
            connectionstyle="arc3,rad=0.15",
            node_size=node_px,
            alpha=0.92,
        )
        # Colourbar for the edge TE scale (Stetter 2012 FIG 7 convention).
        import matplotlib.cm as _cm
        sm = _cm.ScalarMappable(norm=edge_norm, cmap=_EDGE_CMAP)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.040, pad=0.02, shrink=0.72)
        cb.set_label("Transfer entropy", fontsize=6.5)
        cb.ax.tick_params(labelsize=5.5)
        cb.outline.set_linewidth(0.6)

    # Labels offset radially OUTSIDE the node circle.
    if pos:
        cx = float(np.mean([v[0] for v in pos.values()]))
        cy = float(np.mean([v[1] for v in pos.values()]))
        for node, (x, y) in pos.items():
            dx, dy = x - cx, y - cy
            norm = float(np.hypot(dx, dy)) or 1.0
            tx = x + (dx / norm) * label_offset
            ty = y + (dy / norm) * label_offset
            ax.text(tx, ty, str(node), color=p["text"], fontsize=label_pt,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.14",
                              facecolor="white", alpha=0.85,
                              edgecolor="none"))

    ax.set_axis_off()
    # Expand axis box so offset labels are not clipped. Tracks ``node_gap``
    # so a larger circle keeps just enough margin for the radial labels.
    lim = node_gap + label_offset + 0.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    if n_edges == 0:
        ax.text(0.5, 0.02,
                "no FDR-significant edges at this α",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7, color=p["muted"], style="italic")

    # Axes title (not suptitle) so it centres over the circular graph rather
    # than over the figure+colourbar box (which shifts it left).
    if title:
        ax.set_title(title, loc="center", fontweight="bold", fontsize=9, pad=6)
    return fig
