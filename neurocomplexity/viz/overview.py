"""Composite overview figure — 'Phase-2 neurocomplexity summary'.

Layout (Nature double-column, 183 mm × ~140 mm):

    +-----------------------+--------------------+
    | a  P(s) power-law     | b  P(T) power-law  |
    +-----------+-----------+--------------------+
    | c branch  | d shape collapse (raw + col)   |
    +-----------+--------------------------------+
    | e dimensionality scree+cum | f PID atoms   |
    +----------------------------+---------------+

Each panel is driven by the corresponding analysis result dataclass.
``results`` is a dict with keys: criticality, branching, shape_collapse,
dimensionality, pid (any may be None; the panel is then drawn empty).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm

from neurocomplexity.viz._style import PALETTE, panel_label
from neurocomplexity.viz.criticality import _log_pdf


def _empty(ax, msg):
    ax.text(0.5, 0.5, msg, transform=ax.transAxes,
            ha="center", va="center", color=PALETTE["muted"], fontsize=6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def figure_overview(results: dict, *, title: str | None = None):
    """Render the 6-panel overview. Inches: 7.2 x 5.4 (≈ 183 × 137 mm)."""
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.65, wspace=0.55,
                           left=0.07, right=0.97, top=0.93, bottom=0.08)

    # --- Row 1: criticality (sizes | lifetimes) ---
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2:])
    crit = results.get("criticality")
    if crit is not None:
        xs, ps = _log_pdf(crit.sizes)
        ax_a.loglog(xs, ps, "o", ms=3, mec="none", color=PALETTE["signal"])
        if xs.size:
            xx = np.array([xs.min(), xs.max()])
            yy = ps[0] * (xx / xs[0]) ** (-crit.alpha_s)
            ax_a.loglog(xx, yy, "--", lw=0.9, color=PALETTE["accent"],
                        label=fr"$\alpha_s={crit.alpha_s:.2f}$")
        ax_a.set_xlabel("Avalanche size $s$"); ax_a.set_ylabel("$P(s)$")
        ax_a.legend(loc="lower left")
        ax_a.text(0.98, 0.97, f"$R^2={crit.r_squared:.2f}$",
                  transform=ax_a.transAxes, ha="right", va="top",
                  fontsize=6, color=PALETTE["muted"])

        xt, pt = _log_pdf(crit.lifetimes)
        ax_b.loglog(xt, pt, "o", ms=3, mec="none", color=PALETTE["ok"])
        if xt.size:
            xx = np.array([xt.min(), xt.max()])
            yy = pt[0] * (xx / xt[0]) ** (-crit.alpha_t)
            ax_b.loglog(xx, yy, "--", lw=0.9, color=PALETTE["accent"],
                        label=fr"$\alpha_t={crit.alpha_t:.2f}$")
        ax_b.set_xlabel("Lifetime $T$ (s)"); ax_b.set_ylabel("$P(T)$")
        ax_b.legend(loc="lower left")
    else:
        _empty(ax_a, "criticality unavailable")
        _empty(ax_b, "criticality unavailable")
    panel_label(ax_a, "a"); panel_label(ax_b, "b")

    # --- Row 2: branching | shape collapse (wide) ---
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1:])
    br = results.get("branching")
    if br is not None:
        ks = br.k_lags; rks = br.r_values; nz = rks > 0
        ax_c.semilogy(ks[nz], rks[nz], "o", ms=3, mec="none",
                       color=PALETTE["signal"])
        if nz.any() and np.isfinite(br.m):
            k0 = ks[nz][0]; r0 = rks[nz][0]
            b = r0 / (br.m ** k0)
            ax_c.semilogy(ks, b * (br.m ** ks), "--", lw=0.9,
                           color=PALETTE["accent"], label=fr"$m={br.m:.3f}$")
        ax_c.set_xlabel("Lag $k$ (bins)"); ax_c.set_ylabel(r"$r_k$")
        ax_c.legend(loc="upper right")
    else:
        _empty(ax_c, "branching unavailable")
    panel_label(ax_c, "c")

    sc = results.get("shape_collapse")
    if sc is not None:
        durations = np.asarray(sc.durations_used)
        cmap = cm.get_cmap("viridis")
        n = len(durations)
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
        u = sc.rescaled_x
        for y, col in zip(sc.rescaled_y, colors):
            ax_d.plot(u, y, lw=0.8, color=col, alpha=0.85)
        ax_d.set_xlabel(r"$t/T$")
        ax_d.set_ylabel(r"$\langle a\rangle/T^{\gamma-1}$")
        ax_d.text(0.98, 0.97,
                  fr"$\gamma={sc.gamma:.3f}$" + "\n" +
                  f"resid$={sc.residual:.2g}$",
                  transform=ax_d.transAxes, ha="right", va="top",
                  fontsize=6, color=PALETTE["muted"])
    else:
        _empty(ax_d, "shape collapse unavailable")
    panel_label(ax_d, "d")

    # --- Row 3: dimensionality (cumulative) | PID ---
    ax_e = fig.add_subplot(gs[2, :2])
    ax_f = fig.add_subplot(gs[2, 2:])

    dm = results.get("dimensionality")
    if dm is not None:
        eig = np.asarray(dm.eigenvalues, dtype=float)
        eig = np.sort(eig)[::-1]; eig = eig[eig > 0]
        idx = np.arange(1, eig.size + 1)
        cum = np.cumsum(eig) / eig.sum()
        ax_e.plot(idx, cum, "-", lw=1.0, color=PALETTE["signal"])
        ax_e.axhline(0.9, ls="--", lw=0.6, color=PALETTE["muted"])
        ax_e.axvline(dm.participation_ratio, ls="--", lw=0.8,
                      color=PALETTE["accent"],
                      label=f"PR={dm.participation_ratio:.1f}")
        ax_e.set_xlabel("Component index"); ax_e.set_ylabel("Cumulative variance")
        ax_e.set_ylim(0, 1.02); ax_e.legend(loc="lower right")
        ax_e.text(0.02, 0.97,
                  f"N={dm.n_units}  PR/N={dm.participation_ratio/dm.n_units:.2f}",
                  transform=ax_e.transAxes, va="top", fontsize=6,
                  color=PALETTE["muted"])
    else:
        _empty(ax_e, "dimensionality unavailable")
    panel_label(ax_e, "e")

    pid = results.get("pid")
    if pid is not None:
        labels = ["R", f"U({pid.sources[0]})", f"U({pid.sources[1]})", "S"]
        vals = np.array([pid.redundancy, pid.unique_1,
                         pid.unique_2, pid.synergy])
        colors = [PALETTE["muted"], PALETTE["signal"],
                  PALETTE["ok"], PALETTE["accent"]]
        x = np.arange(len(vals))
        ax_f.bar(x, vals, color=colors, edgecolor="none", width=0.7)
        ax_f.set_xticks(x); ax_f.set_xticklabels(labels, fontsize=6.5)
        ax_f.set_ylabel(f"Info about {pid.target} (nats)")
        ymax = max(vals.max(), 1e-12); ax_f.set_ylim(0, ymax * 1.3)
        for xi, v in zip(x, vals):
            ax_f.text(xi, v + ymax * 0.03, f"{v:.4f}",
                       ha="center", va="bottom", fontsize=5.5)
        ax_f.text(0.98, 0.97, f"total MI={pid.total_mi:.4f}",
                  transform=ax_f.transAxes, ha="right", va="top",
                  fontsize=6, color=PALETTE["muted"])
    else:
        _empty(ax_f, "PID unavailable")
    panel_label(ax_f, "f")

    if title:
        fig.suptitle(title, fontsize=9, y=0.995)
    return fig
