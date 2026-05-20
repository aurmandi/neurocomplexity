"""Inference figures for :class:`InferenceResult` objects.

Four canonical inferential exhibits from the neuroscience-statistics literature:

* ``figure_bootstrap`` — bootstrap-distribution histogram + observed marker +
  shaded CI (Efron & Tibshirani 1993 *Intro to the Bootstrap* Fig 6.3;
  Pernet et al. 2011 *Frontiers Psychology*).

* ``figure_null_test`` — null/permutation histogram + observed marker +
  one-sided (default) or two-sided rejection region + p / p_FDR annotation.
  The form follows Nichols & Holmes (2002) *Hum. Brain Mapp.* "Nonparametric
  permutation tests for functional neuroimaging" Fig. 1 and Maris &
  Oostenveld (2007) *J Neurosci Methods*. (Beggs & Plenz 2003 Fig 2A
  actually shows overlaid P(s) histograms for data and shuffled spikes,
  not the canonical "single null + observed" panel — so they are not cited
  here despite being the historical reference for shuffle-based avalanche
  surrogates.) FDR reporting follows Storey & Tibshirani 2003 *PNAS*.

* ``figure_significance_matrix`` — effect-size heatmap (sequential cmap for
  non-negative statistics, diverging for signed) overlaid with ``*`` /
  ``**`` / ``***`` per-cell FDR-significance markers. Standard exhibit for
  pairwise neuroscience statistics: transfer-entropy connectomes (Vicente
  et al. 2011 *J. Comput. Neurosci.*; Wibral, Lizier & Priesemann eds.
  2014, *Directed Information Measures in Neuroscience*) and PID-atom
  matrices in subsequent multi-source PID work (Wollstadt et al. 2019
  *J. Open Source Software* — IDTxl). The original Williams & Beer 2010
  PID paper presents atoms as Venn-style decompositions, not matrices, so
  it is not cited here for the matrix form.

* ``figure_volcano`` — scatter of effect size vs −log₁₀(p_FDR) with the
  significance threshold drawn. Originated in genomics (Cui & Churchill
  2003 *Genome Biology*; Li 2012 *Genomics, Proteomics & Bioinformatics*
  review of volcano-plot conventions). For multi-test neuroscience
  screens we follow the same convention; IDTxl (Wollstadt et al. 2019)
  documents the panel for large-scale TE screens. The Zalesky, Fornito &
  Bullmore (2010) Network-Based Statistic is a *different* multi-test
  procedure (connected-component p-distributions) and is **not** cited
  here for the volcano form.

All functions take an :class:`InferenceResult`, return a matplotlib Figure,
and share the standard ``palette=`` / ``panel_label=`` / ``figsize=`` /
``ax=`` API.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from neurocomplexity.inference.results import InferenceResult
from neurocomplexity.viz._palettes import get_palette, DEFAULT_PALETTE
from neurocomplexity.viz._style import _resolve_palette_and_axes, _apply_panel_label


def _as_scalar(x):
    if x is None:
        return None
    arr = np.asarray(x)
    return float(arr.item()) if arr.size == 1 else None


def figure_bootstrap(
    result: InferenceResult,
    *,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
    nbins: int = 40,
):
    """Bootstrap-distribution histogram with observed marker + shaded CI."""
    boot = result.bootstrap_distribution
    if boot is None:
        raise ValueError(
            "figure_bootstrap requires result.bootstrap_distribution to be set"
        )
    boot = np.asarray(boot).ravel()
    observed = _as_scalar(result.observed)
    lo = _as_scalar(result.ci_lower)
    hi = _as_scalar(result.ci_upper)

    p, fig, ax = _resolve_palette_and_axes(
        palette=palette, ax=ax, figsize=figsize, default_size=(4.0, 2.6),
    )

    ax.hist(boot, bins=nbins, color=p["fill"], edgecolor=p["signal"],
            linewidth=0.4, zorder=2)

    if lo is not None and hi is not None:
        ax.axvspan(lo, hi, color=p["accent"], alpha=0.20, zorder=1,
                   label=f"{int(round(result.ci_level * 100))}% CI")

    if observed is not None:
        ax.axvline(observed, color=p["signal"], lw=1.3, zorder=3,
                   label=f"observed = {observed:.3g}")

    ax.set_xlabel(f"{result.statistic_name} (bootstrap replicates)")
    ax.set_ylabel("Count")

    # Legend + stats annotation placed ABOVE the data so neither obscures the
    # histogram. Stats on the left of the title strip, legend on the right.
    info = f"n = {result.n_resamples}"
    if lo is not None and hi is not None and observed is not None:
        info = f"{observed:.3g} [{lo:.3g}, {hi:.3g}]   n = {result.n_resamples}"
    ax.set_title(info, loc="left", fontsize=6.5, color=p["text"], pad=8)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2,
              frameon=False, handlelength=1.6, borderpad=0.3, fontsize=6.5,
              bbox_transform=ax.transAxes)

    _apply_panel_label(ax, panel_label)
    return fig


def figure_null_test(
    result: InferenceResult,
    *,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
    nbins: int = 40,
    alpha: float = 0.05,
    alternative: str | None = None,
):
    """Null-distribution histogram with observed marker + rejection shading.

    Parameters
    ----------
    alpha : significance level for the shaded rejection region (default 0.05).
        Independent of ``result.ci_level`` (which belongs to bootstrap CIs).
    alternative : 'greater' | 'less' | 'two-sided' | None.
        ``None`` (default) reads ``result.metadata['alternative']`` if present,
        otherwise falls back to 'greater' — the convention for the
        non-negative neuroscience statistics (TE, branching ratio, autonomy)
        that dominate this package's call sites.
    """
    null = result.null_distribution
    if null is None:
        raise ValueError(
            "figure_null_test requires result.null_distribution to be set"
        )
    null = np.asarray(null).ravel()
    observed = _as_scalar(result.observed)
    pval = _as_scalar(result.p_value)
    pval_fdr = _as_scalar(result.p_value_fdr)

    if alternative is None:
        alternative = (result.metadata or {}).get("alternative", "greater")
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError(
            f"alternative must be 'greater'|'less'|'two-sided', got {alternative!r}"
        )

    p, fig, ax = _resolve_palette_and_axes(
        palette=palette, ax=ax, figsize=figsize, default_size=(4.0, 2.6),
    )

    ax.hist(null, bins=nbins, color=p["fill"], edgecolor=p["signal"],
            linewidth=0.4, zorder=2, label="null")

    # Rejection region anchored at the explicit alpha kwarg (NOT ci_level).
    if 0 < alpha < 1 and null.size:
        x_lo = float(min(null.min(), observed if observed is not None else null.min()))
        x_hi = float(max(null.max(), observed if observed is not None else null.max()))
        if alternative == "greater":
            crit = float(np.percentile(null, 100 * (1 - alpha)))
            ax.axvspan(crit, x_hi, color=p["muted"], alpha=0.25, zorder=1,
                       label=f"reject (α={alpha:.2g}, one-sided)")
        elif alternative == "less":
            crit = float(np.percentile(null, 100 * alpha))
            ax.axvspan(x_lo, crit, color=p["muted"], alpha=0.25, zorder=1,
                       label=f"reject (α={alpha:.2g}, one-sided)")
        else:  # two-sided
            lo_crit = float(np.percentile(null, 100 * alpha / 2))
            hi_crit = float(np.percentile(null, 100 * (1 - alpha / 2)))
            ax.axvspan(x_lo, lo_crit, color=p["muted"], alpha=0.25, zorder=1)
            ax.axvspan(hi_crit, x_hi, color=p["muted"], alpha=0.25, zorder=1,
                       label=f"reject (α={alpha:.2g}, two-sided)")

    if observed is not None:
        ax.axvline(observed, color=p["signal"], lw=1.3, zorder=3,
                   label=f"observed = {observed:.3g}")

    ax.set_xlabel(f"{result.statistic_name} (null distribution)")
    ax.set_ylabel("Count")

    # Stats above-left, legend above-right (outside the data, never overlaps).
    info_parts = []
    if pval is not None:
        info_parts.append(f"p = {pval:.3f}")
    if pval_fdr is not None and (pval is None or abs(pval_fdr - pval) > 1e-9):
        info_parts.append(f"p_FDR = {pval_fdr:.3f}")
    eff = _as_scalar(result.effect_size)
    if eff is not None:
        info_parts.append(f"effect = {eff:.3g}")
    info_parts.append(f"n = {result.n_resamples}")
    ax.set_title("   ".join(info_parts), loc="left", fontsize=6.5,
                 color=p["text"], pad=8)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2,
              frameon=False, handlelength=1.6, borderpad=0.3, fontsize=6.5,
              bbox_transform=ax.transAxes)

    _apply_panel_label(ax, panel_label)
    return fig


def _stars(p_value: float, alpha: float) -> str:
    """Conventional significance marker ladder."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < alpha:
        return "*"
    return ""


def figure_significance_matrix(
    result: InferenceResult,
    *,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    alpha: float = 0.05,
    cmap: str | None = None,
):
    """Pairwise effect-size heatmap with FDR significance markers.

    Expects ``result.effect_size`` (or ``result.observed``) to be 2D and a
    matching 2D ``result.p_value_fdr`` (or ``result.p_value``). Diagonal is
    masked. Markers: ``*`` for ``p<alpha``, ``**`` for ``p<0.01``, ``***``
    for ``p<0.001`` (standard convention).

    Colormap selection
    ------------------
    If ``cmap`` is omitted (the default), the cmap is chosen from the data:
      * **non-negative** effects (e.g. transfer entropy, mutual information,
        PID atoms — strictly ≥ 0 by definition) → sequential ``"magma_r"``
        with ``vmin=0`` and ``vmax = 99th percentile``. Divergence around
        zero would be meaningless for these statistics.
      * **signed** effects (e.g. correlation, signed PID contrasts) →
        diverging ``"RdBu_r"`` centred at 0 with ``vmax = 99th percentile``
        of ``|effect|``.
    Pass ``cmap=`` to override.
    """
    eff = np.asarray(
        result.effect_size if result.effect_size is not None else result.observed
    )
    if eff.ndim != 2:
        raise ValueError(
            f"figure_significance_matrix expects a 2D effect_size/observed; "
            f"got shape {eff.shape}"
        )
    pvals = np.asarray(
        result.p_value_fdr if result.p_value_fdr is not None else result.p_value
    )
    if pvals.shape != eff.shape:
        raise ValueError(
            f"p-value shape {pvals.shape} does not match effect-size {eff.shape}"
        )

    p, fig, ax = _resolve_palette_and_axes(
        palette=palette, ax=ax, figsize=figsize, default_size=(4.4, 3.6),
    )

    n_rows, n_cols = eff.shape
    masked = np.array(eff, dtype=float, copy=True)
    if n_rows == n_cols:
        np.fill_diagonal(masked, np.nan)

    finite = masked[np.isfinite(masked)]
    is_non_negative = finite.size > 0 and float(np.min(finite)) >= -1e-12
    if cmap is None:
        cmap = "magma_r" if is_non_negative else "RdBu_r"

    if is_non_negative and cmap not in ("RdBu_r", "RdBu", "PiYG", "PiYG_r",
                                        "PuOr", "PuOr_r", "BrBG", "BrBG_r"):
        # Sequential mode: anchor at 0, clip at the 99th percentile.
        vmax = float(np.nanpercentile(masked, 99)) if finite.size else 1.0
        if vmax <= 0:
            vmax = 1.0
        im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=vmax,
                       interpolation="nearest", aspect="equal")
    else:
        # Diverging mode: centre at 0.
        vmax = float(np.nanpercentile(np.abs(masked), 99)) if finite.size else 1.0
        if vmax <= 0:
            vmax = 1.0
        im = ax.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax,
                       interpolation="nearest", aspect="equal")

    for i in range(n_rows):
        for j in range(n_cols):
            if n_rows == n_cols and i == j:
                continue
            star = _stars(float(pvals[i, j]), alpha)
            if star:
                cell_norm = abs(masked[i, j]) / vmax if vmax else 0.0
                text_color = "white" if cell_norm > 0.55 else p["text"]
                ax.text(j, i, star, ha="center", va="center",
                        fontsize=7, color=text_color)

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels if col_labels else np.arange(n_cols))
    ax.set_yticklabels(row_labels if row_labels else np.arange(n_rows))
    ax.tick_params(top=False, right=False, length=2)

    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label(result.statistic_name, fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.6)

    ax.set_title(
        f"FDR markers: * p<{alpha:.2g}   ** p<0.01   *** p<0.001",
        loc="left", fontsize=6.5, color=p["text"], pad=8,
    )

    _apply_panel_label(ax, panel_label)
    return fig


def figure_volcano(
    result: InferenceResult,
    *,
    palette: str = DEFAULT_PALETTE,
    panel_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
    alpha: float = 0.05,
):
    """Effect size vs −log10(p_FDR) scatter with significance threshold.

    Three-colour scheme — the canonical genomics volcano (Cui & Churchill 2003):

      * **n.s.** (grey, palette ``muted``) — points with ``p_FDR ≥ alpha``.
      * **down** (palette ``signal``) — significant points with ``effect < 0``.
      * **up** (palette ``categorical[1]``) — significant points with ``effect > 0``.

    For strictly non-negative effect sizes (e.g. transfer entropy) only the
    "up" bucket is populated, but the API still produces a 3-bucket scatter so
    figure colour-keys remain consistent across call sites.
    """
    eff = result.effect_size
    pvals = result.p_value_fdr if result.p_value_fdr is not None else result.p_value
    if eff is None or pvals is None:
        raise ValueError(
            "figure_volcano requires both effect_size and p_value (or p_value_fdr)"
        )
    eff = np.asarray(eff, dtype=float).ravel()
    pvals = np.asarray(pvals, dtype=float).ravel()
    if eff.shape != pvals.shape:
        raise ValueError(
            f"effect_size shape {eff.shape} != p_value shape {pvals.shape}"
        )
    if eff.size < 2:
        raise ValueError("figure_volcano expects multiple tests; got 1")

    pvals_safe = np.clip(pvals, 1e-300, 1.0)
    neg_log_p = -np.log10(pvals_safe)
    significant = pvals_safe < alpha
    sig_up = significant & (eff > 0)
    sig_down = significant & (eff < 0)
    nonsig = ~significant

    p, fig, ax = _resolve_palette_and_axes(
        palette=palette, ax=ax, figsize=figsize, default_size=(4.4, 3.0),
    )

    # Pick a distinct second colour for "up". Palette categorical[1] is the
    # next semantic colour after signal; falls back to accent if categorical
    # is too short.
    cats = p.get("categorical", [])
    up_colour = cats[1] if len(cats) > 1 else p["accent"]
    down_colour = p["signal"]

    ax.scatter(eff[nonsig], neg_log_p[nonsig],
               s=14, color=p["muted"], alpha=0.6, edgecolor="none",
               label="n.s.")
    ax.scatter(eff[sig_down], neg_log_p[sig_down],
               s=18, color=down_colour, alpha=0.9, edgecolor="none",
               label=f"down (p<{alpha:.2g})")
    ax.scatter(eff[sig_up], neg_log_p[sig_up],
               s=18, color=up_colour, alpha=0.9, edgecolor="none",
               label=f"up (p<{alpha:.2g})")

    threshold = -np.log10(alpha)
    ax.axhline(threshold, color=p["accent"], lw=0.9, ls="--",
               label=fr"$-\log_{{10}}\,\alpha$ = {threshold:.2f}")
    ax.axvline(0.0, color=p["text"], lw=0.5, ls=":")

    ax.set_xlabel(f"Effect size ({result.statistic_name})")
    ax.set_ylabel(r"$-\log_{10}\,p_{\mathrm{FDR}}$")

    n_up = int(sig_up.sum())
    n_down = int(sig_down.sum())
    ax.set_title(
        f"up: {n_up}   down: {n_down}   total: {eff.size}   n_resamples = {result.n_resamples}",
        loc="left", fontsize=6.5, color=p["text"], pad=8,
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=4,
              frameon=False, handlelength=1.6, borderpad=0.3, fontsize=6.5,
              bbox_transform=ax.transAxes)

    _apply_panel_label(ax, panel_label)
    return fig
