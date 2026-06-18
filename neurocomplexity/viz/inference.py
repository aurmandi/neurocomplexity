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

All functions take an :class:`InferenceResult`, return a matplotlib Figure,
and share the standard ``palette=`` / ``panel_label=`` / ``figsize=`` /
``ax=`` API.
"""
from __future__ import annotations

import numpy as np

from neurocomplexity.inference.results import InferenceResult
from neurocomplexity.viz._palettes import DEFAULT_PALETTE
from neurocomplexity.viz._style import (
    _apply_panel_label,
    _resolve_palette_and_axes,
    stats_box,
)


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
    title: str | None = None,
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

    external_ax = ax is not None
    p, fig, ax = _resolve_palette_and_axes(
        palette=palette, ax=ax, figsize=figsize, default_size=(4.0, 2.6),
    )

    # For small n the bare histogram is noisy: add a rug + KDE overlay so the
    # density estimate is interpretable. Threshold n_resamples < 50 switches
    # bin count down and overlays a Gaussian KDE.
    n_boot = boot.size
    if n_boot < 50:
        ax.hist(boot, bins=min(nbins, max(10, n_boot // 2)),
                density=True, color=p["fill"], edgecolor=p["signal"],
                linewidth=0.4, zorder=2)
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(boot.min(), boot.max(), 200)
            ys = gaussian_kde(boot)(xs)
            ax.plot(xs, ys, "-", lw=1.2, color=p["accent"], zorder=4,
                    label="KDE")
        except Exception:
            pass
        # Rug
        rug_y = 0.0
        ax.plot(boot, np.full_like(boot, rug_y), "|",
                color=p["signal"], ms=6, mew=0.8, zorder=3)
        ax.set_ylabel("Probability density")
    else:
        ax.hist(boot, bins=nbins, color=p["fill"], edgecolor=p["signal"],
                linewidth=0.4, zorder=2)
        ax.set_ylabel("Frequency")

    # CI band: soft neutral grey so the observed-line (signal) remains the
    # visual anchor of the panel.
    if lo is not None and hi is not None:
        ax.axvspan(lo, hi, color="#BFBFBF", alpha=0.30, zorder=1,
                   label=f"{int(round(result.ci_level * 100))}% CI")

    if observed is not None:
        ax.axvline(observed, color=p["accent"], lw=1.3, zorder=3,
                   label=f"observed = {observed:.3g}")

    # Literature convention: x-axis is the estimator itself (Efron &
    # Tibshirani 1993). Render a single-symbol statistic name with an
    # estimator hat (e.g. ``m`` → ``m̂``); leave longer names verbatim.
    sym = result.statistic_name
    xlab = (fr"$\hat{{{sym}}}$"
            if isinstance(sym, str) and len(sym) == 1 and sym.isalpha()
            else str(sym))
    ax.set_xlabel(xlab)

    # Legend top-left (replaces the former stats box). The bootstrap cloud
    # sits centre/right around the observed value, so the top-left corner is
    # the empty one. observed value + CI live in the legend labels, so no
    # separate annotation box is needed.
    ax.legend(loc="upper left", frameon=False, handlelength=1.6,
              borderpad=0.3)

    if title and not external_ax:
        fig.suptitle(title, fontweight="bold", fontsize=9)
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
    lines = []
    if pval is not None:
        lines.append(f"p = {pval:.3f}")
    if pval_fdr is not None and (pval is None or abs(pval_fdr - pval) > 1e-9):
        lines.append(f"$p_\\mathrm{{FDR}}$ = {pval_fdr:.3f}")
    eff = _as_scalar(result.effect_size)
    if eff is not None:
        lines.append(f"effect = {eff:.3g}")
    lines.append(f"n = {result.n_resamples}")
    stats_box(ax, "\n".join(lines), corner="tr")
    ax.legend(loc="upper left", frameon=False, handlelength=1.6,
              borderpad=0.3)

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
    metric: str = "auto",
    title: str | None = "Transfer Entropy Significance (FDR-corrected)",
):
    """Pairwise heatmap with FDR significance markers.

    Parameters
    ----------
    metric
        Which scalar to render in each cell:

        * ``"auto"`` (default) — plot ``result.observed`` when it is 2-D and
          everywhere non-negative (the natural choice for transfer entropy,
          mutual information, PID atoms). Otherwise fall back to
          ``result.effect_size`` (z-score). The colourbar label reflects the
          chosen field: ``result.statistic_name`` for observed, or
          ``"effect size (z)"`` for the z-score.
        * ``"observed"`` — always plot ``result.observed``.
        * ``"effect_size"`` — always plot ``result.effect_size`` (the
          previous default).

    Other notes
    -----------
    Markers: ``*`` for ``p<alpha``, ``**`` for ``p<0.01``, ``***`` for
    ``p<0.001``. Diagonal is masked.

    Colormap selection — if ``cmap`` is omitted:

      * non-negative data → sequential ``"magma_r"`` anchored at 0.
      * signed data → diverging ``"RdBu_r"`` centred at 0.
    """
    if metric not in ("auto", "observed", "effect_size"):
        raise ValueError(
            f"metric must be 'auto'/'observed'/'effect_size'; got {metric!r}"
        )

    obs = result.observed if isinstance(result.observed, np.ndarray) else None
    eff_z = result.effect_size

    if metric == "observed":
        if obs is None or obs.ndim != 2:
            raise ValueError("metric='observed' requires a 2-D result.observed")
        eff = np.asarray(obs)
        cb_label = result.statistic_name
    elif metric == "effect_size":
        if eff_z is None:
            raise ValueError("metric='effect_size' requires result.effect_size")
        eff = np.asarray(eff_z)
        cb_label = "effect size (z)"
    else:  # auto
        if (obs is not None and obs.ndim == 2
                and float(np.nanmin(obs)) >= -1e-12):
            eff = np.asarray(obs)
            cb_label = result.statistic_name
        else:
            eff = np.asarray(eff_z if eff_z is not None else result.observed)
            cb_label = ("effect size (z)" if eff_z is not None
                        else result.statistic_name)
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
    xl = list(col_labels) if col_labels else list(np.arange(n_cols))
    yl = list(row_labels) if row_labels else list(np.arange(n_rows))
    # Auto-rotate long x-tick labels (≥6 unit IDs would otherwise overlap).
    rot_x = 45 if len(xl) >= 6 else 0
    ax.set_xticklabels(xl, rotation=rot_x,
                       ha="right" if rot_x else "center",
                       rotation_mode="anchor")
    ax.set_yticklabels(yl)
    ax.tick_params(top=False, right=False, length=2,
                   labelsize=5.5 if n_cols >= 8 else 6)
    # TE[i, j] is directed flow source i -> target j: rows are senders,
    # columns receivers. Label both so the asymmetry reads off the panel.
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")

    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label(cb_label, fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.6)

    # Asterisk ladder + significant-cell count belong in the figure caption,
    # not on the panel itself; the bold title alone marks the figure.
    # Use the axes title (not suptitle) so it centres over the heatmap, not
    # over the figure+colourbar bounding box (which pulls it left).
    if title:
        ax.set_title(title, loc="center", fontweight="bold", fontsize=9, pad=8)

    _apply_panel_label(ax, panel_label)
    return fig
