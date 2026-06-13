"""Tests for inference figures (figure_bootstrap, figure_null_test)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurocomplexity.inference.results import InferenceResult


def _bootstrap_result(seed=0) -> InferenceResult:
    rng = np.random.default_rng(seed)
    boot = rng.normal(0.30, 0.05, size=400)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return InferenceResult(
        statistic_name="m",
        observed=0.30,
        null_distribution=None,
        bootstrap_distribution=boot,
        p_value=None,
        p_value_fdr=None,
        effect_size=None,
        ci_lower=float(lo),
        ci_upper=float(hi),
        ci_level=0.95,
        method="bootstrap",
        n_resamples=int(boot.size),
        seed=seed,
        metadata={},
    )


def _null_result(seed=1) -> InferenceResult:
    rng = np.random.default_rng(seed)
    null = rng.normal(0.0, 1.0, size=500)
    observed = 2.4
    p = float(np.mean(np.abs(null) >= abs(observed)))
    return InferenceResult(
        statistic_name="TE",
        observed=observed,
        null_distribution=null,
        bootstrap_distribution=None,
        p_value=p,
        p_value_fdr=p,
        effect_size=float(observed),
        ci_lower=None,
        ci_upper=None,
        ci_level=0.95,
        method="permutation",
        n_resamples=int(null.size),
        seed=seed,
        metadata={},
    )


# -- figure_bootstrap --------------------------------------------------------

def test_figure_bootstrap_returns_figure_with_one_axes():
    from neurocomplexity.viz import figure_bootstrap
    fig = figure_bootstrap(_bootstrap_result())
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_figure_bootstrap_marks_observed_value():
    from neurocomplexity.viz import figure_bootstrap
    r = _bootstrap_result()
    fig = figure_bootstrap(r)
    ax = fig.axes[0]
    # Observed marker is drawn as a vertical line via axvline
    vline_xs = [ln.get_xdata()[0] for ln in ax.get_lines()
                if len(set(ln.get_xdata())) == 1]
    assert any(abs(x - r.observed) < 1e-9 for x in vline_xs)
    plt.close(fig)


def test_figure_bootstrap_renders_histogram_of_resamples():
    from neurocomplexity.viz import figure_bootstrap
    r = _bootstrap_result()
    fig = figure_bootstrap(r)
    ax = fig.axes[0]
    # A histogram is drawn as Rectangle patches; expect at least 5 bins
    n_rects = sum(1 for p in ax.patches
                  if type(p).__name__ == "Rectangle" and p.get_height() > 0)
    assert n_rects >= 5
    plt.close(fig)


def test_figure_bootstrap_shades_confidence_interval():
    from neurocomplexity.viz import figure_bootstrap
    r = _bootstrap_result()
    fig = figure_bootstrap(r)
    ax = fig.axes[0]
    # axvspan adds a Rectangle whose x-span equals [ci_lower, ci_upper].
    expected_width = r.ci_upper - r.ci_lower
    ci_rects = [p for p in ax.patches
                if type(p).__name__ == "Rectangle"
                and abs(p.get_width() - expected_width) < 1e-9]
    assert ci_rects, "expected a shaded CI band on the bootstrap figure"
    plt.close(fig)


def test_figure_bootstrap_accepts_palette_and_ax():
    from neurocomplexity.viz import figure_bootstrap
    fig, ax = plt.subplots()
    out = figure_bootstrap(_bootstrap_result(), palette="wine", ax=ax)
    assert out is fig
    plt.close(fig)


def test_figure_bootstrap_raises_if_no_bootstrap_distribution():
    from neurocomplexity.viz import figure_bootstrap
    bad = _bootstrap_result()
    bad = InferenceResult(**{**bad.__dict__, "bootstrap_distribution": None})
    with pytest.raises(ValueError, match="bootstrap_distribution"):
        figure_bootstrap(bad)


# -- figure_null_test --------------------------------------------------------

def test_figure_null_test_returns_figure_with_one_axes():
    from neurocomplexity.viz import figure_null_test
    fig = figure_null_test(_null_result())
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_figure_null_test_marks_observed_value():
    from neurocomplexity.viz import figure_null_test
    r = _null_result()
    fig = figure_null_test(r)
    ax = fig.axes[0]
    vline_xs = [ln.get_xdata()[0] for ln in ax.get_lines()
                if len(set(ln.get_xdata())) == 1]
    assert any(abs(x - r.observed) < 1e-9 for x in vline_xs)
    plt.close(fig)


def test_figure_null_test_renders_histogram_of_null():
    from neurocomplexity.viz import figure_null_test
    fig = figure_null_test(_null_result())
    ax = fig.axes[0]
    n_rects = sum(1 for p in ax.patches
                  if type(p).__name__ == "Rectangle" and p.get_height() > 0)
    assert n_rects >= 5
    plt.close(fig)


def test_figure_null_test_annotates_p_value():
    from neurocomplexity.viz import figure_null_test
    r = _null_result()
    fig = figure_null_test(r)
    ax = fig.axes[0]
    blobs = [t.get_text() for t in ax.texts]
    blobs.append(ax.get_title(loc="left"))
    blobs.append(ax.get_title(loc="center"))
    blobs.append(ax.get_title(loc="right"))
    text_blob = " ".join(blobs).lower()
    assert "p" in text_blob and (f"{r.p_value:.3f}" in text_blob or "p =" in text_blob)
    plt.close(fig)


def test_figure_null_test_raises_if_no_null_distribution():
    from neurocomplexity.viz import figure_null_test
    bad = _null_result()
    bad = InferenceResult(**{**bad.__dict__, "null_distribution": None})
    with pytest.raises(ValueError, match="null_distribution"):
        figure_null_test(bad)


# -- figure_significance_matrix ---------------------------------------------

def _matrix_result(n=5, seed=2) -> InferenceResult:
    rng = np.random.default_rng(seed)
    effect = rng.standard_normal((n, n)) * 0.3
    np.fill_diagonal(effect, 0.0)
    p = rng.uniform(0, 1, size=(n, n))
    # Force a couple of strongly significant cells for marker assertion
    p[0, 1] = 0.0005
    p[2, 3] = 0.005
    p[1, 4] = 0.03
    return InferenceResult(
        statistic_name="TE",
        observed=effect,
        null_distribution=None, bootstrap_distribution=None,
        p_value=p, p_value_fdr=p, effect_size=effect,
        ci_lower=None, ci_upper=None,
        ci_level=0.95, method="permutation", n_resamples=500,
        seed=seed, metadata={},
    )


def test_figure_significance_matrix_returns_figure_with_image():
    from neurocomplexity.viz import figure_significance_matrix
    fig = figure_significance_matrix(_matrix_result())
    assert isinstance(fig, matplotlib.figure.Figure)
    assert fig.axes[0].get_images(), "expected an imshow heatmap"
    plt.close(fig)


def test_figure_significance_matrix_shape_matches_effect_size():
    from neurocomplexity.viz import figure_significance_matrix
    r = _matrix_result(n=6)
    fig = figure_significance_matrix(r)
    arr = fig.axes[0].get_images()[0].get_array()
    assert arr.shape == (6, 6)
    plt.close(fig)


def test_figure_significance_matrix_marks_significant_cells():
    from neurocomplexity.viz import figure_significance_matrix
    r = _matrix_result()
    fig = figure_significance_matrix(r, alpha=0.05)
    ax = fig.axes[0]
    star_texts = [t.get_text() for t in ax.texts if "*" in t.get_text()]
    # We injected p=0.0005 (***), p=0.005 (**), p=0.03 (*) → at least 3 markers
    assert len(star_texts) >= 3, f"expected significance markers, got {star_texts}"
    assert any("***" in s for s in star_texts)
    assert any(s == "**" for s in star_texts)
    plt.close(fig)


def test_figure_significance_matrix_raises_on_scalar():
    from neurocomplexity.viz import figure_significance_matrix
    scalar = _bootstrap_result()
    with pytest.raises(ValueError, match="2D"):
        figure_significance_matrix(scalar)


# ---- Fix 1 + 2: figure_null_test alternative-aware + explicit alpha --------

def test_figure_null_test_defaults_to_one_sided_greater():
    """Default alternative is 'greater' — single upper-tail rejection region.

    Most neuroscience null tests (TE>0, branching m>=random, autonomy) are
    one-sided. Shading both tails by default would mislead reviewers about
    which direction is "significant".
    """
    from neurocomplexity.viz import figure_null_test
    # _null_result has p_value pre-computed; we only inspect the figure here.
    r = _null_result()
    fig = figure_null_test(r)
    ax = fig.axes[0]
    # axvspan rectangles whose x-span includes the right tail and excludes
    # the left tail of the null:
    null = np.asarray(r.null_distribution).ravel()
    null_lo, null_hi = float(null.min()), float(null.max())
    rects = [p for p in ax.patches if type(p).__name__ == "Rectangle"]
    # histogram bins are Rectangles too; pick the wide span ones (width > 5% of range)
    span_rects = [p for p in rects
                  if p.get_width() > 0.05 * (null_hi - null_lo)]
    # Exactly one wide rejection rect (the upper tail); no lower-tail rect.
    assert len(span_rects) == 1, (
        f"expected exactly 1 one-sided rejection rect, found {len(span_rects)}"
    )
    rect = span_rects[0]
    # Its left edge should be ABOVE the median of the null (upper tail).
    assert rect.get_x() > float(np.median(null))
    plt.close(fig)


def test_figure_null_test_two_sided_via_alternative_kwarg():
    from neurocomplexity.viz import figure_null_test
    r = _null_result()
    fig = figure_null_test(r, alternative="two-sided")
    ax = fig.axes[0]
    null = np.asarray(r.null_distribution).ravel()
    null_lo, null_hi = float(null.min()), float(null.max())
    rects = [p for p in ax.patches if type(p).__name__ == "Rectangle"]
    span_rects = [p for p in rects
                  if p.get_width() > 0.05 * (null_hi - null_lo)]
    # Two wide rectangles: lower tail + upper tail
    assert len(span_rects) == 2
    plt.close(fig)


def test_figure_null_test_explicit_alpha_independent_of_ci_level():
    """alpha=0.10 should shade ~10% rejection mass, regardless of ci_level=0.95."""
    from neurocomplexity.viz import figure_null_test
    r = _null_result()  # ci_level=0.95
    fig = figure_null_test(r, alpha=0.10)
    ax = fig.axes[0]
    null = np.asarray(r.null_distribution).ravel()
    # One-sided greater at alpha=0.10 → rejection edge at the 90th percentile
    expected_edge = float(np.percentile(null, 90))
    rects = [p for p in ax.patches if type(p).__name__ == "Rectangle"]
    null_lo, null_hi = float(null.min()), float(null.max())
    span_rects = [p for p in rects
                  if p.get_width() > 0.05 * (null_hi - null_lo)]
    assert span_rects, "expected a rejection-region rect"
    # The left edge of the upper-tail rect should match the 90th percentile
    assert any(abs(p.get_x() - expected_edge) < 0.05 for p in span_rects), (
        f"no rejection rect anchored at the 90th percentile = {expected_edge}"
    )
    plt.close(fig)


def test_figure_null_test_reads_alternative_from_metadata():
    """If metadata['alternative'] is set (as inference.test stores), respect it."""
    from neurocomplexity.viz import figure_null_test
    r = _null_result()
    r = InferenceResult(**{**r.__dict__, "metadata": {"alternative": "two-sided"}})
    fig = figure_null_test(r)  # no explicit alternative kwarg
    ax = fig.axes[0]
    null = np.asarray(r.null_distribution).ravel()
    null_lo, null_hi = float(null.min()), float(null.max())
    rects = [p for p in ax.patches if type(p).__name__ == "Rectangle"]
    span_rects = [p for p in rects
                  if p.get_width() > 0.05 * (null_hi - null_lo)]
    assert len(span_rects) == 2
    plt.close(fig)


# ---- Fix 3: figure_significance_matrix auto cmap by sign of effect ---------

def test_figure_significance_matrix_non_negative_uses_sequential():
    """When all effect-size values are >= 0, the cmap should NOT diverge around 0."""
    from neurocomplexity.viz import figure_significance_matrix
    rng = np.random.default_rng(7)
    eff = rng.uniform(0.0, 1.0, size=(5, 5))  # strictly non-negative
    np.fill_diagonal(eff, 0.0)
    p = rng.uniform(0, 1, size=(5, 5))
    r = InferenceResult(
        statistic_name="TE",
        observed=eff,
        null_distribution=None, bootstrap_distribution=None,
        p_value=p, p_value_fdr=p, effect_size=eff,
        ci_lower=None, ci_upper=None, ci_level=0.95,
        method="permutation", n_resamples=500, seed=7, metadata={},
    )
    fig = figure_significance_matrix(r)  # auto cmap
    im = fig.axes[0].get_images()[0]
    cmap_name = im.get_cmap().name
    # vmin should be 0 (sequential, anchored at silence), not -vmax
    assert im.get_clim()[0] == 0.0, f"sequential should have vmin=0, got {im.get_clim()}"
    assert "RdBu" not in cmap_name and "PiYG" not in cmap_name, (
        f"expected sequential cmap for non-negative effects, got {cmap_name!r}"
    )
    plt.close(fig)


def test_figure_significance_matrix_signed_uses_diverging():
    """Signed effects (positive and negative) → diverging cmap centred at 0."""
    from neurocomplexity.viz import figure_significance_matrix
    fig = figure_significance_matrix(_matrix_result())  # _matrix_result has signed effects
    im = fig.axes[0].get_images()[0]
    lo, hi = im.get_clim()
    assert lo < 0 and hi > 0, f"expected diverging clim, got {(lo, hi)}"
    plt.close(fig)


def test_figure_significance_matrix_cmap_kwarg_overrides_auto():
    from neurocomplexity.viz import figure_significance_matrix
    fig = figure_significance_matrix(_matrix_result(), cmap="viridis")
    assert fig.axes[0].get_images()[0].get_cmap().name == "viridis"
    plt.close(fig)

