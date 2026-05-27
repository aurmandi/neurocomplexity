"""Tests for analysis.manifold."""
from __future__ import annotations

import sys
import warnings as _warnings

import numpy as np
import pandas as pd
import pytest

from neurocomplexity._warnings import QualityControlWarning
from neurocomplexity.analysis.manifold import (
    ManifoldResult,
    bin_units,
    manifold,
    _pca_fit,
    _smooth_gaussian,
)
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording


def _poisson_rec(n_units=10, rate_hz=20.0, duration_s=30.0, seed=0):
    rng = np.random.default_rng(seed)
    units = pd.DataFrame({"id": np.arange(n_units, dtype=np.int64)})
    times = []
    owners = []
    for u in range(n_units):
        n = rng.poisson(rate_hz * duration_s)
        t = np.sort(rng.uniform(0, duration_s, size=n))
        times.append(t)
        owners.append(np.full(n, u, dtype=np.int64))
    st = np.concatenate(times)
    uid = np.concatenate(owners)
    order = np.argsort(st, kind="stable")
    return SpikeRecording(
        spike_times=st[order].astype(np.float64),
        unit_ids=uid[order],
        units=units,
        populations={"all": np.ones(n_units, dtype=bool)},
        duration=float(duration_s),
        sampling_rate=None,
        source=ProvenanceRecord.for_memory("test"),
    )


# ---- _pca_fit ---------------------------------------------------------------

def test_pca_fit_shapes_and_variance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    coords, var = _pca_fit(X, dims=3)
    assert coords.shape == (200, 3)
    assert var.shape == (3,)
    assert var[0] >= var[1] >= var[2]
    assert var.sum() <= 1.0 + 1e-9


def test_pca_fit_centered_inputs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8)) + 5.0
    coords, _ = _pca_fit(X, dims=2)
    assert np.allclose(coords.mean(axis=0), 0.0, atol=1e-9)


# ---- bin_units --------------------------------------------------------------

def test_bin_units_shape_matches_T_and_N():
    rec = _poisson_rec(n_units=6, duration_s=10.0)
    counts = bin_units(rec, bin_size_s=0.05, unit_ids=np.arange(6, dtype=np.int64))
    expected_T = int(np.floor(10.0 / 0.05))
    assert counts.shape == (expected_T, 6)
    assert counts.dtype == np.int32
    assert (counts >= 0).all()


def test_bin_units_preserves_unit_id_order():
    rec = _poisson_rec(n_units=6, duration_s=10.0)
    # Reverse order
    uids = np.array([5, 3, 1, 0, 2, 4], dtype=np.int64)
    counts = bin_units(rec, bin_size_s=0.05, unit_ids=uids)
    # Sanity: column sum for unit 5 (col 0) equals number of spikes for unit 5
    n5 = int(np.sum(rec.unit_ids == 5))
    assert int(counts[:, 0].sum()) == n5


def test_bin_units_invalid_bin_size_raises():
    rec = _poisson_rec()
    with pytest.raises(ValueError):
        bin_units(rec, bin_size_s=0.0, unit_ids=np.array([0], dtype=np.int64))


# ---- _smooth_gaussian -------------------------------------------------------

def test_smooth_gaussian_zero_sigma_identity():
    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    out = _smooth_gaussian(X, sigma_samples=0.0)
    assert np.allclose(out, X)


def test_smooth_gaussian_preserves_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    out = _smooth_gaussian(X, sigma_samples=2.0)
    assert out.shape == X.shape


# ---- manifold (PCA) ---------------------------------------------------------

def test_manifold_pca_smoke():
    rec = _poisson_rec(n_units=8, duration_s=30.0)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        r = manifold(rec, method="pca", dims=2, bin_size_s=0.05)
    assert isinstance(r, ManifoldResult)
    assert r.method == "pca"
    assert r.dims == 2
    assert r.coords.shape[1] == 2
    assert r.explained_variance_ratio is not None
    assert r.explained_variance_ratio.shape == (2,)
    assert r.n_units == 8


def test_manifold_pca_dims_3():
    rec = _poisson_rec(n_units=8, duration_s=30.0)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        r = manifold(rec, method="pca", dims=3, bin_size_s=0.05)
    assert r.coords.shape[1] == 3


def test_manifold_sigma_ms_zero_skips_smoothing():
    rec = _poisson_rec(n_units=8, duration_s=20.0, seed=1)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        r_smooth = manifold(rec, method="pca", dims=2, bin_size_s=0.05, sigma_ms=50.0)
        r_raw = manifold(rec, method="pca", dims=2, bin_size_s=0.05, sigma_ms=0.0)
    assert not np.allclose(r_smooth.coords, r_raw.coords)


def test_manifold_params_round_trip_pca_deterministic():
    rec = _poisson_rec(n_units=8, duration_s=20.0, seed=2)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        r = manifold(rec, method="pca", dims=2, bin_size_s=0.05)
        redo = manifold(rec, **r.params)
    assert np.allclose(r.coords, redo.coords)


# ---- validation -------------------------------------------------------------

def test_manifold_invalid_method_raises():
    rec = _poisson_rec()
    with pytest.raises(ValueError, match="method"):
        manifold(rec, method="bogus")


def test_manifold_invalid_dims_raises():
    rec = _poisson_rec()
    with pytest.raises(ValueError, match="dims"):
        manifold(rec, dims=4)


def test_manifold_invalid_bin_size_raises():
    rec = _poisson_rec()
    with pytest.raises(ValueError):
        manifold(rec, bin_size_s=0.0)


def test_manifold_negative_sigma_raises():
    rec = _poisson_rec()
    with pytest.raises(ValueError):
        manifold(rec, sigma_ms=-1.0)


def test_manifold_too_few_units_raises():
    rec = _poisson_rec(n_units=1, duration_s=20.0)
    with pytest.raises(ValueError, match="too few units"):
        manifold(rec, method="pca", dims=2)


# ---- explained variance only for PCA ----------------------------------------

def test_manifold_explained_variance_none_for_tsne():
    rec = _poisson_rec(n_units=8, duration_s=20.0, seed=3)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        r = manifold(rec, method="tsne", dims=2, bin_size_s=0.1,
                     tsne_perplexity=10.0, random_state=0)
    assert r.method == "tsne"
    assert r.explained_variance_ratio is None
    assert r.coords.shape[1] == 2


def test_manifold_umap_smoke():
    rec = _poisson_rec(n_units=8, duration_s=20.0, seed=4)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        r = manifold(rec, method="umap", dims=2, bin_size_s=0.1,
                     umap_n_neighbors=10, random_state=0)
    assert r.method == "umap"
    assert r.coords.shape[1] == 2


# ---- import errors via monkeypatched sys.modules ----------------------------

def test_manifold_umap_missing_raises_importerror(monkeypatch):
    rec = _poisson_rec(n_units=8, duration_s=20.0, seed=5)
    # Mask the module so the lazy import inside _umap_fit fails.
    monkeypatch.setitem(sys.modules, "umap", None)
    with pytest.raises(ImportError, match="manifold"):
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            manifold(rec, method="umap", dims=2, bin_size_s=0.1)


def test_manifold_tsne_missing_raises_importerror(monkeypatch):
    rec = _poisson_rec(n_units=8, duration_s=20.0, seed=6)
    monkeypatch.setitem(sys.modules, "sklearn.manifold", None)
    with pytest.raises(ImportError, match="scikit-learn"):
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            manifold(rec, method="tsne", dims=2, bin_size_s=0.1)


# ---- warnings ---------------------------------------------------------------

def test_manifold_uncurated_warning_emitted():
    rec = _poisson_rec()
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        manifold(rec, method="pca", dims=2, bin_size_s=0.05)
    assert any(issubclass(w.category, QualityControlWarning) for w in caught)
