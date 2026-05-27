"""Attractor manifold via dimensionality reduction.

Projects a recording's per-unit binned activity into a 2-D or 3-D state space.
Supports PCA (default, no extra deps), UMAP (lazy import), and t-SNE (lazy
import of sklearn). Descriptive viz only — no inference adapter.

References:
  * Cunningham JP, Yu BM (2014). Dimensionality reduction for large-scale
    neural recordings. Nat Neurosci 17:1500-1509.
  * Churchland MM et al. (2007). Stimulus onset quenches neural variability.
    Nat Neurosci 10:1472-1474. (Gaussian smoothing of binned counts.)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from neurocomplexity.core.recording import SpikeRecording


@dataclass(frozen=True)
class ManifoldResult:
    """Output of :func:`manifold`.

    Attributes
    ----------
    coords
        Embedded coordinates of shape ``(n_bins, dims)``. One row per time
        bin, columns are the chosen low-dimensional axes.
    method
        ``"pca"``, ``"umap"`` or ``"tsne"``.
    dims
        Number of embedded dimensions (2 or 3).
    explained_variance_ratio
        Only populated for ``method="pca"`` — fraction of total variance
        captured by each kept axis (length ``dims``). ``None`` for UMAP/t-SNE
        because those are not variance-preserving.
    bin_size_seconds
        Bin size used to build the per-unit count matrix.
    sigma_ms
        Standard deviation of the Gaussian temporal smoothing applied to the
        count matrix before DR (0 disables smoothing).
    n_units
        Number of units pooled.
    populations
        Populations whose units were pooled.
    unit_ids
        ``int64`` array of unit ids used to build the matrix, in column
        order.
    source
        Provenance back-pointer.
    params
        Verbatim copy of the keyword arguments passed to :func:`manifold`.
    """

    coords: np.ndarray
    method: str
    dims: int
    explained_variance_ratio: np.ndarray | None
    bin_size_seconds: float
    sigma_ms: float
    n_units: int
    populations: tuple[str, ...]
    unit_ids: np.ndarray
    source: object
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Binning helper (per-unit)
# ---------------------------------------------------------------------------

def bin_units(rec: SpikeRecording,
              bin_size_s: float,
              unit_ids: np.ndarray) -> np.ndarray:
    """Return (T, N) int32 spike counts per unit per bin, in passed unit-id order."""
    if bin_size_s <= 0:
        raise ValueError("bin_size_s must be > 0")
    unit_ids = np.asarray(unit_ids, dtype=np.int64)
    T = int(np.floor(rec.duration / bin_size_s))
    if T <= 0:
        raise ValueError(f"duration {rec.duration}s shorter than bin {bin_size_s}s")
    N = unit_ids.size
    if N == 0:
        raise ValueError("no units selected")

    # Reuse memory-warning helper (writes int32 -> 4 bytes per cell).
    from neurocomplexity.analysis._binning import _maybe_warn_large_allocation
    _maybe_warn_large_allocation(T, N)

    out = np.zeros((T, N), dtype=np.int32)
    # Map unit id -> output column.
    id_to_col = {int(uid): i for i, uid in enumerate(unit_ids.tolist())}
    # Mask spikes belonging to wanted units only.
    mask = np.isin(rec.unit_ids, unit_ids, assume_unique=False)
    if not mask.any():
        return out
    times = rec.spike_times[mask]
    owners = rec.unit_ids[mask]
    bins = np.floor(times / bin_size_s).astype(np.int64)
    valid = (bins >= 0) & (bins < T)
    bins = bins[valid]
    owners = owners[valid]
    # Convert owners to column index.
    cols = np.fromiter((id_to_col[int(o)] for o in owners.tolist()),
                       dtype=np.int64, count=owners.size)
    flat = bins * N + cols
    np.add.at(out.reshape(-1), flat, 1)
    return out


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def _smooth_gaussian(X: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Apply 1-D Gaussian smoothing along axis 0 (time) of an (T, N) matrix.

    sigma_samples == 0 → identity.
    """
    if sigma_samples <= 0:
        return np.asarray(X, dtype=np.float64).copy()
    # Build Gaussian kernel covering +/-4 sigma.
    half = int(np.ceil(4 * sigma_samples))
    x = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_samples) ** 2)
    kernel /= kernel.sum()
    Xf = np.asarray(X, dtype=np.float64)
    out = np.empty_like(Xf)
    # Column-wise convolution; 'same' length, edge-padded via reflect would
    # bias edges — use np.convolve with mode='same' which zero-pads. That's
    # acceptable for the descriptive viz role.
    for j in range(Xf.shape[1]):
        out[:, j] = np.convolve(Xf[:, j], kernel, mode="same")
    return out


# ---------------------------------------------------------------------------
# DR methods
# ---------------------------------------------------------------------------

def _pca_fit(X: np.ndarray, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """Truncated SVD PCA. Returns (coords (T,dims), explained_variance_ratio (dims,))."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # economy SVD; columns of Vt are principal directions
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    total = float(np.sum(S ** 2))
    if total <= 0:
        var = np.zeros(dims, dtype=np.float64)
        coords = np.zeros((X.shape[0], dims), dtype=np.float64)
        return coords, var
    var = (S[:dims] ** 2) / total
    coords = U[:, :dims] * S[:dims][None, :]
    return coords.astype(np.float64), var.astype(np.float64)


def _umap_fit(X: np.ndarray, dims: int, n_neighbors: int, min_dist: float,
              random_state: int | None) -> np.ndarray:
    mod = sys.modules.get("umap", "missing")
    if mod is None:
        raise ImportError(
            "method='umap' requires umap-learn. Install with: "
            "pip install 'neurocomplexity[manifold]'"
        )
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "method='umap' requires umap-learn. Install with: "
            "pip install 'neurocomplexity[manifold]'"
        ) from exc
    reducer = UMAP(n_components=dims, n_neighbors=int(n_neighbors),
                   min_dist=float(min_dist), random_state=random_state)
    return np.asarray(reducer.fit_transform(X), dtype=np.float64)


def _tsne_fit(X: np.ndarray, dims: int, perplexity: float,
              random_state: int | None) -> np.ndarray:
    mod = sys.modules.get("sklearn.manifold", "missing")
    if mod is None:
        raise ImportError(
            "method='tsne' requires scikit-learn. Install with: "
            "pip install scikit-learn"
        )
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "method='tsne' requires scikit-learn. Install with: "
            "pip install scikit-learn"
        ) from exc
    reducer = TSNE(n_components=dims, perplexity=float(perplexity),
                   random_state=random_state, init="pca")
    return np.asarray(reducer.fit_transform(X), dtype=np.float64)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def manifold(rec: SpikeRecording,
             populations: Sequence[str] | None = None,
             *,
             method: str = "pca",
             dims: int = 2,
             bin_size_s: float = 0.05,
             sigma_ms: float = 50.0,
             umap_n_neighbors: int = 15,
             umap_min_dist: float = 0.1,
             tsne_perplexity: float = 30.0,
             random_state: int | None = 0,
             ) -> ManifoldResult:
    """Project per-unit binned activity into a 2-D or 3-D state space.

    Bins each unit in ``populations`` into ``bin_size_s``-wide counts,
    smooths each unit's count series with a Gaussian of std ``sigma_ms``,
    z-scores per unit (dropping zero-variance units), then projects with
    PCA / UMAP / t-SNE.

    Parameters
    ----------
    rec
        Spike recording.
    populations
        Names of populations to include. ``None`` → all populations.
    method
        Dimensionality-reduction method:

        * ``"pca"`` — variance-preserving linear projection. Default. No
          extra dependencies; returns ``explained_variance_ratio``.
        * ``"umap"`` — non-linear, manifold-preserving. Requires
          ``umap-learn`` (``pip install 'neurocomplexity[manifold]'``).
        * ``"tsne"`` — non-linear, neighbourhood-preserving. Requires
          ``scikit-learn``.
    dims
        Number of output dimensions, 2 or 3.
    bin_size_s
        Bin size in seconds for the per-unit count matrix (default 50 ms).
    sigma_ms
        Standard deviation of the Gaussian temporal smoothing applied to
        each unit's count series before DR. Set to 0 to disable smoothing
        (default 50 ms, following Churchland et al. 2007).
    umap_n_neighbors, umap_min_dist
        UMAP hyperparameters; ignored for other methods.
    tsne_perplexity
        t-SNE perplexity; ignored for other methods.
    random_state
        Seed for stochastic methods (UMAP / t-SNE). ``None`` → non-reproducible.

    Returns
    -------
    :class:`ManifoldResult`

    Raises
    ------
    ValueError
        If ``method`` / ``dims`` / ``bin_size_s`` / ``sigma_ms`` is invalid,
        or if no units have any spikes.
    ImportError
        If ``method="umap"`` and ``umap-learn`` is not installed, or
        ``method="tsne"`` and ``scikit-learn`` is not installed.

    Notes
    -----
    Descriptive visualisation tool — no surrogate-test adapter is provided.
    The geometry of the manifold is interpreted by eye or used as input to
    downstream analyses (e.g. tangling, decoding) outside the scope of this
    package.

    References
    ----------
    * Cunningham JP, Yu BM (2014). *Dimensionality reduction for large-scale
      neural recordings.* Nat Neurosci 17:1500-1509.
    * Gallego JA et al. (2018). *Cortical population activity within a
      preserved neural manifold...* Nat Commun 9:4233.
    """
    from neurocomplexity._warnings import _warn_if_uncurated, _warn_if_nonstationary

    _warn_if_uncurated(rec, "manifold")
    _warn_if_nonstationary(rec, "manifold")

    if method not in ("pca", "umap", "tsne"):
        raise ValueError(f"method must be one of pca/umap/tsne; got {method!r}")
    if dims not in (2, 3):
        raise ValueError(f"dims must be 2 or 3; got {dims}")
    if bin_size_s <= 0:
        raise ValueError("bin_size_s must be > 0")
    if sigma_ms < 0:
        raise ValueError("sigma_ms must be >= 0")

    if populations is None:
        populations = list(rec.populations.keys())
    populations = list(populations)
    if not populations:
        raise ValueError("no populations selected")

    # Union of unit ids across requested populations.
    keep_mask = np.zeros(len(rec.units), dtype=bool)
    for name in populations:
        if name not in rec.populations:
            from neurocomplexity.core.exceptions import PopulationError
            raise PopulationError(f"unknown population {name!r}")
        keep_mask |= rec.populations[name]
    unit_ids = rec.units.loc[keep_mask, "id"].to_numpy(dtype=np.int64)
    N = unit_ids.size
    if N == 0:
        raise ValueError("no units selected")
    if N < dims:
        raise ValueError(f"too few units (N={N}) for dims={dims}")

    counts = bin_units(rec, bin_size_s, unit_ids)  # (T, N) int32
    T = counts.shape[0]
    if T < dims + 2:
        raise ValueError(f"too few bins (T={T}) for dims={dims}")

    sigma_samples = (sigma_ms / 1000.0) / bin_size_s
    X = _smooth_gaussian(counts.astype(np.float64), sigma_samples=sigma_samples)
    # Per-column z-score (skip zero-std).
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd_safe = np.where(sd > 0, sd, 1.0)
    X = (X - mu) / sd_safe

    explained: np.ndarray | None
    if method == "pca":
        coords, explained = _pca_fit(X, dims)
    elif method == "umap":
        coords = _umap_fit(X, dims, umap_n_neighbors, umap_min_dist, random_state)
        explained = None
    else:  # tsne
        coords = _tsne_fit(X, dims, tsne_perplexity, random_state)
        explained = None

    params = {
        "populations": list(populations),
        "method": method,
        "dims": int(dims),
        "bin_size_s": float(bin_size_s),
        "sigma_ms": float(sigma_ms),
        "umap_n_neighbors": int(umap_n_neighbors),
        "umap_min_dist": float(umap_min_dist),
        "tsne_perplexity": float(tsne_perplexity),
        "random_state": random_state,
    }

    return ManifoldResult(
        coords=coords,
        method=method,
        dims=int(dims),
        explained_variance_ratio=explained,
        bin_size_seconds=float(bin_size_s),
        sigma_ms=float(sigma_ms),
        n_units=int(N),
        populations=tuple(populations),
        unit_ids=unit_ids,
        source=rec.source,
        params=params,
    )
