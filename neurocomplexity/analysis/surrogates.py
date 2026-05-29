"""Deprecated: use `neurocomplexity.inference` instead.

Thin shim for backwards compatibility with code written against v1.0.
Will be removed in v2.0.
"""
import warnings

from neurocomplexity.inference.surrogates import (
    interval_shuffle,
    isi_shuffle,
    spike_dither,
)


def jitter_recording(rec, jitter_ms=10.0, rng=None):
    """**Deprecated.** Forwards to
    :func:`neurocomplexity.inference.surrogates.spike_dither`.

    Will be removed in v2.0. Emits :class:`DeprecationWarning`.
    """
    warnings.warn(
        "jitter_recording is deprecated; use "
        "neurocomplexity.inference.surrogates.spike_dither",
        DeprecationWarning, stacklevel=2,
    )
    seed = None if rng is None else int(rng.integers(0, 2**31 - 1))
    return spike_dither(rec, delta_ms=jitter_ms, seed=seed)


def shuffle_isis(rec, rng=None):
    """**Deprecated.** Forwards to
    :func:`neurocomplexity.inference.surrogates.isi_shuffle`.

    Will be removed in v2.0. Emits :class:`DeprecationWarning`.
    """
    warnings.warn(
        "shuffle_isis is deprecated; use "
        "neurocomplexity.inference.surrogates.isi_shuffle",
        DeprecationWarning, stacklevel=2,
    )
    seed = None if rng is None else int(rng.integers(0, 2**31 - 1))
    return isi_shuffle(rec, seed=seed)


def make_surrogate(rec, method, **kwargs):
    """**Deprecated.** Dispatcher kept for v1.x compatibility.

    Use :class:`neurocomplexity.inference.SurrogatePool` (lazy generation,
    deterministic seed handling) or call
    :mod:`neurocomplexity.inference.surrogates` directly.

    Will be removed in v2.0. Emits :class:`DeprecationWarning`.
    """
    warnings.warn(
        "make_surrogate is deprecated; use "
        "neurocomplexity.inference.SurrogatePool",
        DeprecationWarning, stacklevel=2,
    )
    if method == "jitter":
        return jitter_recording(
            rec, **{k: v for k, v in kwargs.items() if k in ("jitter_ms", "rng")}
        )
    if method in ("shuffle", "isi_match", "isi_shuffle"):
        return shuffle_isis(rec, rng=kwargs.get("rng"))
    raise ValueError(f"unknown surrogate method: {method!r}")


__all__ = [
    "spike_dither", "isi_shuffle", "interval_shuffle",
    "jitter_recording", "shuffle_isis", "make_surrogate",
]
