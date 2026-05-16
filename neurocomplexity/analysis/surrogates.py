"""Deprecated: use `neurocomplexity.inference` instead.

Thin shim for backwards compatibility with code written against v1.0.
Will be removed in v2.0.
"""
import warnings

from neurocomplexity.inference.surrogates import (
    spike_dither, isi_shuffle, interval_shuffle,
)


def jitter_recording(rec, jitter_ms=10.0, rng=None):
    warnings.warn(
        "jitter_recording is deprecated; use "
        "neurocomplexity.inference.surrogates.spike_dither",
        DeprecationWarning, stacklevel=2,
    )
    seed = None if rng is None else int(rng.integers(0, 2**31 - 1))
    return spike_dither(rec, delta_ms=jitter_ms, seed=seed)


def shuffle_isis(rec, rng=None):
    warnings.warn(
        "shuffle_isis is deprecated; use "
        "neurocomplexity.inference.surrogates.isi_shuffle",
        DeprecationWarning, stacklevel=2,
    )
    seed = None if rng is None else int(rng.integers(0, 2**31 - 1))
    return isi_shuffle(rec, seed=seed)


def make_surrogate(rec, method, **kwargs):
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
