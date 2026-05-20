"""Block-average + binary-median-split discretiser for ContinuousSignal.

Mirrors the ``binary`` Schreiber TE estimator that the package uses for
spike counts: each bin collapses to {0, 1}, so the joint state space for a
mix of population and signal variables is uniform.
"""
from __future__ import annotations

import numpy as np

from neurocomplexity.core.continuous import ContinuousSignal


def bin_signal_average(sig: ContinuousSignal, *,
                       bin_size_s: float,
                       n_bins: int) -> np.ndarray:
    """Return shape (n_bins,) float array of per-bin block-averages.

    Bins that fall outside the signal's coverage carry ``np.nan`` so callers
    can decide how to handle them. ``bin_size_s`` must be an integer multiple
    of ``1/sig.sampling_rate``.
    """
    fs = float(sig.sampling_rate)
    samples_per_bin = bin_size_s * fs
    nearest = int(round(samples_per_bin))
    if nearest <= 0 or abs(samples_per_bin - nearest) > 1e-9 * max(1.0, samples_per_bin):
        raise ValueError(
            f"bin_size_s ({bin_size_s*1000:.6f} ms) is not an integer multiple "
            f"of 1/sampling_rate ({1000.0/fs:.6f} ms) for signal "
            f"{sig.label!r}; nearest integer ratio = {nearest}."
        )
    out = np.full(int(n_bins), np.nan, dtype=np.float64)
    start_offset = max(0, int(np.ceil(sig.t_start / bin_size_s)))
    end_bin = min(int(n_bins),
                  int(np.floor((sig.t_start + sig.duration) / bin_size_s)))
    if end_bin <= start_offset:
        return out
    sample_start = int(round((start_offset * bin_size_s - sig.t_start) * fs))
    n_covered_bins = end_bin - start_offset
    block = sig.values[sample_start: sample_start + n_covered_bins * nearest]
    if block.size != n_covered_bins * nearest:
        n_covered_bins = block.size // nearest
        block = block[: n_covered_bins * nearest]
        end_bin = start_offset + n_covered_bins
    if n_covered_bins == 0:
        return out
    out[start_offset: start_offset + n_covered_bins] = \
        block.reshape(n_covered_bins, nearest).mean(axis=1)
    return out


def bin_signal_binary(sig: ContinuousSignal, *,
                      bin_size_s: float,
                      n_bins: int,
                      threshold: float | None = None) -> np.ndarray:
    """Return shape (n_bins,) int array of {0, 1}.

    Parameters
    ----------
    sig
        Uniformly-sampled ContinuousSignal.
    bin_size_s
        Width of the analysis bin. Must be an integer multiple of
        ``1 / sig.sampling_rate`` within 1e-9 relative tolerance.
    n_bins
        Length of the output (== T from ``bin_spikes``).
    threshold
        If None, use the median of the per-bin block-averages so the output
        is exactly half-ones, half-zeros (binary median split). Otherwise,
        use the provided absolute threshold.
    """
    fs = float(sig.sampling_rate)
    samples_per_bin = bin_size_s * fs
    nearest = int(round(samples_per_bin))
    if nearest <= 0 or abs(samples_per_bin - nearest) > 1e-9 * max(1.0, samples_per_bin):
        raise ValueError(
            f"bin_size_s ({bin_size_s*1000:.6f} ms) is not an integer multiple "
            f"of 1/sampling_rate ({1000.0/fs:.6f} ms) for signal "
            f"{sig.label!r}; nearest integer ratio = {nearest}."
        )

    # Determine the sample index range we can fully populate.
    out = np.zeros(int(n_bins), dtype=np.int32)
    n_samples = sig.values.size
    # The first analysis bin starts at t=0; signal sample 0 starts at sig.t_start.
    # First analysis-bin index that lies entirely within the signal coverage:
    start_offset = max(0, int(np.ceil(sig.t_start / bin_size_s)))
    # Last (exclusive) analysis-bin index:
    end_bin = min(int(n_bins),
                  int(np.floor((sig.t_start + sig.duration) / bin_size_s)))
    if end_bin <= start_offset:
        return out

    # For each covered bin i, signal indices are
    #   [(i*bin_size_s - sig.t_start) * fs,
    #    ((i+1)*bin_size_s - sig.t_start) * fs)
    # Slice the contiguous block of usable samples and reshape.
    sample_start = int(round((start_offset * bin_size_s - sig.t_start) * fs))
    n_covered_bins = end_bin - start_offset
    block = sig.values[sample_start: sample_start + n_covered_bins * nearest]
    if block.size != n_covered_bins * nearest:
        # Trailing partial bin: drop it.
        n_covered_bins = block.size // nearest
        block = block[: n_covered_bins * nearest]
        end_bin = start_offset + n_covered_bins
    if n_covered_bins == 0:
        return out

    means = block.reshape(n_covered_bins, nearest).mean(axis=1)
    if threshold is None:
        threshold = float(np.median(means))
        # Use strict `>` so already-binary or heavily-tied inputs (e.g. a
        # square-wave stim sampled into {0, 1}) split into both classes
        # rather than collapsing to all-1s when the median equals the
        # minority value.
        out[start_offset: start_offset + n_covered_bins] = (means > threshold).astype(np.int32)
    else:
        out[start_offset: start_offset + n_covered_bins] = (means >= threshold).astype(np.int32)
    return out
