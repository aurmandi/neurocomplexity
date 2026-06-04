"""Pairwise transfer entropy across populations.

Estimator: Schreiber (2000) on binary-thresholded spike counts, with
Miller-Madow bias correction. Robust on integer counts where Kraskov k-NN
degenerates (zero distances).

Recommended granularity
-----------------------
For binned spike-train data the field convention is **per-unit** (or per-
channel) TE — every node in the TE graph is a single unit. This is the
recipe in Shimono & Beggs (2015 J. Neurosci.) and Timme et al. (2016
PLoS Comp. Biol.), both of which use exactly the Schreiber binary
estimator implemented here.

Pooling many units into one "population" stream (``populations=["VISp"]``
where VISp has hundreds of units) creates a binary marginal that
saturates near 1 in every bin, collapses the entropy, and underpowers
the TE test. Use per-unit populations whenever the population would
otherwise contain more than a few tens of units. ``transfer_entropy``
emits a ``UserWarning`` at runtime when any input stream's binary
marginal saturates beyond ``[0.05, 0.95]``.

KSG / continuous-valued TE (Kraskov–Stögbauer–Grassberger; the TRENTOOL3
and IDTxl default for LFP/EEG/MEG) is a different estimator family and
is not implemented in v0.1 — open an issue if you need it.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from neurocomplexity.analysis._binning import bin_spikes
from neurocomplexity.core.recording import SpikeRecording


@dataclass(frozen=True)
class TransferEntropyResult:
    """Output of :func:`transfer_entropy`.

    Attributes
    ----------
    matrix
        ``(P, P)`` pairwise transfer-entropy matrix in **nats**. Convention:
        ``matrix[i, j]`` = TE from population ``i`` (source, row) to
        population ``j`` (target, column). Diagonal is zero. Estimator is
        Schreiber (2000) on binary-thresholded counts with Miller-Madow
        bias correction.
    populations
        Population names, in the order they index ``matrix``.
    bin_size_seconds
        Bin size used to discretise spike counts.
    delay_bins
        Lag in bins between source past and target future (default 1).
    source
        Provenance back-pointer.
    params
        Verbatim copy of the keyword arguments passed to
        :func:`transfer_entropy`.

    Notes
    -----
    Significance is assessed via :func:`neurocomplexity.inference.test`,
    typically with ``surrogate="spike_dither"`` or
    ``surrogate="isi_shuffle"``. FDR correction across the off-diagonal
    entries is applied automatically when ``fdr=True``.
    """

    matrix: np.ndarray       # (P, P) TE in nats; row=source, col=target
    populations: tuple[str, ...]
    bin_size_seconds: float
    delay_bins: int
    source: object
    params: dict = field(default_factory=dict)


def _binary_schreiber_te(source_ts: np.ndarray, target_ts: np.ndarray,
                          delay: int = 1, bias: str = "miller_madow") -> float:
    """TE(source -> target) in nats.

    ``bias`` selects the analytic bias correction subtracted from the plug-in
    estimate (see :func:`transfer_entropy` for the rationale):

    - ``"miller_madow"`` (default): ``(m - 1) / (2N)`` where ``m`` is the
      number of *occupied* cells in the 8-state joint histogram. Simple and
      conservative; the form shipped since v0.1.
    - ``"roulston"``: ``(m_X - 1)(m_Y - 1) / (2N)``, the Roulston (2002)
      product form over the occupied source-past (``m_X``) and target-past
      (``m_Y``) alphabets. Smaller correction when one stream is near-degenerate.
    - ``"none"``: raw plug-in estimate, no correction.
    """
    y = (np.asarray(target_ts) > 0).astype(np.int8)
    x = (np.asarray(source_ts) > 0).astype(np.int8)
    T = len(y)
    if T <= delay + 1:
        return 0.0

    y_future = y[delay:]
    y_past = y[:-delay]
    x_past = x[:-delay]
    N = len(y_future)

    idx = 4 * y_future + 2 * y_past + x_past
    counts = np.bincount(idx, minlength=8).astype(np.float64)
    p_joint = counts / N

    te = 0.0
    for yf in (0, 1):
        for yp in (0, 1):
            for xp in (0, 1):
                i = 4 * yf + 2 * yp + xp
                if counts[i] == 0:
                    continue
                p_yyx = p_joint[i]
                p_yx = p_joint[4 * 0 + 2 * yp + xp] + p_joint[4 * 1 + 2 * yp + xp]
                p_yy = p_joint[4 * yf + 2 * yp + 0] + p_joint[4 * yf + 2 * yp + 1]
                p_y = sum(p_joint[4 * yfp + 2 * yp + xpp]
                          for yfp in (0, 1) for xpp in (0, 1))
                if p_yx == 0 or p_yy == 0 or p_y == 0:
                    continue
                cond_full = p_yyx / p_yx
                cond_red = p_yy / p_y
                if cond_full == 0 or cond_red == 0:
                    continue
                te += p_yyx * np.log(cond_full / cond_red)

    if bias == "none":
        correction = 0.0
    elif bias == "roulston":
        m_x = int(np.unique(x_past).size)
        m_y = int(np.unique(y_past).size)
        correction = (m_x - 1) * (m_y - 1) / (2.0 * N)
    elif bias == "miller_madow":
        m = int(np.sum(counts > 0))
        correction = (m - 1) / (2.0 * N)
    else:
        raise ValueError(
            f"unknown bias={bias!r}; choose 'miller_madow', 'roulston', or 'none'"
        )
    # Two-stage clamp (A9): the plug-in TE itself can drift slightly
    # negative on finite samples; we clamp it to zero BEFORE subtracting
    # the Miller-Madow correction so that the correction is never
    # double-applied to an already-zero estimator. Without this, an
    # already-negative plug-in TE could be pushed further negative and
    # then re-clamped, biasing the final estimator upward in low-signal
    # regimes (because the "raw" zero gets transferred to the corrected
    # estimator without ever paying the correction cost).
    te_raw = max(0.0, float(te))
    te_corrected = te_raw - correction
    return max(0.0, te_corrected)


def transfer_entropy(rec: SpikeRecording,
                     populations: Sequence[str] | None = None,
                     bin_size_ms: float = 5.0,
                     delay_bins: int = 1,
                     estimator: str = "binary",
                     signals: Sequence[str] | None = None,
                     bias: str = "miller_madow",
                     n_jobs: int = 1,
                     ) -> TransferEntropyResult:
    """Pairwise TE matrix across the given populations and (optionally) signals.

    Signals are discretised via a binary median split at the chosen bin width
    and concatenated after populations in the matrix axis order. The result's
    ``populations`` tuple carries the unified ordered list of names.

    Parameters
    ----------
    bias
        Analytic bias correction subtracted from each plug-in TE estimate:
        ``"miller_madow"`` (default, ``(m-1)/(2N)`` over occupied joint
        cells), ``"roulston"`` (``(m_X-1)(m_Y-1)/(2N)`` product form,
        Roulston 2002), or ``"none"``. See :func:`_binary_schreiber_te`.
    n_jobs
        Number of worker processes for the pairwise loop. ``1`` (default)
        runs serially with the usual progress bar; ``>1`` (or ``-1`` for all
        cores) dispatches the ``P*(P-1)`` ordered pairs via
        :func:`joblib.Parallel`.

    Notes
    -----
    **Granularity.** For binned spike-train data the field-standard
    granularity is per-unit (Shimono & Beggs 2015; Timme et al. 2016).
    Pooling many units into one population stream saturates the binary
    marginal and underpowers the test. To run per-unit TE within an area,
    pass ``populations`` as a dict of singleton masks (one mask per unit) to
    :meth:`~neurocomplexity.core.SpikeRecording.with_populations`. A
    ``UserWarning`` is emitted when any stream's binary marginal falls
    outside ``[0.05, 0.95]``.

    Complexity
    ----------
    The pairwise loop is ``O(P^2 * T)`` — ``P*(P-1)`` ordered population
    pairs, each a single ``O(T)`` pass over the binned series. Memory is
    ``O(P*T)`` for the binned counts plus ``O(P^2)`` for the output matrix.
    With ``n_jobs>1`` the ``O(P^2)`` pair work is split across workers; the
    ``O(P*T)`` binning is done once up front and shared.
    """
    from neurocomplexity._warnings import _warn_if_nonstationary, _warn_if_uncurated
    _warn_if_uncurated(rec, "transfer_entropy")
    _warn_if_nonstationary(rec, "transfer_entropy")
    if estimator != "binary":
        raise ValueError(f"only estimator='binary' is implemented in v0.1; got {estimator!r}")
    if populations is None:
        populations = list(rec.populations.keys())
    populations = list(populations)
    signals = list(signals) if signals else []
    if len(populations) + len(signals) < 2:
        raise ValueError("need at least 2 streams (populations + signals) for TE")
    for s in signals:
        if s not in rec.signals:
            raise ValueError(f"unknown signal {s!r}; available: {list(rec.signals.keys())}")

    bs = float(bin_size_ms) / 1000.0
    if populations:
        counts = bin_spikes(rec, populations, bs)  # (T, P)
        T = counts.shape[0]
    else:
        # signals-only path: still need T from the spike grid; use rec.duration.
        T = int(np.floor(rec.duration / bs))
        counts = np.zeros((T, 0), dtype=np.int32)

    if signals:
        from neurocomplexity.analysis._continuous import bin_signal_binary
        sig_cols = [
            bin_signal_binary(rec.signals[name], bin_size_s=bs, n_bins=T)
            for name in signals
        ]
        sig_block = np.stack(sig_cols, axis=1) if sig_cols else np.zeros((T, 0), dtype=np.int32)
        counts = np.concatenate([counts, sig_block], axis=1)
    names = populations + signals
    P = counts.shape[1]

    # Saturation diagnostic: binary-symbol TE collapses when any input
    # stream's binarized marginal is near 0 or 1. This is the classic failure
    # mode of pooling hundreds of units into one population. Warn so the user
    # can switch to per-unit granularity (see module docstring).
    if P:
        bin_marg = (counts > 0).mean(axis=0)
        bad = [
            (names[i], float(bin_marg[i]))
            for i in range(P)
            if bin_marg[i] < 0.05 or bin_marg[i] > 0.95
        ]
        if bad:
            import warnings
            warnings.warn(
                "transfer_entropy: binary marginal saturates for "
                + ", ".join(f"{n!r}={v:.3f}" for n, v in bad)
                + " (outside [0.05, 0.95]); binary-symbol TE will be near "
                "zero and the surrogate test will be underpowered. Use "
                "per-unit populations (one mask per unit) or a finer bin "
                "size — see transfer_entropy module docstring.",
                stacklevel=2,
            )

    mat = np.zeros((P, P), dtype=np.float64)
    pairs = [(s, t) for s in range(P) for t in range(P) if s != t]

    if n_jobs == 1:
        from neurocomplexity._progress import progress_iter
        for s, t in progress_iter(pairs, total=len(pairs), desc="TE matrix"):
            mat[s, t] = _binary_schreiber_te(
                counts[:, s], counts[:, t], delay=delay_bins, bias=bias)
    else:
        from joblib import Parallel, delayed
        vals = Parallel(n_jobs=n_jobs)(
            delayed(_binary_schreiber_te)(
                counts[:, s], counts[:, t], delay=delay_bins, bias=bias)
            for s, t in pairs
        )
        for (s, t), v in zip(pairs, vals):
            mat[s, t] = v

    return TransferEntropyResult(
        matrix=mat,
        populations=tuple(names),
        bin_size_seconds=bs,
        delay_bins=delay_bins,
        source=rec.source,
        params={"populations": list(populations),
                "signals": list(signals),
                "bin_size_ms": float(bin_size_ms),
                "delay_bins": int(delay_bins),
                "estimator": estimator,
                "bias": bias,
                "n_jobs": int(n_jobs)},
    )
