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


def _schreiber_te_general(source_d: np.ndarray, target_d: np.ndarray,
                           K: int, delay: int = 1,
                           bias: str = "miller_madow",
                           history_k: int = 1,
                           history_l: int = 1) -> float:
    """Plug-in TE for arbitrary-alphabet discrete sequences with optional
    history embedding.

    Both inputs must be integer arrays in ``[0, K)`` of equal length. The
    estimator matches the binary-state form (Schreiber 2000) at the legacy
    ``k = l = 1`` setting, and generalises to embeddings ``k = history_k``
    (target past length) and ``l = history_l`` (source past length) by
    forming the joint state index over

        (y_future,
         y_{t-delay}, ..., y_{t-delay-k+1},
         x_{t-delay}, ..., x_{t-delay-l+1})

    of total alphabet size ``K^(1 + k + l)``. Miller-Madow
    ``(m - 1) / (2N)`` correction is applied to the occupied joint cells
    with the two-stage clamp (raw plug-in >= 0, then correction, then
    final clamp).

    Setting ``history_k = history_l = 1`` recovers the legacy three-cell
    joint exactly, modulo float arithmetic; the regression test
    ``test_k1_l1_matches_legacy_binary_estimator`` enforces parity
    against :func:`_binary_schreiber_te` on a binary alphabet.
    """
    y = np.asarray(target_d, dtype=np.int64)
    x = np.asarray(source_d, dtype=np.int64)
    if K < 2:
        raise ValueError("alphabet K must be >= 2")
    if y.size != x.size:
        raise ValueError("source and target must be the same length")
    if history_k < 1 or history_l < 1:
        raise ValueError("history_k and history_l must be >= 1")
    if delay < 1:
        raise ValueError("delay must be >= 1")

    k, l = int(history_k), int(history_l)
    T = y.size
    # Earliest sample index needed for any past lag.
    embed = delay + max(k, l) - 1
    N = T - embed
    if N <= 0:
        return 0.0
    base = np.arange(N) + embed

    y_future = y[base]
    y_past = np.stack(
        [y[base - delay - h] for h in range(k)], axis=1,
    )  # (N, k)
    x_past = np.stack(
        [x[base - delay - h] for h in range(l)], axis=1,
    )  # (N, l)

    n_dims = 1 + k + l
    cols = np.concatenate([y_future[:, None], y_past, x_past], axis=1)
    # Flat index: most-significant = y_future, then y_past_0..y_past_{k-1},
    # then x_past_0..x_past_{l-1}. Total alphabet = K^(1 + k + l).
    powers = K ** np.arange(n_dims - 1, -1, -1, dtype=np.int64)
    flat = (cols * powers[None, :]).sum(axis=1)
    M_total = int(K ** n_dims)
    counts = np.bincount(flat, minlength=M_total).astype(np.float64)
    p = counts / N

    shape = (K,) * n_dims
    p_full = p.reshape(shape)
    y_past_axes = tuple(range(1, 1 + k))
    x_past_axes = tuple(range(1 + k, 1 + k + l))

    # Marginals on the (1 + k + l)-dim table.
    p_yf_yp = p_full.sum(axis=x_past_axes)              # (1+k)-dim
    p_yp_xp = p_full.sum(axis=(0,))                     # (k+l)-dim
    p_yp = p_full.sum(axis=(0,) + x_past_axes)          # (k)-dim

    # Broadcast back to full shape using None inserts.
    p_yp_b = p_yp[(None,) + (slice(None),) * k + (None,) * l]
    p_yp_xp_b = p_yp_xp[(None,) + (slice(None),) * (k + l)]
    p_yf_yp_b = p_yf_yp[(slice(None),) * (1 + k) + (None,) * l]

    num = p_full * p_yp_b
    den = p_yp_xp_b * p_yf_yp_b
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = (p_full > 0) & (den > 0) & (num > 0)
        ratio = np.where(mask, num / den, 1.0)
        log_ratio = np.where(mask, np.log(ratio), 0.0)
    te = float(np.sum(p_full * log_ratio))

    if bias == "none":
        correction = 0.0
    elif bias == "roulston":
        # Roulston-style product correction over occupied alphabets in each
        # marginal (yf x yp x xp).
        m_yf = int(np.unique(y_future).size)
        m_yp = int(np.unique(y_past).size)
        m_xp = int(np.unique(x_past).size)
        correction = (max(m_yf - 1, 0) * max(m_yp - 1, 0)
                      * max(m_xp - 1, 0) / (2.0 * N))
    elif bias == "miller_madow":
        m = int(np.sum(counts > 0))
        correction = (m - 1) / (2.0 * N)
    else:
        raise ValueError(
            f"unknown bias={bias!r}; choose 'miller_madow', 'roulston', or 'none'"
        )
    te_raw = max(0.0, te)
    return max(0.0, te_raw - correction)


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
                     *,
                     discretize: str = "binary",
                     n_quantile_bins: int = 3,
                     history_k: int = 1,
                     history_l: int = 1,
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
    discretize
        How spike counts are mapped to symbols before the plug-in:

        * ``"binary"`` (default) — Schreiber's ``> 0`` threshold; the
          paper-tested estimator. Saturates on pooled populations (see
          module docstring).
        * ``"quantile"`` — quantile-equal discretisation into
          ``n_quantile_bins`` levels via :func:`pid._quantile_discretise`.
          Preserves rate-coded information; appropriate when populations
          are pooled or fire near the upper saturation limit.
        * ``"none"`` — caller is responsible for discretisation. ``counts``
          must already be integer in ``[0, n_quantile_bins)``; the value
          of ``n_quantile_bins`` is taken as the alphabet size ``K``.
    n_quantile_bins
        Alphabet size for ``discretize`` modes other than ``"binary"``.
        Default 3 matches the PID default.
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
    if discretize not in {"binary", "quantile", "none"}:
        raise ValueError(
            f"unknown discretize={discretize!r}; choose 'binary', 'quantile', or 'none'"
        )
    if discretize != "binary" and int(n_quantile_bins) < 2:
        raise ValueError("n_quantile_bins must be >= 2")
    if int(history_k) < 1 or int(history_l) < 1:
        raise ValueError("history_k and history_l must be >= 1")
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

    # Saturation diagnostic: only meaningful for the binary path, since
    # quantile-discretised streams are by construction multi-level.
    if P and discretize == "binary":
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
                "per-unit populations (one mask per unit), a finer bin "
                "size, or discretize='quantile' — see module docstring.",
                stacklevel=2,
            )

    # Build the discretised symbol matrix (T, P). For the binary path the
    # binary threshold lives inside `_binary_schreiber_te`; for the other
    # paths we pre-discretise here so the worker just sees integer streams.
    if discretize == "binary":
        symbols = None  # the legacy path takes raw counts
        K = 2
    elif discretize == "quantile":
        from neurocomplexity.analysis.pid import _quantile_discretise
        if P:
            symbols = np.stack(
                [_quantile_discretise(counts[:, p].astype(np.float64),
                                       int(n_quantile_bins))
                 for p in range(P)],
                axis=1,
            ).astype(np.int64)
        else:
            symbols = np.zeros((T, 0), dtype=np.int64)
        K = int(n_quantile_bins)
    else:  # "none"
        if not np.issubdtype(counts.dtype, np.integer):
            raise ValueError(
                "discretize='none' requires integer-typed counts in [0, n_quantile_bins)"
            )
        if P and (counts.min() < 0 or counts.max() >= int(n_quantile_bins)):
            raise ValueError(
                f"discretize='none' requires counts in [0, {int(n_quantile_bins)}); "
                f"observed range [{int(counts.min())}, {int(counts.max())}]"
            )
        symbols = counts.astype(np.int64)
        K = int(n_quantile_bins)

    def _te_pair(s_idx: int, t_idx: int) -> float:
        if discretize == "binary" and history_k == 1 and history_l == 1:
            return _binary_schreiber_te(
                counts[:, s_idx], counts[:, t_idx],
                delay=delay_bins, bias=bias,
            )
        if discretize == "binary":
            # binary alphabet but k>1 or l>1 — binarise on the fly and use
            # the general plug-in.
            src = (counts[:, s_idx] > 0).astype(np.int64)
            tgt = (counts[:, t_idx] > 0).astype(np.int64)
            return _schreiber_te_general(
                src, tgt, K=2, delay=delay_bins, bias=bias,
                history_k=history_k, history_l=history_l,
            )
        return _schreiber_te_general(
            symbols[:, s_idx], symbols[:, t_idx], K=K,
            delay=delay_bins, bias=bias,
            history_k=history_k, history_l=history_l,
        )

    mat = np.zeros((P, P), dtype=np.float64)
    pairs = [(s, t) for s in range(P) for t in range(P) if s != t]

    if n_jobs == 1:
        from neurocomplexity._progress import progress_iter
        for s, t in progress_iter(pairs, total=len(pairs), desc="TE matrix"):
            mat[s, t] = _te_pair(s, t)
    else:
        from joblib import Parallel, delayed
        vals = Parallel(n_jobs=n_jobs)(
            delayed(_te_pair)(s, t) for s, t in pairs
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
                "n_jobs": int(n_jobs),
                "discretize": discretize,
                "n_quantile_bins": int(n_quantile_bins),
                "history_k": int(history_k),
                "history_l": int(history_l)},
    )


@dataclass(frozen=True)
class ConditionalTransferEntropyResult:
    """Output of :func:`conditional_transfer_entropy`.

    Attributes
    ----------
    value
        Scalar CTE(source -> target | conditions) in **nats**. Estimator
        is the multi-symbol Schreiber plug-in (Miller-Madow corrected) on
        the joint state ``(y_future, y_past_k, x_past_l, z1_past_l, ...,
        zM_past_l)``.
    source, target
        Population names.
    conditions
        Names of conditioning populations, in order.
    bin_size_seconds
        Bin size used to discretise.
    delay_bins
        Lag between source / condition past and target future.
    history_k, history_l
        Target-past and source/condition-past lengths.
    source_rec
        Provenance back-pointer.
    params
        Verbatim copy of keyword arguments passed to
        :func:`conditional_transfer_entropy`.
    """

    value: float
    source: str
    target: str
    conditions: tuple[str, ...]
    bin_size_seconds: float
    delay_bins: int
    history_k: int
    history_l: int
    source_rec: object
    params: dict = field(default_factory=dict)


def _cte_general(y: np.ndarray, x: np.ndarray, z_list: list[np.ndarray],
                 K: int, delay: int, bias: str,
                 history_k: int, history_l: int) -> float:
    """Multi-symbol conditional TE: TE(x -> y | z_list).

    Joint table over
    ``(y_future, y_past_k, x_past_l, z1_past_l, ..., zM_past_l)`` of
    alphabet ``K``, total size ``K^(1 + k + (1 + M) * l)``. Miller-Madow
    on occupied cells. The CTE identity used is

        CTE = H(y_future | y_past, z_past)
              - H(y_future | y_past, x_past, z_past)

    Symbolically, with joint ``q(y_f, y_p, x_p, z_p)``:

        CTE = sum q * log[ q(y_f, y_p, x_p, z_p) * q(y_p, z_p)
                           / ( q(y_p, x_p, z_p) * q(y_f, y_p, z_p) ) ]
    """
    if K < 2:
        raise ValueError("alphabet K must be >= 2")
    if history_k < 1 or history_l < 1:
        raise ValueError("history_k and history_l must be >= 1")
    if delay < 1:
        raise ValueError("delay must be >= 1")
    M = len(z_list)
    k, l = int(history_k), int(history_l)
    embed = delay + max(k, l) - 1
    T = y.size
    N = T - embed
    if N <= 0:
        return 0.0
    base = np.arange(N) + embed

    y_future = y[base]
    y_past = np.stack(
        [y[base - delay - h] for h in range(k)], axis=1,
    )  # (N, k)
    x_past = np.stack(
        [x[base - delay - h] for h in range(l)], axis=1,
    )  # (N, l)
    # Condition past uses lags 1..l (independent of the source delay), so
    # that the immediate screener of an indirect-coupling chain
    # ``X[t-delay] -> Z[t-1] -> Y[t]`` is captured even when ``delay > 1``.
    # This is the same convention IDTxl uses for multivariate-TE conditioning.
    z_past = [
        np.stack([z[base - 1 - h] for h in range(l)], axis=1)
        for z in z_list
    ]  # list of (N, l)
    z_block = (np.concatenate(z_past, axis=1) if z_past
               else np.zeros((N, 0), dtype=np.int64))

    n_dims = 1 + k + l + M * l
    cols = np.concatenate([y_future[:, None], y_past, x_past, z_block], axis=1)
    powers = K ** np.arange(n_dims - 1, -1, -1, dtype=np.int64)
    flat = (cols * powers[None, :]).sum(axis=1)
    M_total = int(K ** n_dims)
    counts = np.bincount(flat, minlength=M_total).astype(np.float64)
    p = counts / N
    shape = (K,) * n_dims
    p_full = p.reshape(shape)

    yf_axis = 0
    x_past_axes = tuple(range(1 + k, 1 + k + l))

    p_yp_xp_zp = p_full.sum(axis=yf_axis)
    p_yf_yp_zp = p_full.sum(axis=x_past_axes)
    p_yp_zp = p_full.sum(axis=(yf_axis,) + x_past_axes)

    full_slice = (slice(None),) * n_dims

    def _expand(arr: np.ndarray, drop_axes: tuple[int, ...]) -> np.ndarray:
        idx = list(full_slice)
        for a in drop_axes:
            idx[a] = None
        return arr[tuple(idx)]

    p_yp_xp_zp_b = _expand(p_yp_xp_zp, (yf_axis,))
    p_yf_yp_zp_b = _expand(p_yf_yp_zp, x_past_axes)
    p_yp_zp_b = _expand(p_yp_zp, (yf_axis,) + x_past_axes)

    num = p_full * p_yp_zp_b
    den = p_yp_xp_zp_b * p_yf_yp_zp_b
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = (p_full > 0) & (den > 0) & (num > 0)
        ratio = np.where(mask, num / den, 1.0)
        log_ratio = np.where(mask, np.log(ratio), 0.0)
    cte = float(np.sum(p_full * log_ratio))

    if bias == "none":
        correction = 0.0
    elif bias in ("miller_madow", "roulston"):
        # Miller-Madow on the expanded joint. Roulston is aliased here
        # because the multi-factor product form has no canonical extension
        # to the multi-condition case.
        m = int(np.sum(counts > 0))
        correction = (m - 1) / (2.0 * N)
    else:
        raise ValueError(
            f"unknown bias={bias!r}; choose 'miller_madow', 'roulston', or 'none'"
        )
    cte_raw = max(0.0, cte)
    return max(0.0, cte_raw - correction)


def conditional_transfer_entropy(
    rec: SpikeRecording,
    *,
    source: str,
    target: str,
    conditions: Sequence[str],
    bin_size_ms: float = 5.0,
    delay_bins: int = 1,
    bias: str = "miller_madow",
    history_k: int = 1,
    history_l: int = 1,
    discretize: str = "binary",
    n_quantile_bins: int = 3,
) -> ConditionalTransferEntropyResult:
    """Conditional transfer entropy ``CTE(source -> target | conditions)``.

    Removes the part of bivariate TE(source -> target) explained by the
    past of ``conditions``. Use this to dissolve indirect-coupling false
    positives in a pairwise TE matrix when a common driver or relay is
    observed: if ``X -> Z -> Y`` with no direct ``X -> Y`` arrow, pairwise
    ``TE(X -> Y)`` is positive but ``CTE(X -> Y | Z)`` is near zero.

    Parameters
    ----------
    source, target
        Population names.
    conditions
        One or more population names to condition on (at least one).
    bin_size_ms, delay_bins, bias, discretize, n_quantile_bins
        Same semantics as :func:`transfer_entropy`.
    history_k, history_l
        Target-past and source-/condition-past embedding lengths.

    Notes
    -----
    Joint-table memory is ``K^(1 + k + (1 + M) * l)`` where
    ``M = len(conditions)``, ``k = history_k``, ``l = history_l``. Stay
    near ``K = 2``, ``k = l = 1``, ``M = 1`` for spike-train workloads;
    larger embeddings will exhaust memory.
    """
    from neurocomplexity._warnings import (
        _warn_if_nonstationary,
        _warn_if_uncurated,
    )
    _warn_if_uncurated(rec, "conditional_transfer_entropy")
    _warn_if_nonstationary(rec, "conditional_transfer_entropy")
    conditions = list(conditions)
    if not conditions:
        raise ValueError(
            "conditional_transfer_entropy requires at least one condition; "
            "for pairwise TE call transfer_entropy()."
        )
    if source == target:
        raise ValueError("source and target must differ")
    if source in conditions or target in conditions:
        raise ValueError("conditions must not include source or target")
    if discretize not in {"binary", "quantile"}:
        raise ValueError(
            f"unknown discretize={discretize!r}; choose 'binary' or 'quantile'"
        )
    bs = float(bin_size_ms) / 1000.0
    streams = [target, source] + list(conditions)
    counts = bin_spikes(rec, streams, bs)  # (T, len(streams))
    if discretize == "binary":
        symbols = (counts > 0).astype(np.int64)
        K = 2
    else:
        from neurocomplexity.analysis.pid import _quantile_discretise
        K = int(n_quantile_bins)
        if K < 2:
            raise ValueError("n_quantile_bins must be >= 2")
        symbols = np.stack(
            [
                _quantile_discretise(counts[:, p].astype(np.float64), K)
                for p in range(counts.shape[1])
            ],
            axis=1,
        ).astype(np.int64)

    y = symbols[:, 0]
    x = symbols[:, 1]
    z_list = [symbols[:, 2 + i] for i in range(len(conditions))]
    value = _cte_general(
        y, x, z_list, K=K, delay=int(delay_bins), bias=bias,
        history_k=int(history_k), history_l=int(history_l),
    )

    return ConditionalTransferEntropyResult(
        value=float(value),
        source=source,
        target=target,
        conditions=tuple(conditions),
        bin_size_seconds=bs,
        delay_bins=int(delay_bins),
        history_k=int(history_k),
        history_l=int(history_l),
        source_rec=rec.source,
        params={
            "source": source,
            "target": target,
            "conditions": list(conditions),
            "bin_size_ms": float(bin_size_ms),
            "delay_bins": int(delay_bins),
            "bias": bias,
            "discretize": discretize,
            "n_quantile_bins": int(n_quantile_bins),
            "history_k": int(history_k),
            "history_l": int(history_l),
        },
    )
