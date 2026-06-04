"""Partial Information Decomposition (Williams & Beer 2010, I_min).

Decomposes I(target ; (source1, source2)) into four atoms:
  * unique_1, unique_2 — information about target only one source carries
  * redundancy        — information both sources share about target
  * synergy           — information only their joint observation reveals

Implementation:
  * Spike counts in each bin are discretised into ``n_levels`` states using
    quantile-equal bins per population. Default ``n_levels=3`` avoids the
    binary-saturation failure mode (when a busy population is active in
    nearly every bin, the binary entropy collapses and all PID atoms vanish).
  * The joint distribution lives on a ``L x L x L`` lattice where L = n_levels.
  * Williams-Beer I_min redundancy is used; the other three atoms follow from
    the PID identities.
  * Every MI term is Miller-Madow bias corrected:  I_MM = I_plug - (m-1)/(2N)
    where m is the number of joint cells with non-zero probability and N the
    sample size — same correction we apply to the binary TE estimator.

References
----------
Williams P.L. & Beer R.D. (2010). Nonnegative Decomposition of Multivariate
Information. arXiv:1004.2515.
Miller G. (1955). Note on the bias of information estimates.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from neurocomplexity.analysis._binning import bin_spikes
from neurocomplexity.core.recording import SpikeRecording


@dataclass(frozen=True)
class PIDResult:
    """Output of :func:`partial_information` (Williams & Beer I_min PID).

    Decomposes the joint mutual information
    ``I(target ; (source1, source2))`` into four non-negative atoms whose
    sum equals ``total_mi``.

    Attributes
    ----------
    redundancy
        Information about ``target`` carried by **both** sources, estimated
        by the Williams & Beer (2010) ``I_min`` redundancy functional.
        **Known limitation.** ``I_min`` measures redundancy as the *minimum*
        specific information any source carries about each target outcome
        and therefore *upper-bounds* the redundant information when sources
        carry information about *different* target outcomes. Synergy
        reported by this estimator is correspondingly biased downward.
        See ``docs/complexity_measures.md`` § "Williams-Beer I_min
        limitation" and Bertschinger et al. (2014, *Entropy* 16) for the
        BROJA alternative. The estimator is retained because it is the
        only PID with a closed form on three variables; treat large
        ``redundancy`` values cautiously when sources are likely to encode
        different aspects of the target.
    unique_1, unique_2
        Information about ``target`` carried by source 1 (resp. source 2)
        and not by the other.
    synergy
        Information about ``target`` that only the **joint** observation of
        both sources reveals.
    total_mi
        ``redundancy + unique_1 + unique_2 + synergy``. Equals
        ``I(target ; (s1, s2))`` up to bias correction.
    target
        Name of the target population.
    sources
        Names of the two source populations.
    bin_size_seconds
        Bin size used to discretise spike counts.
    n_levels
        Number of quantile-equal discretisation levels per population
        (default 3; binary saturation avoided).
    source
        Provenance back-pointer.
    params
        Verbatim copy of the keyword arguments passed to
        :func:`partial_information`.
    """

    redundancy: float
    unique_1: float
    unique_2: float
    synergy: float
    total_mi: float
    target: str
    sources: tuple[str, str]
    bin_size_seconds: float
    n_levels: int
    source: object
    params: dict = field(default_factory=dict)


def _quantile_discretise(x: np.ndarray, n_levels: int) -> np.ndarray:
    """Map integer counts to {0,..,n_levels-1} using quantile edges.

    Falls back gracefully when the distribution has fewer distinct values than
    levels (e.g. mostly-silent population): unique values are used as cut
    points, never producing more than ``n_levels`` states.
    """
    if n_levels < 2:
        raise ValueError("n_levels must be >= 2")
    x = x.astype(np.float64)
    if np.all(x == x[0]):
        return np.zeros_like(x, dtype=np.int8)
    qs = np.linspace(0, 1, n_levels + 1)[1:-1]
    edges = np.unique(np.quantile(x, qs))
    # right=True: edges[i-1] < x <= edges[i].  This is the convention that
    # works for integer count data (e.g. edges=[0,1] gives 3 bins
    # {x<=0, 0<x<=1, x>1}); right=False would collapse x=0 and x=1 into
    # the same upper bin and destroy half the dynamic range.
    return np.digitize(x, edges, right=True).astype(np.int8)


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _mi_mm(joint_counts: np.ndarray) -> float:
    """Mutual information from a 2-D joint count table, Miller-Madow corrected.

    joint_counts shape (a, b); returns I(A;B) in nats.
    """
    N = joint_counts.sum()
    if N <= 0:
        return 0.0
    p = joint_counts / N
    pa = p.sum(axis=1)
    pb = p.sum(axis=0)
    h_a = _entropy(pa)
    h_b = _entropy(pb)
    h_ab = _entropy(p.ravel())
    mi = h_a + h_b - h_ab
    # Miller-Madow bias correction on the joint (the dominant bias term)
    m = int(np.count_nonzero(p))
    mi -= (m - 1) / (2.0 * N)
    return float(max(mi, 0.0))


def _specific_info(target_val: int, src_axis: int,
                   joint: np.ndarray) -> float:
    """I(Y=y ; S_axis). joint shape (L_y, L_s1, L_s2)."""
    j = joint / joint.sum()
    p_y = j.sum(axis=(1, 2))
    if p_y[target_val] == 0:
        return 0.0
    other = 3 - src_axis  # 1<->2
    p_sy = j.sum(axis=other)             # (L_y, L_s)
    p_s = p_sy.sum(axis=0)               # (L_s,)
    out = 0.0
    p_yv = p_y[target_val]
    for sv in range(p_sy.shape[1]):
        p_sv_yv = p_sy[target_val, sv]
        if p_sv_yv == 0:
            continue
        p_sv = p_s[sv]
        if p_sv == 0:
            continue
        p_s_given_y = p_sv_yv / p_yv
        out += p_s_given_y * (np.log(1.0 / p_sv) - np.log(1.0 / p_s_given_y))
    return float(out)


def _redundancy_imin(joint: np.ndarray) -> float:
    p_y = (joint / joint.sum()).sum(axis=(1, 2))
    r = 0.0
    for yv in range(p_y.size):
        if p_y[yv] == 0:
            continue
        i1 = _specific_info(yv, 1, joint)
        i2 = _specific_info(yv, 2, joint)
        r += p_y[yv] * min(i1, i2)
    return float(max(r, 0.0))


def _redundancy_iccs(joint: np.ndarray) -> float:
    """Ince (2017) I_ccs redundancy: pointwise common change in surprisal.

    For each joint outcome ``(y, s1, s2)`` with positive probability,
    compute the pointwise informations
    ``i_k = log p(y | s_k) - log p(y)`` from each source ``k``. The
    pointwise common change in surprisal is

        Δh_common = sign · min(|i1|, |i2|)   if sign(i1) == sign(i2)
                  = 0                         otherwise

    and ``I_ccs = sum p(y, s1, s2) * Δh_common``. This is the redundancy
    measure that, unlike Williams & Beer I_min, correctly assigns zero
    redundancy to the canonical XOR distribution (Ince 2017, §3).

    Reference: Ince, R. A. A. (2017). Measuring Multivariate Redundant
    Information with Pointwise Common Change in Surprisal. *Entropy*,
    19(7), 318. https://doi.org/10.3390/e19070318
    """
    total = float(joint.sum())
    if total <= 0:
        return 0.0
    p_xyz = joint / total                       # (L_y, L_s1, L_s2)
    p_y = p_xyz.sum(axis=(1, 2))                # (L_y,)
    p_s1 = p_xyz.sum(axis=(0, 2))               # (L_s1,)
    p_s2 = p_xyz.sum(axis=(0, 1))               # (L_s2,)
    p_y_s1 = p_xyz.sum(axis=2)                  # (L_y, L_s1)
    p_y_s2 = p_xyz.sum(axis=1)                  # (L_y, L_s2)

    r = 0.0
    L_y, L_s1, L_s2 = p_xyz.shape
    for yv in range(L_y):
        if p_y[yv] <= 0:
            continue
        log_py = np.log(p_y[yv])
        for v1 in range(L_s1):
            if p_y_s1[yv, v1] <= 0 or p_s1[v1] <= 0:
                continue
            # i1 = log p(y | s1) - log p(y)
            i1 = np.log(p_y_s1[yv, v1] / p_s1[v1]) - log_py
            for v2 in range(L_s2):
                if p_xyz[yv, v1, v2] <= 0:
                    continue
                if p_y_s2[yv, v2] <= 0 or p_s2[v2] <= 0:
                    continue
                i2 = np.log(p_y_s2[yv, v2] / p_s2[v2]) - log_py
                if i1 >= 0 and i2 >= 0:
                    common = min(i1, i2)
                elif i1 < 0 and i2 < 0:
                    common = max(i1, i2)  # both negative → least-negative
                else:
                    continue  # disagreeing signs → no common change
                r += float(p_xyz[yv, v1, v2]) * float(common)
    return float(max(r, 0.0))


def partial_information(rec: SpikeRecording,
                         target_pop: str,
                         sources: Sequence[str],
                         bin_size_ms: float = 5.0,
                         delay_bins: int = 1,
                         n_levels: int = 3,
                         *,
                         redundancy: str = "imin",
                         ) -> PIDResult:
    """Bivariate PID — sources must be exactly 2 populations.

    Parameters
    ----------
    bin_size_ms : bin width in ms (default 5).
    delay_bins  : how many bins to shift sources back relative to target
                  (default 1 — predictive PID).
    n_levels    : number of quantile-equal discretisation levels per signal
                  (default 3). Use 2 to recover the legacy binary estimator,
                  but expect zero MI on busy populations.
    redundancy  : redundancy functional.

                  * ``"imin"`` (default, backwards-compatible) — the
                    Williams & Beer (2010) ``I_min`` measure: takes the
                    minimum specific information about each target outcome.
                    Closed-form and conservative; reports nonzero
                    redundancy on canonical XOR.
                  * ``"iccs"`` — the Ince (2017) ``I_ccs`` measure:
                    pointwise common change in surprisal. Assigns
                    redundancy only when both sources change a target's
                    pointwise information in the same direction, and
                    correctly returns ≈ 0 redundancy on canonical XOR.

                  Both estimators respect the PID identities
                  ``U_k = I(Y; S_k) − R`` and
                  ``S = I(Y; S1, S2) − R − U1 − U2`` with the standard
                  clipping at zero.
    """
    from neurocomplexity._warnings import _warn_if_uncurated
    _warn_if_uncurated(rec, "partial_information")
    if len(sources) != 2:
        raise ValueError("PID v0 supports exactly 2 sources")
    if n_levels < 2:
        raise ValueError("n_levels must be >= 2")
    if redundancy not in {"imin", "iccs"}:
        raise ValueError(
            f"unknown redundancy={redundancy!r}; choose 'imin' or 'iccs'"
        )
    s1, s2 = sources
    bs = float(bin_size_ms) / 1000.0

    def _series_for(name: str) -> np.ndarray:
        """Return a 1-D float array per bin for either a population or a signal."""
        if name in rec.populations:
            return bin_spikes(rec, [name], bs)[:, 0].astype(np.float64)
        if name in rec.signals:
            from neurocomplexity.analysis._continuous import bin_signal_average
            T = int(np.floor(rec.duration / bs))
            arr = bin_signal_average(rec.signals[name], bin_size_s=bs, n_bins=T)
            # Fill NaN with the global mean so quantile-discretise sees a clean array
            mean = float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else 0.0
            arr = np.where(np.isnan(arr), mean, arr)
            return arr
        raise ValueError(
            f"unknown stream {name!r}; not in rec.populations or rec.signals"
        )

    y_series = _series_for(target_pop)
    s1_series = _series_for(s1)
    s2_series = _series_for(s2)
    Y  = _quantile_discretise(y_series, n_levels)
    S1 = _quantile_discretise(s1_series, n_levels)
    S2 = _quantile_discretise(s2_series, n_levels)
    if delay_bins > 0:
        Y  = Y[delay_bins:]
        S1 = S1[:-delay_bins]
        S2 = S2[:-delay_bins]
    N = Y.size
    if N < 10:
        raise ValueError("too few bins for PID")

    L_y  = int(Y.max()) + 1
    L_s1 = int(S1.max()) + 1
    L_s2 = int(S2.max()) + 1

    flat = (Y.astype(np.int64) * (L_s1 * L_s2)
            + S1.astype(np.int64) * L_s2
            + S2.astype(np.int64))
    bc = np.bincount(flat, minlength=L_y * L_s1 * L_s2).astype(np.float64)
    joint = bc.reshape(L_y, L_s1, L_s2)

    # Marginal joints for unique terms
    j_s1 = joint.sum(axis=2)  # (L_y, L_s1)
    j_s2 = joint.sum(axis=1)  # (L_y, L_s2)
    i_y_s1 = _mi_mm(j_s1)
    i_y_s2 = _mi_mm(j_s2)
    # Joint (S1, S2) as a single discrete variable for total MI
    total_mi = _mi_mm(joint.reshape(L_y, L_s1 * L_s2))

    if redundancy == "imin":
        red_value = _redundancy_imin(joint)
    else:  # "iccs"
        red_value = _redundancy_iccs(joint)
    # Clip redundancy to be no larger than min(I(Y;S1), I(Y;S2)) — guaranteed
    # for both I_min and I_ccs in theory but float arithmetic can violate it
    # after MM correction; the clip preserves the standard PID identities.
    red_value = min(red_value, i_y_s1, i_y_s2)

    unique_1 = max(0.0, i_y_s1 - red_value)
    unique_2 = max(0.0, i_y_s2 - red_value)
    synergy  = max(0.0, total_mi - red_value - unique_1 - unique_2)

    return PIDResult(
        redundancy=float(red_value),
        unique_1=float(unique_1),
        unique_2=float(unique_2),
        synergy=float(synergy),
        total_mi=float(total_mi),
        target=target_pop,
        sources=(s1, s2),
        bin_size_seconds=bs,
        n_levels=n_levels,
        source=rec.source,
        params={"target_pop": target_pop, "sources": list(sources),
                "bin_size_ms": float(bin_size_ms),
                "delay_bins": int(delay_bins), "n_levels": int(n_levels),
                "redundancy": redundancy},
    )
