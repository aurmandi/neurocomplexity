"""QualityControlWarning and helpers for guarding analyses against uncurated recordings."""
from __future__ import annotations

import threading
import warnings as _warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neurocomplexity.core.recording import SpikeRecording

# Single lock guarding all dedup sets; warnings are emitted off the analysis
# hot path so contention is negligible, and a shared lock keeps the two sets
# coherent under thread-pool dispatch (e.g. transfer_entropy(n_jobs>1)).
_dedup_lock = threading.Lock()


def _dedup_key(rec: SpikeRecording, analysis_name: str) -> tuple[str, str]:
    """Stable per-recording dedup key.

    Uses ``rec.source.source_hash`` (content hash of the on-disk source) so
    the key survives object re-creation and pickling, unlike the old
    ``id(rec)``. Memory recordings carry an empty ``source_hash``; for those
    we fall back to the object id so distinct in-memory recordings still
    dedup independently within the session.
    """
    source_hash = getattr(getattr(rec, "source", None), "source_hash", "") or ""
    anchor = source_hash if source_hash else f"mem:{id(rec)}"
    return (anchor, analysis_name)


class QualityControlWarning(UserWarning):
    """Emitted when an analysis runs on a recording showing no evidence of curation or QC filtering.

    Trigger condition: none of {quality, presence_ratio, isi_violations_ratio,
    KSLabel} columns are present in ``rec.units`` AND ``rec._filtered`` is False.

    Deduplicated per ``(rec.source.source_hash, analysis_name)`` within a
    Python session (object id used as fallback for memory recordings). Call
    ``nc.warnings.reset()`` to clear the dedup state and re-surface warnings.
    """


class MemoryAllocationWarning(UserWarning):
    """Emitted when ``bin_spikes`` is about to allocate a counts matrix
    larger than 25% of available RAM. Suggest ``chunk_seconds=...`` or
    ``rec.crop(...)``."""


class StationarityWarning(UserWarning):
    """Emitted when a stationarity-sensitive analysis (criticality, branching,
    transfer-entropy, shape-collapse) runs on a recording that
    ``analysis.stationarity`` has flagged as non-stationary.

    Deduplicated per ``(rec.source.source_hash, analysis_name)`` (object id
    fallback for memory recordings). Call ``nc.warnings.reset()`` to clear.
    """


_CURATION_COLUMNS = ("quality", "presence_ratio", "isi_violations_ratio", "KSLabel")

_MESSAGE = (
    "Running {analysis} on a recording with no curation or QC columns and "
    "no filtering applied. Results from uncurated sorter output are dominated "
    "by noise units and MUA; published analyses should run on curated/QC-filtered "
    "units only. Attach quality with `nc.io.add_quality(...)` then call "
    "`rec.filter_units(quality='good')`, or curate in Phy and reload with "
    "`nc.io.from_phy(...)`. Suppress with "
    "`warnings.filterwarnings('ignore', category=nc.warnings.QualityControlWarning)` "
    "only if you are certain."
)

_seen: set[tuple[str, str]] = set()


def _reset_dedup() -> None:
    """Clear the per-session quality-control dedup set."""
    with _dedup_lock:
        _seen.clear()


def _warn_if_uncurated(rec: SpikeRecording, analysis_name: str) -> None:
    if rec._filtered:
        return
    cols = set(rec.units.columns)
    if any(c in cols for c in _CURATION_COLUMNS):
        return
    key = _dedup_key(rec, analysis_name)
    with _dedup_lock:
        if key in _seen:
            return
        _seen.add(key)
    _warnings.warn(_MESSAGE.format(analysis=analysis_name),
                   category=QualityControlWarning, stacklevel=3)


_STATIONARITY_MESSAGE = (
    "Running {analysis} on a recording flagged as non-stationary by "
    "`nc.analysis.stationarity` ({reasons}). Stationarity-sensitive statistics "
    "(criticality exponents, branching ratio, transfer entropy, shape-collapse "
    "gamma) can be biased by rate drift or heteroskedasticity. Inspect with "
    "`nc.analysis.stationarity(rec)`, restrict to a stationary epoch via "
    "`rec.crop(...)`, or suppress with "
    "`warnings.filterwarnings('ignore', category=nc.warnings.StationarityWarning)` "
    "if you are certain."
)

_stationarity_seen: set[tuple[str, str]] = set()


def _reset_stationarity_dedup() -> None:
    """Clear the per-session stationarity dedup set."""
    with _dedup_lock:
        _stationarity_seen.clear()


def reset() -> None:
    """Clear all per-session warning dedup state.

    Public counterpart to the internal ``_reset_*`` helpers: after this call
    every guarded analysis will re-emit its quality-control / stationarity
    warning the next time it runs, even on a recording already seen this
    session. Useful in notebooks and long-lived processes that re-run the
    same recording and want the disclosure surfaced again.
    """
    with _dedup_lock:
        _seen.clear()
        _stationarity_seen.clear()


def _warn_if_nonstationary(rec: SpikeRecording, analysis_name: str) -> None:
    """Run a default ``stationarity`` check and warn once if non-stationary."""
    key = _dedup_key(rec, analysis_name)
    with _dedup_lock:
        if key in _stationarity_seen:
            return
        _stationarity_seen.add(key)
    try:
        from neurocomplexity.analysis.stationarity import stationarity
        result = stationarity(rec)
    except Exception:
        return
    if result.is_stationary:
        return
    reasons = "; ".join(result.flags) if result.flags else "see flags"
    _warnings.warn(
        _STATIONARITY_MESSAGE.format(analysis=analysis_name, reasons=reasons),
        category=StationarityWarning, stacklevel=3,
    )
