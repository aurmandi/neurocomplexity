"""QualityControlWarning and helpers for guarding analyses against uncurated recordings."""
from __future__ import annotations

import warnings as _warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neurocomplexity.core.recording import SpikeRecording


class QualityControlWarning(UserWarning):
    """Emitted when an analysis runs on a recording showing no evidence of curation or QC filtering.

    Trigger condition: none of {quality, presence_ratio, isi_violations_ratio,
    KSLabel} columns are present in ``rec.units`` AND ``rec._filtered`` is False.

    Deduplicated per ``(id(rec), analysis_name)`` within a Python session; re-creating
    the recording object resets the dedup state for it.
    """


class MemoryAllocationWarning(UserWarning):
    """Emitted when ``bin_spikes`` is about to allocate a counts matrix
    larger than 25% of available RAM. Suggest ``chunk_seconds=...`` or
    ``rec.crop(...)``."""


class StationarityWarning(UserWarning):
    """Emitted when a stationarity-sensitive analysis (criticality, branching,
    transfer-entropy, shape-collapse) runs on a recording that
    ``analysis.stationarity`` has flagged as non-stationary.

    Deduplicated per ``(id(rec), analysis_name)``.
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

_seen: set[tuple[int, str]] = set()


def _reset_dedup() -> None:
    """Test-only helper: clear the per-session dedup set."""
    _seen.clear()


def _warn_if_uncurated(rec: SpikeRecording, analysis_name: str) -> None:
    if rec._filtered:
        return
    cols = set(rec.units.columns)
    if any(c in cols for c in _CURATION_COLUMNS):
        return
    key = (id(rec), analysis_name)
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

_stationarity_seen: set[tuple[int, str]] = set()


def _reset_stationarity_dedup() -> None:
    """Test-only helper."""
    _stationarity_seen.clear()


def _warn_if_nonstationary(rec: SpikeRecording, analysis_name: str) -> None:
    """Run a default ``stationarity`` check and warn once if non-stationary."""
    key = (id(rec), analysis_name)
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
