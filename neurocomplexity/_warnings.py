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


def _warn_if_uncurated(rec: "SpikeRecording", analysis_name: str) -> None:
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
