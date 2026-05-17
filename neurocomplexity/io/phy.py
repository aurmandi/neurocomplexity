"""Loader for Phy curation directories.

Phy is the standard interactive curation GUI applied on top of Kilosort
output. After curation, Phy writes ``cluster_info.tsv`` with the final
``group`` column (good / mua / noise / unsorted). This loader prefers
that file and falls back to ``cluster_group.tsv`` if the user has only
the minimal export.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.io._sorter_output import _load_sorter_output


def from_phy(
    directory: str | Path,
    *,
    duration: float | None = None,
    populations: Mapping[str, np.ndarray] | None = None,
) -> SpikeRecording:
    """Build a SpikeRecording from a Phy curation directory.

    Parameters
    ----------
    directory
        Path containing ``spike_times.npy``, ``spike_clusters.npy``,
        ``params.py``, and ``cluster_info.tsv`` (or ``cluster_group.tsv``).
    duration
        Override the recording duration in seconds. Default is
        ``max(spike_times) + 1.0``.
    populations
        Override the default ``{"all": ones}`` population mask.

    Notes
    -----
    No quality filtering is applied at load time; call
    ``rec.filter_units(quality=['good'])`` downstream to drop MUA/noise.
    """
    return _load_sorter_output(
        Path(directory),
        label_source="phy",
        duration=duration,
        populations=populations,
    )
