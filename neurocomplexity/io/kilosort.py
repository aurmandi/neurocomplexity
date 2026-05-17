"""Loader for raw Kilosort output directories (no Phy curation)."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.io._sorter_output import _load_sorter_output


def from_kilosort(
    directory: str | Path,
    *,
    duration: float | None = None,
    populations: Mapping[str, np.ndarray] | None = None,
) -> SpikeRecording:
    """Build a SpikeRecording from a raw Kilosort output directory.

    Reads automatic quality labels from ``cluster_KSLabel.tsv``. If you
    have already run Phy curation on this directory use ``from_phy``
    instead so the curated ``group`` column is used.
    """
    return _load_sorter_output(
        Path(directory),
        label_source="kilosort",
        duration=duration,
        populations=populations,
    )
