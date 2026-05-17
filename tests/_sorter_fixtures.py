"""Helpers that write a synthetic Phy/Kilosort sorter output directory."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

DEFAULT_SAMPLE_RATE = 30000.0


def write_sorter_directory(
    directory: Path,
    spike_trains_sec: Mapping[int, np.ndarray],
    *,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    cluster_info: pd.DataFrame | None = None,
    cluster_group: pd.DataFrame | None = None,
    cluster_kslabel: pd.DataFrame | None = None,
    omit_spike_clusters: bool = False,
) -> Path:
    """Write a Phy/Kilosort-format directory at ``directory``.

    ``spike_trains_sec`` is a {cluster_id: spike_times_seconds} mapping. The
    helper converts to integer samples, builds the per-spike cluster array,
    sorts by sample index, and writes the .npy files. Any of the TSV
    arguments that are not None are written verbatim.
    """
    directory.mkdir(parents=True, exist_ok=True)

    sample_chunks: list[np.ndarray] = []
    cluster_chunks: list[np.ndarray] = []
    for cid, st in spike_trains_sec.items():
        samples = np.round(np.asarray(st, dtype=np.float64) * sample_rate).astype(np.int64)
        sample_chunks.append(samples)
        cluster_chunks.append(np.full(samples.shape, int(cid), dtype=np.int32))

    if sample_chunks:
        all_samples = np.concatenate(sample_chunks)
        all_clusters = np.concatenate(cluster_chunks)
        order = np.argsort(all_samples, kind="stable")
        all_samples = all_samples[order]
        all_clusters = all_clusters[order]
    else:
        all_samples = np.empty(0, dtype=np.int64)
        all_clusters = np.empty(0, dtype=np.int32)

    np.save(directory / "spike_times.npy", all_samples)
    if not omit_spike_clusters:
        np.save(directory / "spike_clusters.npy", all_clusters)
    np.save(directory / "spike_templates.npy", all_clusters.astype(np.int32))

    (directory / "params.py").write_text(
        f"dat_path = 'continuous.dat'\n"
        f"n_channels_dat = 384\n"
        f"dtype = 'int16'\n"
        f"offset = 0\n"
        f"sample_rate = {sample_rate}\n"
        f"hp_filtered = True\n",
        encoding="utf-8",
    )

    if cluster_info is not None:
        cluster_info.to_csv(directory / "cluster_info.tsv", sep="\t", index=False)
    if cluster_group is not None:
        cluster_group.to_csv(directory / "cluster_group.tsv", sep="\t", index=False)
    if cluster_kslabel is not None:
        cluster_kslabel.to_csv(directory / "cluster_KSLabel.tsv", sep="\t", index=False)

    return directory
