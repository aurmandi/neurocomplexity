"""Shared loader for Phy / Kilosort sorter output directories."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from neurocomplexity.core.exceptions import RecordingValidationError
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording

LabelSource = Literal["phy", "kilosort"]

_RENAME = {
    "cluster_id": "id",
    "group": "quality",
    "KSLabel": "quality",
    "fr": "firing_rate",
    "ch": "peak_channel",
    "Amplitude": "amplitude",
    "ContamPct": "contam_pct",
}


def _parse_params(path: Path) -> dict:
    """Execute params.py in an empty namespace and return its globals.

    Phy/Kilosort write params.py as a small Python file. This matches the
    convention used by SpikeInterface and Phy itself. Trust assumption:
    the sorter directory is owned by the user running the loader.
    """
    if not path.exists():
        raise RecordingValidationError(f"missing params.py at {path}")
    ns: dict = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    if "sample_rate" not in ns or not isinstance(ns["sample_rate"], (int, float)):
        raise RecordingValidationError(
            f"params.py at {path} missing numeric sample_rate"
        )
    return ns


def _load_label_table(directory: Path, label_source: LabelSource) -> pd.DataFrame:
    """Load the cluster label table for the requested sorter mode."""
    if label_source == "phy":
        info_path = directory / "cluster_info.tsv"
        group_path = directory / "cluster_group.tsv"
        if info_path.exists():
            df = pd.read_csv(info_path, sep="\t")
        elif group_path.exists():
            df = pd.read_csv(group_path, sep="\t")
        else:
            raise RecordingValidationError(
                f"Phy directory {directory} has neither cluster_info.tsv "
                f"nor cluster_group.tsv"
            )
    elif label_source == "kilosort":
        ks_path = directory / "cluster_KSLabel.tsv"
        if not ks_path.exists():
            raise RecordingValidationError(
                f"Kilosort directory {directory} missing cluster_KSLabel.tsv"
            )
        df = pd.read_csv(ks_path, sep="\t")
    else:
        raise ValueError(f"unknown label_source {label_source!r}")

    # Disambiguate quality source before rename: Phy's curated `group` wins
    # over Kilosort's automatic `KSLabel` when both are present; on a raw
    # Kilosort directory only KSLabel exists.
    if label_source == "phy" and "group" in df.columns and "KSLabel" in df.columns:
        df = df.drop(columns=["KSLabel"])

    return df.rename(columns=_RENAME)


def _load_spike_clusters(directory: Path) -> np.ndarray:
    """Prefer spike_clusters.npy (Phy merges applied); fall back to spike_templates.npy."""
    sc = directory / "spike_clusters.npy"
    if sc.exists():
        return np.load(sc).astype(np.int64)
    st = directory / "spike_templates.npy"
    if st.exists():
        warnings.warn(
            f"{directory}: spike_clusters.npy missing; using spike_templates.npy. "
            "Any Phy merges or splits will not be reflected.",
            UserWarning, stacklevel=3,
        )
        return np.load(st).astype(np.int64)
    raise RecordingValidationError(
        f"{directory} has neither spike_clusters.npy nor spike_templates.npy"
    )


def _load_sorter_output(
    directory: Path,
    *,
    label_source: LabelSource,
    duration: float | None,
    populations: Mapping[str, np.ndarray] | None,
) -> SpikeRecording:
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"sorter directory not found: {directory}")

    params = _parse_params(directory / "params.py")
    sample_rate = float(params["sample_rate"])

    samples_path = directory / "spike_times.npy"
    if not samples_path.exists():
        raise RecordingValidationError(f"missing spike_times.npy at {samples_path}")
    samples = np.load(samples_path).astype(np.int64).ravel()
    if samples.size == 0:
        raise RecordingValidationError(f"{directory} contains no spikes")

    clusters = _load_spike_clusters(directory).ravel()
    if clusters.shape != samples.shape:
        raise RecordingValidationError(
            f"spike_times shape {samples.shape} != clusters shape {clusters.shape}"
        )

    order = np.argsort(samples, kind="stable")
    samples = samples[order]
    clusters = clusters[order]
    spike_times = samples.astype(np.float64) / sample_rate

    units_df = _load_label_table(directory, label_source)
    if "id" not in units_df.columns:
        raise RecordingValidationError(
            f"label table for {directory} has no cluster_id/id column"
        )
    units_df["id"] = units_df["id"].astype(np.int64)

    # Add synthetic rows for any cluster present in spike_clusters but absent
    # from the label table — preserves the SpikeRecording invariant.
    known_ids = set(units_df["id"].tolist())
    extra_ids = sorted(set(int(c) for c in np.unique(clusters)) - known_ids)
    if extra_ids:
        extras = pd.DataFrame({"id": extra_ids, "quality": ["unsorted"] * len(extra_ids)})
        units_df = pd.concat([units_df, extras], ignore_index=True)

    if "quality" not in units_df.columns:
        units_df["quality"] = "unsorted"
    units_df["quality"] = units_df["quality"].fillna("unsorted")

    if duration is None:
        duration = float(spike_times.max()) + 1.0
    duration = float(duration)

    if populations is None:
        populations = {"all": np.ones(len(units_df), dtype=bool)}

    provenance = ProvenanceRecord.for_file(
        directory / "params.py", source_format=label_source,
    )

    return SpikeRecording(
        spike_times=spike_times,
        unit_ids=clusters,
        units=units_df.reset_index(drop=True),
        populations=populations,
        duration=duration,
        sampling_rate=sample_rate,
        source=provenance,
    )
