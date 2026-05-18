"""add_quality: attach QC tables from Bombcell / ecephys_spike_sorting / SpikeInterface."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.io._sniff import sniff_qc_format


FormatLiteral = Literal["auto", "bombcell", "ecephys", "spikeinterface"]


_BOMBCELL_QUALITY_MAP = {1: "good", 2: "mua", 0: "noise"}


def _normalise_bombcell(df: pd.DataFrame) -> pd.DataFrame:
    """Map Bombcell columns to the canonical schema. ``cluster_id`` becomes ``id``."""
    out = pd.DataFrame({"id": df["cluster_id"].to_numpy(dtype=np.int64)})
    out["quality"] = df["unitType"].map(_BOMBCELL_QUALITY_MAP).astype("string")
    out["presence_ratio"] = df.get("presenceRatio", pd.Series([np.nan] * len(df))).astype(float)
    out["isi_violations_ratio"] = df.get(
        "fractionRPVs_estimatedTauR",
        pd.Series([np.nan] * len(df))
    ).astype(float)
    out["refractory_period_violations_ratio"] = out["isi_violations_ratio"]
    out["amplitude_cutoff"] = df.get(
        "percentageSpikesMissing_gaussian",
        pd.Series([np.nan] * len(df))
    ).astype(float) / 100.0
    out["signal_to_noise"] = df.get(
        "signalToNoiseRatio",
        pd.Series([np.nan] * len(df))
    ).astype(float)
    out["firing_rate"] = df.get("firingRate", pd.Series([np.nan] * len(df))).astype(float)
    out["qc_source"] = "bombcell"
    return out


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ""}:
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported QC file extension: {suffix}")


def add_quality(
    rec: SpikeRecording,
    path: str | os.PathLike,
    *,
    format: FormatLiteral = "auto",
) -> SpikeRecording:
    """Left-join a QC table onto ``rec.units`` by unit id.

    Returns a new immutable recording with canonical QC columns and an
    appended provenance attachment.
    """
    path = Path(path)
    raw = _load_table(path)

    detected = format
    if format == "auto":
        detected = sniff_qc_format(raw)
        if detected is None:
            raise ValueError(
                f"could not auto-detect QC format for {path.name}; "
                f"columns present: {list(raw.columns)[:8]}...; "
                f"pass format='bombcell'|'ecephys'|'spikeinterface' explicitly"
            )

    if detected == "bombcell":
        norm = _normalise_bombcell(raw)
    elif detected == "ecephys":
        norm = _normalise_ecephys(raw)
    elif detected == "spikeinterface":
        norm = _normalise_spikeinterface(raw)
    else:
        raise ValueError(f"unknown QC format: {detected!r}")

    qc_ids = set(norm["id"].tolist())
    rec_ids = set(rec.units["id"].tolist())
    missing_in_qc = rec_ids - qc_ids
    missing_in_rec = qc_ids - rec_ids
    if missing_in_qc or missing_in_rec:
        m1 = sorted(missing_in_qc)[:10]
        m2 = sorted(missing_in_rec)[:10]
        raise ValueError(
            f"unit_id mismatch between recording and QC file: "
            f"{len(missing_in_qc)} ids in recording but not QC (first: {m1}); "
            f"{len(missing_in_rec)} ids in QC but not recording (first: {m2})"
        )

    drop_cols = [c for c in norm.columns if c != "id" and c in rec.units.columns]
    base = rec.units.drop(columns=drop_cols)
    merged = base.merge(norm, on="id", how="left").reset_index(drop=True)

    attachment = ProvenanceRecord.for_file(path, source_format=f"quality:{detected}")
    return replace(rec, units=merged, attachments=rec.attachments + (attachment,))


# Stubs that raise NotImplementedError until Tasks 6 & 7 fill them in.
def _normalise_ecephys(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("ecephys QC normalisation pending Task 6")


def _normalise_spikeinterface(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("spikeinterface QC normalisation pending Task 7")
