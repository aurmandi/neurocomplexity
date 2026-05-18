"""add_anatomy: attach channel -> brain-region tables from SHARP-Track / Brainglobe / Pinpoint / CSV."""
from __future__ import annotations

import json
import os
import warnings as _warnings
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.io._sniff import sniff_anatomy_format


FormatLiteral = Literal["auto", "sharptrack", "brainglobe", "pinpoint", "csv"]


def _normalise_brainglobe(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "peak_channel": df["Channel"].to_numpy(dtype=np.int64),
        "brain_area": df["Brain region acronym"].astype("string"),
        "brain_area_full": df["Brain region"].astype("string"),
        "ccf_ap": df.get("AP", pd.Series([np.nan] * len(df))).astype(float),
        "ccf_dv": df.get("DV", pd.Series([np.nan] * len(df))).astype(float),
        "ccf_ml": df.get("ML", pd.Series([np.nan] * len(df))).astype(float),
        "anatomy_source": "brainglobe",
    })
    return out


def _normalise_csv(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {c.lower(): c for c in df.columns}
    chan_col = colmap.get("channel")
    area_col = colmap.get("area") or colmap.get("brain_area")
    full_col = colmap.get("brain_area_full") or colmap.get("area_full")
    out = pd.DataFrame({
        "peak_channel": df[chan_col].to_numpy(dtype=np.int64),
        "brain_area": df[area_col].astype("string"),
        "brain_area_full": (
            df[full_col].astype("string") if full_col else df[area_col].astype("string")
        ),
        "ccf_ap": df[colmap["ap"]].astype(float) if "ap" in colmap else np.nan,
        "ccf_dv": df[colmap["dv"]].astype(float) if "dv" in colmap else np.nan,
        "ccf_ml": df[colmap["ml"]].astype(float) if "ml" in colmap else np.nan,
        "anatomy_source": "csv",
    })
    return out


def _normalise_pinpoint(df: pd.DataFrame) -> pd.DataFrame:
    coords = df["coordinates"].tolist()
    ap = [c[0] if c is not None and len(c) > 0 else np.nan for c in coords]
    dv = [c[1] if c is not None and len(c) > 1 else np.nan for c in coords]
    ml = [c[2] if c is not None and len(c) > 2 else np.nan for c in coords]
    out = pd.DataFrame({
        "peak_channel": df["channel"].to_numpy(dtype=np.int64),
        "brain_area": df["area"].astype("string"),
        "brain_area_full": df.get("area_full", df["area"]).astype("string"),
        "ccf_ap": ap, "ccf_dv": dv, "ccf_ml": ml,
        "anatomy_source": "pinpoint",
    })
    return out


def _load_anatomy_table(path: Path, format_hint: str) -> tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()
    if format_hint == "sharptrack" or suffix == ".mat":
        from neurocomplexity.io._anatomy_sharptrack import load_sharptrack
        return load_sharptrack(path), "sharptrack"
    if format_hint == "pinpoint" or suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df, "pinpoint"
    if suffix in {".csv", ""}:
        return pd.read_csv(path), "auto"
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t"), "auto"
    raise ValueError(f"unsupported anatomy file extension: {suffix}")


def add_anatomy(
    rec: SpikeRecording,
    path: str | os.PathLike,
    *,
    format: FormatLiteral = "auto",
) -> SpikeRecording:
    path = Path(path)
    raw, format_after_load = _load_anatomy_table(path, format)

    if format == "auto":
        if format_after_load != "auto":
            detected = format_after_load
        else:
            detected = sniff_anatomy_format(raw)
            if detected is None:
                raise ValueError(
                    f"could not auto-detect anatomy format for {path.name}; "
                    f"columns present: {list(raw.columns)[:8]}...; "
                    f"pass format='brainglobe'|'pinpoint'|'sharptrack'|'csv' explicitly"
                )
    else:
        detected = format

    if detected == "brainglobe":
        norm = _normalise_brainglobe(raw)
    elif detected == "csv":
        norm = _normalise_csv(raw)
    elif detected == "pinpoint":
        norm = _normalise_pinpoint(raw)
    elif detected == "sharptrack":
        norm = _normalise_brainglobe(raw)
        norm["anatomy_source"] = "sharptrack"
    else:
        raise ValueError(f"unknown anatomy format: {detected!r}")

    if "peak_channel" not in rec.units.columns:
        raise ValueError("recording.units must have a 'peak_channel' column to attach anatomy")

    rec_channels = set(rec.units["peak_channel"].dropna().astype(int).tolist())
    anat_channels = set(norm["peak_channel"].tolist())
    missing = rec_channels - anat_channels
    if missing:
        _warnings.warn(
            f"{len(missing)} channel(s) in recording have no anatomy entry "
            f"(first 5: {sorted(missing)[:5]}); their brain_area will be NaN",
            UserWarning, stacklevel=2,
        )

    drop_cols = [c for c in norm.columns if c != "peak_channel" and c in rec.units.columns]
    base = rec.units.drop(columns=drop_cols)
    merged = base.merge(norm, on="peak_channel", how="left").reset_index(drop=True)

    attachment = ProvenanceRecord.for_file(path, source_format=f"anatomy:{detected}")
    return replace(rec, units=merged, attachments=rec.attachments + (attachment,))
