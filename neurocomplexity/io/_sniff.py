"""Format-detection helpers for add_quality / add_anatomy.

Each sniffer returns the format name (string) or None. The dispatch functions
try detectors in priority order (most-specific first) and return the first hit.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


# --- QC detectors ----------------------------------------------------------

_BOMBCELL_REQUIRED = {
    "useTheseTimesStart", "nPeaks", "rawAmplitude",
    "percentageSpikesMissing_gaussian",
}
_ECEPHYS_REQUIRED = {"cluster_id", "isi_viol", "amplitude_cutoff", "presence_ratio"}
_SI_REQUIRED = {"isi_violations_ratio", "presence_ratio"}


def _detect_bombcell(df: pd.DataFrame) -> Optional[str]:
    cols = set(df.columns)
    return "bombcell" if _BOMBCELL_REQUIRED.issubset(cols) else None


def _detect_ecephys(df: pd.DataFrame) -> Optional[str]:
    cols = set(df.columns)
    return "ecephys" if _ECEPHYS_REQUIRED.issubset(cols) else None


def _detect_spikeinterface(df: pd.DataFrame) -> Optional[str]:
    cols = set(df.columns)
    if not _SI_REQUIRED.issubset(cols):
        return None
    # Disambiguate from ecephys: SI does not use the ecephys-specific column names.
    if "isi_viol" in cols or "cluster_id" in cols:
        return None
    return "spikeinterface"


def sniff_qc_format(df: pd.DataFrame) -> Optional[str]:
    """Try detectors in priority order; return the first matching format name."""
    for detector in (_detect_bombcell, _detect_ecephys, _detect_spikeinterface):
        result = detector(df)
        if result is not None:
            return result
    return None


# --- Anatomy detectors -----------------------------------------------------

_BRAINGLOBE_REQUIRED = {"Channel", "Brain region acronym", "Brain region"}
_PINPOINT_REQUIRED = {"area", "coordinates"}  # JSON loaded into a DataFrame


def _detect_brainglobe(df: pd.DataFrame) -> Optional[str]:
    cols = set(df.columns)
    return "brainglobe" if _BRAINGLOBE_REQUIRED.issubset(cols) else None


def _detect_pinpoint(df: pd.DataFrame) -> Optional[str]:
    cols = set(df.columns)
    return "pinpoint" if _PINPOINT_REQUIRED.issubset(cols) else None


def _detect_csv(df: pd.DataFrame) -> Optional[str]:
    cols = {c.lower() for c in df.columns}
    if "channel" in cols and ("area" in cols or "brain_area" in cols):
        return "csv"
    return None


def sniff_anatomy_format(df: pd.DataFrame) -> Optional[str]:
    for detector in (_detect_brainglobe, _detect_pinpoint, _detect_csv):
        result = detector(df)
        if result is not None:
            return result
    return None
