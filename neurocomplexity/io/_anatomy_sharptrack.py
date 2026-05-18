"""SHARP-Track .mat loader. Returns a Brainglobe-shaped DataFrame."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _to_str_list(arr) -> list[str]:
    """MATLAB cell-string arrays decode as ndarray of dtype=object; flatten to Python strings."""
    a = np.asarray(arr).ravel()
    return [str(x) for x in a]


def load_sharptrack(path: Path) -> pd.DataFrame:
    from scipy.io import loadmat

    raw = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "probe_ccf" not in raw:
        keys = [k for k in raw if not k.startswith("__")]
        raise ValueError(
            f"SHARP-Track .mat missing 'probe_ccf' struct; top-level keys: {keys}"
        )
    pc = raw["probe_ccf"]

    def _get(attr):
        if hasattr(pc, attr):
            return getattr(pc, attr)
        if isinstance(pc, dict) and attr in pc:
            return pc[attr]
        return None

    channels_1idx = np.asarray(_get("channels")).ravel().astype(int)
    channels = channels_1idx - 1  # MATLAB -> Python 0-indexed
    areas = _to_str_list(_get("areas"))
    areas_full_raw = _get("areas_full")
    areas_full = _to_str_list(areas_full_raw) if areas_full_raw is not None else areas

    return pd.DataFrame({
        "Channel": channels,
        "Brain region acronym": areas,
        "Brain region": areas_full,
        "AP": np.asarray(_get("ap_um")).ravel().astype(float),
        "DV": np.asarray(_get("dv_um")).ravel().astype(float),
        "ML": np.asarray(_get("ml_um")).ravel().astype(float),
    })
