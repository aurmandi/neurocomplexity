"""I/O loaders that materialise a ``SpikeRecording`` from disk or memory.

Heavy loaders (NWB / Phy / Kilosort / SpikeInterface) are lazy-imported so
installing without the corresponding extra does not pull in pynwb /
spikeinterface / hdmf. Pure-NumPy/pandas curation helpers
(``add_quality``, ``add_anatomy``, ``add_trials``) and ``from_dict`` import
eagerly — they depend only on the always-installed runtime deps, so making
them lazy bought nothing and split the import contract (C P1-10).
"""
from __future__ import annotations

from neurocomplexity.io._anatomy import add_anatomy
from neurocomplexity.io._qc import add_quality
from neurocomplexity.io._trials import add_trials
from neurocomplexity.io.dict_loader import from_dict

__all__ = [
    "from_dict",
    "from_nwb",
    "to_nwb",
    "from_phy",
    "from_kilosort",
    "from_spikeinterface",
    "add_quality",
    "add_anatomy",
    "add_trials",
]


def __getattr__(name):
    if name == "from_nwb":
        from neurocomplexity.io.nwb import from_nwb as _f
        return _f
    if name == "to_nwb":
        from neurocomplexity.io._ndx import to_nwb as _f
        return _f
    if name == "from_phy":
        from neurocomplexity.io.phy import from_phy as _f
        return _f
    if name == "from_kilosort":
        from neurocomplexity.io.kilosort import from_kilosort as _f
        return _f
    if name == "from_spikeinterface":
        from neurocomplexity.io.spikeinterface import from_spikeinterface as _f
        return _f
    raise AttributeError(f"module 'neurocomplexity.io' has no attribute {name!r}")
