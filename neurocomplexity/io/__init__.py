"""I/O loaders that materialise a ``SpikeRecording`` from disk or memory.

Heavy loaders are lazy-imported so installing without the corresponding
extra does not pull in pynwb / spikeinterface / hdmf. Pure-NumPy loaders
import eagerly.
"""
from __future__ import annotations

from neurocomplexity.io.dict_loader import from_dict
from neurocomplexity.io._qc import add_quality

__all__ = [
    "from_dict",
    "from_nwb",
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
    if name == "from_phy":
        from neurocomplexity.io.phy import from_phy as _f
        return _f
    if name == "from_kilosort":
        from neurocomplexity.io.kilosort import from_kilosort as _f
        return _f
    if name == "from_spikeinterface":
        from neurocomplexity.io.spikeinterface import from_spikeinterface as _f
        return _f
    if name == "add_anatomy":
        from neurocomplexity.io._anatomy import add_anatomy as _f
        return _f
    if name == "add_trials":
        from neurocomplexity.io._trials import add_trials as _f
        return _f
    raise AttributeError(f"module 'neurocomplexity.io' has no attribute {name!r}")
