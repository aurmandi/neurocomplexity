"""neurocomplexity — criticality, transfer entropy, and autonomy for spike data."""

from neurocomplexity._version import __version__
from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity import core, io, analysis, inference

# viz is optional (requires matplotlib); import lazily so headless installs
# without matplotlib still work for headless analyses.
try:
    from neurocomplexity import viz  # noqa: F401
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False

__all__ = ["__version__", "SpikeRecording", "ProvenanceRecord",
           "core", "io", "analysis", "inference"]
if _HAS_VIZ:
    __all__.append("viz")
