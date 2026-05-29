"""neurocomplexity — criticality, transfer entropy, and autonomy for spike data."""

from neurocomplexity import analysis, core, inference, io, warnings
from neurocomplexity._progress import set_progress
from neurocomplexity._version import __version__
from neurocomplexity.analysis._binning import estimate_bin_spikes_bytes
from neurocomplexity.core.continuous import ContinuousSignal
from neurocomplexity.core.provenance import ProvenanceRecord
from neurocomplexity.core.recording import SpikeRecording

# viz is optional (requires matplotlib); import lazily so headless installs
# without matplotlib still work for headless analyses.
try:
    from neurocomplexity import viz  # noqa: F401
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False

__all__ = ["__version__", "SpikeRecording", "ContinuousSignal", "ProvenanceRecord",
           "set_progress", "estimate_bin_spikes_bytes",
           "core", "io", "analysis", "inference", "warnings"]
if _HAS_VIZ:
    __all__.append("viz")
