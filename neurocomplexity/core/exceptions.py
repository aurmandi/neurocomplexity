class NeurocomplexityError(Exception):
    """Base class for all neurocomplexity errors."""


class NWBSchemaError(NeurocomplexityError):
    """Raised when an NWB file is missing required schema elements."""


class PopulationError(NeurocomplexityError):
    """Raised on invalid population definitions or membership."""


class RecordingValidationError(NeurocomplexityError):
    """Raised when SpikeRecording invariants are violated."""
