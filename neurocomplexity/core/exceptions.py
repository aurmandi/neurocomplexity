"""Domain-specific exception types.

Every exception raised by ``neurocomplexity`` (except for type errors and
``ValueError`` from argument validation) is a subclass of
:class:`NeurocomplexityError`, so callers can do
``except NeurocomplexityError:`` to catch the lot.
"""


class NeurocomplexityError(Exception):
    """Base class for all errors raised by the ``neurocomplexity`` package."""


class NWBSchemaError(NeurocomplexityError):
    """Raised when an NWB file is missing required schema elements.

    Example: no ``Units`` table (calcium / ophys data), missing
    ``spike_times`` column, malformed ``electrodes`` table.
    """


class PopulationError(NeurocomplexityError):
    """Raised on invalid population definitions or membership.

    Example: a population mask whose length does not match ``len(units)``,
    an empty population, or a unit assigned to multiple populations when
    that is not permitted.
    """


class RecordingValidationError(NeurocomplexityError):
    """Raised when :class:`~neurocomplexity.core.recording.SpikeRecording`
    invariants are violated at construction or transformation time.

    Example: spike times not monotonic, mismatched array lengths,
    ``duration <= 0``, or a unit id in ``spike_times`` that does not
    appear in ``units``.
    """
