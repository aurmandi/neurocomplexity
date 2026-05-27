"""Analysis layer — measures of complexity, criticality and information flow.

Every public function in this package takes a
:class:`~neurocomplexity.core.recording.SpikeRecording` (immutable) and
returns a frozen ``*Result`` dataclass that carries both the numerical
output and the parameters used to produce it.

Top-level functions are re-exported at the package root (``nc.criticality``,
``nc.transfer_entropy``, ...) for convenience.

Statistical inference (surrogate tests, bootstrap CIs, FDR) lives in
:mod:`neurocomplexity.inference`; analysis functions return point estimates
only.

Sub-modules
-----------
- :mod:`~neurocomplexity.analysis.criticality` — avalanche exponents,
  Sethna consistency test.
- :mod:`~neurocomplexity.analysis.branching` — Wilting-Priesemann branching
  ratio.
- :mod:`~neurocomplexity.analysis.shape_collapse` — Friedman avalanche
  shape collapse.
- :mod:`~neurocomplexity.analysis.dimensionality` — participation ratio.
- :mod:`~neurocomplexity.analysis.manifold` — PCA / UMAP / t-SNE
  state-space embedding.
- :mod:`~neurocomplexity.analysis.mse` — Costa multiscale entropy.
- :mod:`~neurocomplexity.analysis.complexity` — López-Ruiz LMC complexity.
- :mod:`~neurocomplexity.analysis.transfer_entropy` — binary Schreiber TE.
- :mod:`~neurocomplexity.analysis.pid` — Williams-Beer I_min PID.
- :mod:`~neurocomplexity.analysis.autonomy` — VAR-Granger self-predictability.
- :mod:`~neurocomplexity.analysis.stationarity` — population-level
  stationarity diagnostics.
"""
from neurocomplexity.analysis.criticality import criticality, CriticalityResult
from neurocomplexity.analysis.transfer_entropy import (
    transfer_entropy,
    TransferEntropyResult,
)
from neurocomplexity.analysis.autonomy import autonomy, AutonomyResult
from neurocomplexity.analysis.dimensionality import (
    dimensionality,
    DimensionalityResult,
)
from neurocomplexity.analysis.branching import wilting_mr, branching_ratio, BranchingResult
from neurocomplexity.analysis.shape_collapse import (
    shape_collapse,
    ShapeCollapseResult,
)
from neurocomplexity.analysis.pid import partial_information, PIDResult
from neurocomplexity.analysis.surrogates import (
    make_surrogate,
    jitter_recording,
    shuffle_isis,
)
from neurocomplexity.analysis.stationarity import stationarity, StationarityResult
from neurocomplexity.analysis.complexity import lmc_complexity, LMCResult
from neurocomplexity.analysis.mse import multiscale_entropy, MSEResult
from neurocomplexity.analysis.manifold import manifold, ManifoldResult

__all__ = [
    "criticality", "CriticalityResult",
    "transfer_entropy", "TransferEntropyResult",
    "autonomy", "AutonomyResult",
    "dimensionality", "DimensionalityResult",
    "wilting_mr", "branching_ratio", "BranchingResult",
    "shape_collapse", "ShapeCollapseResult",
    "partial_information", "PIDResult",
    "make_surrogate", "jitter_recording", "shuffle_isis",
    "stationarity", "StationarityResult",
    "lmc_complexity", "LMCResult",
    "multiscale_entropy", "MSEResult",
    "manifold", "ManifoldResult",
]
