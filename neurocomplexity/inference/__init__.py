"""Statistical inference for neurocomplexity analyses.

Surrogate-based null tests and bootstrap confidence intervals.
See docs/inference.md for method choices and citations.
"""
from neurocomplexity.inference.results import InferenceResult
from neurocomplexity.inference.pool import SurrogatePool
from neurocomplexity.inference.null_test import test
from neurocomplexity.inference.bootstrap import bootstrap

__all__ = ["InferenceResult", "SurrogatePool", "test", "bootstrap"]
