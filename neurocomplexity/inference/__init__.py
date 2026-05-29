"""Statistical inference for neurocomplexity analyses.

Surrogate-based null tests and bootstrap confidence intervals.
See docs/inference.md for method choices and citations.
"""
from neurocomplexity.inference.bootstrap import bootstrap
from neurocomplexity.inference.null_test import pvalue_from_null, test
from neurocomplexity.inference.pool import SurrogatePool
from neurocomplexity.inference.results import InferenceResult

__all__ = ["InferenceResult", "SurrogatePool", "test", "bootstrap", "pvalue_from_null"]
