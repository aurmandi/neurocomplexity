"""Public re-export of internal warning classes.

Allows users to write::

    import warnings, neurocomplexity as nc
    warnings.filterwarnings("ignore", category=nc.warnings.QualityControlWarning)
"""
from neurocomplexity._warnings import QualityControlWarning

__all__ = ["QualityControlWarning"]
