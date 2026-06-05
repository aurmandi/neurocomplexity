import numpy as np


def test_top_level_complexity_exports():
    import neurocomplexity as nc
    assert hasattr(nc.analysis, "lmc_complexity")
    assert hasattr(nc.analysis, "LMCResult")
    assert hasattr(nc.analysis, "multiscale_entropy")
    assert hasattr(nc.analysis, "MSEResult")


def test_top_level_viz_exports():
    import neurocomplexity as nc
    assert hasattr(nc.viz, "figure_lmc_complexity")
    assert hasattr(nc.viz, "figure_mse")
