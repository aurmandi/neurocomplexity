"""Freeze lock for the SemVer-covered public surface.

Every symbol named as *stable* in ``docs/api_stability.md`` must resolve at
the path the contract advertises. This catches contract/code drift — e.g. a
symbol documented as stable but never exported, or a phantom symbol that the
contract names but no longer exists. Both classes of bug were found and fixed
during the Phase 8 API freeze; this test stops them recurring.

If you intentionally add, rename, or remove a stable symbol, update BOTH this
list and ``docs/api_stability.md`` in the same change.
"""
import importlib

import pytest

# (module path, attribute) pairs for the stable surface.
STABLE_SYMBOLS = [
    # Core data types
    ("neurocomplexity", "SpikeRecording"),
    ("neurocomplexity.core", "SpikeRecording"),
    ("neurocomplexity.core", "ProvenanceRecord"),
    ("neurocomplexity.core", "hash_file"),
    # Result dataclasses
    ("neurocomplexity.analysis", "CriticalityResult"),
    ("neurocomplexity.analysis", "BranchingResult"),
    ("neurocomplexity.analysis", "ShapeCollapseResult"),
    ("neurocomplexity.analysis", "DimensionalityResult"),
    ("neurocomplexity.analysis", "TransferEntropyResult"),
    ("neurocomplexity.analysis", "PIDResult"),
    ("neurocomplexity.analysis", "MSEResult"),
    ("neurocomplexity.analysis", "LMCResult"),
    ("neurocomplexity.analysis", "AutonomyResult"),
    ("neurocomplexity.analysis", "StationarityResult"),
    ("neurocomplexity.analysis", "ManifoldResult"),
    ("neurocomplexity.inference", "InferenceResult"),
    ("neurocomplexity.benchmarks", "BenchmarkResult"),
    # Loaders
    ("neurocomplexity.io", "from_nwb"),
    ("neurocomplexity.io", "to_nwb"),
    ("neurocomplexity.io", "from_kilosort"),
    ("neurocomplexity.io", "from_phy"),
    ("neurocomplexity.io", "from_spikeinterface"),
    ("neurocomplexity.io", "from_dict"),
    ("neurocomplexity.io", "add_quality"),
    ("neurocomplexity.io", "add_anatomy"),
    ("neurocomplexity.io", "add_trials"),
    # Top-level analyses
    ("neurocomplexity.analysis", "criticality"),
    ("neurocomplexity.analysis", "wilting_mr"),
    ("neurocomplexity.analysis", "shape_collapse"),
    ("neurocomplexity.analysis", "dimensionality"),
    ("neurocomplexity.analysis", "manifold"),
    ("neurocomplexity.analysis", "multiscale_entropy"),
    ("neurocomplexity.analysis", "lmc_complexity"),
    ("neurocomplexity.analysis", "transfer_entropy"),
    ("neurocomplexity.analysis", "partial_information"),
    ("neurocomplexity.analysis", "autonomy"),
    ("neurocomplexity.analysis", "stationarity"),
    ("neurocomplexity.analysis", "extract_avalanches"),
    ("neurocomplexity.analysis", "fit_avalanche_exponents"),
    # Inference machinery
    ("neurocomplexity.inference", "test"),
    ("neurocomplexity.inference", "bootstrap"),
    ("neurocomplexity.inference", "pvalue_from_null"),
    ("neurocomplexity.inference", "SurrogatePool"),
]


@pytest.mark.parametrize("module_path,attr", STABLE_SYMBOLS)
def test_stable_symbol_resolves(module_path, attr):
    mod = importlib.import_module(module_path)
    assert hasattr(mod, attr), f"{module_path}.{attr} named stable but missing"


def test_merge_probes_is_recording_method():
    # Contract lists merge_probes as SpikeRecording.merge_probes, not nc.io.*.
    from neurocomplexity import SpikeRecording

    assert callable(getattr(SpikeRecording, "merge_probes", None))


def test_viz_figure_functions_resolve():
    # viz is optional (matplotlib); skip if unavailable.
    viz = pytest.importorskip("neurocomplexity.viz")
    for name in ("save_publication", "figure_criticality", "figure_te_network"):
        assert hasattr(viz, name), f"neurocomplexity.viz.{name} named stable but missing"


def test_no_phantom_figure_panel():
    # figure_panel was a phantom in the contract; ensure it stays gone so the
    # contract and code do not re-diverge.
    viz = pytest.importorskip("neurocomplexity.viz")
    assert not hasattr(viz, "figure_panel")
