import numpy as np
import pytest
from neurocomplexity.inference import InferenceResult


def test_inference_result_is_frozen():
    r = InferenceResult(
        statistic_name="TE", observed=0.1,
        null_distribution=np.zeros(10), bootstrap_distribution=None,
        p_value=0.5, p_value_fdr=None, effect_size=0.0,
        ci_lower=None, ci_upper=None, ci_level=0.95,
        method="isi_shuffle", n_resamples=10, seed=0, metadata={},
    )
    with pytest.raises(Exception):
        r.observed = 0.2  # frozen


def test_inference_result_to_dict_roundtrips_scalars():
    r = InferenceResult(
        statistic_name="m_hat", observed=0.95,
        null_distribution=None, bootstrap_distribution=np.array([0.94, 0.95, 0.96]),
        p_value=None, p_value_fdr=None, effect_size=None,
        ci_lower=0.94, ci_upper=0.96, ci_level=0.95,
        method="block_bootstrap", n_resamples=3, seed=0, metadata={"block_s": 10.0},
    )
    d = r.to_dict()
    assert d["statistic_name"] == "m_hat"
    assert d["ci_lower"] == 0.94
    assert d["metadata"]["block_s"] == 10.0
