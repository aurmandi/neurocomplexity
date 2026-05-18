import numpy as np
import pandas as pd

from neurocomplexity.core.recording import SpikeRecording
from neurocomplexity.core.provenance import ProvenanceRecord


def _minimal_rec():
    return SpikeRecording(
        spike_times=np.array([0.1, 0.2], dtype=np.float64),
        unit_ids=np.array([0, 1], dtype=np.int64),
        units=pd.DataFrame({"id": [0, 1]}),
        populations={"all": np.array([True, True])},
        duration=1.0,
        sampling_rate=None,
        source=ProvenanceRecord.for_memory("test"),
    )


def test_attachments_default_empty():
    rec = _minimal_rec()
    assert rec.attachments == ()


def test_filtered_default_false():
    rec = _minimal_rec()
    assert rec._filtered is False


def test_filter_units_sets_filtered_flag():
    rec = _minimal_rec().with_populations({"a": np.array([True, False])},
                                          on_unassigned="drop")
    # filter_units uses keyword form
    rec2 = rec.filter_units(id=[0])
    assert rec2._filtered is True


def test_attachments_preserved_through_filter():
    extra = ProvenanceRecord.for_memory("quality")
    from dataclasses import replace
    rec = _minimal_rec()
    rec = replace(rec, attachments=(extra,))
    rec2 = rec.filter_units(id=[0])
    assert rec2.attachments == (extra,)
