import pandas as pd
import pytest

from neurocomplexity.io._sniff import (
    sniff_qc_format,
    sniff_anatomy_format,
)


def test_sniff_bombcell_by_column_signature():
    df = pd.DataFrame(columns=[
        "useTheseTimesStart", "nPeaks", "rawAmplitude",
        "percentageSpikesMissing_gaussian", "unitType",
    ])
    assert sniff_qc_format(df) == "bombcell"


def test_sniff_ecephys_by_column_signature():
    df = pd.DataFrame(columns=[
        "cluster_id", "isi_viol", "amplitude_cutoff", "presence_ratio",
    ])
    assert sniff_qc_format(df) == "ecephys"


def test_sniff_spikeinterface_by_column_signature():
    df = pd.DataFrame(columns=[
        "snr", "isi_violations_ratio", "presence_ratio", "firing_rate",
    ])
    assert sniff_qc_format(df) == "spikeinterface"


def test_sniff_qc_returns_none_for_unknown():
    df = pd.DataFrame(columns=["foo", "bar"])
    assert sniff_qc_format(df) is None


def test_sniff_anatomy_brainglobe():
    df = pd.DataFrame(columns=[
        "Channel", "Brain region acronym", "Brain region", "AP", "DV", "ML",
    ])
    assert sniff_anatomy_format(df) == "brainglobe"


def test_sniff_anatomy_generic_csv_with_channel_and_area():
    df = pd.DataFrame(columns=["channel", "area"])
    assert sniff_anatomy_format(df) == "csv"


def test_sniff_anatomy_returns_none_without_channel():
    df = pd.DataFrame(columns=["foo", "bar"])
    assert sniff_anatomy_format(df) is None
