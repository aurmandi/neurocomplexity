from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from neurocomplexity.viz._save import save_publication


def _tiny_fig():
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([0, 1, 2], [0, 1, 0.5])
    return fig


def test_writes_svg_tiff_jpg(tmp_path):
    fig = _tiny_fig()
    paths = save_publication(fig, tmp_path / "x")
    assert set(paths) == {"svg", "tiff", "jpg"}
    for fmt, p in paths.items():
        assert p.exists(), f"{fmt} not written"
    assert paths["svg"].suffix == ".svg"
    assert paths["tiff"].suffix == ".tiff"
    assert paths["jpg"].suffix == ".jpg"
    plt.close(fig)


def test_tiff_dpi_default_600(tmp_path):
    from PIL import Image
    fig = _tiny_fig()
    paths = save_publication(fig, tmp_path / "x")
    with Image.open(paths["tiff"]) as im:
        dpi = im.info.get("dpi")
    assert int(round(dpi[0])) == 600
    plt.close(fig)


def test_tiff_dpi_1200(tmp_path):
    from PIL import Image
    fig = _tiny_fig()
    paths = save_publication(fig, tmp_path / "x", tiff_dpi=1200)
    with Image.open(paths["tiff"]) as im:
        dpi = im.info.get("dpi")
    assert int(round(dpi[0])) == 1200
    plt.close(fig)


def test_jpg_always_600_dpi(tmp_path):
    from PIL import Image
    fig = _tiny_fig()
    paths = save_publication(fig, tmp_path / "x", tiff_dpi=1200)
    with Image.open(paths["jpg"]) as im:
        dpi = im.info.get("dpi")
    assert int(round(dpi[0])) == 600
    plt.close(fig)


def test_formats_filter(tmp_path):
    fig = _tiny_fig()
    paths = save_publication(fig, tmp_path / "x", formats=("svg",))
    assert set(paths) == {"svg"}
    assert paths["svg"].exists()
    plt.close(fig)


def test_rejects_pdf(tmp_path):
    fig = _tiny_fig()
    with pytest.raises(ValueError, match="pdf"):
        save_publication(fig, tmp_path / "x", formats=("svg", "pdf"))
    plt.close(fig)


def test_tiff_dpi_must_be_600_or_1200(tmp_path):
    fig = _tiny_fig()
    with pytest.raises(ValueError, match="600|1200"):
        save_publication(fig, tmp_path / "x", tiff_dpi=300)
    plt.close(fig)


def test_path_stem_with_directory(tmp_path):
    fig = _tiny_fig()
    sub = tmp_path / "figs" / "branching"
    sub.parent.mkdir(parents=True, exist_ok=True)
    paths = save_publication(fig, sub)
    assert (tmp_path / "figs" / "branching.svg").exists()
    assert (tmp_path / "figs" / "branching.tiff").exists()
    plt.close(fig)
