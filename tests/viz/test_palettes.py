import pytest
import matplotlib

matplotlib.use("Agg")

from neurocomplexity.viz._palettes import (
    PALETTES, DEFAULT_PALETTE, get_palette, _lighten,
)


def test_named_palettes_present():
    assert set(PALETTES) == {"nature", "forest", "wine", "sage"}


def test_palette_role_keys_consistent():
    required = {"text", "signal", "accent", "muted", "fill", "categorical"}
    for name, p in PALETTES.items():
        assert required.issubset(p), f"{name} missing roles"


def test_palette_hex_colours_canonical():
    # Text is forced to black across all palettes for print legibility
    for name in ("forest", "wine", "sage"):
        assert PALETTES[name]["text"] == "#000000"
    assert PALETTES["forest"]["signal"] == "#2C2A4A"
    assert PALETTES["forest"]["accent"] == "#A6A867"
    assert PALETTES["forest"]["muted"] == "#9D91A3"
    assert PALETTES["wine"]["signal"] == "#66232A"
    assert PALETTES["wine"]["accent"] == "#C39B60"
    assert PALETTES["sage"]["signal"] == "#723D46"
    assert PALETTES["sage"]["accent"] == "#C9CBA3"
    # nature (default): Okabe-Ito blue signal + vermilion accent
    assert PALETTES["nature"]["text"] == "#000000"
    assert PALETTES["nature"]["signal"] == "#0072B2"
    assert PALETTES["nature"]["accent"] == "#D55E00"


def test_categorical_is_list_of_hex():
    for name, p in PALETTES.items():
        cats = p["categorical"]
        assert isinstance(cats, list)
        assert all(isinstance(c, str) and c.startswith("#") for c in cats)
        assert len(cats) >= 3


def test_default_palette_is_nature():
    assert DEFAULT_PALETTE == "nature"


def test_get_palette_returns_dict():
    p = get_palette("forest")
    assert p["signal"] == "#2C2A4A"


def test_get_palette_unknown_raises():
    with pytest.raises(KeyError, match="forest|wine|sage"):
        get_palette("unknown")


def test_lighten_pushes_to_higher_lightness():
    out = _lighten("#723D46", target_l=0.75)
    assert out.startswith("#")
    from neurocomplexity.viz._palettes import _hex_to_rgb
    orig = sum(_hex_to_rgb("#723D46")) / 3
    new = sum(_hex_to_rgb(out)) / 3
    assert new > orig


def test_apply_style_sets_text_color_from_palette():
    import matplotlib as mpl
    from neurocomplexity.viz._style import apply_style
    for pal in ("forest", "wine", "sage"):
        apply_style(palette=pal)
        assert mpl.rcParams["text.color"] == "#000000"
        assert mpl.rcParams["axes.edgecolor"] == "#000000"


def test_set_palette_changes_current():
    from neurocomplexity.viz._style import set_palette, current_palette
    set_palette("sage")
    assert current_palette()["signal"] == "#723D46"
    set_palette("forest")


def test_apply_style_no_pdf_fonttype_42():
    import matplotlib as mpl
    from neurocomplexity.viz._style import apply_style
    apply_style()
    assert mpl.rcParams["pdf.fonttype"] == 42
    assert mpl.rcParams["svg.fonttype"] == "none"
