"""Sphinx configuration for neurocomplexity documentation."""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure the package is importable for autodoc.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import neurocomplexity  # noqa: E402

project = "neurocomplexity"
author = "Sazgar Arman Dinarvand"
copyright = "2026, Sazgar Arman Dinarvand"
release = neurocomplexity.__version__
version = release

extensions = [
    # myst_nb subclasses myst_parser; only list myst_nb to avoid the
    # duplicate-config-registration error.
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

nb_execution_mode = "off"

myst_enable_extensions = [
    "deflist",
    "colon_fence",
    "linkify",
    "substitution",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "paper/**",
    "plans/**",
    # Internal development / audit reports — kept in-repo for provenance but
    # not part of the rendered user documentation.
    "phase*.md",
    "*_review_*.md",
    "*_revision_*.md",
    "publication_plan.md",
    "tutorial_agent_prompt.md",
]

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"neurocomplexity {release}"

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Render dataclass / docstring "Attributes" as :ivar: fields on the class
# rather than separate py:attribute directives. Without this, napoleon emits
# both the class-member attribute and an Attributes-section copy, producing
# "duplicate object description" warnings for every dataclass field.
napoleon_use_ivar = True

# Optional dependencies that the [docs] extra does not install. Mock them
# so autodoc can walk modules that top-import them (viz.* needs
# matplotlib; io.nwb needs pynwb/h5py; io.spikeinterface needs
# spikeinterface) without their wheels actually being present on the
# Read the Docs builder.
autodoc_mock_imports = [
    "matplotlib",
    "pynwb",
    "h5py",
    "spikeinterface",
    "plotly",
    "dash",
    "netpyne",
    "mpi4py",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
