"""Sphinx configuration for kawow documentation."""

import os
import sys

# Make the package importable without installing it
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -------------------------------------------------------
project = "Kawow"
author = "Luc Miaz"
release = "0.1.0"
copyright = "2025, Luc Miaz"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # NumPy / Google style docstrings
    "sphinx.ext.viewcode",       # [source] links
    "sphinx.ext.intersphinx",
    "myst_parser",               # Markdown support
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "rdkit": ("https://www.rdkit.org/docs", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- Options for HTML output --------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 3,
    "titles_only": False,
}
