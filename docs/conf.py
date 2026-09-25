# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Ensure src/ is on sys.path so autodoc can discover softpotato modules
sys.path.insert(0, os.path.abspath("../src"))

import softpotato

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "softpotato"
copyright = "2026, Oliver Rodriguez"
author = "Oliver Rodriguez"
version = softpotato.__version__
release = softpotato.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_parser",
]

# Syntax highlighting configuration
pygments_style = "sphinx"
highlight_language = "python3"

# MyST Markdown parser configuration
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "dollarmath",
]

# nbsphinx configuration
# Do not re-execute notebooks during doc build; render saved outputs/plots
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# Todo settings
todo_include_todos = True

# Autosummary settings
autosummary_generate = True

# Napoleon settings (supports Google and NumPy docstring styles)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

# Theme options
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}
