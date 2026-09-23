#!/usr/bin/env bash
set -euo pipefail

# Directory where the repository will be generated (default: current directory)
TARGET_DIR="${1:-.}"

echo "Generating clean Python repository for softpotato in '${TARGET_DIR}'..."
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

# 1. Create directory structure (standard src-layout)
mkdir -p src/softpotato tests .github/workflows docs/_static docs/_templates
touch docs/_static/.gitkeep docs/_templates/.gitkeep

# 2. pyproject.toml (PEP 517 / PEP 621 compliant, installable via pip / PyPI)
cat << 'EOF' > pyproject.toml
[build-system]
requires = ["setuptools>=61.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "softpotato"
version = "3.0.0a1"
description = "Electrochemical simulation and analysis toolkit in Python."
readme = "README.md"
license = { text = "BSD-3-Clause" }
requires-python = ">=3.10"
authors = [
    { name = "Oliver Rodriguez", email = "oliver.rdz@gmail.com" }
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Chemistry",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: Implementation :: PyPy",
]
keywords = ["electrochemistry", "voltammetry", "simulation"]
dependencies = [
    "numpy>=1.22.0",
    "scipy>=1.8.0",
    "matplotlib>=3.5.0",
]

[project.optional-dependencies]
test = [
    "pytest>=7.0.0",
]
docs = [
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=2.0.0",
    "sphinx-autodoc-typehints>=1.24.0",
]
dev = [
    "pytest>=7.0.0",
    "ruff>=0.1.0",
    "build>=1.0.0",
    "twine>=4.0.0",
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=2.0.0",
    "sphinx-autodoc-typehints>=1.24.0",
]

[project.urls]
Homepage = "https://github.com/oliverrdz/softpotato"
Repository = "https://github.com/oliverrdz/softpotato.git"
"Bug Tracker" = "https://github.com/oliverrdz/softpotato/issues"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
EOF

# 3. Source package files
cat << 'EOF' > src/softpotato/__init__.py
"""Soft Potato: Electrochemical simulation and analysis toolkit."""

__version__ = "3.0.0a1"
__all__ = ["__version__"]
EOF

# PEP 561 type marker
touch src/softpotato/py.typed

# 4. Basic test suite
cat << 'EOF' > tests/__init__.py
EOF

cat << 'EOF' > tests/test_basic.py
import softpotato


def test_version():
    assert softpotato.__version__ == "3.0.0a1"
EOF

# 5. README.md
cat << 'EOF' > README.md
# Soft Potato

[![CI](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/soft-potato/badge/?version=latest)](https://soft-potato.readthedocs.io/en/latest)
[![PyPI version](https://img.shields.io/pypi/v/softpotato.svg)](https://pypi.org/project/softpotato/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Soft Potato** is an open-source electrochemical simulation and analysis toolkit in Python.

> [!WARNING]
> **Active Rewrite in Progress:** Soft Potato is currently undergoing a complete rewrite (`v3.0.0a1`) and is **incomplete**. The API is experimental, under rapid development, and subject to breaking changes. It is not yet ready for production use.

## Installation

### From PyPI
```bash
pip install softpotato
```

### From Source (Development)
```bash
git clone https://github.com/oliverrdz/softpotato.git
cd softpotato
pip install -e ".[dev]"
```

## Quick Start

```python
import softpotato as sp

print(f"Soft Potato version: {sp.__version__}")
```

## Testing

Run the test suite using `pytest`:

```bash
pytest
```

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
EOF

# 6. LICENSE (BSD 3-Clause)
cat << 'EOF' > LICENSE
BSD 3-Clause License

Copyright (c) 2026, Oliver Rodriguez

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
EOF

# 7. .gitignore
cat << 'EOF' > .gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
build/
dist/
*.egg-info/
*.egg

# Virtual environments
.venv/
venv/
env/

# Testing & coverage
.pytest_cache/
.coverage
htmlcov/

# Static type checkers & linters
.mypy_cache/
.ruff_cache/

# IDE & Editor artifacts
.idea/
.vscode/
*.swp
*.swo
*~
.DS_Store
EOF

# 8. GitHub Actions CI workflow
cat << 'EOF' > .github/workflows/ci.yml
name: CI

on:
  push:
  pull_request:
    branches: [ main ]

jobs:
  test:
    name: Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Lint with Ruff
        run: |
          ruff check .

      - name: Run test suite
        run: |
          pytest -v
EOF

# 9. Sphinx Documentation
cat << 'EOF' > docs/conf.py
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
import softpotato

project = "softpotato"
copyright = "2026, Oliver Rodriguez"
author = "Oliver Rodriguez"
version = softpotato.__version__
release = softpotato.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
]

todo_include_todos = True
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}
EOF

cat << 'EOF' > docs/index.rst
.. Soft Potato documentation master file

Welcome to Soft Potato's Documentation!
=======================================

**Soft Potato** is an open-source electrochemical simulation and analysis toolkit designed for electrochemists, materials scientists, and engineers.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
EOF

cat << 'EOF' > docs/installation.rst
Installation
============

From PyPI
---------

Install the latest release of ``softpotato`` from PyPI:

.. code-block:: bash

   pip install softpotato

For pre-releases (such as alpha/beta versions):

.. code-block:: bash

   pip install --pre softpotato

From Source (Development)
-------------------------

Clone the repository and install in editable mode with development and documentation dependencies:

.. code-block:: bash

   git clone https://github.com/oliverrdz/softpotato.git
   cd softpotato
   pip install -e ".[dev,docs]"
EOF

cat << 'EOF' > docs/api.rst
API Reference
=============

.. automodule:: softpotato
   :members:
   :undoc-members:
   :show-inheritance:
EOF

cat << 'EOF' > docs/Makefile
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
EOF

cat << 'EOF' > docs/make.bat
@ECHO OFF
pushd %~dp0
if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build
%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx installed.
	exit /b 1
)
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end
:end
popd
EOF

# 10. Read the Docs configuration (.readthedocs.yaml)
cat << 'EOF' > .readthedocs.yaml
version: 2

build:
  os: ubuntu-24.04
  tools:
    python: "3.11"

sphinx:
  configuration: docs/conf.py

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs
EOF

# 11. GitHub Actions Documentation Workflow
cat << 'EOF' > .github/workflows/docs.yml
name: Documentation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

jobs:
  build-and-upload:
    name: Build & Upload Docs
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[docs]"

      - name: Build documentation with Sphinx
        run: |
          sphinx-build -b html -W --keep-going docs docs/_build/html

      - name: Upload HTML documentation artifact
        uses: actions/upload-artifact@v4
        with:
          name: documentation-html
          path: docs/_build/html

      - name: Upload to Read the Docs
        if: github.event_name == 'push' && env.READTHEDOCS_TOKEN != ''
        uses: readthedocs/upload-action@v1
        env:
          READTHEDOCS_TOKEN: ${{ secrets.READTHEDOCS_TOKEN }}
        with:
          token: ${{ secrets.READTHEDOCS_TOKEN }}
          project-slug: "soft-potato"
          html-dir: "docs/_build/html"
EOF

echo "✓ softpotato repository structure generated successfully in '${TARGET_DIR}'."
echo ""
echo "Next steps:"
echo "  1. Install in editable mode:  pip install -e '.[dev]'"
echo "  2. Run tests:                 pytest"
echo "  3. Build package for PyPI:    python -m build"
echo "  4. Upload to PyPI:            python -m twine upload dist/*"

