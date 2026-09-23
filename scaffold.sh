#!/usr/bin/env bash
set -euo pipefail

# Directory where the repository will be generated (default: current directory)
TARGET_DIR="${1:-.}"

echo "Generating clean Python repository for softpotato in '${TARGET_DIR}'..."
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

# 1. Create directory structure (standard src-layout)
mkdir -p src/softpotato tests .github/workflows

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
dev = [
    "pytest>=7.0.0",
    "ruff>=0.1.0",
    "build>=1.0.0",
    "twine>=4.0.0",
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
[![PyPI version](https://img.shields.io/pypi/v/softpotato.svg)](https://pypi.org/project/softpotato/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Soft Potato** is an open-source electrochemical simulation and analysis toolkit in Python.

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

echo "✓ softpotato repository structure generated successfully in '${TARGET_DIR}'."
echo ""
echo "Next steps:"
echo "  1. Install in editable mode:  pip install -e '.[dev]'"
echo "  2. Run tests:                 pytest"
echo "  3. Build package for PyPI:    python -m build"
echo "  4. Upload to PyPI:            python -m twine upload dist/*"

