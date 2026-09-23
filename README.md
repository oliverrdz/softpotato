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
