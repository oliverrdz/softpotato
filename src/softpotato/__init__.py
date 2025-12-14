"""
SoftPotato v3.0

Electrochemistry simulator and toolbox (core numerical engine lives in
:mod:`softpotato.core`).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("softpotato")
except PackageNotFoundError:  # pragma: no cover
    # Allows imports when running from source tree without installation.
    __version__ = "0+unknown"


__all__ = ["__version__"]

