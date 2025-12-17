from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("softpotato")
except PackageNotFoundError:  # pragma: no cover
    # Allows `pytest` from a fresh checkout before editable install.
    __version__ = "3.0.0-alpha2"
