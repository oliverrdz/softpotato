"""
Soft Potato 3.0
Next-generation open-source electrochemical simulator and toolkit.
"""

__version__ = "3.0.0"
__author__ = "Soft Potato Developers"

# Expose primary submodules for easy access
from . import core

try:
    from . import geometry
    from . import simulate
    from . import techniques
    from . import kinetics
    from . import analytical
except ImportError:
    pass

