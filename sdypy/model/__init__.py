from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sdypy-model")
except PackageNotFoundError:  # source checkout without installed metadata
    __version__ = "0+unknown"

import pyLump as lumped
from .shell.shell import Shell
from .tetrahedron.tet10 import Tetrahedron
from .beam.beam import Beam
from .eigenvalue_solution import solve_eigenvalue

from . import mesh