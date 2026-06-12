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

# beam, shell, tetrahedron, eigenvalue_solution remain accessible as module
# attributes (they are registered as submodule references by the imports above)
# but are intentionally excluded from the curated public surface.
__all__ = ["Beam", "Shell", "Tetrahedron", "solve_eigenvalue", "lumped", "mesh"]
