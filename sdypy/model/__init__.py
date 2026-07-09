__version__ = "0.1.4"

import pyLump as lumped
from .shell.shell import Shell
from .tetrahedron.tet10 import Tetrahedron
from .beam.beam import Beam
from .eigenvalue_solution import solve_eigenvalue
from .acoustic_external import AcousticExternalProblem

from . import mesh