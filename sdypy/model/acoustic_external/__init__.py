"""
Boundary Element Method for Exterior Acoustic problems
"""

__version__ = "0.0.1"
from .acoustic_external_problem import AcousticExternalProblem
from .geometry import Body, Field, box_mesh
from .mesh import Mesh
from .elements import (ContinuousP1Mesh, DiscontinuousP1Mesh)
from .integrators import (ElementIntegratorCollocation)
from .matrix_assembly import (CollocationAssembler)
from .solve import BEMSolver

from .kernels import (r_vec, G, dG_dr, 
                          dG_dn_y, dG_dn_x, 
                          d2G_dn_x_dn_y)

__all__ = ["Mesh", "BEMSolver", "AcousticExternalProblem",
            "CollocationAssembler",
            "ElementIntegratorCollocation",
            "ContinuousP1Mesh", "DiscontinuousP1Mesh",
            "Body", "Field", "box_mesh",]