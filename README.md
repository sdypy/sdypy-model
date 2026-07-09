# `sdypy-model`

A namespace Python package for the SDyPy project.

The package currently contains the following finite elements:

- `beam`: Euler-Bernoulli and Timoshenko beam elements.
- `shell`: MITC4 shell elements.
- `tet`: Quadratic (10 node) tetrahedral elements.

It also provides an exterior acoustics solver:

- `acoustic_external`: exterior acoustic radiation/scattering with a collocation
  boundary element method (BEM), including an optional Burton–Miller formulation
  to suppress spurious internal resonances.

**Note**: The package is still under development and may not be fully functional.

## Beam elements

```python
import sdypy as sd

beam_obj = sd.model.Beam(...)

M = beam_obj.M
K = beam_obj.K
```

## Shell elements

```python
import sdypy as sd

shell_obj = sd.model.Shell(...)

M = shell_obj.M
K = shell_obj.K
```

## Tetrahedral elements
```python

import sdypy as sd

tet_obj = sd.model.Tetrahedron(...)

M = tet_obj.M
K = tet_obj.K
```

## Acoustic radiation (external BEM)

Solve for the acoustic field radiated by a vibrating surface. The surface is a
triangular mesh (e.g. a `pyvista.PolyData`) with coordinates in **metres**. Define the boundary condition (either Dirichlet or Neumann). With a Neumann boundary condition you prescribe the normal velocity at each node; the solver
returns the velocity potential `φ` on the boundary, from which the pressure
`p = jωρφ` is recovered anywhere in the exterior field.

```python
import numpy as np
import pyvista as pv
import sdypy as sd

# Vibrating surface (triangle mesh, coordinates in metres)
sphere = pv.Sphere(radius=0.15, theta_resolution=30, phi_resolution=30)
vn = 0.01 * np.ones(sphere.n_points)        # normal velocity [m/s]

prob = sd.model.AcousticExternalProblem(
    mesh=sphere,
    rho=1.225, c0=343.0,                    # air: density [kg/m3], sound speed [m/s]
    boundary_condition=vn,                  # (n_points,) scalar, or (n_points, 3) vector
    boundary_condition_type="Neumann",      # "Neumann" (velocity) | "Dirichlet" (pressure)
    frequency=500.0,                        # [Hz]
    use_burton_miller=False,                # combined (Burton–Miller) formulation, optional
)

phi, q = prob.solve_problem()               # boundary solution: potential + ∂φ/∂n

# Radiated velocity potential at field points of shape (N, 3) [m]
pts = np.array([[0.3, 0.0, 0.0], [0.0, 0.0, 0.5]])
phi_field = prob.evaluate_field(pts)        # velocity potential φ
p = 1j * 2 * np.pi * prob.frequency * prob.rho * phi_field   # pressure p = jωρφ [Pa]
```

Use `set_frequency()` / `set_boundary_condition()` to reuse the assembled model across
a frequency sweep without rebuilding the mesh. All quantities are in SI units and the
complex time convention is `e^{jωt}`. A complete, validated example (pulsating sphere vs.
the analytical solution) is in [`examples/acoustic_external_example.py`](https://github.com/sdypy/sdypy-model/blob/master/examples/acoustic_external_example.py).

<!-- TODO: embed BEM validation animations (mode-shape / monopole gifs) here -->

# Disclaimer

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

Use of this software is at your own risk.

This software is distributed under the terms of the MIT License. See the LICENSE file for more details.