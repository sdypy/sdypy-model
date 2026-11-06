Code documentation
==================

FEM models
----------

Beam
~~~~

Euler–Bernoulli / Timoshenko beam elements.

.. autoclass:: sdypy.model.Beam
   :members:

Shell
~~~~~

MITC4 shell elements.

.. autoclass:: sdypy.model.Shell
   :members:

Tetrahedron
~~~~~~~~~~~

10-node tetrahedron elements.

.. autoclass:: sdypy.model.Tetrahedron
   :members:

Eigenvalue solver
-----------------

.. autofunction:: sdypy.model.solve_eigenvalue

Meshing
-------

Shared 2D/3D mesh helpers.

.. autoclass:: sdypy.model.mesh.Mesh3D
   :members:
.. autofunction:: sdypy.model.mesh.generate_quadrilateral_mesh

Acoustic radiation (external BEM)
---------------------------------

The high-level driver for exterior acoustic radiation/scattering problems.

.. autoclass:: sdypy.model.AcousticExternalProblem
   :members:

Lower-level building blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sdypy.model.acoustic_external.BEMSolver
   :members:

.. autoclass:: sdypy.model.acoustic_external.Mesh
   :members:

.. autoclass:: sdypy.model.acoustic_external.CollocationAssembler
   :members:

.. autoclass:: sdypy.model.acoustic_external.Body
   :members:

.. autoclass:: sdypy.model.acoustic_external.Field
   :members:

.. autofunction:: sdypy.model.acoustic_external.box_mesh

Elements
~~~~~~~~

Boundary-element types (P1 triangles): continuous and discontinuous.

.. autoclass:: sdypy.model.acoustic_external.ContinuousP1Mesh
   :members:
.. autoclass:: sdypy.model.acoustic_external.DiscontinuousP1Mesh
   :members:

Integrators
~~~~~~~~~~~

.. autoclass:: sdypy.model.acoustic_external.ElementIntegratorCollocation
   :members:

Kernels
~~~~~~~

Green's function and its derivatives.

.. autofunction:: sdypy.model.acoustic_external.G

Quadrature
~~~~~~~~~~

Singular / near-singular triangle quadrature.

.. TODO: document the quadrature rules in
   ``sdypy.model.acoustic_external.quadrature``.

Solve
~~~~~

.. TODO: document the solve routines in
   ``sdypy.model.acoustic_external.solve``.
