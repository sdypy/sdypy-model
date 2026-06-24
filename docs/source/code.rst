Code documentation
==================

.. TODO: this page is a structural skeleton. Fill each section with
   ``autoclass`` / ``autofunction`` directives as the docstrings are finalised.

FEM models
----------

Beam
~~~~

Euler–Bernoulli / Timoshenko beam elements.

.. TODO: .. autoclass:: sdypy.model.Beam

Shell
~~~~~

MITC4 shell elements.

.. TODO: .. autoclass:: sdypy.model.Shell

Tetrahedron
~~~~~~~~~~~

10-node tetrahedron elements.

.. TODO: .. autoclass:: sdypy.model.Tetrahedron

Eigenvalue solver
-----------------

.. TODO: .. autofunction:: sdypy.model.solve_eigenvalue

Meshing
-------

Shared 2D/3D mesh helpers.

.. TODO: .. autoclass:: sdypy.model.mesh.Mesh3D
.. TODO: .. autofunction:: sdypy.model.mesh.generate_quadrilateral_mesh

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

.. TODO: .. autoclass:: sdypy.model.acoustic_external.ContinuousP1Mesh
.. TODO: .. autoclass:: sdypy.model.acoustic_external.DiscontinuousP1Mesh

Integrators
~~~~~~~~~~~

.. TODO: .. autoclass:: sdypy.model.acoustic_external.ElementIntegratorCollocation

Kernels
~~~~~~~

Green's function and its derivatives.

.. TODO: .. autofunction:: sdypy.model.acoustic_external.G

Quadrature
~~~~~~~~~~

Singular / near-singular triangle quadrature.

.. TODO: document the quadrature rules in
   ``sdypy.model.acoustic_external.quadrature``.

Solve
~~~~~

.. TODO: document the solve routines in
   ``sdypy.model.acoustic_external.solve``.
