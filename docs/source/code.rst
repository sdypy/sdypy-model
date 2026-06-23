Code documentation
==================

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
