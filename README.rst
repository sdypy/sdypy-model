``sdypy-model``

A namespace Python package for the SDyPy project.

The package currently contains the following finite elements:

- ``beam``: Euler-Bernoulli and Timoshenko beam elements.
- ``shell``: MITC4 shell elements.
- ``tet``: Quadratic (10 node) tetrahedral elements.

Beam elements
-------------

.. code-block:: python

    import sdypy as sd

    beam_obj = sd.model.Beam(...)

    M = beam_obj.M
    K = beam_obj.K


Shell elements
--------------

.. code-block:: python

    import sdypy as sd

    shell_obj = sd.model.Shell(...)

    M = shell_obj.M
    K = shell_obj.K


Tetrahedral elements
--------------------

.. code-block:: python

    import sdypy as sd

    tet_obj = sd.model.Tetrahedron(...)

    M = tet_obj.M
    K = tet_obj.K