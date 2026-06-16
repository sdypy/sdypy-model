"""Public-API conformance tests for sdypy-model 0.2.0.

Covers:
  (a-b) Tetrahedron canonical + deprecated kwargs
  (c)   Beam canonical + deprecated kwargs
  (d)   Shell canonical + deprecated kwargs
  (e)   All __all__ entries resolve via getattr
  (f)   lumped and mesh are module objects
  (g)   beam, tetrahedron, eigenvalue_solution NOT in __all__
"""
import inspect
import warnings
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_tet_args():
    """Return (org, conec, young_modulus, density, poisson_ratio) for a single
    10-node tetrahedron element.  Values are physically plausible but tiny."""
    # 10 nodes of a tet10 element (4 corners + 6 midpoints), unit tetrahedron
    org = np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0.5, 0., 0.],
        [0.5, 0.5, 0.],
        [0., 0.5, 0.],
        [0., 0., 0.5],
        [0.5, 0., 0.5],
        [0., 0.5, 0.5],
    ], dtype=float)
    conec = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    return org, conec


def _patched_tetrahedron(young_modulus, density, poisson_ratio, **kwargs):
    """Construct Tetrahedron with assembly methods monkeypatched to no-ops so
    the constructor completes without real FEM computation on toy data."""
    from unittest.mock import patch
    from scipy import sparse
    import sdypy.model

    org, conec = _make_minimal_tet_args()

    dummy_K = sparse.eye(30, format="csc")
    dummy_M = sparse.eye(30, format="csc")

    with patch.object(
        sdypy.model.Tetrahedron, "assemble_matrices", return_value=(dummy_K, dummy_M)
    ), patch.object(
        sdypy.model.Tetrahedron, "assemble_matrices_lumped", return_value=(dummy_K, dummy_M)
    ):
        obj = sdypy.model.Tetrahedron(
            org, conec, young_modulus, density, poisson_ratio, **kwargs
        )
    return obj


def _patched_tetrahedron_deprecated(Young, Density, Poisson):
    """Same but using deprecated keyword-only spellings."""
    from unittest.mock import patch
    from scipy import sparse
    import sdypy.model

    org, conec = _make_minimal_tet_args()

    dummy_K = sparse.eye(30, format="csc")
    dummy_M = sparse.eye(30, format="csc")

    with patch.object(
        sdypy.model.Tetrahedron, "assemble_matrices", return_value=(dummy_K, dummy_M)
    ), patch.object(
        sdypy.model.Tetrahedron, "assemble_matrices_lumped", return_value=(dummy_K, dummy_M)
    ):
        # positional org/conec, then new-style positional args as None (deprecated path)
        obj = sdypy.model.Tetrahedron(
            org, conec, None, None, None,
            Young=Young, Density=Density, Poisson=Poisson,
        )
    return obj


def _make_minimal_shell_args():
    """Minimal flat square mesh (2 quad elements, 6 nodes)."""
    nodes = np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [2., 0., 0.],
        [0., 1., 0.],
        [1., 1., 0.],
        [2., 1., 0.],
    ], dtype=float)
    elements = np.array([
        [0, 1, 4, 3],
        [1, 2, 5, 4],
    ])
    return nodes, elements


def _patched_shell(young_modulus, poisson_ratio, density, thickness=0.001, **kwargs):
    from unittest.mock import patch
    import sdypy.model

    nodes, elements = _make_minimal_shell_args()

    with patch.object(sdypy.model.Shell, "construct_global_matrices"):
        obj = sdypy.model.Shell(
            nodes, elements, young_modulus, poisson_ratio, density, thickness,
            **kwargs
        )
    return obj


def _patched_shell_deprecated(E, nu, rho, thickness=0.001):
    from unittest.mock import patch
    import sdypy.model

    nodes, elements = _make_minimal_shell_args()

    with patch.object(sdypy.model.Shell, "construct_global_matrices"):
        obj = sdypy.model.Shell(
            nodes, elements, None, None, None, thickness,
            E=E, nu=nu, rho=rho,
        )
    return obj


# ---------------------------------------------------------------------------
# (a) Tetrahedron with canonical kwargs — no warning
# ---------------------------------------------------------------------------

class TestTetrahedronCanonical:
    def test_no_deprecation_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            obj = _patched_tetrahedron(
                young_modulus=210e9, density=7850.0, poisson_ratio=0.3
            )
        assert obj.young_modulus is not None
        assert obj.poisson_ratio == 0.3
        assert obj.ro == 7850.0

    def test_attributes_stored(self):
        obj = _patched_tetrahedron(
            young_modulus=210e9, density=7800.0, poisson_ratio=0.28
        )
        # After broadcast, young_modulus becomes an array
        assert np.all(obj.young_modulus == 210e9)
        assert np.all(obj.ro == 7800.0)
        assert obj.poisson_ratio == 0.28


# ---------------------------------------------------------------------------
# (b) Tetrahedron with deprecated kwargs — warns DeprecationWarning
# ---------------------------------------------------------------------------

class TestTetrahedronDeprecated:
    def test_warns_young(self):
        with pytest.warns(DeprecationWarning, match="Young is deprecated"):
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                _patched_tetrahedron_deprecated(
                    Young=210e9, Density=7850.0, Poisson=0.3
                )

    def test_warns_density(self):
        with pytest.warns(DeprecationWarning, match="Density is deprecated"):
            _patched_tetrahedron_deprecated(
                Young=210e9, Density=7850.0, Poisson=0.3
            )

    def test_warns_poisson(self):
        with pytest.warns(DeprecationWarning, match="Poisson is deprecated"):
            _patched_tetrahedron_deprecated(
                Young=210e9, Density=7850.0, Poisson=0.3
            )

    def test_constructs_identically(self):
        """Deprecated path stores same values as canonical path."""
        obj_canon = _patched_tetrahedron(
            young_modulus=210e9, density=7850.0, poisson_ratio=0.3
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            obj_dep = _patched_tetrahedron_deprecated(
                Young=210e9, Density=7850.0, Poisson=0.3
            )
        assert np.all(obj_canon.young_modulus == obj_dep.young_modulus)
        assert np.all(obj_canon.ro == obj_dep.ro)
        assert obj_canon.poisson_ratio == obj_dep.poisson_ratio

    def test_both_spellings_raises(self):
        from unittest.mock import patch
        from scipy import sparse
        import sdypy.model

        org, conec = _make_minimal_tet_args()
        dummy = sparse.eye(30, format="csc")

        with patch.object(sdypy.model.Tetrahedron, "assemble_matrices", return_value=(dummy, dummy)):
            with pytest.raises(TypeError):
                sdypy.model.Tetrahedron(
                    org, conec, 210e9, 7850.0, 0.3, Young=200e9
                )


# ---------------------------------------------------------------------------
# (c) Beam canonical + deprecated
# ---------------------------------------------------------------------------

class TestBeam:
    def _make_beam(self, **override):
        import sdypy.model
        n_elements = 5
        length = np.ones(n_elements) * 0.1
        density = np.ones(n_elements) * 7850.0
        kwargs = dict(
            org=None, conec=None,
            length=length, width=0.02, height=0.01,
            density=density, young_modulus=210e9,
            n_nodes=n_elements + 1,
        )
        kwargs.update(override)
        return sdypy.model.Beam(**kwargs)

    def test_canonical_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            obj = self._make_beam()
        assert obj.young_modulus is not None

    def test_deprecated_young_warns(self):
        import sdypy.model
        n_elements = 5
        length = np.ones(n_elements) * 0.1
        density = np.ones(n_elements) * 7850.0
        with pytest.warns(DeprecationWarning, match="Young is deprecated"):
            sdypy.model.Beam(
                org=None, conec=None,
                length=length, width=0.02, height=0.01,
                density=density, young_modulus=None,
                n_nodes=n_elements + 1, Young=210e9,
            )

    def test_both_spellings_raises(self):
        import sdypy.model
        n_elements = 5
        length = np.ones(n_elements) * 0.1
        density = np.ones(n_elements) * 7850.0
        with pytest.raises(TypeError):
            sdypy.model.Beam(
                org=None, conec=None,
                length=length, width=0.02, height=0.01,
                density=density, young_modulus=210e9,
                n_nodes=n_elements + 1, Young=200e9,
            )

    def test_attribute_stored(self):
        obj = self._make_beam()
        assert np.all(obj.young_modulus == 210e9)


# ---------------------------------------------------------------------------
# (d) Shell canonical + deprecated
# ---------------------------------------------------------------------------

class TestShell:
    def test_canonical_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            obj = _patched_shell(
                young_modulus=210e9, poisson_ratio=0.3, density=7850.0
            )
        assert obj.young_modulus is not None

    def test_deprecated_warns(self):
        with pytest.warns(DeprecationWarning):
            _patched_shell_deprecated(E=210e9, nu=0.3, rho=7850.0)

    def test_deprecated_e_warns(self):
        with pytest.warns(DeprecationWarning, match="E is deprecated"):
            _patched_shell_deprecated(E=210e9, nu=0.3, rho=7850.0)

    def test_deprecated_nu_warns(self):
        with pytest.warns(DeprecationWarning, match="nu is deprecated"):
            _patched_shell_deprecated(E=210e9, nu=0.3, rho=7850.0)

    def test_deprecated_rho_warns(self):
        with pytest.warns(DeprecationWarning, match="rho is deprecated"):
            _patched_shell_deprecated(E=210e9, nu=0.3, rho=7850.0)

    def test_constructs_identically(self):
        """Canonical and deprecated paths produce same stored attributes."""
        obj_canon = _patched_shell(
            young_modulus=210e9, poisson_ratio=0.3, density=7850.0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            obj_dep = _patched_shell_deprecated(E=210e9, nu=0.3, rho=7850.0)

        assert np.all(obj_canon.young_modulus == obj_dep.young_modulus)
        assert np.all(obj_canon.poisson_ratio == obj_dep.poisson_ratio)
        assert np.all(obj_canon.density == obj_dep.density)

    def test_both_spellings_raises(self):
        from unittest.mock import patch
        import sdypy.model

        nodes, elements = _make_minimal_shell_args()
        with patch.object(sdypy.model.Shell, "construct_global_matrices"):
            with pytest.raises(TypeError):
                sdypy.model.Shell(
                    nodes, elements, 210e9, 0.3, 7850.0, 0.001,
                    E=200e9,
                )


# ---------------------------------------------------------------------------
# (e) Every entry in __all__ resolves via getattr
# ---------------------------------------------------------------------------

class TestPublicAll:
    def test_all_names_resolve(self):
        import sdypy.model
        for name in sdypy.model.__all__:
            assert hasattr(sdypy.model, name), f"{name!r} not found in sdypy.model"
            assert getattr(sdypy.model, name) is not None, f"{name!r} is None in sdypy.model"

    def test_all_exact_contents(self):
        import sdypy.model
        assert sorted(sdypy.model.__all__) == sorted(
            ["Beam", "Shell", "Tetrahedron", "solve_eigenvalue", "lumped", "mesh"]
        )


# ---------------------------------------------------------------------------
# (f) lumped and mesh are module objects
# ---------------------------------------------------------------------------

class TestModuleObjects:
    def test_lumped_is_module(self):
        import sdypy.model
        assert inspect.ismodule(sdypy.model.lumped)

    def test_mesh_is_module(self):
        import sdypy.model
        assert inspect.ismodule(sdypy.model.mesh)


# ---------------------------------------------------------------------------
# (g) beam, tetrahedron, eigenvalue_solution NOT in __all__
# ---------------------------------------------------------------------------

class TestNotInAll:
    def test_beam_submodule_not_in_all(self):
        import sdypy.model
        assert "beam" not in sdypy.model.__all__

    def test_tetrahedron_submodule_not_in_all(self):
        import sdypy.model
        assert "tetrahedron" not in sdypy.model.__all__

    def test_eigenvalue_solution_not_in_all(self):
        import sdypy.model
        assert "eigenvalue_solution" not in sdypy.model.__all__

    def test_shell_submodule_not_in_all(self):
        import sdypy.model
        assert "shell" not in sdypy.model.__all__
