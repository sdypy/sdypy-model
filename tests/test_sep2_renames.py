"""
SEP 2 renames keep working deprecated aliases (sdypy hub, public-api spec,
"Evidenced divergences carry deprecated aliases to the canonical names").

Every old name emits DeprecationWarning and gives the same value as its new
name; the new name and positional calls emit no DeprecationWarning. Passing
an old and a new name together raises TypeError.
"""

import warnings

import numpy as np
import pytest
from scipy import sparse

import sdypy.model as model
from sdypy.model.acoustic_external import quadrature, kernels
from sdypy.model.acoustic_external.integrators import ElementIntegratorCollocation
from sdypy.model.beam import beam as beam_mod
from sdypy.model.shell import shell as shell_mod
from sdypy.model.tetrahedron import tet10


def _call_new(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        return fn(*args, **kwargs)


def _call_old(fn, new, *args, **kwargs):
    with pytest.warns(DeprecationWarning, match=new):
        return fn(*args, **kwargs)


def _same(a, b):
    if sparse.issparse(a):
        a, b = a.toarray(), b.toarray()
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
    else:
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Functions: (function, positional args, old keyword, new keyword)
# ---------------------------------------------------------------------------

def _tet_data():
    jac = np.tile(np.array([[2., 0, 0], [0, 3, 0], [0, 0, 4]]), (2, 1, 1))
    return jac, tet10.Jacobi_det(jac)


_JAC, _JAC_DET = _tet_data()
_K = sparse.eye(4, format="csc") * 2.0
_M = sparse.eye(4, format="csc")

# (function, fixed kwargs, old, new, value)
FUNCTION_ALIASES = [
    (beam_mod.matrices_k_e_timoshenko, dict(l=0.1, area_moment=1e-8, area=1e-4, poisson_ratio=0.3), "E", "young_modulus", 2e11),
    (beam_mod.matrices_k_e_timoshenko, dict(l=0.1, young_modulus=2e11, area=1e-4, poisson_ratio=0.3), "I", "area_moment", 1e-8),
    (beam_mod.matrices_k_e_timoshenko, dict(l=0.1, young_modulus=2e11, area_moment=1e-8, poisson_ratio=0.3), "A", "area", 1e-4),
    (beam_mod.matrices_k_e_timoshenko, dict(l=0.1, young_modulus=2e11, area_moment=1e-8, area=1e-4), "nu", "poisson_ratio", 0.3),
    (beam_mod.matrices_k_e, dict(l=0.1), "EI", "bending_stiffness", 2e3),
    (tet10.H_matrix, dict(poisson_ratio=np.array([0.3, 0.3])), "E", "young_modulus", np.array([1e9, 2e9])),
    (tet10.H_matrix, dict(young_modulus=np.array([1e9, 2e9])), "nu", "poisson_ratio", np.array([0.3, 0.3])),
    (tet10.Jacobi_det, dict(), "J", "jacobian", _JAC),
    (tet10.Jacobi_inv, dict(jacobian_det=_JAC_DET), "J", "jacobian", _JAC),
    (tet10.Jacobi_inv, dict(jacobian=_JAC), "J_det", "jacobian_det", _JAC_DET),
    (shell_mod.get_constitutive_tensor, dict(poisson_ratio=0.3), "E", "young_modulus", 210e9),
    (shell_mod.get_constitutive_tensor, dict(young_modulus=210e9), "nu", "poisson_ratio", 0.3),
    (shell_mod.lump_mass_matrix, dict(), "M", "mass_matrix", np.eye(4)),
    (quadrature.gauss_legendre_1d, dict(), "n", "n_points", 4),
    (kernels.dG_dr, dict(r_norm=np.array([1.0, 2.0]), k=3.0), "G", "greens_function", np.array([1.0 + 0j, 0.5])),
]


@pytest.mark.parametrize("fn, fixed, old, new, value", FUNCTION_ALIASES)
def test_old_keyword_warns_and_matches_new(fn, fixed, old, new, value):
    expected = _call_new(fn, **fixed, **{new: value})
    got = _call_old(fn, new, **fixed, **{old: value})
    _same(got, expected)


@pytest.mark.parametrize("fn, fixed, old, new, value", FUNCTION_ALIASES)
def test_old_and_new_together_raise(fn, fixed, old, new, value):
    with pytest.raises(TypeError):
        fn(**fixed, **{old: value, new: value})


@pytest.mark.parametrize("fn, fixed, old, new, value", FUNCTION_ALIASES)
def test_missing_required_raises(fn, fixed, old, new, value):
    if fn is beam_mod.matrices_k_e_timoshenko and new == "poisson_ratio":
        pytest.skip("poisson_ratio defaults to 0.3")
    with pytest.raises(TypeError):
        fn(**fixed)


def test_positional_calls_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        beam_mod.matrices_k_e_timoshenko(0.1, 2e11, 1e-8, 1e-4, 0.3)
        beam_mod.matrices_k_e(0.1, 2e3)
        tet10.H_matrix(np.array([1e9]), np.array([0.3]))
        tet10.Jacobi_det(_JAC)
        shell_mod.get_constitutive_tensor(210e9, 0.3)
        quadrature.gauss_legendre_1d(4)


def test_solve_eigenvalue_aliases():
    expected = _call_new(model.solve_eigenvalue, stiffness_matrix=_K, mass_matrix=_M,
                         n_modes=2, convert_to_hz=False)
    for old, new in [("K", "stiffness_matrix"), ("M", "mass_matrix")]:
        kwargs = dict(stiffness_matrix=_K, mass_matrix=_M, n_modes=2, convert_to_hz=False)
        value = kwargs.pop(new)
        got = _call_old(model.solve_eigenvalue, new, **kwargs, **{old: value})
        _same(got[0], expected[0])  # eigenvectors are defined up to sign
    with pytest.raises(TypeError):
        model.solve_eigenvalue(_K, _M, 2, False, K=_K)
    with pytest.raises(TypeError):
        model.solve_eigenvalue(n_modes=2)


def test_solve_eigenvalue_positional_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        model.solve_eigenvalue(_K, _M, 2, False)


# ---------------------------------------------------------------------------
# Acoustic integrator and problem
# ---------------------------------------------------------------------------

def _integrator_args():
    x = np.array([0.0, 0.0, 1.0])
    y = np.array([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    w = np.array([[0.5, 0.5]])
    n = np.array([[0.3, 0.3, 0.4], [0.2, 0.2, 0.6]])
    return x, y, w, n


def test_integrator_single_layer_N_alias():
    integ = ElementIntegratorCollocation(2.0)
    x, y, w, n = _integrator_args()
    expected = _call_new(integ.single_layer, x, y, w, shape_functions=n)
    got = _call_old(integ.single_layer, "shape_functions", x, y, w, N=n)
    _same(got, expected)
    _same(_call_new(integ.single_layer, x, y, w, n), expected)
    with pytest.raises(TypeError):
        integ.single_layer(x, y, w, shape_functions=n, N=n)


# ---------------------------------------------------------------------------
# Mesh3D.transform
# ---------------------------------------------------------------------------

def _mesh():
    pts = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    return model.mesh.Mesh3D(pts, faces)


def test_mesh_transform_R_alias():
    mesh = _mesh()
    rot = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    expected = _call_new(mesh.transform, 2.0, rotation_matrix=rot)
    got = _call_old(mesh.transform, "rotation_matrix", 2.0, R=rot)
    np.testing.assert_array_equal(got.points, expected.points)
    _call_new(mesh.transform, 2.0, rot)
    with pytest.raises(TypeError):
        mesh.transform(2.0, rotation_matrix=rot, R=rot)


# ---------------------------------------------------------------------------
# Beam
# ---------------------------------------------------------------------------

def _beam_kwargs():
    n_el = 5
    return dict(length=np.full(n_el, 0.2), width=0.02, height=0.01,
                density=np.full(n_el, 7850.0), young_modulus=np.full(n_el, 2e11),
                n_nodes=n_el + 1)


@pytest.fixture(scope="module")
def beam():
    b = model.Beam(nodes=None, elements=None, **_beam_kwargs())
    b.solve(n_modes=4)
    return b


BEAM_ATTRIBUTES = [
    ("org", "nodes"), ("conec", "elements"), ("I", "area_moment"),
    ("EI", "bending_stiffness"), ("M", "mass_matrix"), ("K", "stiffness_matrix"),
    ("nat_freq", "natural_freq"), ("A", "mode_shape"),
]


@pytest.mark.parametrize("old, new", BEAM_ATTRIBUTES)
def test_beam_old_attribute(beam, old, new):
    with pytest.warns(DeprecationWarning, match=new):
        value = getattr(beam, old)
    _same(value, getattr(beam, new))


@pytest.mark.parametrize("old, new", [("org", "nodes"), ("conec", "elements")])
def test_beam_old_constructor_keyword(old, new):
    ref = model.Beam(nodes=None, elements=None, **_beam_kwargs())
    nodes = ref.nodes
    elements = ref.elements
    value = {"org": nodes, "conec": elements}[old]
    kwargs = _beam_kwargs()
    kwargs.update({"nodes": nodes, "elements": elements})
    kwargs.pop(new)
    got = _call_old(model.Beam, new, **kwargs, **{old: value})
    _same(getattr(got, new), getattr(ref, new))


@pytest.mark.parametrize("old, new", [("Young", "young_modulus")])
def test_beam_old_property_keyword(old, new):
    kwargs = _beam_kwargs()
    value = kwargs.pop(new)
    got = _call_old(model.Beam, new, nodes=None, elements=None, **kwargs, **{old: value})
    ref = model.Beam(nodes=None, elements=None, **_beam_kwargs())
    _same(got.stiffness_matrix, ref.stiffness_matrix)
    with pytest.raises(TypeError):
        model.Beam(nodes=None, elements=None, **_beam_kwargs(), **{old: value})


def test_beam_missing_required_raises():
    kwargs = _beam_kwargs()
    kwargs.pop("density")
    with pytest.raises(TypeError):
        model.Beam(nodes=None, elements=None, **kwargs)


def test_beam_solve_n_alias():
    b = model.Beam(nodes=None, elements=None, **_beam_kwargs())
    expected = _call_new(b.solve, n_modes=4)
    got = _call_old(b.solve, "n_modes", n=4)
    # eigenvectors are defined up to sign; the Lanczos solver is not bit-exact
    np.testing.assert_allclose(got[0], expected[0], rtol=1e-6, atol=1e-4)
    with pytest.raises(TypeError):
        b.solve(n_modes=4, n=4)
    _call_new(b.solve, True, 4)


def test_beam_attribute_setter_forwards(beam):
    with pytest.warns(DeprecationWarning, match="natural_freq"):
        beam.nat_freq = beam.natural_freq
    # restore nothing: value is identical


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shell():
    nodes = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    return model.Shell(nodes, np.array([[0, 1, 2, 3]]), 210e9, 0.3, 7850.0, 0.01)


@pytest.mark.parametrize("old, new", [("K", "stiffness_matrix"), ("M", "mass_matrix")])
def test_shell_old_attribute(shell, old, new):
    with pytest.warns(DeprecationWarning, match=new):
        value = getattr(shell, old)
    _same(value, getattr(shell, new))


# ---------------------------------------------------------------------------
# Tetrahedron
# ---------------------------------------------------------------------------

def _tet_nodes_elements():
    nodes = np.array([
        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        [0.5, 0., 0.], [0.5, 0.5, 0.], [0., 0.5, 0.], [0., 0., 0.5],
        [0.5, 0., 0.5], [0., 0.5, 0.5]])
    return nodes, np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])


@pytest.fixture(scope="module")
def tet():
    nodes, elements = _tet_nodes_elements()
    t = model.Tetrahedron(nodes, elements, 210e9, 7850.0, 0.3)
    t.solve()
    return t


TET_ATTRIBUTES = [
    ("org", "nodes"), ("conec", "elements"), ("ro", "density"),
    ("K", "stiffness_matrix"), ("M", "mass_matrix"),
]


@pytest.mark.parametrize("old, new", TET_ATTRIBUTES)
def test_tet_old_attribute(tet, old, new):
    with pytest.warns(DeprecationWarning, match=new):
        value = getattr(tet, old)
    _same(value, getattr(tet, new))


@pytest.mark.parametrize("old, new, value", [
    ("Young", "young_modulus", 210e9),
    ("Density", "density", 7850.0),
    ("Poisson", "poisson_ratio", 0.3),
])
def test_tet_old_constructor_keyword(old, new, value):
    nodes, elements = _tet_nodes_elements()
    base = dict(young_modulus=210e9, density=7850.0, poisson_ratio=0.3)
    ref = model.Tetrahedron(nodes, elements, **base)
    base.pop(new)
    got = _call_old(model.Tetrahedron, new, nodes, elements, **base, **{old: value})
    _same(got.stiffness_matrix, ref.stiffness_matrix)
    _same(got.mass_matrix, ref.mass_matrix)
    with pytest.raises(TypeError):
        model.Tetrahedron(nodes, elements, 210e9, 7850.0, 0.3, **{old: value})


def test_tet_old_nodes_elements_keyword():
    nodes, elements = _tet_nodes_elements()
    got = _call_old(model.Tetrahedron, "nodes", org=nodes, conec=elements,
                    young_modulus=210e9, density=7850.0, poisson_ratio=0.3)
    ref = model.Tetrahedron(nodes, elements, 210e9, 7850.0, 0.3)
    _same(got.stiffness_matrix, ref.stiffness_matrix)


def test_tet_construct_loce_aliases():
    nodes, elements = _tet_nodes_elements()
    expected = _call_new(tet10.construct_loce, nodes, elements)
    got = _call_old(tet10.construct_loce, "nodes", org=nodes, conec=elements)
    _same(got, expected)
    with pytest.raises(TypeError):
        tet10.construct_loce(nodes, elements, org=nodes)


def test_tet_assemble_aliases(tet):
    E = np.full(1, 210e9)
    rho = np.full(1, 7850.0)
    expected = _call_new(tet.assemble_matrices, young_modulus=E, density=rho)
    got = _call_old(tet.assemble_matrices, "young_modulus", E=E, density=rho)
    _same(got, expected)
    got = _call_old(tet.assemble_matrices, "density", young_modulus=E, ro=rho)
    _same(got, expected)
    with pytest.raises(TypeError):
        tet.assemble_matrices(E, rho, E=E)


def test_tet_matrix_derivative_aliases(tet):
    idx = np.array([0])
    expected = _call_new(tet.matrix_derivative, derivative_young_modulus_idx=idx,
                         derivative_density_idx=idx)
    got = _call_old(tet.matrix_derivative, "derivative_young_modulus_idx",
                    derivative_E_ind=idx, derivative_density_idx=idx)
    _same(got, expected)
    got = _call_old(tet.matrix_derivative, "derivative_density_idx",
                    derivative_young_modulus_idx=idx, derivative_ro_ind=idx)
    _same(got, expected)
    with pytest.raises(TypeError):
        tet.matrix_derivative(idx, idx, derivative_E_ind=idx)


def test_tet_S_matrix_aliases(tet):
    pytest.importorskip("ipywidgets")  # tqdm.notebook progress bar
    kw = dict(compute_vector_sensitivity=False)
    idx = np.array([0])
    mode = np.array([6])
    expected = _call_new(tet.S_matrix, mode_idx=mode, derivative_young_modulus_idx=idx, **kw)
    got = _call_old(tet.S_matrix, "mode_idx", eig_ind=mode,
                    derivative_young_modulus_idx=idx, **kw)
    _same(got, expected)
    got = _call_old(tet.S_matrix, "derivative_young_modulus_idx", mode_idx=mode,
                    derivative_E_ind=idx, **kw)
    _same(got, expected)
    with pytest.raises(TypeError):
        tet.S_matrix(mode, idx, eig_ind=mode)
