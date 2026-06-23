"""
Fast pytest for AcousticExternalProblem.

Uses a small pulsating sphere (optimised for speed).
Runs with verbose=False for silent execution.
"""

import numpy as np
import pyvista as pv

from sdypy.model.acoustic_external import AcousticExternalProblem


def pulsating_sphere_surface_pressure(
    a: float, k: float, v0: float, rho0: float, c0: float
) -> complex:
    """
    Analytical surface pressure for a uniformly pulsating sphere.
    """
    return (rho0 * c0 * v0) * (1j * k * a) / (1 + 1j * k * a)


def _small_sphere(radius: float):
    """Create a small, valid closed sphere mesh for testing."""
    sphere = pv.Sphere(radius=radius, theta_resolution=8, phi_resolution=8)
    sphere.compute_normals(
        point_normals=True, cell_normals=False,
        consistent_normals=True, auto_orient_normals=True,
        inplace=True,
    )
    return sphere


def test_pulsating_sphere_surface_pressure():
    """
    BEM on a pulsating sphere: verify solve completes, surface pressure is
    non-zero finite, and field pressure decays with distance.
    """
    rho0 = 1.225
    c0 = 343.0
    freq = 500.0
    v0 = 0.01
    a = 0.15

    sphere = _small_sphere(a)
    vn = v0 * np.ones(sphere.n_points, dtype=np.float64)

    prob = AcousticExternalProblem(
        mesh=sphere, rho=rho0, c0=c0,
        boundary_condition=vn, boundary_condition_type="Neumann",
        frequency=freq,
        assembler_type="continuous",
        use_burton_miller=True,
    )

    phi = prob.solve_problem(verbose=False)
    assert phi is not None
    assert phi.shape[0] == sphere.n_points

    # Surface pressure should be non-zero and finite
    p_surf = 1j * 2 * np.pi * freq * rho0 * phi
    assert np.all(np.isfinite(p_surf)), "Non-finite surface pressure"
    assert np.max(np.abs(p_surf)) > 0, "Zero surface pressure"

    # Field should decay with distance
    r_near = np.array([[0.2, 0.0, 0.0]])
    r_far  = np.array([[1.0, 0.0, 0.0]])
    p_near = prob.evaluate_field(r_near, result_type="p", verbose=False)
    p_far  = prob.evaluate_field(r_far,  result_type="p", verbose=False)
    assert np.abs(p_far[0]) < np.abs(p_near[0]), (
        "Pressure should decay with distance"
    )


def test_pulsating_sphere_quantitative_accuracy():
    """
    Regression test: on a moderately refined mesh the BEM field pressure must
    match the analytical pulsating-sphere solution to within a few percent.
    Guards the physics (the qualitative test above only checks decay).
    """
    rho0, c0, freq, v0, a = 1.225, 343.0, 500.0, 0.01, 0.15
    k = 2 * np.pi * freq / c0

    sphere = pv.Sphere(radius=a, theta_resolution=16, phi_resolution=16)
    sphere.compute_normals(
        point_normals=True, cell_normals=False,
        consistent_normals=True, auto_orient_normals=True, inplace=True,
    )
    vn = v0 * np.ones(sphere.n_points, dtype=np.float64)

    prob = AcousticExternalProblem(
        mesh=sphere, rho=rho0, c0=c0,
        boundary_condition=vn, boundary_condition_type="Neumann",
        frequency=freq, assembler_type="continuous", use_burton_miller=False,
    )
    prob.solve_problem(verbose=False)

    # Analytical surface and radiated field pressure at r = 0.5 m
    p_surf_ana = pulsating_sphere_surface_pressure(a, k, v0, rho0, c0)
    r = 0.5
    p_field_ana = p_surf_ana * (a / r) * np.exp(-1j * k * (r - a))

    p_field_bem = prob.evaluate_field(
        np.array([[r, 0.0, 0.0]]), result_type="p", verbose=False
    )[0]

    rel_err = abs(abs(p_field_bem) - abs(p_field_ana)) / abs(p_field_ana)
    assert rel_err < 0.10, f"Field pressure rel. error too high: {rel_err:.1%}"


def test_set_frequency():
    """Changing frequency should reset the assembled-matrices flag."""
    sphere = _small_sphere(0.15)
    vn = np.ones(sphere.n_points, dtype=np.float64)

    prob = AcousticExternalProblem(
        mesh=sphere, rho=1.225, c0=343.0,
        boundary_condition=vn, boundary_condition_type="Neumann",
        frequency=500.0, use_burton_miller=False,
    )

    # First solve — auto-assembles
    prob.solve_problem(verbose=False)
    assert prob._matrices_assembled

    # Changing frequency resets the flag
    prob.set_frequency(600.0)
    assert not prob._matrices_assembled


def test_set_boundary_condition():
    """Updating BC should store it correctly."""
    sphere = _small_sphere(0.15)
    vn_init = 0.01 * np.ones(sphere.n_points, dtype=np.float64)

    prob = AcousticExternalProblem(
        mesh=sphere, rho=1.225, c0=343.0,
        boundary_condition=vn_init, boundary_condition_type="Neumann",
        frequency=500.0,
    )

    # Update with different scalar BC
    vn_new = 0.05 * np.ones(sphere.n_points)
    prob.set_boundary_condition(vn_new)
    np.testing.assert_allclose(prob.bem_mesh.Neumann_BC, vn_new)

    # Update with 3-D radial vector BC (should auto-project onto normals)
    # For a sphere centered at origin, outward normal = position / radius
    pts = sphere.points
    r = np.linalg.norm(pts, axis=1, keepdims=True)
    rad_vec = 0.01 * pts / r  # shape (N, 3), radial outward
    prob.set_boundary_condition(rad_vec)
    # After projection, all values should be close to 0.01 (radial dot normal)
    np.testing.assert_allclose(
        prob.bem_mesh.Neumann_BC, 0.01 * np.ones(sphere.n_points),
        atol=2e-3,
    )


def test_evaluate_field():
    """Field evaluation should return correct shape and type."""
    sphere = _small_sphere(0.15)
    vn = 0.01 * np.ones(sphere.n_points, dtype=np.float64)

    prob = AcousticExternalProblem(
        mesh=sphere, rho=1.225, c0=343.0,
        boundary_condition=vn, boundary_condition_type="Neumann",
        frequency=500.0,
    )
    prob.solve_problem(verbose=False)

    # Evaluate at a single point outside the sphere
    pts = np.array([[0.3, 0.0, 0.0]])
    p = prob.evaluate_field(pts, result_type="p", verbose=False)
    assert p.shape == (1,), f"Expected shape (1,), got {p.shape}"
    assert np.issubdtype(p.dtype, np.complexfloating)

    # Evaluate with result_type="phi"
    phi_pts = prob.evaluate_field(pts, result_type="phi", verbose=False)
    assert phi_pts.shape == (1,)
    assert np.issubdtype(phi_pts.dtype, np.complexfloating)


def test_assembler_types():
    """Both continuous and discontinuous assemblers should work."""
    sphere = _small_sphere(0.15)
    vn = 0.01 * np.ones(sphere.n_points, dtype=np.float64)

    for atype in ("continuous", "discontinuous"):
        prob = AcousticExternalProblem(
            mesh=sphere, rho=1.225, c0=343.0,
            boundary_condition=vn, boundary_condition_type="Neumann",
            frequency=500.0, assembler_type=atype,
        )
        phi = prob.solve_problem(verbose=False)
        assert phi is not None
        assert len(phi) > 0


def test_invalid_parameters():
    """Invalid parameters should raise ValueError."""
    import pytest

    sphere = _small_sphere(0.15)
    vn = np.ones(sphere.n_points, dtype=np.float64)

    # Invalid boundary condition type
    with pytest.raises(ValueError):
        AcousticExternalProblem(
            mesh=sphere, rho=1.225, c0=343.0,
            boundary_condition=vn, boundary_condition_type="InvalidType",
            frequency=500.0,
        )

    # Invalid assembler type
    with pytest.raises(ValueError):
        AcousticExternalProblem(
            mesh=sphere, rho=1.225, c0=343.0,
            boundary_condition=vn, boundary_condition_type="Neumann",
            frequency=500.0, assembler_type="invalid",
        )

    # evaluate_field before solve
    prob = AcousticExternalProblem(
        mesh=sphere, rho=1.225, c0=343.0,
        boundary_condition=vn, boundary_condition_type="Neumann",
        frequency=500.0,
    )
    with pytest.raises(RuntimeError):
        prob.evaluate_field(np.array([[0.3, 0.0, 0.0]]), verbose=False)
