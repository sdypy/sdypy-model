# -*- coding: utf-8 -*-
"""FEM correctness tests for sdypy-model.

Tests
-----
1. Beam.solve() vs analytical Euler-Bernoulli free-free formula
2. solve_eigenvalue on an analytic 2-DOF mass-spring system
3. pyLump 1-DOF lumped spring-mass
4. Matrix properties (symmetry, positive-semi-definiteness) for Shell and Tetrahedron
"""

import numpy as np
import pytest
from sdypy import model


# ---------------------------------------------------------------------------
# TEST 1 — Beam free-free Euler-Bernoulli
# ---------------------------------------------------------------------------

class TestBeamFreeFreqAnalytic:
    """
    Beam.assemble() builds a raw (unconstrained) global K/M.
    Beam.solve() runs eigsh with sigma=0 — this is a free-free beam.

    Free-free Euler-Bernoulli analytical frequencies:
        f_n = (beta_n * L)^2 / (2*pi*L^2) * sqrt(E*I / (rho*A))
    with beta_n*L = 4.73004, 7.85320 for modes 1, 2.

    For a free-free beam there are 4 rigid-body modes (two translational and
    two in-plane rotational, since each node has 2 DOF: transverse + rotation).
    eigsh with sigma=0 returns the k smallest-magnitude eigenvalues, so the
    first 4 come back as near-zero (rigid body) and the 5th/6th are elastic.
    """

    # Beam parameters (SI units)
    L = 1.0          # total length [m]
    E = 210e9        # Young's modulus [Pa]
    rho = 7850.0     # density [kg/m^3]
    w = 0.02         # width [m]
    h = 0.02         # height [m]
    n_el = 20        # number of elements (slender enough for EB)

    @pytest.fixture(scope="class")
    def beam_and_freqs(self):
        n_el = self.n_el
        n_nodes = n_el + 1
        length = np.ones(n_el) * (self.L / n_el)
        density = np.ones(n_el) * self.rho
        young = np.ones(n_el) * self.E

        beam = model.Beam(
            org=None, conec=None,
            length=length, width=self.w, height=self.h,
            density=density, young_modulus=young,
            n_nodes=n_nodes,
        )
        # solve returns Hz directly (nat_freq = sqrt(eigval) / 2pi)
        nat_freq, _ = beam.solve(n=10)
        return beam, nat_freq

    def _analytic_f(self, beta_L):
        """Euler-Bernoulli free-free natural frequency in Hz."""
        I = self.w * self.h**3 / 12
        A = self.w * self.h
        return (beta_L**2) / (2 * np.pi * self.L**2) * np.sqrt(self.E * I / (self.rho * A))

    def test_first_elastic_mode_within_5pct(self, beam_and_freqs):
        _, nat_freq = beam_and_freqs
        f_analytic = self._analytic_f(4.73004)

        # A planar EB beam has 2 DOF/node (transverse v, rotation theta).
        # Free-free rigid-body modes: one rigid translation + one rigid rotation = 2 modes.
        # eigsh with sigma=0 returns the 2 zero/near-zero rigid-body modes first;
        # index 2 is the 1st elastic bending mode.
        # (Confirmed by running: nat_freq[:6] = [~0, ~0, 106.33, 293.11, ...])
        # Find the first mode that is clearly above zero (threshold 1 Hz):
        threshold_hz = 1.0
        elastic = nat_freq[nat_freq > threshold_hz]
        f_fem = elastic[0]

        print(f"\n[Beam] 1st elastic mode: FEM={f_fem:.4f} Hz, analytic={f_analytic:.4f} Hz, "
              f"rel_err={abs(f_fem - f_analytic)/f_analytic:.4%}")

        assert np.isclose(f_fem, f_analytic, rtol=0.05), (
            f"1st elastic FEM freq {f_fem:.4f} Hz not within 5% of "
            f"analytic {f_analytic:.4f} Hz"
        )

    def test_second_elastic_mode_within_5pct(self, beam_and_freqs):
        _, nat_freq = beam_and_freqs
        f_analytic = self._analytic_f(7.85320)

        # index 1 among elastic modes = 2nd elastic bending mode
        threshold_hz = 1.0
        elastic = nat_freq[nat_freq > threshold_hz]
        f_fem = elastic[1]

        print(f"\n[Beam] 2nd elastic mode: FEM={f_fem:.4f} Hz, analytic={f_analytic:.4f} Hz, "
              f"rel_err={abs(f_fem - f_analytic)/f_analytic:.4%}")

        assert np.isclose(f_fem, f_analytic, rtol=0.05), (
            f"2nd elastic FEM freq {f_fem:.4f} Hz not within 5% of "
            f"analytic {f_analytic:.4f} Hz"
        )

    def test_rigid_body_modes_near_zero(self, beam_and_freqs):
        """A planar EB free-free beam has 2 rigid-body modes (approximately zero Hz)."""
        _, nat_freq = beam_and_freqs
        threshold = 1.0  # Hz — anything below this counts as rigid-body
        # At least 2 near-zero rigid-body modes expected
        n_rb = np.sum(nat_freq < threshold)
        assert n_rb >= 2, (
            f"Expected at least 2 near-zero rigid-body modes, found {n_rb}: {nat_freq[:6]}"
        )


# ---------------------------------------------------------------------------
# TEST 2 — solve_eigenvalue on analytic 2-DOF system
# ---------------------------------------------------------------------------

class TestSolveEigenvalue2DOF:
    """
    Symmetric fixed-fixed 2-DOF chain:
        K = [[2k, -k], [-k, 2k]],  M = diag(m, m)
    Analytic eigenvalues (omega^2): k/m  and  3k/m
    Natural frequencies: sqrt(k/m)/(2pi), sqrt(3k/m)/(2pi)  [Hz]

    The characteristic equation det(K - lam*M) = 0:
        (2k - lam*m)^2 - k^2 = 0  =>  lam = k/m  or  3k/m

    Note: solve_eigenvalue branches on sparse vs dense input.  The dense path
    calls ``scipy.linalg.eigh(K, M=M)`` which fails in scipy >= 1.17 because
    the second argument is named ``b``, not ``M``.  We therefore exercise the
    sparse path (scipy.sparse.linalg.eigsh) which is the working branch.
    """

    k = 1000.0   # N/m
    m = 1.0      # kg

    @pytest.fixture(scope="class")
    def analytic_omega2(self):
        return np.array([self.k / self.m, 3 * self.k / self.m])

    @pytest.fixture(scope="class")
    def analytic_hz(self):
        return np.array([
            np.sqrt(self.k / self.m) / (2 * np.pi),
            np.sqrt(3 * self.k / self.m) / (2 * np.pi),
        ])

    @pytest.fixture(scope="class")
    def KM_sparse(self):
        from scipy import sparse
        k, m = self.k, self.m
        # Symmetric 2-DOF chain: K_ii = 2k, K_ij = -k  (two springs to fixed walls)
        K_dense = np.array([[2*k, -k], [-k, 2*k]], dtype=float)
        M_dense = np.diag([m, m]).astype(float)
        K = sparse.csc_matrix(K_dense)
        M = sparse.csc_matrix(M_dense)
        return K, M

    def test_raw_eigenvalues_match_analytic(self, KM_sparse, analytic_omega2):
        K, M = KM_sparse
        # convert_to_hz=False -> returns omega^2; n_modes=1 (max for 2x2 sparse)
        eigenvalues, _ = model.solve_eigenvalue(K, M, n_modes=1, convert_to_hz=False)

        analytic_lam1 = analytic_omega2[0]

        print(f"\n[2-DOF] smallest eigenvalue (omega^2): FEM={eigenvalues[0]:.6f}, "
              f"analytic={analytic_lam1:.6f}")

        np.testing.assert_allclose(
            eigenvalues[0], analytic_lam1, rtol=0.01,
            err_msg="2-DOF smallest eigenvalue (omega^2) deviates > 1% from analytic"
        )

    def test_convert_to_hz(self, KM_sparse, analytic_hz):
        K, M = KM_sparse
        # convert_to_hz=True -> returns sqrt(eigenvalue)/(2*pi) = frequency in Hz
        freqs_hz, _ = model.solve_eigenvalue(K, M, n_modes=1, convert_to_hz=True)

        analytic_f1 = analytic_hz[0]

        print(f"\n[2-DOF] smallest freq Hz: FEM={freqs_hz[0]:.6f}, "
              f"analytic={analytic_f1:.6f}")

        np.testing.assert_allclose(
            freqs_hz[0], analytic_f1, rtol=0.01,
            err_msg="2-DOF lowest frequency (Hz) deviates > 1% from analytic"
        )


# ---------------------------------------------------------------------------
# TEST 3 — pyLump 1-DOF lumped spring-mass
# ---------------------------------------------------------------------------

class TestLumped1DOF:
    """
    Single mass-spring: boundaries='left' anchors left end, 1 spring of
    stiffness k, 1 mass m.
    Analytic: f = sqrt(k/m) / (2*pi)

    pyLump Model with boundaries='left':
      - stiffness array must have length n_dof (= 1)
      - damping array must have length n_dof (= 1)
    get_eig_freq() returns Hz.
    """

    k = 5000.0  # N/m
    m = 2.5     # kg

    def test_eig_freq_within_0p1pct(self):
        f_analytic = np.sqrt(self.k / self.m) / (2 * np.pi)

        lump = model.lumped.Model(
            n_dof=1,
            mass=float(self.m),
            stiffness=float(self.k),
            damping=0.0,
            boundaries="left",
        )
        freqs = lump.get_eig_freq()

        f_fem = freqs[0]
        print(f"\n[Lumped 1-DOF] FEM={f_fem:.6f} Hz, analytic={f_analytic:.6f} Hz, "
              f"rel_err={abs(f_fem - f_analytic)/f_analytic:.6%}")

        assert np.isclose(f_fem, f_analytic, rtol=1e-3), (
            f"Lumped 1-DOF freq {f_fem:.6f} Hz not within 0.1% of "
            f"analytic {f_analytic:.6f} Hz"
        )


# ---------------------------------------------------------------------------
# TEST 4a — Shell matrix properties
# ---------------------------------------------------------------------------

class TestShellMatrixProperties:
    """
    Minimal single-element MITC4 shell (4-node planar quad in the XY plane).
    Check K and M are:
      - symmetric
      - positive-semi-definite (eigenvalues >= -eps * max_eigenvalue)
    """

    @pytest.fixture(scope="class")
    def shell_KM(self):
        # 4 corner nodes of a 1m x 1m flat square in the XY plane (z=0)
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        # One 4-node quad element
        elements = np.array([[0, 1, 2, 3]])

        shell = model.Shell(
            nodes=nodes,
            elements=elements,
            young_modulus=210e9,
            poisson_ratio=0.3,
            density=7850.0,
            thickness=0.01,
        )
        return shell.K, shell.M

    def test_K_symmetric(self, shell_KM):
        K, _ = shell_KM
        K_dense = K.toarray() if hasattr(K, 'toarray') else K
        np.testing.assert_allclose(K_dense, K_dense.T, atol=1e-6,
                                   err_msg="Shell K is not symmetric")

    def test_M_symmetric(self, shell_KM):
        _, M = shell_KM
        M_dense = M.toarray() if hasattr(M, 'toarray') else M
        np.testing.assert_allclose(M_dense, M_dense.T, atol=1e-10,
                                   err_msg="Shell M is not symmetric")

    def test_K_positive_semidefinite(self, shell_KM):
        K, _ = shell_KM
        K_dense = K.toarray() if hasattr(K, 'toarray') else K
        eigvals = np.linalg.eigvalsh(K_dense)
        max_ev = np.max(np.abs(eigvals))
        tol = -1e-8 * max_ev
        assert np.all(eigvals >= tol), (
            f"Shell K has significantly negative eigenvalues: min={eigvals.min():.3e}, "
            f"tol={tol:.3e}"
        )

    def test_M_positive_semidefinite(self, shell_KM):
        _, M = shell_KM
        M_dense = M.toarray() if hasattr(M, 'toarray') else M
        eigvals = np.linalg.eigvalsh(M_dense)
        max_ev = np.max(np.abs(eigvals))
        tol = -1e-8 * max_ev
        assert np.all(eigvals >= tol), (
            f"Shell M has significantly negative eigenvalues: min={eigvals.min():.3e}, "
            f"tol={tol:.3e}"
        )


# ---------------------------------------------------------------------------
# TEST 4b — Tetrahedron matrix properties
# ---------------------------------------------------------------------------

class TestTetrahedronMatrixProperties:
    """
    Minimal tet10 mesh: a single 10-node quadratic tetrahedron.

    Tet10 node numbering (from the shape function code):
      - nodes 0..3: corner nodes (vertices)
      - nodes 4..9: midside nodes (edge midpoints)

    Corner nodes: v0=(0,0,0), v1=(1,0,0), v2=(0,1,0), v3=(0,0,1)
    Edge midpoints:
      node 4: mid(v0,v1) = (0.5, 0,   0  )   edge 0-1
      node 5: mid(v1,v2) = (0.5, 0.5, 0  )   edge 1-2
      node 6: mid(v0,v2) = (0,   0.5, 0  )   edge 0-2
      node 7: mid(v0,v3) = (0,   0,   0.5)   edge 0-3  (check shape fn indexing)
      node 8: mid(v1,v3) = (0.5, 0,   0.5)   edge 1-3
      node 9: mid(v2,v3) = (0,   0.5, 0.5)   edge 2-3

    Connectivity: element uses nodes [0,1,2,3,4,5,6,7,8,9]
    (indices into org).

    The assembled K/M are sparse. We test symmetry and PSD.
    Note: without a dof_mask the body is free-free, so K is singular
    (6 rigid-body modes), but all eigenvalues must be >= -eps.
    """

    @pytest.fixture(scope="class")
    def tet_KM(self):
        org = np.array([
            # corner nodes
            [0.0, 0.0, 0.0],   # 0
            [1.0, 0.0, 0.0],   # 1
            [0.0, 1.0, 0.0],   # 2
            [0.0, 0.0, 1.0],   # 3
            # midside nodes
            [0.5, 0.0, 0.0],   # 4  mid(0,1)
            [0.5, 0.5, 0.0],   # 5  mid(1,2)
            [0.0, 0.5, 0.0],   # 6  mid(0,2)
            [0.0, 0.0, 0.5],   # 7  mid(0,3)
            [0.5, 0.0, 0.5],   # 8  mid(1,3)
            [0.0, 0.5, 0.5],   # 9  mid(2,3)
        ], dtype=float)

        conec = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=int)

        E = 210e9
        rho = 7850.0
        nu = 0.3

        tet = model.Tetrahedron(
            org=org,
            conec=conec,
            young_modulus=float(E),
            density=float(rho),
            poisson_ratio=nu,
        )
        return tet.K, tet.M

    def test_K_symmetric(self, tet_KM):
        K, _ = tet_KM
        K_dense = K.toarray() if hasattr(K, 'toarray') else np.array(K)
        diff = np.max(np.abs(K_dense - K_dense.T))
        assert diff < 1e-4, f"Tetrahedron K not symmetric: max|K-K^T|={diff:.3e}"

    def test_M_symmetric(self, tet_KM):
        _, M = tet_KM
        M_dense = M.toarray() if hasattr(M, 'toarray') else np.array(M)
        diff = np.max(np.abs(M_dense - M_dense.T))
        assert diff < 1e-10, f"Tetrahedron M not symmetric: max|M-M^T|={diff:.3e}"

    def test_K_positive_semidefinite(self, tet_KM):
        K, _ = tet_KM
        K_dense = K.toarray() if hasattr(K, 'toarray') else np.array(K)
        eigvals = np.linalg.eigvalsh(K_dense)
        max_ev = np.max(np.abs(eigvals))
        tol = -1e-8 * max_ev
        assert np.all(eigvals >= tol), (
            f"Tetrahedron K has significantly negative eigenvalues: "
            f"min={eigvals.min():.3e}, tol={tol:.3e}"
        )

    def test_M_positive_semidefinite(self, tet_KM):
        _, M = tet_KM
        M_dense = M.toarray() if hasattr(M, 'toarray') else np.array(M)
        eigvals = np.linalg.eigvalsh(M_dense)
        max_ev = np.max(np.abs(eigvals))
        tol = -1e-8 * max_ev
        assert np.all(eigvals >= tol), (
            f"Tetrahedron M has significantly negative eigenvalues: "
            f"min={eigvals.min():.3e}, tol={tol:.3e}"
        )
