"""
Acoustic External Problem class for boundary element method.

Provides a high-level interface for solving exterior acoustic problems
using the boundary element method (BEM). Wraps the lower-level ``Body``,
``Field``, ``Mesh``, assembler, and solver classes into a single
configurable workflow.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from .geometry import Body, Field
from .mesh import Mesh
from .elements import ContinuousP1Mesh, DiscontinuousP1Mesh
from .integrators import ElementIntegratorCollocation
from .matrix_assembly import ContinuousAssembler, DiscontinuousAssembler
from .solve import BEMSolver


class AcousticExternalProblem:
    """
    High-level driver for solving exterior acoustic BEM problems.

    Accepts a triangular surface mesh (``pv.PolyData``), medium properties,
    boundary conditions, and solver parameters.  Handles the full workflow:

    #. Builds the underlying :class:`Body`, :class:`Field`, and :class:`Mesh`.
    #. Creates the appropriate collocation assembler.
    #. Assembles operator matrices on demand.
    #. Solves the boundary integral equation (direct or Burton–Miller).
    #. Evaluates the acoustic pressure or potential at arbitrary field points.

    Parameters
    ----------
    mesh : pv.PolyData
        Triangular surface mesh of the scattering / radiating body.
        Vertex coordinates must be in **metres** [m].
    rho : float
        Density of the surrounding medium [kg/m³].
    c0 : float
        Speed of sound in the surrounding medium [m/s].
    boundary_condition : np.ndarray, optional
        Boundary condition values at geometric nodes.  Shape ``(n_points,)``
        for a scalar field, or ``(n_points, 3)`` for a 3-D velocity vector
        field (which is automatically projected onto the nodal normals).
    boundary_condition_type : str, optional
        ``"Neumann"`` (default) or ``"Dirichlet"``.
    frequency : float, optional
        Frequency of the problem [Hz].  Can be set later via
        :meth:`set_frequency`.
    assembler_type : str, optional
        ``"continuous"`` (default) or ``"discontinuous"``.
    use_burton_miller : bool, optional
        If ``True`` (default) solves the Burton–Miller combined formulation
        to suppress spurious internal resonances.
    alpha_bm : complex, optional
        Coupling parameter for the Burton–Miller formulation.
        Default is ``1j``.
    quad_order : int, optional
        Order of the standard triangle quadrature rule (1, 3, or 7).
        Default is 3.
    near_threshold : float, optional
        Factor (× characteristic element length) used to detect near-singular
        elements.  Default is 2.0.

    Attributes
    ----------
    body : Body
        Underlying geometric body holding boundary conditions.
    field : Field
        Medium properties (density, sound speed).
    bem_mesh : Mesh
        Merged BEM mesh object with precomputed geometry and jump coefficients.
    integrator : ElementIntegratorCollocation
        Element-level integration engine.
    assembler : ContinuousAssembler | DiscontinuousAssembler
        Collocation matrix assembler.
    solver : BEMSolver
        High-level solver (assembles, solves, evaluates field).

    Examples
    --------
    >>> import numpy as np
    >>> import pyvista as pv
    >>> # Sphere mesh with unit radius
    >>> sphere = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)
    >>> # Uniform normal velocity BC
    >>> vn = np.ones(sphere.n_points, dtype=complex)
    >>> prob = AcousticExternalProblem(
    ...     mesh=sphere, rho=1.225, c0=343.0,
    ...     boundary_condition=vn, boundary_condition_type="Neumann",
    ...     frequency=500.0,
    ... )
    >>> phi, q = prob.solve_problem(verbose=False)
    >>> field_pts = np.array([[0.0, 0.0, 2.0]])
    >>> phi_field = prob.evaluate_field(field_pts, verbose=False)
    >>> # acoustic pressure: p = j ω ρ₀ φ
    >>> p = 1j * 2 * np.pi * prob.frequency * prob.rho * phi_field
    """

    def __init__(
        self,
        mesh: pv.PolyData,
        rho: float,
        c0: float,
        boundary_condition: np.ndarray | None = None,
        boundary_condition_type: str = "Neumann",
        frequency: float | None = None,
        assembler_type: str = "continuous",
        use_burton_miller: bool = True,
        alpha_bm: complex = 1j,
        quad_order: int = 3,
        near_threshold: float = 2.0,
    ) -> None:
        # ── Store parameters ──────────────────────────────────────────────────
        self._mesh_pv = mesh
        self.rho = rho
        self.c0 = c0
        self.boundary_condition_type = boundary_condition_type
        self.use_burton_miller = use_burton_miller
        self.alpha_bm = alpha_bm
        self.quad_order = quad_order
        self.near_threshold = near_threshold
        self._frequency = frequency
        self._assembler_type = assembler_type

        # ── Extract geometry ──────────────────────────────────────────────────
        nodes_bem = mesh.points.astype(np.float64)               # (N, 3) [m]
        elements_bem = mesh.faces.reshape(-1, 4)[:, 1:].copy()   # (M, 3)

        # ── Prepare boundary condition arrays ─────────────────────────────────
        if boundary_condition is not None:
            bc = np.asarray(boundary_condition, dtype=complex)
        else:
            bc = None

        if boundary_condition_type == "Neumann":
            Neumann_BC = bc
            Dirichlet_BC = None
        elif boundary_condition_type == "Dirichlet":
            Neumann_BC = None
            Dirichlet_BC = bc
        else:
            raise ValueError(
                f"boundary_condition_type must be 'Neumann' or 'Dirichlet', "
                f"got '{boundary_condition_type}'."
            )

        # ── Build Body, Field, Mesh ───────────────────────────────────────────
        self.body = Body(
            mesh_nodes=nodes_bem,
            mesh_elements=elements_bem,
            Neumann_BC=Neumann_BC,
            Dirichlet_BC=Dirichlet_BC,
            frequency=frequency,
        )
        self.field = Field(rho0=rho, c0=c0)
        self.bem_mesh = Mesh(
            self.body, peripheral_objects=None, field=self.field
        )

        # ── Build integrator and assembler ────────────────────────────────────
        self.integrator = ElementIntegratorCollocation(k=self.bem_mesh.k)

        if assembler_type == "continuous":
            element_mesh = ContinuousP1Mesh(self.bem_mesh)
            self.assembler = ContinuousAssembler(
                self.bem_mesh, self.integrator,
                quad_order=quad_order, near_threshold=near_threshold,
            )
        elif assembler_type == "discontinuous":
            element_mesh = DiscontinuousP1Mesh(
                self.bem_mesh, collocation_strategy="interior_shifted",
            )
            self.assembler = DiscontinuousAssembler(
                element_mesh, self.integrator,
                quad_order=quad_order, near_threshold=near_threshold,
            )
        else:
            raise ValueError(
                f"assembler_type must be 'continuous' or 'discontinuous', "
                f"got '{assembler_type}'."
            )
        self._element_mesh = element_mesh
        self.solver = BEMSolver(self.assembler)

        # ── Internal state ────────────────────────────────────────────────────
        self._matrices_assembled = False
        self._matrices: dict[str, np.ndarray] = {}
        self._phi: np.ndarray | None = None
        self._q: np.ndarray | None = None

    # ── Public helpers ────────────────────────────────────────────────────────

    def set_frequency(self, frequency: float) -> None:
        """
        Update the problem frequency.

        Updates the wavenumber ``k`` in the mesh, the integrator, and the
        underlying ``Body``.  Resets the assembled-matrices flag so that a
        subsequent :meth:`solve_problem` call re-assembles everything.

        Parameters
        ----------
        frequency : float
            Frequency in Hertz [Hz].
        """
        self._frequency = frequency
        omega = 2.0 * np.pi * frequency
        k = omega / self.c0

        self.bem_mesh.k = k
        self.bem_mesh.frequency = frequency
        self.bem_mesh.source_object.frequency = frequency
        self.integrator.k = k

        self._matrices_assembled = False
        self._matrices = {}

    def set_boundary_condition(self, boundary_condition: np.ndarray) -> None:
        """
        Update the boundary condition values.

        Parameters
        ----------
        boundary_condition : np.ndarray
            Shape ``(n_points,)`` for a scalar field, or
            ``(n_points, 3)`` for a 3-D velocity vector field (which is
            automatically projected onto the nodal normals).
        """
        bc = np.asarray(boundary_condition, dtype=complex)

        # Project 3-D vector BC onto angle-weighted node normals
        if bc.ndim == 2 and bc.shape[1] == 3:
            bc = np.einsum("ij,ij->i", bc, self.bem_mesh.node_n_hat)

        if self.boundary_condition_type == "Neumann":
            self.bem_mesh.Neumann_BC = bc
            self.bem_mesh.Dirichlet_BC = None
            self.body.Neumann_BC = bc
            self.body.Dirichlet_BC = None
        else:
            self.bem_mesh.Dirichlet_BC = bc
            self.bem_mesh.Neumann_BC = None
            self.body.Dirichlet_BC = bc
            self.body.Neumann_BC = None

    # ── Assembly ──────────────────────────────────────────────────────────────

    def assemble_matrices(
        self,
        ops: tuple[str, ...] = ("S", "D", "Kp", "NReg"),
        verbose: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Assemble BEM operator matrices.

        Can be called independently to inspect or reuse the boundary
        operator matrices without performing a solve.  Assembled matrices
        are stored as instance attributes (``self.S``, ``self.D``, …) and
        also returned as a dictionary.

        Parameters
        ----------
        ops : tuple of str, optional
            Operators to assemble.  Any subset of
            ``{"S", "D", "Kp", "N", "NReg"}``.
            Default is ``("S", "D", "Kp", "NReg")``.
        verbose : bool, optional
            Show progress bars during assembly.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping operator names to dense matrices.
        """
        mats = self.solver.assemble_matrices(ops=ops, verbose=verbose)
        self._matrices.update(mats)
        for key, val in mats.items():
            setattr(self, key, val)
        self._matrices_assembled = True
        return mats

    # ── Solve ─────────────────────────────────────────────────────────────────

    def solve_problem(
        self, verbose: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the BEM problem.

        If matrices have not been assembled yet (or the frequency has been
        changed since the last assembly), they are assembled automatically
        before solving.  The result is also stored on the instance as
        ``self.phi`` (velocity potential) and ``self.q`` (normal derivative).

        Parameters
        ----------
        verbose : bool, optional
            Show progress information.

        Returns
        -------
        phi : np.ndarray
            Velocity potential ``φ`` on the boundary degrees of freedom,
            shape ``(num_dofs,)``.
        q : np.ndarray
            Normal derivative ``q = ∂φ/∂n`` on the boundary degrees of
            freedom, shape ``(num_dofs,)``.

        Notes
        -----
        The returned ``(phi, q)`` pair is the complete boundary solution: it
        can be stored and later handed to :meth:`evaluate_field` (via its
        ``solution`` / ``phi`` / ``q`` arguments) to evaluate the field
        without solving again.

        The surface (boundary) acoustic pressure follows from the potential as
        ``p = j ω ρ₀ φ`` with ``ω = 2π·frequency`` [Pa].

        Examples
        --------
        >>> phi, q = prob.solve_problem(verbose=False)        # doctest: +SKIP
        >>> p_surface = 1j * 2 * np.pi * prob.frequency * prob.rho * phi
        """
        if not self._matrices_assembled:
            if self.use_burton_miller:
                self.assemble_matrices(
                    ops=("S", "D", "Kp", "NReg"), verbose=verbose,
                )
            else:
                self.assemble_matrices(ops=("S", "D"), verbose=verbose)

        if self.use_burton_miller:
            phi = self.solver.solve_burton_miller(
                matrices=self._matrices,
                alpha=self.alpha_bm,
                verbose=verbose,
            )
        else:
            phi = self.solver.solve_direct(
                matrices=self._matrices,
                verbose=verbose,
            )

        self._phi = phi
        self._q = self.solver.velocity_BC
        return self._phi, self._q

    # ── Field evaluation ──────────────────────────────────────────────────────

    def evaluate_field(
        self,
        field_points: np.ndarray,
        solution: tuple[np.ndarray, np.ndarray] | None = None,
        phi: np.ndarray | None = None,
        q: np.ndarray | None = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Evaluate the acoustic velocity potential at points in the domain.

        Uses the boundary solution ``(phi, q)`` and the Helmholtz boundary
        integral representation.  By default the solution stored by
        :meth:`solve_problem` is used, so a solve must have been run first.
        Alternatively a **precomputed** boundary solution can be passed in —
        either as a tuple via ``solution=(phi, q)`` or as the individual
        ``phi`` and ``q`` arrays — which lets you reuse a stored solution
        without solving again (see Notes).

        Parameters
        ----------
        field_points : np.ndarray
            Array of query points, shape ``(M, 3)``, in metres [m].
        solution : tuple of np.ndarray, optional
            Precomputed boundary solution ``(phi, q)`` at the boundary degrees
            of freedom, as returned by :meth:`solve_problem`.  Mutually
            exclusive with the ``phi`` / ``q`` arguments.
        phi : np.ndarray, optional
            Precomputed boundary velocity potential at the DOFs.  Use together
            with ``q`` instead of ``solution``.
        q : np.ndarray, optional
            Precomputed boundary normal derivative ``∂φ/∂n`` at the DOFs.
        verbose : bool, optional
            Show a progress bar during integration.

        Returns
        -------
        np.ndarray
            Complex velocity potential ``φ`` at the query points, shape
            ``(M,)``.  Obtain the acoustic pressure with
            ``p = 1j * 2 * np.pi * prob.frequency * prob.rho * phi`` [Pa].

        Notes
        -----
        Precompute-and-reuse workflow: call :meth:`solve_problem` once and keep
        the returned ``(phi, q)`` (the expensive part is the matrix assembly and
        linear solve).  Later, rebuild an :class:`AcousticExternalProblem` with
        the **same mesh and frequency** and pass the stored solution::

            phi, q = prob.solve_problem()
            # ... store phi, q (e.g. np.savez) ...
            # later, after rebuilding `prob` identically:
            phi_field = prob.evaluate_field(pts, solution=(phi, q))

        Only the solve is skipped; the surface geometry is reconstructed from
        the rebuilt problem.

        Raises
        ------
        ValueError
            If both ``solution`` and ``phi``/``q`` are supplied.
        RuntimeError
            If no boundary solution is available (no precomputed input given
            and :meth:`solve_problem` has not been run).
        """
        if solution is not None:
            if phi is not None or q is not None:
                raise ValueError(
                    "Pass either solution=(phi, q) or phi=/q=, not both."
                )
            phi, q = solution

        phi = self._phi if phi is None else phi
        q = self._q if q is None else q

        if phi is None or q is None:
            raise RuntimeError(
                "No boundary solution available.  Call solve_problem() first, "
                "or pass a precomputed solution=(phi, q) (or phi=, q=)."
            )

        return self.solver.evaluate_field(
            field_points=np.asarray(field_points, dtype=np.float64),
            phi=phi,
            q=q,
            quad_order=self.quad_order,
            verbose=verbose,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def phi(self) -> np.ndarray | None:
        """Velocity potential on boundary DOFs (set after :meth:`solve_problem`)."""
        return self._phi

    @property
    def q(self) -> np.ndarray | None:
        """Normal derivative ``∂φ/∂n`` on boundary DOFs."""
        return self._q

    @property
    def frequency(self) -> float | None:
        """Current frequency [Hz]."""
        return self._frequency

    def __repr__(self) -> str:
        return (
            f"AcousticExternalProblem("
            f"n_nodes={self.bem_mesh.num_nodes}, "
            f"n_elements={self.bem_mesh.num_elements}, "
            f"assembler='{self._assembler_type}', "
            f"BM={self.use_burton_miller})"
        )