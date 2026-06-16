import numpy as np

from .kernels import (G, dG_dn_y, 
                      dG_dr, r_vec)
from .quadrature import (standard_triangle_quad, 
                         map_to_physical_triangle_batch,
                         shape_functions_P1)
from .matrix_assembly import ContinuousAssembler, DiscontinuousAssembler
from .elements import ContinuousP1Mesh, DiscontinuousP1Mesh

from tqdm.auto import tqdm


class BEMSolver:
    """
    Boundary Element Method solver for acoustic problems.

    Handles matrix assembly, application of boundary conditions, and
    solution of the linear system for both continuous and discontinuous
    element formulations.

    Attributes:
        assembler (ContinuousAssembler | DiscontinuousAssembler): Assembler 
            object for building matrices.
        element_type (str): Type of elements ("continuous_p1" or 
            "discontinuous_p1")
    """

    def __init__(self, assembler: ContinuousAssembler | DiscontinuousAssembler):
        """Initialize the solver.

        Args:
            assembler (ContinuousAssembler | DiscontinuousAssembler): 
                Collocation assembler for either continuous or discontinuous 
                elements.
        """
        self.assembler = assembler
        self.element_mesh = assembler.element_mesh
        self.mesh = assembler.mesh
        
        # Detect element type
        if isinstance(assembler, DiscontinuousAssembler):
            self.element_type = "discontinuous_p1"
            self.is_discontinuous = True
        else:
            self.element_type = "continuous_p1"
            self.is_discontinuous = False
            
        # Check for overdetermined system (centroid collocation)
        if self.is_discontinuous:
            self.is_overdetermined = (
                self.element_mesh.collocation_strategy == "centroid"
            )
        else:
            self.is_overdetermined = False

    def assemble_matrices(self, 
                          ops: tuple[str, ...] = ("S","D","Kp","N", "NReg"),
                          verbose: bool = True,
                          ) -> dict[str, np.ndarray]:
        """
        Assemble selected operator matrices.

        Args:
            ops (tuple[str, ...]): Any subset of {"S","D","Kp","N", "NReg"}.
            verbose (bool): Show progress bars if True.

        Returns:
            dict[str, np.ndarray]: Dictionary of assembled matrices.
        """
        A: dict[str, np.ndarray] = {}
        for op in ops:
            A[op] = self.assembler.assemble(op, verbose=verbose)
        return A

    def solve_direct(self,
                     matrices: dict[str, np.ndarray] | None = None,
                     jump_coeff: np.ndarray | None = None,
                     verbose: bool = True) -> np.ndarray:
        """
        Solve equation:

            (D - C) φ = S q
        
        for the unknown boundary quantity. For exterior problems:

        - bc_type="Dirichlet": given φ on Γ, solve for q = ∂φ/∂n on Γ
        - bc_type="Neumann": given q = ∂φ/∂n on Γ, solve for φ on Γ

        Args:
            matrices (dict[str, np.ndarray] | None): Pre-assembled operator 
                matrices {"S","D"}. If None, assembles them.
            jump_coeff (np.ndarray | None): Jump coefficients at collocation
                points. If None, uses mesh's jump_coefficients or defaults 
                to 0.5.
            verbose (bool): Show progress information.

        Returns:
            np.ndarray: Solution vector for the unknown boundary quantity
                at DOF nodes.
        """
        if matrices is None:
            matrices = self.assemble_matrices(ops=("S","D"), verbose=verbose)

        # Determine boundary condition type
        if self.mesh.Dirichlet_BC is not None:
            bc_type = "Dirichlet"
            bc_values_geom = self.mesh.Dirichlet_BC
        elif self.mesh.Neumann_BC is not None:
            bc_type = "Neumann"
            bc_values_geom = self.mesh.Neumann_BC
        else:
            raise ValueError("No boundary condition values provided.")
        
        # Map BC to DOF structure
        bc_values = self._map_bc_to_dofs(bc_values_geom)
            
        S = matrices["S"]
        D = matrices["D"]

        # Build jump coefficient matrix
        C = self._build_jump_matrix(jump_coeff)

        # Solve system
        if bc_type == "Neumann":
            q = bc_values
            A_sys = D - C
            rhs = S @ q
            
            if self.is_overdetermined:
                sol, _, _, _ = np.linalg.lstsq(A_sys, rhs, rcond=None)
            else:
                sol = np.linalg.solve(A_sys, rhs)
                
            self.potential_BC = sol
            self.velocity_BC = bc_values
            return sol

        if bc_type == "Dirichlet":            
            phi = bc_values
            A_sys = S
            rhs = (D - C) @ phi
            
            if self.is_overdetermined:
                sol, _, _, _ = np.linalg.lstsq(A_sys, rhs, rcond=None)
            else:
                sol = np.linalg.solve(A_sys, rhs)
                
            self.velocity_BC = sol
            self.potential_BC = bc_values
            return sol
    
    def solve_burton_miller(self,
                            matrices: dict[str, np.ndarray] | None = None,
                            jump_coeff: np.ndarray | None = None,
                            alpha: complex = 1j,
                            verbose: bool = True,
                            ) -> np.ndarray:
        """
        Solve the BIE via the Burton–Miller combined formulation.

        This method forms a linear combination of the standard boundary equation
        and its normal-derivative equation to remove spurious resonances.

        The combined equation is taken (for exterior problems) in the form:
            (D - C) φ + α N φ = S q + α (C + K') q,
            [(D - C) + α N] φ = [S + α (C + K')] q

        where C is the double-layer jump term (typically 0.5·I on closed smooth
        surfaces).
            S  : single layer (G)
            D  : double layer (∂G/∂n_y)
            K' : adjoint double layer (∂G/∂n_x)
            N  : hypersingular (∂²G/∂n_x∂n_y)

        Given boundary data, the method solves for the complementary unknown:
        - bc_type="Neumann": given q = ∂φ/∂n, solve for φ on Γ
        - bc_type="Dirichlet": given φ on Γ, solve for q = ∂φ/∂n on Γ

        Args:
            matrices (dict[str, np.ndarray] | None): Pre-assembled operator 
                matrices {"S","D","Kp","NReg"}. If None, assembles them.
            jump_coeff (np.ndarray | None): Jump coefficients at collocation
                points. If None, uses mesh's jump_coefficients or defaults 
                to 0.5.
            alpha (complex): Coupling parameter α. Defaults to 1j.
            verbose (bool): Show progress information.

        Returns:
            np.ndarray: Solution vector (φ for Neumann input, 
                or q for Dirichlet input).
        """

        if matrices is None:
            matrices = self.assemble_matrices(ops=("S","D","Kp","NReg"), 
                                             verbose=verbose)

        # Determine boundary condition type
        if self.mesh.Dirichlet_BC is not None:
            bc_type = "Dirichlet"
            bc_values_geom = self.mesh.Dirichlet_BC
        elif self.mesh.Neumann_BC is not None:
            bc_type = "Neumann"
            bc_values_geom = self.mesh.Neumann_BC
        else:
            raise ValueError("No boundary condition values provided.")

        # Map BC to DOF structure
        bc_values = self._map_bc_to_dofs(bc_values_geom)

        S = matrices["S"]
        D = matrices["D"]
        Kp = matrices["Kp"]
        N  = matrices["NReg"]  # Use regularized hypersingular

        # Build jump coefficient matrix
        C = self._build_jump_matrix(jump_coeff)

        if bc_type == "Neumann":
            q = bc_values.astype(complex, copy=False)
            A_sys = (D - C) + alpha * N
            rhs = (S + alpha * (C + Kp)) @ q
            
            if self.is_overdetermined:
                phi, _, _, _ = np.linalg.lstsq(A_sys, rhs, rcond=None)
            else:
                phi = np.linalg.solve(A_sys, rhs)
                
            self.potential_BC = phi
            self.velocity_BC = bc_values
            return phi

        if bc_type == "Dirichlet":
            phi = bc_values.astype(complex, copy=False)
            A_sys = S + alpha * (C + Kp)
            rhs = (D - C + alpha * N) @ phi
            
            if self.is_overdetermined:
                q, _, _, _ = np.linalg.lstsq(A_sys, rhs, rcond=None)
            else:
                q = np.linalg.solve(A_sys, rhs)
                
            self.velocity_BC = q
            self.potential_BC = bc_values
            return q

        raise ValueError("bc_type must be 'Dirichlet' or 'Neumann'")
    
    def evaluate_field(self,
                       field_points: np.ndarray,
                       phi: np.ndarray | None = None,
                       q: np.ndarray | None = None,
                       quad_order: int = 3,
                       verbose: bool = True) -> np.ndarray:
        """
        Evaluate the potential at domain points using boundary solution.

        For discontinuous elements, the solution at DOFs is automatically
        interpolated within each element using P1 shape functions.

        Args:
            field_points (np.ndarray): Array of M points, shape (M,3).
            phi (np.ndarray | None): Boundary potential at DOF nodes, 
                shape (num_dofs,), or None.
            q (np.ndarray | None): Boundary normal derivative at DOF nodes, 
                shape (num_dofs,), or None.
            quad_order (int, optional): Triangle quadrature order. 
                Defaults to 3.
            verbose (bool): Show progress bar.

        Returns:
            np.ndarray: Complex potential at field points, shape (M,).
        """
        if phi is None:
            try: 
                phi = self.potential_BC
            except AttributeError:
                raise ValueError("Boundary potential not found. Provide as " 
                "Dirichlet BC or run solve_direct to compute from Neumann " 
                "boundary conditions.")
            
        if q is None:
            try:
                q = self.velocity_BC
            except AttributeError:
                raise ValueError("Boundary normal derivative not found. " 
                "Provide as Neumann BC or run solve_direct to compute from " 
                "Dirichlet boundary conditions.")

        # Set up quadrature
        xi_eta, w = standard_triangle_quad(quad_order)
        Nq = shape_functions_P1(xi_eta)
        yq, a2 = map_to_physical_triangle_batch(xi_eta,
                                               self.mesh.v0, 
                                               self.mesh.e1, 
                                               self.mesh.e2)
        w_phys = w[None, :, None] * a2[:, None, None]

        # Compute kernels
        r_norm, r_hat = r_vec(field_points[:, None, None, :], 
                             yq[None, :, :, :])[1:]

        Gvals = G(r_norm, self.mesh.k)
        dGr = dG_dr(r_norm, Gvals, self.mesh.k)

        ny_b = self.mesh.n_hat[None, :, None, :]
        dGdnY = dG_dn_y(r_hat, dGr, ny_b)

        u = np.zeros(field_points.shape[0], dtype=complex)

        # Integrate over elements
        for e in tqdm(range(self.mesh.num_elements), 
                      desc="Evaluating pressure field at points",
                      disable = not verbose):
            
            # Get DOF values for this element
            if self.is_discontinuous:
                dof_indices = self.element_mesh.get_dof_in_element(e)
                phi_elem = phi[dof_indices]
                q_elem = q[dof_indices]
            else:
                # Continuous elements: DOFs are node indices
                conn = self.mesh.mesh_elements[e]
                phi_elem = phi[conn]
                q_elem = q[conn]

            # Interpolate to quadrature points using shape functions
            phi_q = Nq @ phi_elem
            q_q = Nq @ q_elem

            wq = w_phys[e, :, 0]

            # Accumulate contributions
            u += np.sum(dGdnY[:, e, :] * (phi_q[None, :] * wq[None, :]), axis=1)
            u -= np.sum(Gvals[:, e, :] * (q_q[None, :] * wq[None, :]), axis=1)

        return u
    
    def get_solution_at_geometric_nodes(self,
                                       solution_type: str = "potential"
                                       ) -> np.ndarray:
        """
        Get the solution mapped to geometric mesh nodes.
        
        For continuous elements, this is identity.
        For discontinuous elements, averages DOF values at shared vertices.
        
        Args:
            solution_type (str): Either "potential" or "velocity"
            
        Returns:
            np.ndarray: Solution at geometric nodes, shape (N,)
        """
        if solution_type == "potential":
            try:
                dof_solution = self.potential_BC
            except AttributeError:
                raise ValueError("No potential solution available")
        elif solution_type == "velocity":
            try:
                dof_solution = self.velocity_BC
            except AttributeError:
                raise ValueError("No velocity solution available")
        else:
            raise ValueError("solution_type must be 'potential' or 'velocity'")
            
        if self.is_discontinuous:
            return self.element_mesh.map_dof_to_geometric(dof_solution)
        else:
            return dof_solution.copy()
    
    def _map_bc_to_dofs(self, bc_geom: np.ndarray) -> np.ndarray:
        """
        Map boundary conditions from geometric nodes to DOF structure.
        
        Args:
            bc_geom (np.ndarray): BC at geometric nodes, shape (N,)
            
        Returns:
            np.ndarray: BC at DOF nodes, shape (num_dofs,)
        """
        if self.is_discontinuous:
            return self.element_mesh.map_bc_from_geometric(bc_geom)
        else:
            return bc_geom.copy()
    
    def _build_jump_matrix(self, jump_coeff: np.ndarray | None) -> np.ndarray:
        """
        Build the jump coefficient matrix C.
        
        For continuous elements, can use solid angle computation.
        For discontinuous elements, typically use 0.5 (smooth closed surface).
        
        Args:
            jump_coeff (np.ndarray | None): Custom jump coefficients, or None
                to use defaults.
                
        Returns:
            np.ndarray: Diagonal matrix C, shape (num_dofs, num_dofs)
        """
        num_dofs = self.element_mesh.num_dofs
        
        if jump_coeff is not None:
            if jump_coeff.shape[0] != num_dofs:
                raise ValueError(f"jump_coeff size {jump_coeff.shape[0]} "
                               f"does not match num_dofs {num_dofs}")
            return np.diag(jump_coeff)
        
        # Use defaults
        if self.is_discontinuous:
            # For discontinuous elements on smooth closed surfaces, use 0.5
            # (collocation points are interior, not on boundary)
            return 0.5 * np.eye(num_dofs, dtype=complex)
        else:
            # For continuous elements, try to use solid angle computation
            try:
                # Map geometric node jump coefficients to DOFs
                jump_geom = self.mesh.jump_coefficients
                return np.diag(self._map_bc_to_dofs(jump_geom))
            except AttributeError:
                # Fallback to 0.5 if not available
                return 0.5 * np.eye(num_dofs, dtype=complex)
    
    def __repr__(self) -> str:
        overdetermined_str = " (overdetermined)" if self.is_overdetermined else ""
        return (f"BEMSolver(element_type='{self.element_type}'{overdetermined_str}, "
                f"num_elements={self.mesh.num_elements}, "
                f"num_dofs={self.element_mesh.num_dofs})")