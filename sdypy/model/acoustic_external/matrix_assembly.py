import numpy as np
from tqdm.auto import tqdm

from .quadrature import (standard_triangle_quad, 
                         duffy_rule, 
                         telles_rule,
                         barycentric_projection,
                         shape_functions_P1,
                         map_to_physical_triangle_batch)

from .mesh import Mesh
from .elements import ContinuousP1Mesh, DiscontinuousP1Mesh
from .integrators import ElementIntegratorCollocation


class _CollocationCache:
    """
    Geometry + quadrature cache for Collocation. Useful for k-sweeping.
    Caches quadrature points mapped to physical elements for:
      - regular elements (all elements at once)
      - Telles rule (per node-element pair)
      - Duffy rule (per node-element pair)
    """
    def __init__(self, 
                 mesh: Mesh, 
                 quad_order: int):
        self.m = mesh
        self.quad_order = quad_order

        self.xi_eta_reg, self.w_reg = standard_triangle_quad(self.quad_order)

        self.N_reg = shape_functions_P1(self.xi_eta_reg) 

        xi = self.xi_eta_reg; w = self.w_reg
        yq, a2 = map_to_physical_triangle_batch(
            xi, self.m.v0, self.m.e1, self.m.e2
        )
        self._y_reg = yq
        self._w_reg_phys = (w[None, :] * a2[:, None])

        self._telles = {}
        self._duffy  = {}

    def get_regular(self, 
                    elem_idx: np.ndarray):

        return (self._y_reg[elem_idx], 
                self._w_reg_phys[elem_idx], 
                self.N_reg)

    def get_telles(self, 
                   node_idx: int, 
                   elem: int):
        key = (node_idx, elem)
        if key not in self._telles:
            x = self.m.mesh_nodes[node_idx]
            xi_star, eta_star = barycentric_projection(x, 
                                                       self.m.v0[elem], 
                                                       self.m.e1[elem], 
                                                       self.m.e2[elem])
            xi_eta, w = telles_rule(u_star=xi_star, v_star=eta_star, n_leg=10)
            yq, a2 = map_to_physical_triangle_batch(xi_eta,
                                                    self.m.v0[elem:elem+1],
                                                    self.m.e1[elem:elem+1],
                                                    self.m.e2[elem:elem+1])
            Nq = shape_functions_P1(xi_eta)
            self._telles[key] = (yq[0], w * a2[0], Nq)
        return self._telles[key]

    def get_duffy(self, node_idx: int, elem: int):
        key = (node_idx, elem)
        if key not in self._duffy:

            conn = self.m.mesh_elements[elem]
            loc = int(np.where(conn == node_idx)[0][0])
            xi_eta, w = duffy_rule(n_leg=10, sing_vert_int=loc)
            yq, a2 = map_to_physical_triangle_batch(xi_eta,
                                                    self.m.v0[elem:elem+1],
                                                    self.m.e1[elem:elem+1],
                                                    self.m.e2[elem:elem+1])
            Nq = shape_functions_P1(xi_eta)
            self._duffy[key] = (yq[0], w * a2[0], Nq)
        return self._duffy[key]


class ContinuousAssembler:
    """
    Collocation BEM assembler for continuous P1 elements.

    This class computes collocation matrices for:

    - Single-layer potential (``S``)
    - Double-layer potential (``D``)
    - Adjoint double-layer (``Kp``)
    - Hypersingular operator (``N``)
    - Regularized hypersingular operator (``NReg``)

    The assembler holds the mesh and integrator so that expensive data
    (geometry, connectivity, etc.) is reused across operators.
    
    This is your original CollocationAssembler, renamed for clarity.
    """

    def __init__(self,
                 mesh: ContinuousP1Mesh | Mesh,
                 integrator: ElementIntegratorCollocation,
                 quad_order: int = 3,
                 near_threshold: float = 2.0):
        """
        Initialize the continuous collocation assembler.

        Args:
            mesh (ContinuousP1Mesh | Mesh): Geometry and discretization data.
                Can accept either the new wrapper or legacy Mesh object.
            integrator (ElementIntegratorCollocation): Local integration engine.
            quad_order (int, optional): Order of standard triangle quadrature.
                Defaults to 3.
            near_threshold (float, optional): Distance factor for near-singular
                detection. Defaults to 2.0.
        """
        # Handle both legacy Mesh and new ContinuousP1Mesh
        if isinstance(mesh, ContinuousP1Mesh):
            self.element_mesh = mesh
            self.mesh = mesh.mesh
        else:
            # Legacy support: wrap bare Mesh object
            self.element_mesh = ContinuousP1Mesh(mesh)
            self.mesh = mesh
            
        self.integrator = integrator
        self.quad_order = quad_order
        self.near_threshold = near_threshold

        self.Nn = self.mesh.num_nodes
        self.Ne = self.mesh.num_elements

        self.cache = _CollocationCache(self.mesh, quad_order)

    def assemble(self, operator: str, verbose: bool = True) -> np.ndarray:
        """
        Assemble the collocation matrix for a boundary operator.

        Args:
            operator (str): One of ``{"S", "D", "Kp", "N", "NReg"}``.
            verbose (bool): Show progress bar if True.

        Returns:
            np.ndarray: Dense matrix of shape (num_nodes, num_nodes) containing
            the collocation coefficients for the selected operator.
        """
        if operator not in {"S", "D", "Kp", "N", "NReg"}:
            raise ValueError(f"Unknown operator {operator}")

        # prepare an empty matrix to fill
        A = np.zeros((self.Nn, self.Nn), dtype=np.complex128)

        # for each node:
        for node_idx in tqdm(range(self.Nn), 
                             desc=f"Assembling {operator} matrix",
                             disable = not verbose):
            x = self.mesh.mesh_nodes[node_idx]
            n_x = self.mesh.node_n_hat[node_idx]

            sing, near, reg = self.classify_elements(x, node_idx)

            if len(sing) > 0:
                for elem in sing:
                    if operator == "NReg":
                        conn = self.mesh.mesh_elements[elem]
                        try:
                            loc = int(np.where(conn == node_idx)[0][0])
                        except Exception:
                            loc = 0  # fallback
                        xi_eta, w = duffy_rule(n_leg=10, sing_vert_int=loc)
                        row = self.call_integrator(operator, x, n_x,
                                                   np.array([elem]),
                                                xi_eta, w, Nq=None, n_y=None)
                    else:
                        yq, w_phys, Nq = self.cache.get_duffy(node_idx, elem)
                        n_y = self.mesh.n_hat[elem:elem+1] if operator in \
                            {"D", "N"} else None
                        row = self.call_integrator(operator, x, n_x, 
                                                np.array([elem]), 
                                                yq[None, :, :], 
                                                w_phys[None, :], 
                                                Nq,
                                                n_y)
                    nodes = self.mesh.mesh_elements[elem]
                    for local, node in enumerate(nodes):
                        A[node_idx, node] += row[0, local]

            for elem in near:
                if operator == "NReg":
                    xi_star, eta_star = barycentric_projection(
                        x, self.mesh.v0[elem], 
                        self.mesh.e1[elem], self.mesh.e2[elem]
                    )
                    xi_eta, w = telles_rule(u_star=xi_star, 
                                            v_star=eta_star, 
                                            n_leg=10)
                    row = self.call_integrator(operator, x, n_x,
                                            np.array([elem]),
                                            xi_eta, w, Nq=None, n_y=None)
                else:
                    yq, w_phys, Nq = self.cache.get_telles(node_idx, elem)
                    n_y = self.mesh.n_hat[elem:elem+1] if operator in {"D", "N"}\
                        else None
                    row = self.call_integrator(operator, x, n_x, 
                                            np.array([elem]), 
                                            yq[None, :, :], 
                                            w_phys[None, :], 
                                            Nq,
                                            n_y)
                nodes = self.mesh.mesh_elements[elem]
                for local, node in enumerate(nodes):
                    A[node_idx, node] += row[0, local]

            if len(reg) > 0:
                if operator == "NReg":
                    xi_eta, w = standard_triangle_quad(self.quad_order)
                    vals = self.call_integrator(operator, x, n_x, 
                                                reg, 
                                                xi_eta, 
                                                w, 
                                                Nq=None, 
                                                n_y=None)
                    for el, row in zip(self.mesh.mesh_elements[reg], vals):
                        for local, node in enumerate(el):
                            A[node_idx, node] += row[local]

                else:
                    y_phys, w_phys, N = self.cache.get_regular(reg)
                    n_y = self.mesh.n_hat[reg] if operator in {"D", "N"} else None
                    vals = self.call_integrator(operator, x, n_x, reg,
                                                y_phys, w_phys, N,
                                                n_y)
                    
                    for el, row in zip(self.mesh.mesh_elements[reg], vals):
                        for local, node in enumerate(el):
                            A[node_idx, node] += row[local]
        return A
    
    def call_integrator(self,
                        operator: str,
                        x: np.ndarray,
                        n_x: np.ndarray | None,
                        elem_idx: np.ndarray | None,
                        xi_eta: np.ndarray,
                        w: np.ndarray,
                        Nq: np.ndarray | None,
                        n_y: np.ndarray | None = None) -> np.ndarray:
        """
        Dispatch to the correct integrator method.
        Args:
            operator (str): Operator key (``S``, ``D``, ``Kp``, ``N`` or 
                "NReg").
            x (np.ndarray): Collocation point, shape (3,).
            n_x (np.ndarray): Outward normal at the collocation point, shape 
                (3,).
            elem_idx (np.ndarray): Indices of source elements.
            xi_eta (np.ndarray): Quadrature points, shape (K, Q, 3).
            w (np.ndarray): Quadrature weights, shape (K, Q).
            Nq (np.ndarray): Shape functions at quadrature points, shape (Q, 3).
            n_y (np.ndarray | None): Outward normals at source elements, 
                shape (len(elem_idx), 3). Required for ``D``, ``N`` and 
                ``NReg``.

        Returns:
            np.ndarray: Local element contributions, shape (len(elem_idx), 3).
        """
        
        if operator == "S":
            return self.integrator.single_layer(x = x,
                                                y_phys = xi_eta,
                                                w_phys = w,
                                                N = Nq)
        if operator == "D":
            if n_y is None:
                raise ValueError("n_y must be provided for double-layer \
                                 operator")
            return self.integrator.double_layer(x = x,
                                               y_phys = xi_eta,
                                               w_phys = w,
                                               N = Nq,
                                               n_y = n_y)
        if operator == "Kp":
            return self.integrator.adjoint_double_layer(x = x,
                                                       x_normal = n_x,
                                                       y_phys = xi_eta,
                                                       w_phys = w,
                                                       N = Nq)     
        if operator == "N":
            if n_y is None:
                raise ValueError("n_y must be provided for hypersingular \
                                 operator")
            return self.integrator.hypersingular_layer(x = x,
                                                      x_normal = n_x,
                                                      y_phys = xi_eta,
                                                      w_phys = w,
                                                      N = Nq,
                                                      n_y = n_y)
        if operator == "NReg":
            return self.integrator.hypersingular_layer_reg(x = x,
                                        x_normal = n_x,
                                        y_v0 = self.mesh.v0[elem_idx],
                                        y_e1 = self.mesh.e1[elem_idx],
                                        y_e2 = self.mesh.e2[elem_idx],
                                        y_normals= self.mesh.n_hat[elem_idx],
                                        xi_eta = xi_eta,
                                        w = w,)
        raise ValueError(f"Unsupported operator: {operator}")

    def classify_elements(self, 
                          x: np.ndarray, 
                          node_idx: int) -> tuple[np.ndarray, 
                                                  np.ndarray, 
                                                  np.ndarray]:
        """
        Classify elements relative to a collocation node.

        Args:
            x (np.ndarray): Coordinates of the collocation node, shape (3,).
            node_idx (int): Index of the node in the mesh.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of element indices
            corresponding to singular, near-singular, and regular categories.
        """
        singular = self.mesh.node_in_el[node_idx]
        d = np.linalg.norm(self.mesh.centroids - x, axis=1)
        near = np.where(d < self.near_threshold * self.mesh.char_length)[0]
        near = np.setdiff1d(near, singular)
        regular = np.setdiff1d(np.arange(self.Ne), np.union1d(singular, near))
        return singular, near, regular


# For backward compatibility, keep old name as alias
CollocationAssembler = ContinuousAssembler


class DiscontinuousAssembler:
    """
    Collocation BEM assembler for discontinuous P1 elements.
    
    Key differences from continuous assembler:
    - Collocation points are interior to elements (no singular integrals)
    - System size is 3M × 3M (M = num_elements)
    - No need for Duffy quadrature (collocation points avoid vertices)
    - Telles quadrature still used for near-singular integrals
    
    Supports the same operators: S, D, Kp, N, NReg
    """
    
    def __init__(self,
                 mesh: DiscontinuousP1Mesh,
                 integrator: ElementIntegratorCollocation,
                 quad_order: int = 3,
                 near_threshold: float = 2.0,
                 precompute_telles: bool = True):
        """
        Initialize the discontinuous collocation assembler.
        
        Args:
            mesh (DiscontinuousP1Mesh): Discontinuous element mesh.
            integrator (ElementIntegratorCollocation): Local integration engine.
            quad_order (int): Order of standard triangle quadrature. Default 3.
            near_threshold (float): Distance factor for near-singular detection.
                Default 2.0.
            precompute_telles (bool): Pre-compute Telles quadrature for all
                near-singular pairs. Default True for better performance.
        """
        self.element_mesh = mesh
        self.mesh = mesh.mesh  # Base geometric mesh
        self.integrator = integrator
        self.quad_order = quad_order
        self.near_threshold = near_threshold
        
        self.num_dofs = mesh.num_dofs
        self.num_elements = mesh.num_elements
        self.num_collocation = mesh.get_num_collocation_points()
        
        # Pre-compute quadrature on all elements (CRITICAL OPTIMIZATION)
        self.xi_eta_reg, self.w_reg = standard_triangle_quad(quad_order)
        self.N_reg = shape_functions_P1(self.xi_eta_reg)
        
        # Pre-compute physical quadrature points for ALL elements
        # This saves ~20s per assembly by avoiding redundant computation!
        yq, a2 = map_to_physical_triangle_batch(
            self.xi_eta_reg, self.mesh.v0, self.mesh.e1, self.mesh.e2
        )
        self._y_reg = yq  # (M, Q, 3) - physical quad points
        self._w_reg_phys = self.w_reg[None, :] * a2[:, None]  # (M, Q) - physical weights
        
        # Cache for Telles quadrature (geometry + weights)
        self._telles_cache = {}  # Maps (coll_point_tuple, elem) -> (y_phys, w_phys, N)
        
        # Pre-compute Telles for all near-singular pairs
        if precompute_telles:
            self._precompute_telles_cache()
            
    def _precompute_telles_cache(self):
        """
        Pre-compute Telles quadrature for all near-singular element pairs.
        
        This can significantly speed up assembly by avoiding on-the-fly
        computation during the main assembly loop.
        """
        print("Pre-computing Telles quadrature cache...")
        
        if self.element_mesh.collocation_strategy == "centroid":
            # For centroid, collocation points are at element centroids
            collocation_points = self.mesh.centroids
        else:
            collocation_points = self.element_mesh.collocation_points
            
        # Build cache
        for coll_idx in range(len(collocation_points)):
            x = collocation_points[coll_idx]
            near, _ = self._classify_elements_discontinuous(x)
            
            for elem in near:
                # This populates the cache
                _ = self._get_telles_quad(x, elem)
                
        print(f"Cached {len(self._telles_cache)} Telles quadrature rules.")
        
    def assemble(self, operator: str, verbose: bool = True) -> np.ndarray:
        """
        Assemble the collocation matrix for a boundary operator.
        
        Args:
            operator (str): One of {"S", "D", "Kp", "N", "NReg"}
            verbose (bool): Show progress bar if True
            
        Returns:
            np.ndarray: Dense matrix of shape (num_collocation, num_dofs)
                For interior_shifted: (3M, 3M)
                For centroid: (M, 3M) - overdetermined
        """
        if operator not in {"S", "D", "Kp", "N", "NReg"}:
            raise ValueError(f"Unknown operator: {operator}")
            
        A = np.zeros((self.num_collocation, self.num_dofs), dtype=np.complex128)
        
        # Handle different collocation strategies
        if self.element_mesh.collocation_strategy == "centroid":
            self._assemble_centroid(A, operator, verbose)
        else:
            self._assemble_interior_shifted(A, operator, verbose)
            
        return A
    
    def _assemble_interior_shifted(self, 
                                   A: np.ndarray, 
                                   operator: str,
                                   verbose: bool):
        """
        Assembly for interior_shifted and vertex collocation strategies.
        
        Each DOF has its own collocation point.
        """
        for dof_idx in tqdm(range(self.num_dofs),
                           desc=f"Assembling {operator} (discontinuous)",
                           disable=not verbose):
            x, n_x = self.element_mesh.get_collocation_point(dof_idx=dof_idx)
            
            # Classify elements (no singular elements since collocation is interior)
            near, regular = self._classify_elements_discontinuous(x)
            
            # Process near elements with Telles quadrature
            for elem in near:
                yq, w_phys, Nq = self._get_telles_quad(x, elem)
                
                n_y = self.mesh.n_hat[elem:elem+1] if operator in {"D", "N"} \
                    else None
                
                if operator == "NReg":
                    # NReg uses different integration approach
                    xi_eta, w = self._get_telles_quad_ref(x, elem)  # Get reference quad
                    row = self._call_integrator(operator, x, n_x,
                                               np.array([elem]),
                                               xi_eta, w, None, n_y)
                else:
                    # Use pre-computed physical quadrature
                    row = self._call_integrator(operator, x, n_x,
                                               np.array([elem]),
                                               yq[None, :, :],
                                               w_phys[None, :],
                                               Nq, n_y)
                
                # Add to matrix
                dof_indices = self.element_mesh.get_dof_in_element(elem)
                for local_dof, global_dof in enumerate(dof_indices):
                    A[dof_idx, global_dof] += row[0, local_dof]
            
            # Process regular elements with standard quadrature
            if len(regular) > 0:
                if operator == "NReg":
                    xi_eta, w = standard_triangle_quad(self.quad_order)
                    vals = self._call_integrator(operator, x, n_x,
                                                regular, xi_eta, w, None, None)
                else:
                    y_phys = self._y_reg[regular]
                    w_phys = self._w_reg_phys[regular]
                    n_y = self.mesh.n_hat[regular] if operator in {"D", "N"} \
                        else None
                    
                    vals = self._call_integrator(operator, x, n_x, regular,
                                                y_phys, w_phys, self.N_reg, n_y)
                
                # Add to matrix
                for elem_idx, row in zip(regular, vals):
                    dof_indices = self.element_mesh.get_dof_in_element(elem_idx)
                    for local_dof, global_dof in enumerate(dof_indices):
                        A[dof_idx, global_dof] += row[local_dof]
    
    def _assemble_centroid(self, 
                          A: np.ndarray, 
                          operator: str,
                          verbose: bool):
        """
        Assembly for centroid collocation strategy.
        
        One collocation point per element at centroid.
        Results in overdetermined M × 3M system.
        """
        for elem_idx in tqdm(range(self.num_elements),
                            desc=f"Assembling {operator} (discontinuous-centroid)",
                            disable=not verbose):
            x, n_x = self.element_mesh.get_collocation_point(elem_idx=elem_idx)
            
            near, regular = self._classify_elements_discontinuous(x)
            
            # Process all elements (near and regular)
            all_elems = np.concatenate([near, regular]) if len(near) > 0 else regular
            
            for elem in all_elems:
                # Use Telles for near, standard for regular
                if elem in near:
                    xi_eta, w = self._get_telles_quad(x, elem)
                    yq, a2 = map_to_physical_triangle_batch(
                        xi_eta,
                        self.mesh.v0[elem:elem+1],
                        self.mesh.e1[elem:elem+1],
                        self.mesh.e2[elem:elem+1]
                    )
                    w_phys = w * a2[0]
                    Nq = shape_functions_P1(xi_eta)
                else:
                    yq = self._y_reg[elem:elem+1]
                    w_phys = self._w_reg_phys[elem:elem+1]
                    Nq = self.N_reg
                
                n_y = self.mesh.n_hat[elem:elem+1] if operator in {"D", "N", "NReg"} \
                    else None
                
                if operator == "NReg":
                    if elem in near:
                        xi_eta, w = self._get_telles_quad(x, elem)
                    else:
                        xi_eta, w = standard_triangle_quad(self.quad_order)
                    row = self._call_integrator(operator, x, n_x,
                                               np.array([elem]),
                                               xi_eta, w, None, n_y)
                else:
                    row = self._call_integrator(operator, x, n_x,
                                               np.array([elem]),
                                               yq[0][None, :, :] if elem in near 
                                               else yq,
                                               w_phys[None, :] if elem in near 
                                               else w_phys,
                                               Nq, n_y)
                
                dof_indices = self.element_mesh.get_dof_in_element(elem)
                for local_dof, global_dof in enumerate(dof_indices):
                    A[elem_idx, global_dof] += row[0, local_dof]
    
    def _classify_elements_discontinuous(self, 
                                        x: np.ndarray) -> tuple[np.ndarray, 
                                                                np.ndarray]:
        """
        Classify elements as near or regular (no singular elements).
        
        Args:
            x (np.ndarray): Collocation point, shape (3,)
            
        Returns:
            tuple: (near_elements, regular_elements)
        """
        d = np.linalg.norm(self.mesh.centroids - x, axis=1)
        near = np.where(d < self.near_threshold * self.mesh.char_length)[0]
        regular = np.setdiff1d(np.arange(self.num_elements), near)
        return near, regular
    
    def _get_telles_quad(self, 
                        x: np.ndarray, 
                        elem: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get or compute Telles quadrature for near-singular element.
        
        Now returns (y_phys, w_phys, N_vals) - fully pre-computed physical quadrature.
        
        Args:
            x (np.ndarray): Collocation point
            elem (int): Element index
            
        Returns:
            tuple: (y_phys, w_phys, N_vals)
                y_phys: Physical quad points, shape (Q, 3)
                w_phys: Physical weights, shape (Q,)
                N_vals: Shape function values, shape (Q, 3)
        """
        # Create cache key from position and element
        key = (tuple(np.round(x, 8)), elem)  # Round to avoid float precision issues
        
        if key not in self._telles_cache:
            # Compute Telles transformation in reference space
            xi_star, eta_star = barycentric_projection(
                x, self.mesh.v0[elem], self.mesh.e1[elem], self.mesh.e2[elem]
            )
            xi_eta, w = telles_rule(u_star=xi_star, v_star=eta_star, n_leg=10)
            
            # Map to physical space ONCE
            yq, a2 = map_to_physical_triangle_batch(
                xi_eta,
                self.mesh.v0[elem:elem+1],
                self.mesh.e1[elem:elem+1],
                self.mesh.e2[elem:elem+1]
            )
            w_phys = w * a2[0]
            N_vals = shape_functions_P1(xi_eta)
            
            # Cache physical quantities AND reference quadrature (for NReg)
            self._telles_cache[key] = (yq[0], w_phys, N_vals, xi_eta, w)
            
        result = self._telles_cache[key]
        return result[0], result[1], result[2]  # y_phys, w_phys, N_vals
    
    def _get_telles_quad_ref(self, x: np.ndarray, elem: int) -> tuple[np.ndarray, np.ndarray]:
        """Get reference Telles quadrature (for NReg operator)."""
        key = (tuple(np.round(x, 8)), elem)
        if key in self._telles_cache:
            result = self._telles_cache[key]
            return result[3], result[4]  # xi_eta, w
        
        # Compute if not cached
        xi_star, eta_star = barycentric_projection(
            x, self.mesh.v0[elem], self.mesh.e1[elem], self.mesh.e2[elem]
        )
        return telles_rule(u_star=xi_star, v_star=eta_star, n_leg=10)
    
    def _call_integrator(self,
                        operator: str,
                        x: np.ndarray,
                        n_x: np.ndarray | None,
                        elem_idx: np.ndarray,
                        xi_eta: np.ndarray,
                        w: np.ndarray,
                        Nq: np.ndarray | None,
                        n_y: np.ndarray | None = None) -> np.ndarray:
        """
        Dispatch to the correct integrator method.
        
        Same signature as ContinuousAssembler.call_integrator
        """
        if operator == "S":
            return self.integrator.single_layer(x=x, y_phys=xi_eta,
                                               w_phys=w, N=Nq)
        if operator == "D":
            if n_y is None:
                raise ValueError("n_y required for double-layer operator")
            return self.integrator.double_layer(x=x, y_phys=xi_eta,
                                               w_phys=w, N=Nq, n_y=n_y)
        if operator == "Kp":
            return self.integrator.adjoint_double_layer(x=x, x_normal=n_x,
                                                       y_phys=xi_eta,
                                                       w_phys=w, N=Nq)
        if operator == "N":
            if n_y is None:
                raise ValueError("n_y required for hypersingular operator")
            return self.integrator.hypersingular_layer(x=x, x_normal=n_x,
                                                      y_phys=xi_eta,
                                                      w_phys=w, N=Nq, n_y=n_y)
        if operator == "NReg":
            return self.integrator.hypersingular_layer_reg(
                x=x, x_normal=n_x,
                y_v0=self.mesh.v0[elem_idx],
                y_e1=self.mesh.e1[elem_idx],
                y_e2=self.mesh.e2[elem_idx],
                y_normals=self.mesh.n_hat[elem_idx],
                xi_eta=xi_eta, w=w
            )
        raise ValueError(f"Unsupported operator: {operator}")