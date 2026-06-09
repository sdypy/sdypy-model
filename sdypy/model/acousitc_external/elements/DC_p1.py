"""
Discontinuous P1 (linear) element implementation for acoustic BEM.

Discontinuous elements have independent DOFs for each element, even if
geometrically coincident. This is essential for handling edges, corners,
and the Burton-Miller formulation on complex geometries.
"""

import numpy as np
from acoustic_BEM.mesh import Mesh as BaseMesh


class DiscontinuousP1Mesh:
    """
    Discontinuous P1 element mesh.
    
    In discontinuous elements:
    - Each element has 3 independent DOF nodes
    - DOF nodes may be geometrically coincident but are algebraically independent
    - Collocation points are placed inside elements (not at vertices)
    - M elements × 3 nodes = 3M DOFs
    
    Attributes:
        mesh (Mesh): Underlying geometric mesh
        num_elements (int): Number of elements (M)
        num_dofs (int): Number of degrees of freedom (3M)
        dof_coordinates (np.ndarray): Coordinates of DOF nodes, shape (3M, 3)
        collocation_points (np.ndarray): Collocation points, shape (3M, 3)
        collocation_normals (np.ndarray): Normals at collocation points, shape (3M, 3)
        element_to_dof (np.ndarray): Maps element index to 3 DOF indices, shape (M, 3)
        collocation_strategy (str): Strategy used for placing collocation points
    """
    
    def __init__(self, 
                 base_mesh: BaseMesh,
                 collocation_strategy: str = "interior_shifted",
                 shift_factor: float = 0.15):
        """
        Initialize discontinuous P1 mesh.
        
        Args:
            base_mesh (Mesh): Your existing Mesh object with geometric properties
            collocation_strategy (str): Strategy for collocation point placement.
                Options:
                - "interior_shifted": Place collocation points inside element
                  (recommended, default)
                - "centroid": One collocation point at element centroid
                  (results in overdetermined system)
                - "vertex": Collocation at DOF vertices (can be unstable)
            shift_factor (float): For "interior_shifted", how far to shift
                collocation points toward centroid (0.0 = at vertex, 
                1.0 = at centroid). Typical: 0.1-0.2. Default: 0.15
        """
        self.mesh = base_mesh
        self.num_elements = base_mesh.num_elements
        self.num_dofs = 3 * self.num_elements
        self.collocation_strategy = collocation_strategy
        self.shift_factor = shift_factor
        
        # Build DOF structure
        self._build_dof_nodes()
        self._build_element_to_dof_mapping()
        self._build_collocation_points()
        
        # Store element type identifier
        self.element_type = "discontinuous_p1"
        
    def _build_dof_nodes(self):
        """
        Create independent DOF nodes for each element.
        
        Each element gets its own 3 DOF nodes at the geometric vertices.
        DOF nodes for different elements are algebraically independent even
        if geometrically coincident.
        """
        # For each element, replicate its 3 vertex coordinates
        # Result: (M, 3, 3) -> (3M, 3)
        dof_coords = []
        
        for elem_idx in range(self.num_elements):
            conn = self.mesh.mesh_elements[elem_idx]
            v0 = self.mesh.mesh_nodes[conn[0]]
            v1 = self.mesh.mesh_nodes[conn[1]]
            v2 = self.mesh.mesh_nodes[conn[2]]
            
            dof_coords.append(v0)
            dof_coords.append(v1)
            dof_coords.append(v2)
            
        self.dof_coordinates = np.array(dof_coords)
        
    def _build_element_to_dof_mapping(self):
        """
        Build mapping from element index to its 3 DOF indices.
        
        Element e has DOFs [3*e, 3*e+1, 3*e+2]
        """
        self.element_to_dof = np.arange(self.num_dofs).reshape(
            self.num_elements, 3
        )
        
    def _build_collocation_points(self):
        """
        Build collocation points based on selected strategy.
        """
        if self.collocation_strategy == "interior_shifted":
            self._build_interior_shifted_collocation()
        elif self.collocation_strategy == "centroid":
            self._build_centroid_collocation()
        elif self.collocation_strategy == "vertex":
            self._build_vertex_collocation()
        else:
            raise ValueError(f"Unknown collocation strategy: "
                           f"{self.collocation_strategy}")
            
    def _build_interior_shifted_collocation(self):
        """
        Place collocation points inside elements by shifting from vertices
        toward centroid.
        
        For each vertex j of element e:
            x_coll = (1 - α) * x_vertex + α * x_centroid
        
        where α = shift_factor (typically 0.1-0.2)
        
        This avoids singularities at geometric corners while maintaining
        a square 3M × 3M system.
        """
        alpha = self.shift_factor
        coll_points = []
        coll_normals = []
        
        for elem_idx in range(self.num_elements):
            centroid = self.mesh.centroids[elem_idx]
            elem_normal = self.mesh.n_hat[elem_idx]
            conn = self.mesh.mesh_elements[elem_idx]
            
            for local_node in range(3):
                vertex = self.mesh.mesh_nodes[conn[local_node]]
                
                # Shift toward centroid
                coll_pt = (1.0 - alpha) * vertex + alpha * centroid
                coll_points.append(coll_pt)
                
                # Use element normal (could also use vertex normal)
                coll_normals.append(elem_normal)
                
        self.collocation_points = np.array(coll_points)
        self.collocation_normals = np.array(coll_normals)
        
    def _build_centroid_collocation(self):
        """
        Place one collocation point at each element centroid.
        
        Results in M collocation points but 3M DOFs → overdetermined system.
        Use least-squares solver for this case.
        
        Note: For this strategy, only stores M collocation points, not 3M.
        """
        self.collocation_points = self.mesh.centroids.copy()
        self.collocation_normals = self.mesh.n_hat.copy()
        
        # Warning: This creates an overdetermined system
        if self.num_elements != self.num_dofs:
            import warnings
            warnings.warn(
                f"Centroid collocation creates overdetermined system: "
                f"{self.num_elements} equations, {self.num_dofs} unknowns. "
                f"Use least-squares solver.",
                UserWarning
            )
            
    def _build_vertex_collocation(self):
        """
        Place collocation points at DOF vertices.
        
        This is the most straightforward but can be less stable due to
        singularities at edges/corners.
        """
        self.collocation_points = self.dof_coordinates.copy()
        
        # For normals, use element normals repeated 3 times per element
        coll_normals = []
        for elem_idx in range(self.num_elements):
            elem_normal = self.mesh.n_hat[elem_idx]
            coll_normals.extend([elem_normal] * 3)
            
        self.collocation_normals = np.array(coll_normals)
        
    def get_dof_in_element(self, elem_idx: int) -> np.ndarray:
        """
        Get DOF indices for a given element.
        
        Args:
            elem_idx (int): Element index
            
        Returns:
            np.ndarray: Array of 3 DOF indices [3*e, 3*e+1, 3*e+2]
        """
        return self.element_to_dof[elem_idx]
    
    def get_collocation_point(self, 
                             dof_idx: int | None = None,
                             elem_idx: int | None = None) -> tuple[np.ndarray, 
                                                                   np.ndarray]:
        """
        Get collocation point and normal.
        
        Args:
            dof_idx (int | None): DOF index (for interior_shifted/vertex)
            elem_idx (int | None): Element index (for centroid strategy)
            
        Returns:
            tuple: (position, normal) both shape (3,)
        """
        if self.collocation_strategy == "centroid":
            if elem_idx is None:
                raise ValueError("elem_idx required for centroid collocation")
            return (self.collocation_points[elem_idx], 
                   self.collocation_normals[elem_idx])
        else:
            if dof_idx is None:
                raise ValueError("dof_idx required for this collocation strategy")
            return (self.collocation_points[dof_idx], 
                   self.collocation_normals[dof_idx])
    
    def map_bc_from_geometric(self, geometric_bc: np.ndarray) -> np.ndarray:
        """
        Map boundary conditions from geometric nodes to DOFs.
        
        Strategy: For each element, assign the BC values from its geometric
        vertices to its DOF nodes.
        
        Args:
            geometric_bc (np.ndarray): BC values at geometric nodes, shape (N,)
            
        Returns:
            np.ndarray: BC values at DOF nodes, shape (3M,)
        """
        if geometric_bc.shape[0] != self.mesh.num_nodes:
            raise ValueError(
                f"BC array size {geometric_bc.shape[0]} does not match "
                f"num_geometric_nodes {self.mesh.num_nodes}"
            )
            
        dof_bc = np.zeros(self.num_dofs, dtype=geometric_bc.dtype)
        
        for elem_idx in range(self.num_elements):
            conn = self.mesh.mesh_elements[elem_idx]
            dof_indices = self.get_dof_in_element(elem_idx)
            
            # Copy BC values from geometric vertices to DOF nodes
            dof_bc[dof_indices[0]] = geometric_bc[conn[0]]
            dof_bc[dof_indices[1]] = geometric_bc[conn[1]]
            dof_bc[dof_indices[2]] = geometric_bc[conn[2]]
            
        return dof_bc
    
    def map_dof_to_geometric(self, dof_solution: np.ndarray) -> np.ndarray:
        """
        Map DOF solution back to geometric nodes (averaging at shared vertices).
        
        Since multiple DOFs may correspond to the same geometric vertex,
        we average their values.
        
        Args:
            dof_solution (np.ndarray): Solution at DOF nodes, shape (3M,)
            
        Returns:
            np.ndarray: Solution at geometric nodes, shape (N,)
        """
        if dof_solution.shape[0] != self.num_dofs:
            raise ValueError(
                f"Solution array size {dof_solution.shape[0]} does not match "
                f"num_dofs {self.num_dofs}"
            )
            
        geometric_solution = np.zeros(self.mesh.num_nodes, 
                                      dtype=dof_solution.dtype)
        geometric_counts = np.zeros(self.mesh.num_nodes, dtype=int)
        
        for elem_idx in range(self.num_elements):
            conn = self.mesh.mesh_elements[elem_idx]
            dof_indices = self.get_dof_in_element(elem_idx)
            
            # Accumulate DOF values at geometric nodes
            for local_node, global_node in enumerate(conn):
                geometric_solution[global_node] += dof_solution[
                    dof_indices[local_node]
                ]
                geometric_counts[global_node] += 1
                
        # Average where multiple DOFs map to same geometric node
        mask = geometric_counts > 0
        geometric_solution[mask] /= geometric_counts[mask]
        
        return geometric_solution
    
    def get_element_properties(self) -> dict:
        """
        Get dictionary of element geometric properties.
        
        Returns:
            dict: Contains v0, e1, e2, n_hat, centroids, areas, a2
        """
        return {
            'v0': self.mesh.v0,
            'e1': self.mesh.e1,
            'e2': self.mesh.e2,
            'n_hat': self.mesh.n_hat,
            'centroids': self.mesh.centroids,
            'areas': self.mesh.areas,
            'a2': self.mesh.a2,
            'char_length': self.mesh.char_length,
        }
    
    def get_num_collocation_points(self) -> int:
        """
        Get number of collocation points (may differ from num_dofs for centroid).
        
        Returns:
            int: Number of collocation points
        """
        return self.collocation_points.shape[0]
    
    def __repr__(self) -> str:
        return (f"DiscontinuousP1Mesh(num_elements={self.num_elements}, "
                f"num_geometric_nodes={self.mesh.num_nodes}, "
                f"num_dofs={self.num_dofs}, "
                f"strategy='{self.collocation_strategy}')")