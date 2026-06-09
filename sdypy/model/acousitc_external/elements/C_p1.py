"""
Continuous P1 (linear) element implementation for acoustic BEM.

This module wraps your existing Mesh class to provide a consistent
interface with the discontinuous element implementation.
"""

import numpy as np
from acoustic_BEM.mesh import Mesh as BaseMesh


class ContinuousP1Mesh:
    """
    Continuous P1 element mesh wrapper.
    
    This class provides a thin wrapper around your existing Mesh class
    to maintain API consistency with DiscontinuousP1Mesh.
    
    In continuous elements:
    - Nodes are shared between adjacent elements
    - DOF nodes = geometric nodes
    - Collocation points = nodes
    - N nodes = N DOFs
    
    Attributes:
        mesh (Mesh): Underlying mesh object from your existing implementation
        num_dofs (int): Number of degrees of freedom (= num_nodes)
        dof_coordinates (np.ndarray): Coordinates of DOF points, shape (N, 3)
        collocation_points (np.ndarray): Collocation points, shape (N, 3)
        collocation_normals (np.ndarray): Normals at collocation points, shape (N, 3)
        element_to_dof (np.ndarray): Mapping from elements to DOF indices, shape (M, 3)
    """
    
    def __init__(self, base_mesh: BaseMesh):
        """
        Initialize continuous P1 mesh from existing Mesh object.
        
        Args:
            base_mesh (Mesh): Your existing Mesh object with all geometric
                properties computed.
        """
        self.mesh = base_mesh
        
        # For continuous elements, DOFs are simply the nodes
        self.num_dofs = base_mesh.num_nodes
        self.dof_coordinates = base_mesh.mesh_nodes.copy()
        
        # Collocation points are at the nodes
        self.collocation_points = base_mesh.mesh_nodes.copy()
        self.collocation_normals = base_mesh.node_n_hat.copy()
        
        # Element connectivity maps directly to node indices
        self.element_to_dof = base_mesh.mesh_elements.copy()
        
        # Store element type identifier
        self.element_type = "continuous_p1"
        
    def get_dof_in_element(self, elem_idx: int) -> np.ndarray:
        """
        Get DOF indices for a given element.
        
        Args:
            elem_idx (int): Element index
            
        Returns:
            np.ndarray: Array of 3 DOF indices for this element
        """
        return self.element_to_dof[elem_idx]
    
    def get_collocation_point(self, dof_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get collocation point and normal for a given DOF.
        
        Args:
            dof_idx (int): DOF index
            
        Returns:
            tuple: (position, normal) both shape (3,)
        """
        return self.collocation_points[dof_idx], self.collocation_normals[dof_idx]
    
    def map_bc_from_geometric(self, geometric_bc: np.ndarray) -> np.ndarray:
        """
        Map boundary conditions from geometric nodes to DOFs.
        
        For continuous elements, this is identity mapping since
        DOF nodes = geometric nodes.
        
        Args:
            geometric_bc (np.ndarray): BC values at geometric nodes, shape (N,)
            
        Returns:
            np.ndarray: BC values at DOF nodes, shape (N,)
        """
        if geometric_bc.shape[0] != self.num_dofs:
            raise ValueError(f"BC array size {geometric_bc.shape[0]} does not "
                           f"match num_dofs {self.num_dofs}")
        return geometric_bc.copy()
    
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
    
    def __repr__(self) -> str:
        return (f"ContinuousP1Mesh(num_elements={self.mesh.num_elements}, "
                f"num_nodes={self.mesh.num_nodes}, num_dofs={self.num_dofs})")