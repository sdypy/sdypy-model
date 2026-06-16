import numpy as np

class Body:
    def __init__(self,
                 mesh_nodes: np.ndarray,
                 mesh_elements: np.ndarray,
                 Neumann_BC: np.ndarray | None,
                 Dirichlet_BC: np.ndarray | None,
                 frequency: float | None = None,):
        """
        Initialize the Body class for acoustic boundary element method.
        Args:
            mesh_nodes (np.ndarray): Array of shape (N, 3) representing the
                coordinates of the mesh nodes.
            mesh_elements (np.ndarray): Array of shape (M, 3) representing the
                connectivity of the mesh elements.
            Neumann_BC (np.ndarray | None): Array of shape (N,) or (N, 3)
                representing the Neumann boundary conditions at the mesh nodes.
            Dirichlet_BC (np.ndarray | None): Array of shape (N,) representing
                the Dirichlet boundary conditions at the mesh nodes.
            frequency (float | None): Frequency of the acoustic wave in Hz.
        """
        self.mesh_nodes = mesh_nodes
        self.mesh_elements = mesh_elements
        self.Neumann_BC = Neumann_BC
        self.Dirichlet_BC = Dirichlet_BC
        self.frequency = frequency

        self.num_nodes = self.mesh_nodes.shape[0]
        self.num_elements = self.mesh_elements.shape[0]

        self.precompute_elements()
        # Compute default nodal normals using area weighting - maybe switch to angle wighting
        self.node_normals_area()
        self.get_characteristic_length()
        
        if self.Neumann_BC is not None and self.Neumann_BC.ndim == 2:
            self.project_neumann_bc()

    def precompute_elements(self):
        """
        Precompute geometric properties of the mesh elements. Initializes:
            - v0: First vertex of each triangle.
            - e1: Edge vector from v0 to v1.
            - e2: Edge vector from v0 to v2.
            - a2: Twice the area of each triangle (||e1×e2|| Jacobian).
            - n_hat: Unit normal vector of each triangle.
            - centroids: Centroid of each triangle.
            - areas: Area of each triangle.
            - node_normals: Area-weighted normals at each mesh node.
            - char_length: Characteristic length of the mesh.
        """
        v0 = self.mesh_nodes[self.mesh_elements[:, 0], :]
        v1 = self.mesh_nodes[self.mesh_elements[:, 1], :]
        v2 = self.mesh_nodes[self.mesh_elements[:, 2], :]
        e1 = v1 - v0
        e2 = v2 - v0
        cross = np.cross(e1, e2)
        a2 = np.linalg.norm(cross, axis=1)
        n_hat = cross / (a2[:, np.newaxis] + 1e-300)
        centroids = (v0 + v1 + v2) / 3.0
        areas = 0.5 * a2
        node_in_el = [self.node_in_element(i) for i in range(self.num_nodes)]

        self.v0 = v0
        self.e1 = e1
        self.e2 = e2
        self.a2 = a2
        self.n_hat = n_hat # unit normal vector for each triangle
        self.centroids = centroids
        self.areas = areas
        self.node_in_el = node_in_el

    def node_normals_area(self) -> np.ndarray:
        """
        Compute area-weighted normals at each mesh node and store in 
        self.node_n_hat.
        """
        node_normals = np.zeros((self.num_nodes, 3), dtype=float)

        contrib = self.n_hat * (self.areas[:, None])

        idx0 = self.mesh_elements[:, 0]
        idx1 = self.mesh_elements[:, 1]
        idx2 = self.mesh_elements[:, 2]
        np.add.at(node_normals, idx0, contrib)
        np.add.at(node_normals, idx1, contrib)
        np.add.at(node_normals, idx2, contrib)

        norms = np.linalg.norm(node_normals, axis=1, keepdims=True) + 1e-300
        node_normals = node_normals / norms

        self.node_n_hat = node_normals

    def node_normals_angle(self) -> np.ndarray:
        """
        Compute angle-weighted normals at each mesh node and store in
        self.node_n_hat.

        Each face normal contributes to its three vertices weighted by the
        interior angle at that vertex. This often produces smoother per-vertex
        normals on irregular meshes than pure area weighting.
        """
        a0, b0 = self.e1, self.e2               
        a1, b1 = (self.e2 - self.e1), (-self.e1)
        a2, b2 = (self.e1 - self.e2), (-self.e2)

        ang0 = self._angles(a0, b0)
        ang1 = self._angles(a1, b1)
        ang2 = self._angles(a2, b2)

        node_normals = np.zeros((self.num_nodes, 3), dtype=float)
        idx0 = self.mesh_elements[:, 0]
        idx1 = self.mesh_elements[:, 1]
        idx2 = self.mesh_elements[:, 2]

        np.add.at(node_normals, idx0, self.n_hat * ang0[:, None])
        np.add.at(node_normals, idx1, self.n_hat * ang1[:, None])
        np.add.at(node_normals, idx2, self.n_hat * ang2[:, None])

        norms = np.linalg.norm(node_normals, axis=1, keepdims=True) + 1e-300
        node_normals = node_normals / norms

        self.node_n_hat = node_normals
    
    def _angles(self,
                u: np.ndarray, 
                v: np.ndarray) -> np.ndarray:
        un = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-300)
        vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-300)
        cos = np.clip(np.sum(un * vn, axis=1), -1.0, 1.0)
        return np.arccos(cos)

    def get_characteristic_length(self) -> float:
        """
        Compute the characteristic length of the mesh.

        Returns:
            char_length (float): Characteristic length of the mesh.
        """
        self.char_length = np.maximum.reduce([
            np.linalg.norm(self.e1, axis=1),
            np.linalg.norm(self.e2, axis=1),
            np.linalg.norm(self.e1 - self.e2, axis=1)
        ])

    def node_in_element(self, node_idx: int) -> np.ndarray:
        """
        Get the indices of elements connected to a given node.

        Args:
            node_idx (int): Index of the node.

        Returns:
            elements (np.ndarray): Array of element indices connected to the 
                given node.
        """
        return np.where(self.mesh_elements == node_idx)[0]

    def project_neumann_bc(self) -> None:
        """Project Neumann boundary data onto the mesh normals.

        If ``self.Neumann_BC`` is given as full 3-D Neumann components
        with shape (num_nodes, 3), this replaces it by its scalar
        projection along the nodal normals:

            q_i = v_i · n_i

        where ``v_i`` is the velocity vector at node i and ``n_i`` is the
        unit outward normal stored in ``self.node_n_hat``.

        Does nothing if ``Neumann_BC`` is already 1-D.

        Raises:
            ValueError: if Neumann_BC has an unexpected shape.
        """
        if self.Neumann_BC.ndim == 1:
            return  # already scalar

        if self.Neumann_BC.shape == (self.num_nodes, 3):
            self.Neumann_BC = np.einsum(
                "ij,ij->i", self.Neumann_BC, self.node_n_hat
            )
            return

        raise ValueError(
            f"Neumann_BC has shape {self.Neumann_BC.shape}, "
            f"expected ({self.num_nodes},) or ({self.num_nodes},3)"
        )
    
class Field:
    def __init__(self,
                #  field_extent: np.ndarray,
                #  num_points: np.ndarray,
                 rho0: float = 1.225,
                 c0: float = 343.0,):
        """
        Initialize the Field class for acoustic boundary element method.
        Initializes:
            - field_points: Array of shape (P, 3) representing the field points.

        Args:
            field_extent (np.ndarray): Array of shape (3, 2) defining the min
                and max coordinates of the field region along x, y, z axes.
            num_points (np.ndarray): Array of shape (3,) defining the number
                of points along each axis in the field region.
            rh0 (float): Density of the medium (default is 1.225 kg/m^3 for 
                air).
            c0 (float): Speed of sound in the medium (default is 343 m/s for 
                air).
        """
        # self.field_extent = field_extent
        # self.num_points = num_points
        self.rho0 = rho0
        self.c0 = c0
        
        # if np.any(self.num_points < 1):
        #     raise ValueError("num_points must be positive integers.")
        # if self.field_extent.shape != (3, 2):
        #     raise ValueError("field_extent must be of shape (3, 2).")
        # if np.any(self.field_extent[:, 1] <= self.field_extent[:, 0]):
        #     raise ValueError("In field_extent, max values must be greater "\
        #                      "than min values.")
        
        # xs = np.linspace(self.field_extent[0, 0],
        #                  self.field_extent[0, 1],
        #                  self.num_points[0])
        # ys = np.linspace(self.field_extent[1, 0],
        #                  self.field_extent[1, 1],
        #                  self.num_points[1])
        # zs = np.linspace(self.field_extent[2, 0],
        #                  self.field_extent[2, 1],
        #                  self.num_points[2])
        
        # self.X, self.Y, self.Z = np.meshgrid(xs, ys, zs, indexing='ij')
        # pts = np.vstack([self.X.ravel(),
        #                  self.Y.ravel(),
        #                  self.Z.ravel()]).T

        # self.field_points = pts
        
def box_mesh(center: np.ndarray,
             size: np.ndarray,
             divisions: int | None = None,
             ) -> tuple[np.ndarray, np.ndarray]:

    """
    Create a box mesh centered at 'center' with given 'size'.
    Args:
        center (np.ndarray): Center of the box (3,).
        size (np.ndarray): Size of the box along each axis (3,).
        divisions (int, optional): Number of subdivisions along each edge.
            each edge. If an array, should be of shape (3,) for x, y, z axes.
    Returns:
        v (np.ndarray): Array of shape (N, 3) with vertex coordinates.
        elements (np.ndarray): Array of shape (M, 3) with triangular element
            connectivity.
    """

    c = np.asarray(center).reshape(3)
    if np.any(np.array(size) <= 0):
        raise ValueError("Size dimensions must be positive.")
    
    if divisions is not None:
        if isinstance(divisions, int):
            if divisions < 1:
                raise ValueError("Divisions must be at least 1.")
        else:
            raise ValueError("Divisions must be an integer.")

    h = 0.5 * np.array(size)
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    v = c + signs * h

    elements = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                         [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                         [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]])

    if divisions is not None and divisions > 1:
        v, elements = subdivide_triangles(v, elements, divisions)

    return v, elements

def subdivide_triangles(vertices, elements, divisions=1):
    """
    Subdivide triangular elements into smaller triangles.
    
    Args:
        vertices (np.ndarray): Array of shape (N, 3) containing vertex 
            coordinates.
        elements (np.ndarray): Array of shape (M, 3) containing element 
            connectivity.
        divisions (int): Number of divisions per edge. Must be >= 1.
    
    Returns:
        new_vertices (np.ndarray): Array of shape (N_new, 3) with subdivided 
            vertices.
        new_elements (np.ndarray): Array of shape (M_new, 3) with subdivided 
            elements.
    """
    if divisions < 1:
        raise ValueError("Divisions must be at least 1.")
    
    if divisions == 1:
        return vertices.copy(), elements.copy()
    
    vertex_dict = {}
    vertex_list = []
    new_elements_list = []
    
    for elem in elements:
        v0, v1, v2 = vertices[elem[0]], vertices[elem[1]], vertices[elem[2]]
        
        subdiv_indices = np.zeros((divisions + 1, divisions + 1), dtype=int)
        
        for i in range(divisions + 1):
            for j in range(divisions + 1 - i):
                u = i / divisions
                v = j / divisions
                w = 1 - u - v
                
                point = w * v0 + u * v1 + v * v2
                subdiv_indices[i, j] = get_vertex_index(point, 
                                                        vertex_dict, 
                                                        vertex_list)
        
        for i in range(divisions):
            for j in range(divisions - i):
                idx0 = subdiv_indices[i, j]
                idx1 = subdiv_indices[i + 1, j]
                idx2 = subdiv_indices[i, j + 1]
                new_elements_list.append([idx0, idx1, idx2])
                
                if j < divisions - i - 1:
                    idx0 = subdiv_indices[i + 1, j]
                    idx1 = subdiv_indices[i + 1, j + 1]
                    idx2 = subdiv_indices[i, j + 1]
                    new_elements_list.append([idx0, idx1, idx2])
    
    new_vertices = np.array(vertex_list, dtype=float)
    new_elements = np.array(new_elements_list, dtype=int)
    
    return new_vertices, new_elements

def get_vertex_index(coord, vertex_dict, vertex_list):
        """Get or create vertex index for given coordinates."""
        coord_tuple = tuple(coord)
        if coord_tuple not in vertex_dict:
            vertex_dict[coord_tuple] = len(vertex_list)
            vertex_list.append(coord)
        return vertex_dict[coord_tuple]
