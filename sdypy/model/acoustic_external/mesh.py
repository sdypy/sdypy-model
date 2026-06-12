import numpy as np
from .geometry import Body, Field

class Mesh:
    def __init__(self,
                 source_object: Body,
                 peripheral_objects: list[Body] | None,
                 field: Field):
        
        """
        Initialize the mesh by merging multiple bodies and precomputing
        geometric properties.
        Args:
            source_object (Body): The main body with defined boundary
                conditions and frequency.
            peripheral_objects (list[Body] | None): Additional bodies to be
                merged into the mesh.
            field (Field): Field object defining medium properties.
        Raises:
            ValueError: If source_object does not have a defined frequency or
                boundary conditions.

        Returns:
            None
        """
        if not source_object.frequency:
            raise ValueError("Source object must have a defined frequency of"\
            " oscillation.")
        if source_object.Neumann_BC is None and \
            source_object.Dirichlet_BC is None:
            raise ValueError("Source object must have at least one boundary "
                             "condition defined (Neumann or Dirichlet).")
        
        self.source_object = source_object
        self.peripheral_objects = peripheral_objects if peripheral_objects else []
        self.field = field

        self.frequency = self.source_object.frequency
        self.c0 = self.field.c0
        self.rho0 = self.field.rho0
        self.k = 2.0 * np.pi * self.frequency / self.c0
        
        self.merge()

    def merge(self):
        """
        Merge multiple mesh bodies into a single mesh.
        """
        #create a list off all objects to merge
        components = [self.source_object]
        if self.peripheral_objects is not None:
            for obj in self.peripheral_objects:
                components.append(obj)

        mesh_nodes, mesh_elements, q_all, p_all = self._stack_components(
            components
            )
        
        self.num_nodes = mesh_nodes.shape[0]
        self.num_elements = mesh_elements.shape[0]
        
        if (q_all is not None) and (p_all is not None):
            raise ValueError("Cannot mix global Neumann and Dirichlet in this "\
            "pipeline.") # why?
        if (q_all is None) and (p_all is None):
            q_all = np.zeros(mesh_nodes.shape[0], float)

        self.mesh_nodes = mesh_nodes
        self.mesh_elements = mesh_elements
        self.Neumann_BC = q_all
        self.Dirichlet_BC = p_all

        (v0, e1, e2, a2, n_hat, centroids, areas,
         node_in_el, char_length) = self._precompute_global(
             self.mesh_nodes, 
             self.mesh_elements
             )
        
        self.v0 = v0
        self.e1 = e1
        self.e2 = e2
        self.a2 = a2
        self.n_hat = n_hat
        self.centroids = centroids
        self.areas = areas
        self.node_in_el = node_in_el
        self.char_length = char_length

        self.node_normals_angle()

        #project velocity vector field into scalar normal velocity if not already done by the user.
        if q_all is not None and q_all.ndim == 2 and q_all.shape[1] == 3:
            q_all = np.einsum("ij,ij->i", q_all, self.node_n_hat)

        jump_coefficients = self._compute_jump_coefficients(self.mesh_nodes, 
                                                            self.mesh_elements, 
                                                            self.node_in_el, 
                                                            self.node_n_hat)

        self.jump_coefficients = jump_coefficients

        # self.field_points = self.field.field_points if self.field is not None \
        #     else None
        self.field_c0 = (self.field.c0 if self.field is not None else None)
        self.field_rho0 = (self.field.rho0 if self.field is not None else None)

    def _precompute_global(self,
                              mesh_nodes: np.ndarray,
                              mesh_elements: np.ndarray
                              ) -> tuple[np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray,
                                         np.ndarray, list[np.ndarray],
                                         np.ndarray, np.ndarray]:
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
        v0 = mesh_nodes[mesh_elements[:, 0], :]
        v1 = mesh_nodes[mesh_elements[:, 1], :]
        v2 = mesh_nodes[mesh_elements[:, 2], :]

        e1 = v1 - v0
        e2 = v2 - v0
        cross = np.cross(e1, e2)
        a2 = np.linalg.norm(cross, axis=1)
        n_hat = cross / (a2[:, np.newaxis] + 1e-300)
        centroids = (v0 + v1 + v2) / 3.0
        areas = 0.5 * a2

        node_in_el = [np.where((mesh_elements == i).any(axis=1))[0]
                      for i in range(mesh_nodes.shape[0])]

        char_length = np.maximum.reduce([
            np.linalg.norm(e1, axis=1),
            np.linalg.norm(e2, axis=1),
            np.linalg.norm(e1 - e2, axis=1),
        ])

        return (v0, e1, e2, a2, 
                n_hat, centroids, 
                areas, node_in_el, 
                char_length)

    def _compute_jump_coefficients(self,
                                   mesh_nodes: np.ndarray,
                                   mesh_elements: np.ndarray,
                                   node_in_el: list[np.ndarray],
                                   node_n_hat: np.ndarray) -> np.ndarray:
        """
        Compute jump coefficients C(x) for the double-layer operator. Uses the
        Van Oosterom and Strackee method to compute the solid angle.

        For each node, compute the interior solid angle Ω(x) subtended by
        the adjacent triangular elements, then set

            C(x) = Ω(x) / (4π).

        For smooth closed surfaces, this tends to 0.5 everywhere.
        For open or non-smooth surfaces, values reflect the local geometry.

        Returns:
            np.ndarray: Jump coefficients, shape (num_nodes,).
        """

        coeff = np.zeros(mesh_nodes.shape[0], float)
        for i in range(mesh_nodes.shape[0]):
            elems = node_in_el[i]
            x0 = mesh_nodes[i]
            n0 = node_n_hat[i]
            obs = x0 + 1e-6 * n0
            solid_angle = 0.0
            for e in elems:
                tri = mesh_elements[e]
                v0 = mesh_nodes[tri[0]]
                v1 = mesh_nodes[tri[1]]
                v2 = mesh_nodes[tri[2]]

                r0 = v0 - obs
                r1 = v1 - obs
                r2 = v2 - obs

                a = np.linalg.norm(r0)
                b = np.linalg.norm(r1)
                c = np.linalg.norm(r2)
                triple = np.dot(r0, np.cross(r1, r2))
                denom = (a*b*c + 
                         np.dot(r0, r1)*c + 
                         np.dot(r1, r2)*a + 
                         np.dot(r2, r0)*b)
                angle = 2.0 * np.arctan2(triple, denom)
                solid_angle += angle
            coeff[i] = abs(solid_angle) / (4.0 * np.pi)
        return coeff
    
    def _stack_components(self,
                          objs: list[Body]) -> tuple[np.ndarray, np.ndarray,
                                                     np.ndarray | None,
                                                     np.ndarray | None]:
        """Return stacked (nodes, elements, q, p)."""
        node_blocks, elem_blocks = [], []
        q_blocks, p_blocks = [], []

        node_offset = 0
        for obj in objs:
            nodes = obj.mesh_nodes
            elems = obj.mesh_elements + node_offset
            node_blocks.append(nodes)
            elem_blocks.append(elems)

            if obj.Neumann_BC is not None:
                q = obj.Neumann_BC
                if q.ndim == 2 and q.shape[1] == 3:
                    q = np.einsum("ij,ij->i", q, obj.node_n_hat)
                q_blocks.append(q)
            if obj.Dirichlet_BC is not None:
                p_blocks.append(obj.Dirichlet_BC)

            node_offset += nodes.shape[0]

        mesh_nodes = np.vstack(node_blocks) if node_blocks else \
            np.empty((0, 3), float)
        mesh_elements = (np.vstack(elem_blocks) if elem_blocks else \
                         np.empty((0, 3), int))

        Neumann_BC = np.concatenate(q_blocks) if q_blocks else None
        Dirichlet_BC = np.concatenate(p_blocks) if p_blocks else None

        return mesh_nodes, mesh_elements, Neumann_BC, Dirichlet_BC
    
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