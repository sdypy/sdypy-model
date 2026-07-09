"""
Mesh3D class — a generic 3D triangular surface mesh for BEM/FEM preprocessing.

Wraps :class:`pyvista.PolyData` and provides common mesh-preparation utilities:

- Loading from OBJ files (with optional cleaning / triangulation)
- Boundary-loop extraction (hole detection)
- Hole-collar generation (tapered geometry for closing openings)
- Normal computation and repair
- Mesh quality checks (boundary edges, non-manifold edges)
- Element-size analysis (edge lengths, area, λ/6 BEM criterion)
- Nearest-neighbour BC mapping via KDTree
- Geometric masking (cylinder, sphere, box regions)

Typical workflow::

    mesh = Mesh3D.from_obj("mesh.obj")
    loops = mesh.extract_boundary_loops()
    filled = mesh.build_hole_collars(loops, inset_fraction=0.1)
    filled.repair_normals()
    quality = filled.check_quality()
    stats   = filled.element_size_analysis(c0=343_000, f_max=8*533)
    mask    = filled.cylinder_exclusion_mask(radius=52.0, z_min=10.0, z_max=35.0)

All geometric quantities are *in the units of the loaded mesh* (typically mm
for OBJ files).  Use :meth:`to_meters` / :meth:`to_millimeters` to convert.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from collections import defaultdict
from typing import Any, Sequence


# ============================================================================
# Helper: degenerate-triangle removal for collar fill
# ============================================================================
def _remove_degenerate_triangles(
    faces: np.ndarray,
    points: np.ndarray,
    min_area: float = 1e-12,
) -> np.ndarray:
    """Remove triangles whose area is below *min_area*."""
    v0 = points[faces[:, 0]]
    v1 = points[faces[:, 1]]
    v2 = points[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return faces[areas > min_area]


# ============================================================================
# Main class
# ============================================================================

class Mesh3D:
    """
    A 3D triangular surface mesh for BEM / FEM preprocessing.
    Initialite a Mesh3D object by passing point coordinates and triangle 
    connectivity, or use the classmethod constructors .from_poly_data() or
    .from_obj(). to provide a pyvista PolyData or an OBJ file path, respectively.

    Parameters
    ----------
    points : ndarray of shape (N, 3)
        Vertex coordinates.
    faces : ndarray of shape (M, 3)
        Triangle connectivity (0-based indices).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, points: np.ndarray, faces: np.ndarray) -> None:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {points.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must have shape (M, 3), got {faces.shape}")

        self._points = np.asarray(points, dtype=float)
        self._faces = np.asarray(faces, dtype=int)
        self._normals: np.ndarray | None = None  # lazy

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def points(self) -> np.ndarray:
        """Vertex coordinates, shape ``(N, 3)``."""
        return self._points

    @points.setter
    def points(self, value: np.ndarray) -> None:
        self._points = np.asarray(value, dtype=float)
        self._normals = None

    @property
    def faces(self) -> np.ndarray:
        """Triangle connectivity, shape ``(M, 3)``, 0‑based."""
        return self._faces

    @faces.setter
    def faces(self, value: np.ndarray) -> None:
        self._faces = np.asarray(value, dtype=int)
        self._normals = None

    @property
    def n_points(self) -> int:
        """Number of vertices."""
        return self._points.shape[0]

    @property
    def n_cells(self) -> int:
        """Number of triangles."""
        return self._faces.shape[0]

    # ------------------------------------------------------------------
    # Classmethod constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_poly_data(cls, mesh: pv.PolyData) -> Mesh3D:
        """Construct from an existing :class:`pyvista.PolyData` object.

        The mesh must be triangular (all cells are triangles).
        """
        # pyvista stores faces as [n_pts0, v00, v01, v02, n_pts1, v10, v11, v12, ...]
        if mesh.n_cells == 0:
            return cls(np.empty((0, 3), float), np.empty((0, 3), int))

        faces_arr = mesh.faces.reshape(-1, 4)[:, 1:]  # (M, 3)
        return cls(mesh.points.copy(), faces_arr.copy())

    @classmethod
    def from_obj(
        cls,
        path: str,
        clean: bool = True,
        merge_tol: float = 1e-5,
        triangulate: bool = True,
    ) -> Mesh3D:
        """Load a triangular mesh from an OBJ file.

        Parameters
        ----------
        path : str
            Path to the ``.obj`` file.
        clean : bool
            Whether to merge duplicate points.
        merge_tol : float
            Tolerance for point merging (in OBJ file units).
        triangulate : bool
            Whether to convert non-triangular faces to triangles.

        Returns
        -------
        Mesh3D
        """
        raw = pv.read(path)
        if clean:
            raw = raw.clean(
                point_merging=True,
                merge_tol=merge_tol,
                lines_to_points=False,
                polys_to_lines=False,
                strips_to_polys=False,
                inplace=False,
                absolute=False,
            )
        if triangulate:
            raw = raw.triangulate(inplace=False)
        return cls.from_poly_data(raw)

    @classmethod
    def from_obj_with_transform(
        cls,
        path: str,
        transform_path: str,
        scale_to_mm: float = 1.0,
        clean: bool = True,
        merge_tol: float = 1e-5,
        triangulate: bool = True,
    ) -> Mesh3D:
        """Load an OBJ and apply a Structure-from-Motion (SfM) coordinate
        transform loaded from a ``.npz`` file.

        The transform file must contain the keys:

        - ``s`` : scalar scale factor
        - ``R`` : rotation matrix, shape ``(3, 3)``
        - ``obj_orig`` : origin (centroid) of the object in OBJ coordinates
        - ``obj_mod`` : target offset in world coordinates

        The transform applied to each vertex is::

            x' = s * R @ (x - obj_orig) + obj_mod

        and optionally multiplied by *scale_to_mm* to convert units.

        Parameters
        ----------
        path : str
            Path to the ``.obj`` file.
        transform_path : str
            Path to the ``.npz`` file containing the transform arrays.
        scale_to_mm : float, optional
            Additional scaling factor to apply after the transform.
            Set to ``1000`` to convert from metres to millimetres
            (default ``1.0``).
        clean : bool
            Whether to merge duplicate points (default ``True``).
        merge_tol : float
            Tolerance for point merging (default ``1e-5``).
        triangulate : bool
            Whether to convert non-triangular faces (default ``True``).

        Returns
        -------
        Mesh3D
        """
        raw = pv.read(path)
        if clean:
            raw = raw.clean(
                point_merging=True,
                merge_tol=merge_tol,
                lines_to_points=False,
                polys_to_lines=False,
                strips_to_polys=False,
                inplace=False,
                absolute=False,
            )
        if triangulate:
            raw = raw.triangulate(inplace=False)

        _T = np.load(transform_path)
        s = float(_T["s"])
        R = _T["R"]          # (3, 3)
        obj_orig = _T["obj_orig"]   # (3,)
        obj_mod = _T["obj_mod"]     # (3,)

        points = raw.points.copy()
        points = s * (R @ (points - obj_orig).T).T + obj_mod
        points = points * scale_to_mm

        faces_arr = raw.faces.reshape(-1, 4)[:, 1:]  # (M, 3)
        return cls(points, faces_arr.copy())

    # ------------------------------------------------------------------
    # Geometric transform
    # ------------------------------------------------------------------

    def transform(
        self,
        s: float = 1.0,
        R: np.ndarray | None = None,
        obj_orig: np.ndarray | None = None,
        obj_mod: np.ndarray | None = None,
        scale_to_mm: float = 1.0,
    ) -> Mesh3D:
        """Return a new mesh with a rigid similarity transform applied.

        The transform applied to each vertex is::

            x' = scale_to_mm * (s * R @ (x - obj_orig) + obj_mod)

        Parameters
        ----------
        s : float, optional
            Uniform scale factor (default ``1.0``).
        R : ndarray of shape ``(3, 3)`` or None
            Rotation matrix.  If *None*, identity is used.
        obj_orig : ndarray of shape ``(3,)`` or None
            Origin offset subtracted before rotation.  If *None*, ``(0, 0, 0)``.
        obj_mod : ndarray of shape ``(3,)`` or None
            Translation added after rotation.  If *None*, ``(0, 0, 0)``.
        scale_to_mm : float, optional
            Final unit scaling (default ``1.0``).  Set to ``1000`` to
            convert from metres to millimetres.

        Returns
        -------
        Mesh3D
        """
        R_mat = np.eye(3) if R is None else np.asarray(R, dtype=float)
        orig = np.zeros(3) if obj_orig is None else np.asarray(obj_orig, dtype=float)
        mod = np.zeros(3) if obj_mod is None else np.asarray(obj_mod, dtype=float)

        points = self._points.copy()
        points = float(s) * (R_mat @ (points - orig).T).T + mod
        points = points * float(scale_to_mm)

        return Mesh3D(points, self._faces.copy())

    # ------------------------------------------------------------------
    # Conversion to / from pyvista
    # ------------------------------------------------------------------

    def to_poly_data(self) -> pv.PolyData:
        """Return a :class:`pyvista.PolyData` copy of this mesh."""
        pv_faces = np.hstack(
            [np.full((self.n_cells, 1), 3, dtype=int), self._faces]
        ).ravel()
        return pv.PolyData(self._points.copy(), pv_faces)

    # ------------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------------

    def to_meters(self) -> Mesh3D:
        """Return a new mesh with points converted from mm to m."""
        return Mesh3D(self._points * 1e-3, self._faces.copy())

    def to_millimeters(self) -> Mesh3D:
        """Return a new mesh with points converted from m to mm."""
        return Mesh3D(self._points * 1e3, self._faces.copy())

    # ------------------------------------------------------------------
    # Boundary-loop extraction
    # ------------------------------------------------------------------

    def extract_boundary_loops(self) -> list[list[int]]:
        """Return a list of ordered vertex-index loops for each boundary hole.

        Each loop is a list of vertex indices into :attr:`points`.
        Assumes boundary edges form simple closed loops (no branches).

        Returns
        -------
        list[list[int]]
            One list per hole, each containing the vertex indices in order
            around the hole.
        """
        poly = self.to_poly_data()
        be = poly.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            feature_edges=False,
            manifold_edges=False,
        )

        if be.n_cells == 0:
            return []

        # Map boundary-edge local vertex indices back to original mesh indices
        tree = cKDTree(self._points)
        _, local_to_orig = tree.query(be.points)

        # Build adjacency (original vertex indices)
        lines = be.lines.reshape(-1, 3)[:, 1:]  # (M, 2)
        adj: dict[int, list[int]] = defaultdict(list)
        for lo, hi in lines:
            v0, v1 = int(local_to_orig[lo]), int(local_to_orig[hi])
            adj[v0].append(v1)
            adj[v1].append(v0)

        visited: set[int] = set()
        loops: list[list[int]] = []

        for start in sorted(adj):
            if start in visited:
                continue
            loop = [start]
            visited.add(start)
            prev = -1
            cur = start

            while True:
                nxt = next((v for v in adj[cur] if v != prev), None)
                if nxt is None or nxt == start or nxt in visited:
                    break
                loop.append(nxt)
                visited.add(nxt)
                prev = cur
                cur = nxt

            if len(loop) >= 3:
                loops.append(loop)

        return loops

    # ------------------------------------------------------------------
    # Hole collars
    # ------------------------------------------------------------------

    def build_hole_collars(
        self,
        loops: list[list[int]],
        inset_fraction: float = 0.1,
        min_area: float = 1e-12,
    ) -> Mesh3D:
        """Return a new mesh with collar geometry added at each hole.

        For each boundary loop, a ring of inset collar vertices is created
        between the original boundary and the loop centroid.  The interior
        is filled with triangles fanned from the centroid.

        Parameters
        ----------
        loops : list[list[int]]
            Boundary loops as returned by :meth:`extract_boundary_loops`.
        inset_fraction : float, optional
            Fraction (0–0.5) of the distance from boundary to centroid at
            which the collar ring is placed (default ``0.1``).
        min_area : float, optional
            Minimum triangle area for retaining a face (default ``1e-12``).

        Returns
        -------
        Mesh3D
            New mesh that includes original vertices *plus* extra collar
            and centroid vertices.  The returned mesh is **not** closed
            (call :meth:`repair_normals` and check with :meth:`check_quality`).

        Notes
        -----
        The extra vertices (collar ring + centroid) are appended after the
        original points.  Use ``mesh.points[zero_idx]`` or the ``zero_indices``
        attribute (if stored) to identify vertices that should receive
        zero velocity BC.
        """
        n_orig = self.n_points
        all_pts: list[np.ndarray] = []
        all_tri: list[list[int]] = []
        zero_idx: list[int] = []
        offset = n_orig

        for loop in loops:
            loop_pts = self._points[loop]  # (L, 3)
            centroid = loop_pts.mean(axis=0)  # (3,)
            L = len(loop)

            # Collar vertices: inset fraction of the way toward centroid
            collar_pts = loop_pts + inset_fraction * (centroid[None, :] - loop_pts)
            new_pts = np.vstack([collar_pts, centroid[None, :]])  # (L+1, 3)

            c_start = offset  # global index of first collar vertex
            cen_idx = offset + L  # global index of centroid

            # Collar quads → 2 triangles each
            for i in range(L):
                v0 = loop[i]
                v1 = loop[(i + 1) % L]
                c0 = c_start + i
                c1 = c_start + (i + 1) % L
                all_tri.append([v0, v1, c1])
                all_tri.append([v0, c1, c0])

            # Inner fill: fan triangles from centroid to collar ring
            for i in range(L):
                c0 = c_start + i
                c1 = c_start + (i + 1) % L
                all_tri.append([cen_idx, c0, c1])

            all_pts.append(new_pts)
            zero_idx.extend(range(c_start, cen_idx + 1))
            offset += L + 1

        extra_pts = (
            np.vstack(all_pts) if all_pts else np.empty((0, 3), float)
        )
        extra_faces = (
            np.array(all_tri, dtype=int) if all_tri else np.empty((0, 3), int)
        )

        merged_pts = (
            np.vstack([self._points, extra_pts])
            if len(extra_pts)
            else self._points.copy()
        )
        merged_faces = (
            np.vstack([self._faces, extra_faces])
            if len(extra_faces)
            else self._faces.copy()
        )

        merged_faces = _remove_degenerate_triangles(
            merged_faces, merged_pts, min_area
        )

        result = Mesh3D(merged_pts, merged_faces)
        # Attach metadata about which indices are "zero BC" vertices
        result._zero_collar_idx = zero_idx
        return result

    # ------------------------------------------------------------------
    # Normals
    # ------------------------------------------------------------------

    def repair_normals(
        self,
        point_normals: bool = True,
        cell_normals: bool = True,
        consistent_normals: bool = True,
        auto_orient_normals: bool = True,
    ) -> None:
        """Compute consistent, outward-facing normals (in‑place).

        Parameters
        ----------
        point_normals : bool
            Compute per-vertex normals.
        cell_normals : bool
            Compute per-face normals.
        consistent_normals : bool
            Enforce consistent winding order.
        auto_orient_normals : bool
            Orient normals outward (requires a closed or nearly-closed mesh).

        Notes
        -----
        After calling this method, the internal point normals are cached.
        Access them via :meth:`point_normals`.
        """
        poly = self.to_poly_data()
        poly.compute_normals(
            point_normals=point_normals,
            cell_normals=cell_normals,
            consistent_normals=consistent_normals,
            auto_orient_normals=auto_orient_normals,
            inplace=True,
        )
        # Store computed point normals
        if point_normals and "Normals" in poly.point_data:
            self._normals = poly.point_data["Normals"].copy()
        # Update face connectivity if pyvista rewound it
        self._faces = poly.faces.reshape(-1, 4)[:, 1:].copy()

    @property
    def point_normals(self) -> np.ndarray | None:
        """Per-vertex normals, shape ``(N, 3)``, or *None* if not computed."""
        return self._normals

    # ------------------------------------------------------------------
    # Quality checks
    # ------------------------------------------------------------------

    def check_quality(self) -> dict[str, Any]:
        """Run common mesh-quality checks.

        Returns
        -------
        dict with keys:
            - ``boundary_edges`` : number of boundary edges (0 = closed)
            - ``non_manifold_edges`` : number of non-manifold edges (0 = manifold)
            - ``min_triangle_area`` : smallest triangle area
            - ``max_aspect_ratio`` : maximum triangle aspect ratio
            - ``edges_stats`` : dict ``{mean, std, min, max}`` of edge lengths
        """
        poly = self.to_poly_data()

        be = poly.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            feature_edges=False,
            manifold_edges=False,
        )
        nm = poly.extract_feature_edges(
            boundary_edges=False,
            non_manifold_edges=True,
            feature_edges=False,
            manifold_edges=False,
        )

        # Triangle geometry
        v0 = self._points[self._faces[:, 0]]
        v1 = self._points[self._faces[:, 1]]
        v2 = self._points[self._faces[:, 2]]

        e01 = np.linalg.norm(v1 - v0, axis=1)
        e12 = np.linalg.norm(v2 - v1, axis=1)
        e20 = np.linalg.norm(v0 - v2, axis=1)
        all_edges = np.concatenate([e01, e12, e20])

        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)

        # Aspect ratio = longest edge / (2 * inradius)
        # For a triangle: inradius = 2*area / perimeter
        perimeters = e01 + e12 + e20
        inradii = 2.0 * areas / (perimeters + 1e-300)
        max_edges = np.maximum.reduce([e01, e12, e20])
        aspect_ratios = max_edges / (2.0 * inradii + 1e-300)

        return {
            "boundary_edges": be.n_cells,
            "non_manifold_edges": nm.n_cells,
            "min_triangle_area": float(areas.min()),
            "max_aspect_ratio": float(aspect_ratios.max()),
            "edges_stats": {
                "mean": float(all_edges.mean()),
                "std": float(all_edges.std()),
                "min": float(all_edges.min()),
                "max": float(all_edges.max()),
            },
        }

    # ------------------------------------------------------------------
    # Element-size analysis  (BEM λ/6 criterion)
    # ------------------------------------------------------------------

    def element_size_analysis(
        self,
        c0: float = 343_000.0,
        f_max: float | None = None,
    ) -> dict[str, Any]:
        """Analyse element sizes and check BEM accuracy (λ/6 rule).

        Parameters
        ----------
        c0 : float
            Speed of sound in the mesh units **per second**.
            Default ``343_000.0`` (mm/s when mesh is in mm).
        f_max : float or None
            Highest frequency to check (same unit as c0, i.e. Hz).
            If *None* only raw statistics are returned.

        Returns
        -------
        dict with keys:
            - ``n_triangles`` : int
            - ``edge_length`` : dict ``{mean, std, min, max}``
            - ``equiv_diameter`` : dict ``{mean, std, min, max}``
            - ``area`` : dict ``{mean, std, min, max}``
            - ``lambda_min`` : float (if *f_max* given)
            - ``lambda_over_6`` : float (if *f_max* given)
            - ``mean_over_limit`` : float (if *f_max* given)
            - ``passes_lambda_over_6`` : bool (if *f_max* given)
        """
        v0 = self._points[self._faces[:, 0]]
        v1 = self._points[self._faces[:, 1]]
        v2 = self._points[self._faces[:, 2]]

        # Edge lengths
        e01 = np.linalg.norm(v1 - v0, axis=1)
        e12 = np.linalg.norm(v2 - v1, axis=1)
        e20 = np.linalg.norm(v0 - v2, axis=1)
        all_edges = np.concatenate([e01, e12, e20])

        # Areas and equivalent diameter
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        eq_diam = 2.0 * np.sqrt(areas / np.pi)

        result: dict[str, Any] = {
            "n_triangles": self.n_cells,
            "edge_length": {
                "mean": float(all_edges.mean()),
                "std": float(all_edges.std()),
                "min": float(all_edges.min()),
                "max": float(all_edges.max()),
            },
            "equiv_diameter": {
                "mean": float(eq_diam.mean()),
                "std": float(eq_diam.std()),
                "min": float(eq_diam.min()),
                "max": float(eq_diam.max()),
            },
            "area": {
                "mean": float(areas.mean()),
                "std": float(areas.std()),
                "min": float(areas.min()),
                "max": float(areas.max()),
            },
        }

        if f_max is not None and f_max > 0:
            lam_min = c0 / f_max
            h_limit = lam_min / 6.0
            result["lambda_min"] = lam_min
            result["lambda_over_6"] = h_limit
            result["mean_over_limit"] = all_edges.mean() / h_limit
            result["passes_lambda_over_6"] = all_edges.mean() < h_limit

        return result

    # ------------------------------------------------------------------
    # BC mapping  (KDTree nearest-neighbour)
    # ------------------------------------------------------------------

    def map_to_reference(
        self, ref_points: np.ndarray
    ) -> np.ndarray:
        """Return for each vertex the index of its nearest neighbour in *ref_points*.

        Parameters
        ----------
        ref_points : ndarray of shape ``(K, 3)``
            Reference point cloud (e.g. the original measurement mesh).

        Returns
        -------
        ndarray of shape ``(N,)``
            Indices into *ref_points* for each vertex of current mesh of this instance.
        """
        tree = cKDTree(ref_points)
        _, idx = tree.query(self._points)
        return idx

    # ------------------------------------------------------------------
    # Geometric region masking
    # ------------------------------------------------------------------

    def cylinder_exclusion_mask(
        self,
        radius: float,
        z_min: float,
        z_max: float,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Return a boolean mask for points inside a cylinder.

        The cylinder axis is aligned with the *z* axis and centred at
        ``(cx, cy)`` in the XY plane.  Points are considered inside if
        their radial distance ``r`` from the axis satisfies
        ``r < radius`` **and** ``z_min < z < z_max``.

        Parameters
        ----------
        radius : float
            Cylinder radius (in mesh units).
        z_min : float
            Lower *z* bound (in mesh units).
        z_max : float
            Upper *z* bound (in mesh units).
        center : tuple of float, optional
            ``(cx, cy, cz)`` of the cylinder base centre (default ``(0, 0, 0)``).

        Returns
        -------
        ndarray of bool, shape ``(N,)``
            *True* for vertices inside the exclusion zone.
        """
        cx, cy, cz = center
        r_xy = np.sqrt(
            (self._points[:, 0] - cx) ** 2 + (self._points[:, 1] - cy) ** 2
        )
        mask = (
            (r_xy < radius)
            & (self._points[:, 2] > z_min + cz)
            & (self._points[:, 2] < z_max + cz)
        )
        return mask

    def sphere_exclusion_mask(
        self,
        radius: float,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Return a boolean mask for points inside a sphere.

        Parameters
        ----------
        radius : float
            Sphere radius (in mesh units).
        center : tuple of float, optional
            ``(cx, cy, cz)`` of the sphere centre (default ``(0, 0, 0)``).

        Returns
        -------
        ndarray of bool, shape ``(N,)``
        """
        cx, cy, cz = center
        r = np.linalg.norm(self._points - np.array([cx, cy, cz]), axis=1)
        return r < radius

    def box_exclusion_mask(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
    ) -> np.ndarray:
        """Return a boolean mask for points inside an axis‑aligned box.

        Parameters are bounds in mesh units.  Returns *True* for vertices
        where **all** coordinate bounds are satisfied.

        Returns
        -------
        ndarray of bool, shape ``(N,)``
        """
        pts = self._points
        mask = (
            (pts[:, 0] >= x_min)
            & (pts[:, 0] <= x_max)
            & (pts[:, 1] >= y_min)
            & (pts[:, 1] <= y_max)
            & (pts[:, 2] >= z_min)
            & (pts[:, 2] <= z_max)
        )
        return mask

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Mesh3D({self.n_points} points, {self.n_cells} triangles)"
        )