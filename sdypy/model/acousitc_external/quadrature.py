import warnings
import numpy as np

# ============================================================================
# PERFORMANCE OPTIMIZATION: Cache Gauss-Legendre quadratures
# ============================================================================

_GAUSS_LEGENDRE_CACHE = {}

# Cache for Telles transformation coefficients
_TELLES_COEFF_CACHE = {}

def _precompute_common_quadratures():
    """
    Pre-compute common Gauss-Legendre quadratures at module import.
    
    This dramatically improves performance by avoiding repeated calls to
    np.polynomial.legendre.leggauss, which is very expensive.
    """
    common_orders = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]
    for n in common_orders:
        points, weights = np.polynomial.legendre.leggauss(n)
        points = 0.5 * (points + 1.0)
        weights = 0.5 * weights
        _GAUSS_LEGENDRE_CACHE[n] = (
            np.asarray(points, dtype=np.float64, order='C'),
            np.asarray(weights, dtype=np.float64, order='C')
        )

# Pre-compute on module import
_precompute_common_quadratures()

# ============================================================================
# Mapping functions
# ============================================================================

def map_to_physical_triangle(xi_eta: np.ndarray,
                             v0: np.ndarray,
                             e1: np.ndarray,
                             e2: np.ndarray
                             ) -> tuple[np.ndarray, float]:
    
    """
    Map (xi,eta) in the standard reference triangle (0<=xi,eta, xi+eta<=1)
    to physical coordinates y = v0 + xi*e1 + eta*e2.

    Args:
        xi_eta (np.ndarray): Array of shape (N, 2) representing the
            quadrature points in barycentric coordinates.
        v0 (np.ndarray): Array of shape (3,) representing the first
            vertex of the triangle.
        e1 (np.ndarray): Array of shape (3,) representing the edge
            vector from v0 to v1.
        e2 (np.ndarray): Array of shape (3,) representing the edge
            vector from v0 to v2.
    
    Returns:
        y_phys (np.ndarray): Array of shape (N, 3) representing the
            quadrature points in physical coordinates.
        a2 (float): Jacobian scale (||e1×e2||), i.e. twice the *physical* 
            triangle area.
    """
    y_phys = v0[None, :] + \
            xi_eta[:, [0]] * e1[None, :] + \
            xi_eta[:, [1]] * e2[None, :]
    a2 = np.linalg.norm(np.cross(e1, e2))
    return y_phys, a2

def map_to_physical_triangle_batch(xi_eta: np.ndarray,
                                   v0: np.ndarray,
                                   e1: np.ndarray,
                                   e2: np.ndarray) -> tuple[np.ndarray, 
                                                            np.ndarray]:
    """
    Vectorized mapping for K triangles at once.

    Args:
        xi_eta (np.ndarray): Array of shape (Q, 2) representing the
            quadrature points in barycentric coordinates.
        v0 (np.ndarray): Array of shape (K, 3) representing the first
            vertex of each triangle.
        e1 (np.ndarray): Array of shape (K, 3) representing the edge
            vector from v0 to v1 for each triangle.
        e2 (np.ndarray): Array of shape (K, 3) representing the edge
            vector from v0 to v2 for each triangle.

    Returns:
        y (np.ndarray): Array of shape (K, Q, 3) representing the
            quadrature points in physical coordinates.
        a2 (np.ndarray): Array of shape (K,) representing the Jacobian
            scale (||e1×e2||), i.e. twice the *physical* triangle area.
    """
    xi = xi_eta[:, 0][None, :]
    eta = xi_eta[:, 1][None, :]
    y = v0[:, None, :] + \
        xi[..., None]*e1[:, None, :] + \
        eta[..., None]*e2[:, None, :]
    a2 = np.linalg.norm(np.cross(e1, e2), axis=1)
    return y, a2

# ============================================================================
# Shape functions
# ============================================================================

def shape_functions_P1(xi_eta: np.ndarray) -> np.ndarray:
    """
    Compute the P1 (linear) shape functions at given barycentric coordinates.

    Args:
        xi_eta (np.ndarray): Array of shape (N, 2) representing the 
            barycentric coordinates (xi, eta) in the reference triangle.

    Returns:
        N (np.ndarray): Array of shape (N, 3) representing the values of 
            the three P1 shape functions at the given points.
    """
    N = np.empty((xi_eta.shape[0], 3), dtype=xi_eta.dtype)
    N[:,1] = xi_eta[:,0]
    N[:,2] = xi_eta[:,1]
    N[:,0] = 1.0 - N[:,1] - N[:,2]
    return N

def shape_function_gradients_P1() -> np.ndarray:
    """
    Compute the gradients of the P1 (linear) shape functions in the 
    reference triangle.

    Returns:
        dN_dxi (np.ndarray): Array of shape (3, 2) representing the 
            gradients of the three P1 shape functions with respect to 
            (xi, eta).
    """
    dN_dxi = np.array([[-1.0, -1.0],
                       [ 1.0,  0.0],
                       [ 0.0,  1.0]])
    return dN_dxi

# ============================================================================
# Standard quadrature rules
# ============================================================================

def standard_triangle_quad(order: int = 1,
                           ) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Generate quadrature points and weights for a standard triangle.
    The standard triangle has vertices at (0,0), (1,0), and (0,1).

    Args:
        order (int, optional): Order of the quadrature. Supported orders are 
            1, 3, and 7. Default is 1.


    Returns:
        quad_points (np.ndarray): Array of shape (N, 2) representing the 
            quadrature points in barycentric coordinates.
        quad_weights (np.ndarray): Array of shape (N,) representing the 
            quadrature weights.
    """
    
    if order == 1:
        quad_points = np.array([[1/3, 1/3]])
        quad_weights = np.array([0.5])

    elif order == 3:
        quad_points = np.array([[1/6, 1/6],
                                [2/3, 1/6],
                                [1/6, 2/3]])
        quad_weights = np.array([1/6, 1/6, 1/6])

    elif order == 7:
        quad_points = np.array([[1/3, 1/3],
                                [0.0597158717, 0.4701420641],
                                [0.4701420641, 0.0597158717],
                                [0.4701420641, 0.4701420641],
                                [0.7974269853, 0.1012865073],
                                [0.1012865073, 0.7974269853],
                                [0.1012865073, 0.1012865073]])
        quad_weights = np.array([0.225,
                                 0.1323941527,
                                 0.1323941527,
                                 0.1323941527,
                                 0.1259391805,
                                 0.1259391805,
                                 0.1259391805]) * 0.5
        
    else:
        raise ValueError("Unsupported quadrature order. Supported orders are "
                         "1, 3, and 7.")
    
    pts = np.asarray(quad_points, dtype=np.float64, order='C')
    w = np.asarray(quad_weights, dtype=np.float64, order='C')

    return pts, w

# ============================================================================
# Refinement quadrature
# ============================================================================

def refined_triangle_quad(xi_eta: np.ndarray,
                          weights: np.ndarray,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine a single triangle into four smaller triangles and adjust the
    quadrature points and weights accordingly.

    Args:
        xi_eta (np.ndarray): Array of shape (N, 2) representing the
            quadrature points in barycentric coordinates.
        weights (np.ndarray): Array of shape (N,) representing the
            quadrature weights.

    Returns:
        xi_eta_ref (np.ndarray): Array of shape (4N, 2) representing the
            refined quadrature points in barycentric coordinates.
        w_ref (np.ndarray): Array of shape (4N,) representing the
            refined quadrature weights.
    """
    X = xi_eta
    xi  = X[:, 0]
    eta = X[:, 1]
    Wq = weights * 0.25

    X1 = np.column_stack((      0.5 * xi,            0.5 * eta))
    X2 = np.column_stack((0.5 + 0.5 * xi,            0.5 * eta))
    X3 = np.column_stack((      0.5 * xi,      0.5 + 0.5 * eta))
    X4 = np.column_stack((0.5 - 0.5 * xi, 0.5 * xi + 0.5 * eta))

    xi_eta_ref = np.vstack((X1, X2, X3, X4))
    w_ref      = np.concatenate((Wq, Wq, Wq, Wq))

    return xi_eta_ref, w_ref

def subdivide_triangle_quad(xi_eta: np.ndarray,
                            weights: np.ndarray,
                            levels: int = 1,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """
    Recursively refine a triangle into smaller triangles and adjust the
    quadrature points and weights accordingly.

    Args:
        xi_eta (np.ndarray): Array of shape (N, 2) representing the
            quadrature points in barycentric coordinates.
        weights (np.ndarray): Array of shape (N,) representing the
            quadrature weights.
        levels (int, optional): Number of refinement levels. Each level
            subdivides each triangle into four smaller triangles. Default is 1.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Refined quadrature points and weights.
    """
    pts = np.asarray(xi_eta, dtype=float)
    w   = np.asarray(weights, dtype=float)

    if levels < 1:
        warnings.warn("Number of refinement levels must be at least 1. "
                      "Returning original points and weights.")
        return pts, w

    for _ in range(levels):
        pts, w = refined_triangle_quad(pts, w)

    pts = np.asarray(pts, dtype=np.float64, order='C')
    w   = np.asarray(w, dtype=np.float64, order='C')

    return pts, w

# ============================================================================
# Singular integration: Duffy transformation
# ============================================================================

def duffy_rule(n_leg: int = 8,
               sing_vert_int: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate quadrature points and weights for a triangle using the Duffy 
    transformation from a square.

    Args:
        n_leg (int, optional): Number of Gauss-Legendre points along one edge 
            of the square. The total number of quadrature points will be 
            n_leg*(n_leg+1)/2. Default is 8.
        sing_vert_int (int, optional): Vertex index (0, 1, or 2) where the
            singularity is located. Default is 0.

    Returns:
        quad_points (np.ndarray): Array of shape (N, 2) representing the 
            quadrature points in barycentric coordinates.
        quad_weights (np.ndarray): Array of shape (N,) representing the 
            quadrature weights.
    """
    
    u, wu = gauss_legendre_1d(n_leg)
    v, wv = gauss_legendre_1d(n_leg)

    # vectorized without meshgrid
    XI  = np.multiply.outer(u, (1.0 - v))
    ETA = np.multiply.outer(u, v)
    w   = (np.multiply.outer(wu, wv) * u[:, None]).ravel()

    pts = np.stack([XI.ravel(), ETA.ravel()], axis=1)
    if sing_vert_int != 0:
        pts = permute_to_vertex(pts, sing_vert_int)

    pts = np.asarray(pts, dtype=np.float64, order='C')
    w   = np.asarray(w, dtype=np.float64, order='C')

    return pts, w

# ============================================================================
# Near-singular integration: Telles transformation
# ============================================================================

def telles_rule(u_star: float,
                v_star: float | None = None,
                sing_vert_int: int = 0,
                n_leg: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate quadrature points and weights for a triangle using the Telles 
    transformation from a square.

    'The transformation is a cubic polynomial that maps the Gauss-Legendre 
    points in [0, 1] onto itself, and has zero Jacobian at the singularity
    location u_star. The parameter s0 is a reference GL point that controls
    the clustering of points around the singularity.'

    Args:
        u_star (float): u-coordinate of the singularity in the reference
            triangle (0 <= u_star <= 1).
        v_star (float | None): v-coordinate of the singularity in the reference
            triangle (0 <= v_star <= 1, u_star + v_star <= 1).
        sing_vert_int (int, optional): Vertex index (0, 1, or 2) where the
            singularity is located. Default is 0.
        n_leg (int, optional): Number of Gauss-Legendre points along one edge
            of the square. Default is 8.

    Returns:
        quad_points (np.ndarray): Array of shape (N, 2) representing the
            quadrature points in barycentric coordinates.
        quad_weights (np.ndarray): Array of shape (N,) representing the
            quadrature weights.
    """

    u_nodes, wu = gauss_legendre_1d(n_leg)
    v_nodes, wv = gauss_legendre_1d(n_leg)

    s0_u = np.clip(u_star, 0.1, 0.9)        

    u_map, du = telles_cubic_1d(u_nodes, u_star, s0_u)
    if v_star is not None:
        s0_v = np.clip(v_star, 0.1, 0.9)
        v_map, dv = telles_cubic_1d(v_nodes, v_star, s0_v)
    else:
        v_map, dv = v_nodes, np.ones_like(v_nodes)

    XI  = np.multiply.outer(u_map, (1.0 - v_map))
    ETA = np.multiply.outer(u_map, v_map)
    w   = (np.multiply.outer(wu, wv) * \
           np.multiply.outer(du, dv) * u_map[:, None]).ravel()

    pts = np.stack([XI.ravel(), ETA.ravel()], axis=1)
    if sing_vert_int != 0:
        pts = permute_to_vertex(pts, sing_vert_int)

    pts = np.asarray(pts, dtype=np.float64, order='C')
    w   = np.asarray(w, dtype=np.float64, order='C')
    
    return pts, w

# ============================================================================
# Helper functions
# ============================================================================

def gauss_legendre_1d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Gauss-Legendre quadrature points and weights on [0, 1].
    
    Results are cached for performance. Common orders (1-30) are pre-computed
    at module import time.

    Args:
        n (int): Number of quadrature points.

    Returns:
        points (np.ndarray): Array of shape (n,) representing the quadrature 
            points on [0, 1].
        weights (np.ndarray): Array of shape (n,) representing the quadrature 
            weights (sum(weights) = 1).
    """
    
    if n < 1:
        raise ValueError("Number of quadrature points must be at least 1.")
    
    # Check cache first (includes pre-computed common orders)
    if n in _GAUSS_LEGENDRE_CACHE:
        points, weights = _GAUSS_LEGENDRE_CACHE[n]
        # Return copies to prevent accidental mutation
        return points.copy(), weights.copy()
    
    # Fallback: compute on-demand for uncommon orders
    points, weights = np.polynomial.legendre.leggauss(n)
    points = 0.5 * (points + 1.0)
    weights = 0.5 * weights
    
    # Cache for future use
    _GAUSS_LEGENDRE_CACHE[n] = (
        np.asarray(points, dtype=np.float64, order='C'),
        np.asarray(weights, dtype=np.float64, order='C')
    )
    
    return points.copy(), weights.copy()

def permute_to_vertex(xi_eta: np.ndarray,
                      sing_vert_int: int) -> np.ndarray:
    """
    Permute the barycentric coordinates (xi, eta) to place the singularity
    at the specified vertex.

    Args:
        xi_eta (np.ndarray): Array of shape (N, 2) representing the 
            barycentric coordinates (xi, eta).
        sing_vert_int (int): Vertex index (0, 1, or 2) where the singularity 
            is located.

    Returns:
        xi_eta_perm (np.ndarray): Array of shape (N, 2) representing the 
            permuted barycentric coordinates.
    """
    
    if sing_vert_int == 0:
        return xi_eta
    elif sing_vert_int == 1:
        xi = xi_eta[:, 0]
        eta = xi_eta[:, 1]
        xi_perm = eta
        eta_perm = 1.0 - xi - eta
        return np.column_stack([xi_perm, eta_perm])
    elif sing_vert_int == 2:
        xi = xi_eta[:, 0]
        eta = xi_eta[:, 1]
        xi_perm = 1.0 - xi - eta
        eta_perm = xi
        return np.column_stack([xi_perm, eta_perm])
    else:
        raise ValueError("sing_vert_int must be 0, 1, or 2.")
    
# Cache for Telles transformation coefficients
_TELLES_COEFF_CACHE = {}

def _get_telles_coefficients(t0: float, s0: float = 0.5) -> tuple[float, float, float] | None:
    """
    Compute or retrieve cached Telles transformation coefficients.
    
    The transformation is: u_telles = a*u³ + b*u² + c*u
    
    Args:
        t0: Singularity location in [0,1]
        s0: Reference point (default 0.5)
        
    Returns:
        (a, b, c) coefficients, or None if singular
    """
    # Round to avoid floating point key issues
    key = (round(t0, 6), round(s0, 6))
    
    if key in _TELLES_COEFF_CACHE:
        return _TELLES_COEFF_CACHE[key]
    
    # Pre-compute matrix (constant for given s0)
    s0_2 = s0 * s0
    s0_3 = s0_2 * s0
    
    # Use analytical inverse instead of np.linalg.solve (faster)
    # M = [[1, 1, 1], [s0³, s0², s0], [3s0², 2s0, 1]]
    # For this specific matrix structure, we can compute inverse analytically
    det = 2*s0_3 - 3*s0_2 + 1
    
    if abs(det) < 1e-14:
        _TELLES_COEFF_CACHE[key] = None
        return None
    
    # Solve M @ [a,b,c]^T = [1, t0, 0]^T analytically
    # Using Cramer's rule or direct inversion
    a = (t0 - s0) / (s0_3 - s0_2)
    b = (s0_3 - t0*s0) / (s0_3 - s0_2) 
    c = (t0*s0_2 - s0_2) / (s0_3 - s0_2)
    
    # Verify solution (optional, can remove for production)
    # Actually, let's use the robust solve but cache it
    M = np.array([[1.0, 1.0, 1.0],
                  [s0_3, s0_2, s0],
                  [3*s0_2, 2*s0, 1.0]])
    rhs = np.array([1.0, t0, 0.0])
    
    try:
        coeffs = np.linalg.solve(M, rhs)
        a, b, c = coeffs[0], coeffs[1], coeffs[2]
        _TELLES_COEFF_CACHE[key] = (a, b, c)
        return (a, b, c)
    except np.linalg.LinAlgError:
        _TELLES_COEFF_CACHE[key] = None
        return None

def telles_cubic_1d(u: np.ndarray,
                    t0: float,
                    s0: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the Telles cubic transformation to the 1D Gauss-Legendre points.
    Fallback to original points if the transformation matrix is singular.
    
    Uses cached transformation coefficients for performance.

    Args:
        u (np.ndarray): Array of shape (N,) representing the Gauss-Legendre 
            points in [0, 1].
        t0 (float): Location of the singularity in [0, 1].
        s0 (float, optional): Reference GL point for clustering. Default is 
            0.5.

    Returns:
        u_telles (np.ndarray): Array of shape (N,) representing the 
            transformed points.
        du_telles_du (np.ndarray): Array of shape (N,) representing the 
            derivative of the transformation with respect to u.
    """
    
    # Get cached coefficients
    coeffs = _get_telles_coefficients(t0, s0)
    
    if coeffs is None:
        return u.copy(), np.ones_like(u)
    
    a, b, c = coeffs
    
    # Vectorized polynomial evaluation
    u_telles = a * u**3 + b * u**2 + c * u
    du_telles_du = 3 * a * u**2 + 2 * b * u + c

    if np.any(du_telles_du <= 0):
        return u.copy(), np.ones_like(u)

    return u_telles, du_telles_du

def barycentric_projection(x: np.ndarray,
                           v0: np.ndarray,
                           e1: np.ndarray,
                           e2: np.ndarray,
                           clamp: bool = True,
                           ) -> tuple[float, float]:
    """
    Project point x onto the plane of the triangle (v0, v0+e1, v0+e2)
    and express it in reference barycentric-like coords (xi, eta).
    
    Optimized version using pre-computed metric tensor.

    Returns:
        (xi, eta) with xi, eta >= 0 and xi + eta <= 1 if clamp=True.
        If the 2x2 metric is singular/near-singular, returns (1/3, 1/3).
    """
    b = x - v0
    
    # Pre-compute metric components (dot products)
    g11 = np.dot(e1, e1)
    g12 = np.dot(e1, e2)
    g22 = np.dot(e2, e2)
    
    det = g11 * g22 - g12 * g12
    
    if abs(det) < 1e-14:
        return (1.0/3.0, 1.0/3.0)
    
    # Compute RHS
    r1 = np.dot(b, e1)
    r2 = np.dot(b, e2)
    
    # Solve using analytical inverse (faster than np.linalg.solve for 2x2)
    inv_det = 1.0 / det
    xi = inv_det * (g22 * r1 - g12 * r2)
    eta = inv_det * (-g12 * r1 + g11 * r2)

    if clamp:
        xi = max(0.0, xi)
        eta = max(0.0, eta)
        s = xi + eta
        if s > 1.0 and s > 0.0:
            xi /= s
            eta /= s

    return float(xi), float(eta)