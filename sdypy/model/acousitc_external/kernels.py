import numpy as np

def r_vec(x: np.ndarray, 
          y: np.ndarray) -> np.ndarray:
    """
    Compute the vector from points y to points x.

    r_vec = x - y
    r_norm = ||r_vec||
    r_hat = r_vec / r_norm

    Args:
        x (np.ndarray): Array of shape (..., 3) representing points x.
        y (np.ndarray): Array of shape (..., 3) representing points y.

    Returns:
        r_vec (np.ndarray): Array of shape (..., 3) representing the vector 
            from y to x.
        r_norm (np.ndarray): Array of shape (...) representing the norm of 
            r_vec.
        r_hat (np.ndarray): Array of shape (..., 3) representing the unit 
            vector in the direction of r_vec.
    """
    r_vec_ = x - y
    r_norm = np.linalg.norm(r_vec_, axis=-1)
    r_norm = np.where(r_norm == 0, 1e-16, r_norm)  # Avoid division by zero
    r_hat = r_vec_ / r_norm[..., np.newaxis]
    return r_vec_, r_norm, r_hat

def G(r_norm: np.ndarray, 
      k: float) -> np.ndarray:
    """
    Compute the Green's function for the Helmholtz equation in 3D.

    G(r) = e^{ikr}/(4π r)

    Args:
        r_norm (np.ndarray): Array of shape (...) representing the distance 
            between source and field points.
        k (float): Wavenumber.

    Returns:
        G (np.ndarray): Array of shape (...) representing the Green's function.
    """
    return np.exp(1j * k * r_norm) / (4 * np.pi * r_norm)

def dG_dr(r_norm: np.ndarray, 
          G: np.ndarray,
          k: float) -> np.ndarray:
    """
    Compute the derivative of the Green's function with respect to r.

    For G(r) = e^{ikr}/(4π r):
        dG/dr = (ik - 1/r) * G

    Args:
        r_norm (np.ndarray): Array of shape (...) representing the distance 
            between source and field points.
        G (np.ndarray): Array of shape (...) representing the Green's function.
        k (float): Wavenumber.

    Returns:
        dG_dr (np.ndarray): Array of shape (...) representing the derivative 
            of the Green's function with respect to r.
    """
    return G * (1j * k - 1 / r_norm)

def d2G_dr2(r: np.ndarray, 
            G_vals: np.ndarray, 
            k: float) -> np.ndarray:
    """
    Second radial derivative d^2G/dr^2 for 3D Helmholtz.

    For G(r) = e^{ikr}/(4π r):
        d^2G/dr^2 = (-k^2 - 2i k / r + 2 / r^2) * G

    Args:
        r (np.ndarray): Array of shape (...) representing the distance 
            between source and field points.
        G_vals (np.ndarray): Array of shape (...) representing the Green's
            function values G(r,k).
        k (float): Wavenumber.

    Returns:
        d2G_dr2: Same shape as r (complex).
    """

    eps = np.finfo(float).eps
    r_safe = np.where(r == 0, eps, r)
    return ((-k**2) - (2j * k) / r_safe + 2.0 / (r_safe**2)) * G_vals

def dG_dn_y(r_hat: np.ndarray,
            dG_dr: np.ndarray,
            n_y: np.ndarray) -> np.ndarray:
    """
    Compute the normal derivative of the Green's function with respect to
    the source point y.

    For G(r) = e^{ikr}/(4π r) and dG/dr = (ik - 1/r) G:
        ∂G/∂n_y = -dG/dr (r_hat · n_y)

    Args:
        r_hat (np.ndarray): Array of shape (..., 3) representing the unit 
            vector in the direction from y to x.
        dG_dr (np.ndarray): Array of shape (...) representing the derivative 
            of the Green's function with respect to r.
        n_y (np.ndarray): Array of shape (..., 3) representing the normal 
            vector at point y.

    Returns:
        dG_dn_y (np.ndarray): Array of shape (...) representing the normal 
            derivative of the Green's function with respect to y.
    """
    return -dG_dr * np.einsum('...i,...i->...', r_hat, n_y)

def dG_dn_x(r_hat: np.ndarray,
            dG_dr: np.ndarray,
            n_x: np.ndarray) -> np.ndarray:
    """
    Compute the normal derivative of the Green's function with respect to
    the field point x.

    For G(r) = e^{ikr}/(4π r) and dG/dr = (ik - 1/r) G:
        ∂G/∂n_x = dG/dr (r_hat · n_x)

    Args:
        r_hat (np.ndarray): Array of shape (..., 3) representing the unit 
            vector in the direction from y to x.
        dG_dr (np.ndarray): Array of shape (...) representing the derivative 
            of the Green's function with respect to r.
        n_x (np.ndarray): Array of shape (..., 3) representing the normal 
            vector at point x.

    Returns:
        dG_dn_x (np.ndarray): Array of shape (...) representing the normal 
            derivative of the Green's function with respect to x.
    """
    return dG_dr * np.einsum('...i,...i->...', r_hat, n_x)


def d2G_dn_x_dn_y(r_hat: np.ndarray,
                  r: np.ndarray,
                  n_x: np.ndarray,
                  n_y: np.ndarray,
                  G_vals: np.ndarray,
                  k: float,
                  ) -> np.ndarray:
    """
    Direct hypersingular kernel ∂²G/(∂n_x ∂n_y) for 3D Helmholtz.

    Uses the identity:
        ∂²G/∂n_x∂n_y
        = - f''(r) (r_hat·n_x)(r_hat·n_y)
          - (f'(r)/r) [ (n_x·n_y) - (r_hat·n_x)(r_hat·n_y) ],

    where f'(r) = dG/dr = (ik - 1/r) G, and
          f''(r) = d²G/dr² = (-k² - 2ik/r + 2/r²) G.

    Args:
        r_hat (np.ndarray): Array of shape (..., 3) representing the unit 
            vector in the direction from y to x.
        r (np.ndarray): Array of shape (...,) representing the distance 
            between source and field points.
        n_x (np.ndarray): Array of shape (..., 3) representing the normal 
            vector at point x.
        n_y (np.ndarray): Array of shape (..., 3) representing the normal 
            vector at point y.
        G_vals (np.ndarray): Array of shape (...,) representing the Green's
            function values G(r,k).
        k (float): Wavenumber.

    Returns:
        d2G_dn_x_dn_y (np.ndarray): Array of shape (...) representing the 
            second normal derivative of the Green's function with respect to 
            both x and y.
    """
    r = np.where(r == 0, 1e-16, r)  # Avoid division by zero
    # f'(r) and f''(r)
    dGr = dG_dr(r, G_vals, k)
    d2Gr = d2G_dr2(r, G_vals, k)

    nx_dot_ny = np.einsum("...i,...i->...", n_x, n_y)
    nx_dot_rh = np.einsum("...i,...i->...", n_x, r_hat)
    ny_dot_rh = np.einsum("...i,...i->...", n_y, r_hat)

    term1 = -d2Gr * (nx_dot_rh * ny_dot_rh)
    term2 = -(dGr / r) * (nx_dot_ny - nx_dot_rh * ny_dot_rh)

    return term1 + term2
