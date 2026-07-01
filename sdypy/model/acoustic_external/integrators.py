import numpy as np
from .kernels import (r_vec, G, dG_dr, 
                      dG_dn_y, dG_dn_x, 
                      d2G_dn_x_dn_y,)
from .quadrature import (map_to_physical_triangle_batch, 
                         shape_functions_P1)

from .mesh import Mesh

class ElementIntegratorCollocation:
    """
    Element-wise integrator for acoustic boundary element method.
    
    Provides efficient vectorized integration methods for single layer, double 
    layer, adjoint double layer, and hypersingular boundary integrals.
    """
    
    def __init__(self, 
                 k: float):
        """
        Initialize the element integrator.

        Args:
            k: Wavenumber for Helmholtz kernel.
        """
        self.k = k
        self._dtype = np.complex128

    def single_layer(self, 
                     x: np.ndarray, 
                     y_phys: np.ndarray, 
                     w_phys: np.ndarray, 
                     N: np.ndarray) -> np.ndarray:
        """
        Compute single layer potential integral:
        ∫_T G(x,y) N_j(y) dS_y

        Args:
            x: Observation point, shape (3,).
            y_phys: Physical quadrature points on the source triangle, shape 
                (Q, 3).
            w_phys: Quadrature weights on the source triangle, shape (Q,).
            N: Shape function values at quadrature points, shape (Q, 3).

        Returns:
            Local vector for the triangle, shape (3,).
        """
        r = r_vec(x[None, None, :], y_phys)[1]# = r_norm
        Gv = G(r, self.k)
        acc =  np.sum((w_phys[:, :, None] * Gv[:, :, None]) * N[None, :, :], 
                      axis=1)
            
        return acc

    def double_layer(self, 
                     x: np.ndarray, 
                     y_phys: np.ndarray, 
                     w_phys: np.ndarray, 
                     N: np.ndarray, 
                     n_y: np.ndarray) -> np.ndarray:
        """
        Compute double layer potential integral:
        ∫_T ∂G(x,y)/∂n_y N_j(y) dS_y

        Args:
            x: Observation point, shape (3,).
            y_phys: Physical quadrature points on the source triangle, shape 
                (Q, 3).
            w_phys: Quadrature weights on the source triangle, shape (Q,).
            N: Shape function values at quadrature points, shape (Q, 3).
            n_y: Normal vector at source triangle, shape (3,).

        Returns:
            Local vector for the triangle, shape (3,).
        """

        r, rhat = r_vec(x[None, None, :], y_phys)[1:]
        Gv = G(r, self.k)
        dGr = dG_dr(r, Gv, self.k)
        dGdnY = dG_dn_y(rhat, dGr, n_y[:, None, :])
        acc =  np.sum((w_phys[:, :, None] * dGdnY[:, :, None]) * N[None, :, :], 
                      axis=1)
            
        return acc

    def adjoint_double_layer(self, 
                             x: np.ndarray, 
                             x_normal: np.ndarray, 
                             y_phys: np.ndarray, 
                             w_phys: np.ndarray, 
                             N: np.ndarray) -> np.ndarray:
        """
        Compute adjoint double layer potential integral:
        ∫_T ∂G(x,y)/∂n_x N_j(y) dS_y

        Args:
            x: Observation point, shape (3,).
            x_normal: Normal vector at observation point, shape (3,).
            y_phys: Physical quadrature points on the source triangle, shape
                (Q, 3).
            w_phys: Quadrature weights on the source triangle, shape (Q,).
            N: Shape function values at quadrature points, shape (Q, 3).

        Returns:
            Local vector for the triangle, shape (3,).
        """
        r, rhat = r_vec(x[None, None, :], y_phys)[1:]
        Gv = G(r, self.k)
        dGr = dG_dr(r, Gv, self.k)
        nx = np.broadcast_to(x_normal[None, None, :], y_phys.shape)
        dGdnX = dG_dn_x(rhat, dGr, nx)
        acc =  np.sum((w_phys[:, :, None] * dGdnX[:, :, None]) * N[None, :, :], 
                      axis=1)
            
        return acc

    def hypersingular_layer(self, 
                            x: np.ndarray, 
                            x_normal: np.ndarray, 
                            y_phys: np.ndarray, 
                            w_phys: np.ndarray, 
                            N: np.ndarray, 
                            n_y: np.ndarray) -> np.ndarray:
        """
        Compute adjoint double layer potential integral:
        ∫_T ∂²G(x,y)/(∂n_x∂n_y) N_j(y) dS_y
        Args:
            x: Observation point, shape (3,).
            x_normal: Normal vector at observation point, shape (3,).
            y_phys: Physical quadrature points on the source triangle, shape
                (Q, 3).
            w_phys: Quadrature weights on the source triangle, shape (Q,).
            N: Shape function values at quadrature points, shape (Q, 3).
            n_y: Normal vector at source triangle, shape (3,).

        Returns:
            Local vector for the triangle, shape (3,).
        """

        r, rhat = r_vec(x[None, None, :], y_phys)[1:]
        Gv = G(r, self.k)
        d2 = d2G_dn_x_dn_y(r_hat=rhat, 
                           r=r, 
                           n_x=np.broadcast_to(x_normal[None, None, :], 
                                               y_phys.shape),
                           n_y=n_y[:, None, :], G_vals=Gv, k=self.k)
        acc = np.sum((w_phys[:, :, None] * d2[:, :, None]) * N[None, :, :], 
                      axis=1)
            
        return acc
    
    def hypersingular_layer_reg(self,
                                x: np.ndarray,
                                x_normal: np.ndarray,
                                y_v0: np.ndarray,
                                y_e1: np.ndarray,
                                y_e2: np.ndarray,
                                y_normals: np.ndarray,
                                xi_eta: np.ndarray,
                                w: np.ndarray,) -> np.ndarray:
        """
        Compute hypersingular integrals (regularised).
        
        Uses the identity:
        ∫_T ∂²G/(∂n_x∂n_y) N_j dS = 
            k² ∫_T (n_x·n_y) G N_j dS + ∫_T ∇_y G · ∇_Γ N_j dS

        Args:
            x: Observation point, shape (3,).
            x_normal: Normal vector at observation point, shape (3,).
            y_v0: First vertices of source triangles, shape (K, 3).
            y_e1: First edge vectors of source triangles, shape (K, 3).
            y_e2: Second edge vectors of source triangles, shape (K, 3).
            y_normals: Normal vectors of source triangles, shape (K, 3).
            xi_eta: Quadrature points in reference triangle, shape (Q, 2).
            w: Quadrature weights, shape (Q,).

        Returns:
            Local vectors for each triangle, shape (K, 3).
        """
        K = len(y_v0)
        y_phys, a2 = map_to_physical_triangle_batch(xi_eta, y_v0, y_e1, y_e2)
        w_phys = w[None, :] * a2[:, None]
        N_vals = shape_functions_P1(xi_eta)
        
        nx_dot_ny = np.einsum("i,ki->k", x_normal, y_normals)
        r_norm, r_hat = r_vec(x[None, None, :], y_phys)[1:]
        G_vals = G(r_norm, self.k)
        
        dG_dr_vals = dG_dr(r_norm, G_vals, self.k)
        grad_y_G = -dG_dr_vals[:, :, None] * r_hat

        part1 = np.einsum("kq,k,qj,kq->kj", 
                          G_vals, nx_dot_ny, N_vals, w_phys) * (self.k**2)
        
        dN_ref = np.array([[-1.0, -1.0],
                          [ 1.0,  0.0],
                          [ 0.0,  1.0]], dtype=np.float64)
        
        G11 = np.einsum("ki,ki->k", y_e1, y_e1)
        G12 = np.einsum("ki,ki->k", y_e1, y_e2)
        G22 = np.einsum("ki,ki->k", y_e2, y_e2)
        
        det_G = G11 * G22 - G12 * G12
        det_G = np.where(np.abs(det_G) < 1e-14, 1e-14, det_G)
        
        Ginv = np.zeros((K, 2, 2))
        Ginv[:, 0, 0] = G22 / det_G
        Ginv[:, 0, 1] = -G12 / det_G
        Ginv[:, 1, 0] = -G12 / det_G
        Ginv[:, 1, 1] = G11 / det_G
        
        g1 = Ginv[:, 0, 0, None] * y_e1 + Ginv[:, 0, 1, None] * y_e2
        g2 = Ginv[:, 1, 0, None] * y_e1 + Ginv[:, 1, 1, None] * y_e2 
        g = np.stack([g1, g2], axis=1)
        
        grad_N = np.einsum("kao,ja->kjo", g, dN_ref)
        
        dot_products = np.einsum("kqo,kjo->kjq", grad_y_G, grad_N)
        part2 = np.einsum("kjq,kq->kj", dot_products, w_phys)

        return part1 + part2