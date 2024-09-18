
import numpy as np

from scipy.sparse.linalg import eigsh
from scipy import sparse

from tqdm.notebook import tqdm

from .mass_tet_basis import mass_tet10


def construct_loce(org, conec, n_dof_per_node=3):
    loce = []
    insert = np.tile(np.arange(n_dof_per_node), conec.shape[1])
    for element in conec:
        loce.append(np.repeat(element, n_dof_per_node)*n_dof_per_node + insert)
    return np.array(loce)

def _assemble_B_matrix_tet10(N_diff):
    B = np.zeros((N_diff.shape[0], 6, 30))
    for j in range(10):
        B[:, 0, j*3+0] = N_diff[:, 0, j]
        B[:, 1, j*3+1] = N_diff[:, 1, j]
        B[:, 2, j*3+2] = N_diff[:, 2, j]

        B[:, 3, j*3+0] = N_diff[:, 1, j]
        B[:, 3, j*3+1] = N_diff[:, 0, j]
        B[:, 4, j*3+1] = N_diff[:, 2, j]
        B[:, 4, j*3+2] = N_diff[:, 1, j]
        B[:, 5, j*3+0] = N_diff[:, 2, j]
        B[:, 5, j*3+2] = N_diff[:, 0, j]
    return B

def H_matrix(E, nu):
    d1 = E*(1-nu)/((1+nu)*(1-2*nu))
    d2 = nu*E/((1+nu)*(1-2*nu))
    d3 = E/(2*(1+nu))

    H = np.zeros((E.shape[0], 6, 6))
    H[:, 0, 0] = d1
    H[:, 1, 1] = d1
    H[:, 2, 2] = d1
    H[:, 3, 3] = d3
    H[:, 4, 4] = d3
    H[:, 5, 5] = d3
    H[:, 0, 1] = d2
    H[:, 0, 2] = d2
    H[:, 1, 0] = d2
    H[:, 1, 2] = d2
    H[:, 2, 0] = d2
    H[:, 2, 1] = d2
    return H

def Jacobi_det(J):
    J_det = J[:, 0, 0]*((J[:, 1, 1] * J[:, 2, 2]) - (J[:, 1, 2] * J[:, 2, 1]))\
                - J[:, 0, 1]*((J[:, 1, 0] * J[:, 2, 2]) - (J[:, 1, 2] * J[:, 2, 0]))\
                + J[:, 0, 2]*((J[:, 1, 0] * J[:, 2, 1]) -
                              (J[:, 1, 1] * J[:, 2, 0]))
    return J_det

def Jacobi_inv(J, J_det):
    J_t = np.transpose(J, (0, 2, 1))
    J_inv = np.zeros_like(J)
    J_inv[:, 0, 0] = ((J_t[:, 1, 1] * J_t[:, 2, 2]) -
                      (J_t[:, 1, 2] * J_t[:, 2, 1])) / J_det
    J_inv[:, 0, 1] = -((J_t[:, 1, 0] * J_t[:, 2, 2]) -
                       (J_t[:, 1, 2] * J_t[:, 2, 0])) / J_det
    J_inv[:, 0, 2] = ((J_t[:, 1, 0] * J_t[:, 2, 1]) -
                      (J_t[:, 1, 1] * J_t[:, 2, 0])) / J_det
    J_inv[:, 1, 0] = -((J_t[:, 0, 1] * J_t[:, 2, 2]) -
                       (J_t[:, 0, 2] * J_t[:, 2, 1])) / J_det
    J_inv[:, 1, 1] = ((J_t[:, 0, 0] * J_t[:, 2, 2]) -
                      (J_t[:, 0, 2] * J_t[:, 2, 0])) / J_det
    J_inv[:, 1, 2] = -((J_t[:, 0, 0] * J_t[:, 2, 1]) -
                       (J_t[:, 0, 1] * J_t[:, 2, 0])) / J_det
    J_inv[:, 2, 0] = ((J_t[:, 0, 1] * J_t[:, 1, 2]) -
                      (J_t[:, 0, 2] * J_t[:, 1, 1])) / J_det
    J_inv[:, 2, 1] = -((J_t[:, 0, 0] * J_t[:, 1, 2]) -
                       (J_t[:, 0, 2] * J_t[:, 1, 0])) / J_det
    J_inv[:, 2, 2] = ((J_t[:, 0, 0] * J_t[:, 1, 1]) -
                      (J_t[:, 0, 1] * J_t[:, 1, 0])) / J_det
    return J_inv

class Tetrahedron:
    def __init__(self, org, conec, Young, Density, Poisson, calc_n_freq=20, dof_mask=None, added_masses=None, mass_locations=None, org_rotation=None, lumped=False):
        """Initialize the tetrahedron model.

        Parameters
        ----------
        org : array_like
            Nodes.
        conec : array_like
            Elements.
        Young : array_like
            Stiffnesses of the elements.
        Density : array_like
            Density of the elements.
        Poisson : array_like
            Poisson ratio of the material.
        calc_n_freq : int
            How many natural frequencies to compute.
        dof_mask : array_like
            Mask the degrees of freedom that are rigidly fixated.
        added_masses : array_like
            Masses of point masses.
        mass_locations : array_like
            Locations of the point masses.
        org_rotation : array_like
            Rotation around the x, y, and z axes (in radians). This is a temporary rotation
            for rigid fixation that is not in-line with one of the axes.
        """
        self.n_dof_node = 3
        self.calc_n_freq = calc_n_freq
        
        self.ro = Density
        self.E = Young
        self.nu = Poisson
        self.org = org
        self.conec = conec
        self.loce = construct_loce(org, conec, self.n_dof_node)
        self.dof_mask = dof_mask

        self.added_masses = added_masses
        self.mass_locations = mass_locations
        self.org_rotation = org_rotation

        self.m_base = mass_tet10()
        
        self.n_nodes = self.org.shape[0]  # število vozlišč v mreži
        self.n_el = self.conec.shape[0]  # število elementov v mreži
        self.n_dof = self.n_dof_node * self.n_nodes  # število prostostnih stopenj

        self.n_dof_element = self.conec.shape[1] * self.n_dof_node
        self.Jg = np.repeat(self.loce.flatten(), self.n_dof_element)
        self.Ig = np.tile(self.loce, self.n_dof_element).flatten()
        
        if type(self.E) in [float, int, np.float64]:
            self.E = np.repeat(self.E, self.n_el)
        if type(self.ro) in [float, int, np.float64]:
            self.ro = np.repeat(self.ro, self.n_el)

        if lumped:
            self.K, self.M = self.assemble_matrices_lumped(self.E, self.ro)
        else:
            self.K, self.M = self.assemble_matrices(self.E, self.ro)

        if self.org_rotation is not None:
            pass

        if self.added_masses is not None and self.mass_locations is not None:
            self.add_point_masses()
    
    # def add_point_masses(self):
    #     for i, m in zip(self.mass_locations, self.added_masses):
    #         # self.M[np.ix_(self.loce[i], self.loce[i])][0, 0] += m
    #         for _ in range(self.loce.shape[1]):
    #             self.M[self.loce[i][_], self.loce[i][_]] += (m/self.conec.shape[1])
    #         # self.M[self.loce[i][0], self.loce[i][0]] += m
    
    def add_point_masses(self):
        for i, m in zip(self.mass_locations, self.added_masses):
            add_loc = i*3 + np.array([0, 1, 2])
            self.M[add_loc, add_loc] += m
                # self.M[np.ix_(add_loc, add_loc)] += np.zeros((3, 3)) * m


    def assemble_matrices_lumped(self, E, ro):
        """
        Assembling the element matrices into the global mass and stiffness matrix.
        
        To calculate the derivative of the matrix with respect to rho or E, 
        simply set that parameter to 1. Everything else is calculated the same way.
        
        Parameters
        ----------
        rho : float or array_like
            Density of the elements.
        E : float or array_like
            Young's modulus (stiffness) of the elements.
        n_el : int
            Number of elements.
        """
        Kg = np.zeros((self.n_el * self.n_dof_element**2))
        Mg = np.zeros_like(Kg)

        # Assemble H matrix
        H = H_matrix(E, self.nu)

        # Gauss quadrature locations and weights
        loc1 = 0.58541020
        loc2 = 0.13819660
        xis = [loc1, loc2, loc2, loc2]*2
        etas = [loc2, loc1, loc2, loc2]*2
        zetas = [loc2, loc2, loc1, loc2]*2
        w = 0.041666667

        # Iterate over integration points
        for _, (xi, eta, zeta) in enumerate(zip(xis, etas, zetas)):
            lam = 1 - xi - eta - zeta
            N_diffNat = np.array([[-1+4*xi, 0, 1-4*lam, 0, 4*eta, -4*eta, 4*(lam-xi), 4*zeta, 0, -4*zeta],
                                  [0, -1+4*eta, 1-4*lam, 0, 4*xi, 4 *
                                      (lam-eta), -4*xi, 0, 4*zeta, -4*zeta],
                                  [0, 0, 1-4*lam, -1+4*zeta, 0, -4*eta, -4*xi, 4*xi, 4*eta, 4*(lam-zeta)]])

            J = N_diffNat @ self.org[self.conec]

            # Jacobi determinant
            J_det = Jacobi_det(J)

            if _ >= 4:
                # Mass matrix
                Mg += np.transpose(ro[:, None, None] * J_det[:, None, None]/60/4 * np.identity(30)[None, :, :], (0, 2, 1)).flatten()
            else:
                # Stiffness matrix
                # Jacobi inverse
                J_inv = Jacobi_inv(J, J_det)
                
                # Shape functions
                N_diff = J_inv @ N_diffNat

                # B matrix
                B = _assemble_B_matrix_tet10(N_diff)

                # Add to stiffness matrix
                Kg += np.transpose(w * J_det[:, None, None] * (
                    np.transpose(B, (0, 2, 1)) @ H) @ B, (0, 2, 1)).flatten()
        
        M = sparse.csc_matrix(
            (Mg, (self.Ig, self.Jg)), shape=(self.n_dof, self.n_dof))
        K = sparse.csc_matrix(
            (Kg, (self.Ig, self.Jg)), shape=(self.n_dof, self.n_dof))

        del Kg
        del Mg

        if self.dof_mask is not None:
            K = K[self.dof_mask][:, self.dof_mask]
            M = M[self.dof_mask][:, self.dof_mask]
        
        return K, M


    def assemble_matrices(self, E, ro):
        """
        Assembling the element matrices into the global mass and stiffness matrix.
        
        To calculate the derivative of the matrix with respect to rho or E, simply set that parameter to 1. Everything else is calculated the same way.
        
        Parameters
        ----------
        rho : float or array_like
            Density of the elements.
        E : float or array_like
            Young's modulus (stiffness) of the elements.
        n_el : int
            Number of elements.
        """
        Kg = np.zeros((self.n_el * self.n_dof_element**2))
        Mg = np.zeros_like(Kg)

        # Assemble H matrix
        H = H_matrix(E, self.nu)

        # Gauss quadrature locations and weights
        loc1 = 0.58541020
        loc2 = 0.13819660
        xis = [loc1, loc2, loc2, loc2, 1/4]
        etas = [loc2, loc1, loc2, loc2, 1/4]
        zetas = [loc2, loc2, loc1, loc2, 1/4]
        w = 0.041666667

        # Iterate over integration points
        for _, (xi, eta, zeta) in enumerate(zip(xis, etas, zetas)):
            lam = 1 - xi - eta - zeta
            N_diffNat = np.array([[-1+4*xi, 0, 1-4*lam, 0, 4*eta, -4*eta, 4*(lam-xi), 4*zeta, 0, -4*zeta],
                                  [0, -1+4*eta, 1-4*lam, 0, 4*xi, 4 *
                                      (lam-eta), -4*xi, 0, 4*zeta, -4*zeta],
                                  [0, 0, 1-4*lam, -1+4*zeta, 0, -4*eta, -4*xi, 4*xi, 4*eta, 4*(lam-zeta)]])

            J = N_diffNat @ self.org[self.conec]

            # Jacobi determinant
            J_det = Jacobi_det(J)

            if _ == 4:
                # Mass matrix
                Mg = np.transpose(ro[:, None, None] * J_det[:, None, None]/2520 * self.m_base[None, :, :], (0, 2, 1)).flatten()
            else:
                # Stiffness matrix
                # Jacobi inverse
                J_inv = Jacobi_inv(J, J_det)
                
                # Shape functions
                N_diff = J_inv @ N_diffNat

                # B matrix
                B = _assemble_B_matrix_tet10(N_diff)

                # Add to stiffness matrix
                Kg += np.transpose(w * J_det[:, None, None] * (
                    np.transpose(B, (0, 2, 1)) @ H) @ B, (0, 2, 1)).flatten()
        
        M = sparse.csc_matrix(
            (Mg, (self.Ig, self.Jg)), shape=(self.n_dof, self.n_dof))
        K = sparse.csc_matrix(
            (Kg, (self.Ig, self.Jg)), shape=(self.n_dof, self.n_dof))

        del Kg
        del Mg

        if self.dof_mask is not None:
            K = K[self.dof_mask][:, self.dof_mask]
            M = M[self.dof_mask][:, self.dof_mask]
        
        return K, M
    
    def solve(self):
        """
        Solve the eigenvalue problem.
        
        Parameters
        ----------
        dof_mask : array_like, optional
            Degrees of freedom that are not fixed. If None, the body is free-free.
        """
        self.eigval, self.eigvec = eigsh(self.K, M=self.M, k=self.calc_n_freq, sigma=0, which='LM')

        eig_omega = np.sqrt(self.eigval)
        eig_omega[eig_omega != eig_omega] = 0  # nan element are 0
        self.nat_freq = eig_omega / (2*np.pi)
        # self.A = np.array([v.reshape(self.org.shape) for i, v in enumerate(self.eigvec.T)]) # reshape the eigenvectors

        if self.dof_mask is not None:
            eigvec_full = np.zeros((self.org.flatten().shape[0], self.eigvec.shape[1]))
            eigvec_full[self.dof_mask] = self.eigvec
            self.eigvec_full = eigvec_full
        else:
            self.eigvec_full = self.eigvec

        try:
            self.A = self.reshape_eigenvectors(self.eigvec_full, self.org.shape)
        except:
            pass

    def reshape_eigenvectors(self, eigvec, shape):
        return np.array([v.reshape(shape) for i, v in enumerate(eigvec.T)])
        
    def matrix_derivative(self, derivative_E_ind=None, derivative_ro_ind=None, diff_step=1e-5):
        if derivative_E_ind is not None:
            E = self.E.copy()
            step = np.mean(E[derivative_E_ind] * diff_step)
            
            E[derivative_E_ind] = E[derivative_E_ind] + step
            
            K_, M_ = self.assemble_matrices(E, self.ro)
            
            self.K_diff = (K_ - self.K) / step
            self.M_diff = sparse.csc_matrix(self.K_diff.shape)
            
        elif derivative_ro_ind is not None:
            ro = self.ro.copy()
            step = np.mean(ro[derivative_ro_ind] * diff_step)
            
            ro[derivative_ro_ind] = ro[derivative_ro_ind] + step
            
            K_, M_ = self.assemble_matrices(self.E, ro)
            
            self.M_diff = (M_ - self.M) / step
            self.K_diff = sparse.csc_matrix(self.M_diff.shape)
            
    def split_A(self, j, k_pivot=None):
        """Compute dynamic matrix A and performe LU split.
        
        Matrix is used for v_j computation page 25, Eq. (2.65) in [1].
        
        [1] Friswell, Mottershead. Finite Element Model Updating in Structural
            Dynamics.
        """
        if k_pivot is None:
            k_pivot = np.argmax(np.abs(self.eigvec[:, j]))
            
        A = (self.K - self.eigval[j]*self.M)
#         A = A.toarray()
#         A = np.insert(A, k_pivot, 0, axis=0)
#         A = np.insert(A, k_pivot, 0, axis=1)
        
        A._shape = (A.shape[0]+1, A.shape[1]+1)
        A.indptr = np.insert(A.indptr, k_pivot, A.indptr[k_pivot])
        A.indices[A.indices >= k_pivot] += 1
        A[k_pivot, k_pivot] = 1
        
        lu_A = sparse.linalg.splu(A)
        return lu_A
    
    def S_eigval(self, j):
        return self.eigvec[:, j] @ (self.K_diff - self.eigval[j]*self.M_diff) @ self.eigvec[:, j]
    
    def S_matrix(self, eig_ind, derivative_E_ind, compute_vector_sensitivity=True, coor_inds=None, diff_step=1e-5):
        dMs = []
        dKs = []

        if coor_inds is not None:
            coor_inds1 = construct_loce(None, np.array([coor_inds]), 3)

        S_mtx_val = np.zeros((len(eig_ind), len(derivative_E_ind)))
        S_mtx_vec = np.zeros((self.eigvec.shape[0], len(eig_ind), len(derivative_E_ind)))

        for i1, j in enumerate(tqdm(eig_ind, leave=False)):
            if compute_vector_sensitivity:
                k_pivot = np.argmax(np.abs(self.eigvec[:, j]))
                lu_A = self.split_A(j, k_pivot)

            for i2, dEi in enumerate(tqdm(derivative_E_ind, leave=False)):
                # Assemble derivative matrices
                if i1 == 0:
        #             self.assemble_matrices(derivative_E_ind=dEi)
                    self.matrix_derivative(derivative_E_ind=dEi, diff_step=diff_step)
                    dMs.append(self.M_diff)
                    dKs.append(self.K_diff)
                else:
                    self.M_diff = dMs[i2]
                    self.K_diff = dKs[i2]

                # Eigenvalue sensitivity
                S_eigval = self.S_eigval(j=j)

                if compute_vector_sensitivity:
                    # Eigenvector sensitivity
                    f_j = -(self.K_diff - self.eigval[j]*self.M_diff - S_eigval * self.M) @ self.eigvec[:, j]
                    f_j = np.insert(f_j, k_pivot, 0)

                    v_j = lu_A.solve(f_j)
                    v_j = np.delete(v_j, k_pivot)

                    c_j = -self.eigvec[:, j] @ self.M @ v_j - 1/2*self.eigvec[:, j] @ self.M_diff @ self.eigvec[:, j]

                    S_eigvec = v_j + c_j*self.eigvec[:, j]

                    S_mtx_vec[:, i1, i2] = S_eigvec

                S_mtx_val[i1, i2] = S_eigval

        if compute_vector_sensitivity:
            if coor_inds is not None:
                S_mtx_vec = S_mtx_vec[coor_inds1]

            S_mtx_vec = S_mtx_vec.reshape(-1, len(derivative_E_ind))
            S_mtx = np.concatenate((S_mtx_val, S_mtx_vec))

        else:
            S_mtx = S_mtx_val
            
        return S_mtx    

    # Working version, for reference.
    # def S_matrix(self, eig_ind, derivative_E_ind, compute_vector_sensitivity=True, coor_mask=None):
    #     dMs = []
    #     dKs = []

    #     S_mtx_val = np.zeros((len(eig_ind), len(derivative_E_ind)))
    #     S_mtx_vec = np.zeros((len(eig_ind)*self.eigvec.shape[0], len(derivative_E_ind)))

    #     for i1, j in enumerate(tqdm(eig_ind)):
    #         if compute_vector_sensitivity:
    #             k_pivot = np.argmax(np.abs(self.eigvec[:, j]))
    #             lu_A = self.split_A(j, k_pivot)

    #         for i2, dEi in enumerate(tqdm(derivative_E_ind, leave=False)):
    #             # Assemble derivative matrices
    #             if i1 == 0:
    #     #             self.assemble_matrices(derivative_E_ind=dEi)
    #                 self.matrix_derivative(derivative_E_ind=dEi)
    #                 dMs.append(self.M_diff)
    #                 dKs.append(self.K_diff)
    #             else:
    #                 self.M_diff = dMs[i2]
    #                 self.K_diff = dKs[i2]

    #             # Eigenvalue sensitivity
    #             S_eigval = self.S_eigval(j=j)

    #             if compute_vector_sensitivity:
    #                 # Eigenvector sensitivity
    #                 f_j = -(self.K_diff - self.eigval[j]*self.M_diff - S_eigval * self.M) @ self.eigvec[:, j]
    #                 f_j = np.insert(f_j, k_pivot, 0)

    #                 v_j = lu_A.solve(f_j)
    #                 v_j = np.delete(v_j, k_pivot)

    #                 c_j = -self.eigvec[:, j] @ self.M @ v_j - 1/2*self.eigvec[:, j] @ self.M_diff @ self.eigvec[:, j]

    #                 S_eigvec = v_j + c_j*self.eigvec[:, j]


    #                 S_mtx_vec[i1*self.eigvec.shape[0]:(i1+1)*self.eigvec.shape[0], i2] = S_eigvec

    #             S_mtx_val[i1, i2] = S_eigval

    #     S_mtx = np.concatenate((S_mtx_val, S_mtx_vec))
    #     return S_mtx