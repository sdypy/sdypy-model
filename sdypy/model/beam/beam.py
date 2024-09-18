import numpy as np
import scipy.linalg
from scipy.sparse.linalg import eigsh
from scipy import sparse
import time

def matrices_k_e_timoshenko(l, E, I, A, nu=0.3, k_s=5/6):
    """
    Stiffness matrix for a beam element (Timoshenko theory).
    """
    G = E/(2*(1+nu))
    phi = (12*E*I)/(k_s*A*G*l**2)

    if type(l) == np.ndarray:
        ones = np.ones_like(l)
    else:
        ones = 1

    K = E*I/(l**3*(1+phi)) * np.array([[12*ones,  6*l,          -12*ones,  6*l],
                                    [6*l, (4+phi)*l**2, -6*l, (2-phi)*l**2],
                                    [-12*ones, -6*l,         12*ones,   -6*l],
                                    [6*l, (2-phi)*l**2, -6*l, (4+phi)*l**2]])

    return K

def matrices_m_e(l, m):
    """
    Mass matrix of a beam element.

    """
    if type(l) == np.ndarray:
        ones = np.ones_like(l)
        M = (m)/420 * np.array([[156*ones,   22*l,    54*ones,    -13*l],
                                [22*l,  4*l**2,  13*l,  -3*l**2],
                                [54*ones,    13*l,    156*ones,   -22*l],
                                [-13*l, -3*l**2, -22*l, 4*l**2]])
    else:
        M = (m)/420 * np.array([[156,   22*l,    54,    -13*l],
                                [22*l,  4*l**2,  13*l,  -3*l**2],
                                [54,    13*l,    156,   -22*l],
                                [-13*l, -3*l**2, -22*l, 4*l**2]])
    return M

def matrices_k_e(l, EI):
    """
    Stiffness matrix of a beam element.

    """
    if type(l) == np.ndarray:
        ones = np.ones_like(l)
        K = (EI)/l**3 * np.array([[12*ones,  6*l,    -12*ones,  6*l],
                                [6*l, 4*l**2, -6*l, 2*l**2],
                                [-12*ones, -6*l,   12*ones,   -6*l],
                                [6*l, 2*l**2, -6*l, 4*l**2]])
    
    else:
        K = (EI)/l**3 * np.array([[12,  6*l,    -12,  6*l],
                               [6*l, 4*l**2, -6*l, 2*l**2],
                               [-12, -6*l,   12,   -6*l],
                               [6*l, 2*l**2, -6*l, 4*l**2]])
    return K



class Beam:
    def __init__(self, org, conec, length, width, height, density, Young, n_nodes=None, added_masses=None, mass_locations=None):
        """
        
        Parameters
        ----------
        org : array_like
            Organization matrix.
        conec : array_like
            Connectivity matrix.
        length : array_like
            Lengths of the beam elements.
        width : float
            Width of the beam.
        height : float
            Height of the beam.
        mass : array_like
            Masses of the beam elements.
        Young : array_like
            Young's modulus of each beam.
        n_nodes : int, optional
            Number of nodes to construct org and conec if not given.
        """
        if org is None and n_nodes is not None:
            org = np.linspace(0, np.sum(length), n_nodes)
            conec = np.array([[i, i+1] for i in range(org.shape[0]-1)])
        
        self.org = org
        self.conec = conec

        self.n_dof_node = 2
        self.el_nodes = 2
        self.n_elements = n_nodes - 1
        self.n_dof = self.n_dof_node * n_nodes

        self.length = length
        self.width = width
        self.height = height
        self.mass = width * height * np.array(length) * np.array(density)
        self.Young = Young
        self.area = self.width*self.height

        self.I = (self.height**3*self.width)/12
        self.EI = np.array(self.Young) * self.I

        self.added_masses = added_masses
        self.mass_locations = mass_locations

        self.construct_loce()
        self.assemble()

        if self.added_masses is not None and self.mass_locations is not None:
            self.add_mass()

    def construct_loce(self):
        """
        Construct LOCE matrix from CONEC.
        """
        self.loce = []
        insert = np.arange(self.n_dof_node)
        conec1 = self.conec.flatten()
        for node in conec1:
            self.loce.append(node * self.n_dof_node + insert)

        self.loce = np.asarray(self.loce).flatten()
        self.loce = self.loce.reshape(
            self.conec.shape[0], self.n_dof_node * self.el_nodes)

    def assemble(self, mode="EB"):
        """Assemble mass and stiffness matrices.

        Parameters
        ----------
        mode : str, optional
        The mode of assembly. Default is "EB".
         - "EB": Euler-Bernoulli beam theory.
         - "T": Timoshenko beam theory.
        """
        # self.Ms = np.zeros((self.n_elements, self.n_dof, self.n_dof))
        # self.Ks = np.zeros((self.n_elements, self.n_dof, self.n_dof))

        self.M = np.zeros((self.n_dof, self.n_dof))
        self.K = np.zeros((self.n_dof, self.n_dof))

        self.Ms1 = matrices_m_e(self.length, self.mass)
        if mode == "EB":
            self.Ks1 = matrices_k_e(self.length, self.EI)
        elif mode == "Timoshenko":
            self.Ks1 = matrices_k_e_timoshenko(self.length, self.Young, self.I, self.area)

        for i in range(self.n_elements):
            # self.Ms[i][np.ix_(self.loce[i], self.loce[i])] = self.Ms1[:, :, i]
            # self.Ks[i][np.ix_(self.loce[i], self.loce[i])] = self.Ks1[:, :, i]

            self.M[np.ix_(self.loce[i], self.loce[i])] += self.Ms1[:, :, i]
            self.K[np.ix_(self.loce[i], self.loce[i])] += self.Ks1[:, :, i]

    def add_mass(self):
        for i, m in zip(self.mass_locations, self.added_masses):
            # self.M[np.ix_(self.loce[i], self.loce[i])][0, 0] += m
            self.M[self.loce[i][0], self.loce[i][0]] += m

    def solve(self, lanczos=True, n=10):
        """Solve eigen problem."""
        # if lanczos is True:
        # try:
        #     K = sparse.csc_matrix(self.K)
        #     M = sparse.csc_matrix(self.M)

        #     eigval, eigvec = eigsh(K, M=M, k=n, sigma=0, which='LM')
        #     # print('lanczos')
        # # else:
        # except:
        #     print('full')
        #     eigval, eigvec = scipy.linalg.eig(self.K, self.M)

        try:
            K = sparse.csc_matrix(self.K)
            M = sparse.csc_matrix(self.M)

            eigval, eigvec = eigsh(K, M=M, k=n, sigma=0, which='LM')
        except:
            K = sparse.csc_matrix(self.K)
            M = sparse.csc_matrix(self.M)

            eigval, eigvec = eigsh(K, M=M, k=n, sigma=100, which='LM')
            

        sort_inds = np.argsort(eigval) # sortiramo po velikosti
        eigval = eigval[sort_inds]
        eigvec = eigvec[:, sort_inds]

        eig_omega = np.sqrt(eigval)
        eig_omega[eig_omega != eig_omega] = 0  # nan element are 0
        self.nat_freq = np.real(eig_omega / (2*np.pi))
        self.A = eigvec

        return self.nat_freq, np.real(self.A)




