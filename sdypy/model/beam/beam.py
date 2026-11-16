import numpy as np
import scipy.linalg
from scipy.sparse.linalg import eigsh
from scipy import sparse
import time
import warnings

from .._deprecation import DeprecatedAttribute, reject_both, require


def matrices_k_e_timoshenko(l, young_modulus=None, area_moment=None, area=None,
                            poisson_ratio=None, k_s=5/6, *,
                            E=None, I=None, A=None, nu=None):
    """
    Stiffness matrix for a beam element (Timoshenko theory).

    Parameters
    ----------
    l : float or ndarray
        Element length.
    young_modulus : float or ndarray
        Young's modulus.
    area_moment : float
        Second moment of area of the cross-section.
    area : float
        Cross-section area.
    poisson_ratio : float, optional
        Poisson's ratio (default ``0.3``).
    k_s : float, optional
        Shear correction factor (default ``5/6``).

    Deprecated parameters
    ---------------------
    E : use young_modulus instead.
    I : use area_moment instead.
    A : use area instead.
    nu : use poisson_ratio instead.
    """
    if E is not None:
        reject_both("E", "young_modulus", young_modulus)
        warnings.warn("E is deprecated; use young_modulus instead",
                      DeprecationWarning, stacklevel=2)
        young_modulus = E
    if I is not None:
        reject_both("I", "area_moment", area_moment)
        warnings.warn("I is deprecated; use area_moment instead",
                      DeprecationWarning, stacklevel=2)
        area_moment = I
    if A is not None:
        reject_both("A", "area", area)
        warnings.warn("A is deprecated; use area instead",
                      DeprecationWarning, stacklevel=2)
        area = A
    if nu is not None:
        reject_both("nu", "poisson_ratio", poisson_ratio)
        warnings.warn("nu is deprecated; use poisson_ratio instead",
                      DeprecationWarning, stacklevel=2)
        poisson_ratio = nu
    require("matrices_k_e_timoshenko", young_modulus=young_modulus,
            area_moment=area_moment, area=area)
    if poisson_ratio is None:
        poisson_ratio = 0.3

    E = young_modulus
    I = area_moment
    A = area
    G = E/(2*(1+poisson_ratio))
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

def matrices_k_e(l, bending_stiffness=None, *, EI=None):
    """
    Stiffness matrix of a beam element.

    Parameters
    ----------
    l : float or ndarray
        Element length.
    bending_stiffness : float or ndarray
        Bending stiffness (Young's modulus times area moment).

    Deprecated parameters
    ---------------------
    EI : use bending_stiffness instead.
    """
    if EI is not None:
        reject_both("EI", "bending_stiffness", bending_stiffness)
        warnings.warn("EI is deprecated; use bending_stiffness instead",
                      DeprecationWarning, stacklevel=2)
        bending_stiffness = EI
    require("matrices_k_e", bending_stiffness=bending_stiffness)
    EI = bending_stiffness
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
    # Deprecated attribute names (SEP 2); kept through v1.x.
    org = DeprecatedAttribute("nodes")
    conec = DeprecatedAttribute("elements")
    I = DeprecatedAttribute("area_moment")
    EI = DeprecatedAttribute("bending_stiffness")
    M = DeprecatedAttribute("mass_matrix")
    K = DeprecatedAttribute("stiffness_matrix")
    nat_freq = DeprecatedAttribute("natural_freq")
    A = DeprecatedAttribute("mode_shape")

    def __init__(self, nodes=None, elements=None, length=None, width=None,
                 height=None, density=None, young_modulus=None, n_nodes=None,
                 added_masses=None, mass_locations=None, mode='EB', *,
                 org=None, conec=None, Young=None):
        """

        Parameters
        ----------
        nodes : array_like
            Node coordinates (organization matrix).
        elements : array_like
            Element connectivity matrix.
        length : array_like
            Lengths of the beam elements.
        width : float
            Width of the beam.
        height : float
            Height of the beam.
        density : array_like
            Density of the beam elements.
        young_modulus : array_like
            Young's modulus of each beam.
        n_nodes : int, optional
            Number of nodes to construct nodes and elements if not given.
        mode : str, optional
            The mode of assembly. Default is "EB".
             - "EB": Euler-Bernoulli beam theory.
             - "T": Timoshenko beam theory.

        Deprecated parameters
        ---------------------
        org : use nodes instead.
        conec : use elements instead.
        Young : use young_modulus instead.
        """
        if org is not None:
            reject_both("org", "nodes", nodes)
            warnings.warn("org is deprecated; use nodes instead",
                          DeprecationWarning, stacklevel=2)
            nodes = org
        if conec is not None:
            reject_both("conec", "elements", elements)
            warnings.warn("conec is deprecated; use elements instead",
                          DeprecationWarning, stacklevel=2)
            elements = conec
        if Young is not None:
            if young_modulus is not None:
                raise TypeError(
                    "Cannot pass both 'young_modulus' and deprecated 'Young'. "
                    "Use 'young_modulus' only."
                )
            warnings.warn(
                "Young is deprecated; use young_modulus instead",
                DeprecationWarning,
                stacklevel=2,
            )
            young_modulus = Young

        require("Beam", length=length, width=width, height=height,
                density=density, young_modulus=young_modulus)

        if nodes is None and n_nodes is not None:
            nodes = np.linspace(0, np.sum(length), n_nodes)
            elements = np.array([[i, i+1] for i in range(nodes.shape[0]-1)])

        self.nodes = nodes
        self.elements = elements

        self.n_dof_node = 2
        self.el_nodes = 2
        self.n_elements = n_nodes - 1
        self.n_dof = self.n_dof_node * n_nodes

        self.length = length
        self.width = width
        self.height = height
        self.mass = width * height * np.array(length) * np.array(density)
        self.young_modulus = young_modulus
        self.area = self.width*self.height

        self.area_moment = (self.height**3*self.width)/12
        self.bending_stiffness = (np.array(self.young_modulus)
                                  * self.area_moment)

        self.added_masses = added_masses
        self.mass_locations = mass_locations

        self.construct_loce()
        self.assemble(mode=mode)

        if self.added_masses is not None and self.mass_locations is not None:
            self.add_mass()

    def construct_loce(self):
        """
        Construct LOCE matrix from CONEC.
        """
        self.loce = []
        insert = np.arange(self.n_dof_node)
        conec1 = self.elements.flatten()
        for node in conec1:
            self.loce.append(node * self.n_dof_node + insert)

        self.loce = np.asarray(self.loce).flatten()
        self.loce = self.loce.reshape(
            self.elements.shape[0], self.n_dof_node * self.el_nodes)

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

        self.mass_matrix = np.zeros((self.n_dof, self.n_dof))
        self.stiffness_matrix = np.zeros((self.n_dof, self.n_dof))

        self.Ms1 = matrices_m_e(self.length, self.mass)
        if mode == "EB":
            self.Ks1 = matrices_k_e(self.length, self.bending_stiffness)
        elif mode == "T":
            self.Ks1 = matrices_k_e_timoshenko(self.length, self.young_modulus, self.area_moment, self.area)

        for i in range(self.n_elements):
            # self.Ms[i][np.ix_(self.loce[i], self.loce[i])] = self.Ms1[:, :, i]
            # self.Ks[i][np.ix_(self.loce[i], self.loce[i])] = self.Ks1[:, :, i]

            self.mass_matrix[np.ix_(self.loce[i], self.loce[i])] += self.Ms1[:, :, i]
            self.stiffness_matrix[np.ix_(self.loce[i], self.loce[i])] += self.Ks1[:, :, i]

    def add_mass(self):
        for i, m in zip(self.mass_locations, self.added_masses):
            # self.M[np.ix_(self.loce[i], self.loce[i])][0, 0] += m
            self.mass_matrix[self.loce[i][0], self.loce[i][0]] += m

    def solve(self, lanczos=True, n_modes=10, *, n=None):
        """Solve eigen problem.

        Parameters
        ----------
        lanczos : bool, optional
            Unused; kept for backward compatibility.
        n_modes : int, optional
            Number of modes to compute (default ``10``).

        Deprecated parameters
        ---------------------
        n : use n_modes instead.
        """
        if n is not None:
            if n_modes != 10:
                raise TypeError(
                    "Cannot pass both 'n_modes' and deprecated 'n'. "
                    "Use 'n_modes' only."
                )
            warnings.warn("n is deprecated; use n_modes instead",
                          DeprecationWarning, stacklevel=2)
            n_modes = n
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
            K = sparse.csc_matrix(self.stiffness_matrix)
            M = sparse.csc_matrix(self.mass_matrix)

            eigval, eigvec = eigsh(K, M=M, k=n_modes, sigma=0, which='LM')
        except:
            K = sparse.csc_matrix(self.stiffness_matrix)
            M = sparse.csc_matrix(self.mass_matrix)

            eigval, eigvec = eigsh(K, M=M, k=n_modes, sigma=100, which='LM')
            

        sort_inds = np.argsort(eigval) # sortiramo po velikosti
        eigval = eigval[sort_inds]
        eigvec = eigvec[:, sort_inds]

        eig_omega = np.sqrt(eigval)
        eig_omega[eig_omega != eig_omega] = 0  # nan element are 0
        self.natural_freq = np.real(eig_omega / (2*np.pi))
        self.mode_shape = eigvec

        return self.natural_freq, np.real(self.mode_shape)




