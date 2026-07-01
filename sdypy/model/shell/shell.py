import numpy as np
from scipy.sparse import coo_array
from scipy.sparse import diags_array
from collections.abc import Iterable
from tqdm import tqdm

import warnings

def get_normal_vector(nodes_el):
    """Based on the nodal coordinates of the element, calculate the normal vector and the
    orthogonal vectors to the normal vector.
    
    The code is based on MATLAB code and the book "Finite element procedures"
    by Bathe.

    Parameters
    ----------
    nodes_el : np.ndarray
        array of the nodal coordinates of the element, shaped (4, 3).
    """
    Vn = np.zeros((4, 3))
    V1 = np.zeros((4, 3))
    V2 = np.zeros((4, 3))

    # calculate normal at each node
    R = np.array([-1, 1, 1, -1])
    S = np.array([-1, -1, 1, 1])

    for i in range(4):
        # derivative of the shape functions in the parent domain
        dN1dr = -0.25 * (1 - S[i])
        dN2dr = 0.25 * (1 - S[i])
        dN3dr = 0.25 * (1 + S[i])
        dN4dr = -0.25 * (1 + S[i])

        dN1ds = -0.25 * (1 - R[i])
        dN2ds = -0.25 * (1 + R[i])
        dN3ds = 0.25 * (1 + R[i])
        dN4ds = 0.25 * (1 - R[i])

        # construct first two columns of Jacobian 
        Jm = np.array([[dN1dr, dN2dr, dN3dr, dN4dr],
                    [dN1ds, dN2ds, dN3ds, dN4ds]]) @ nodes_el

        # normal vector 3 at the point
        Vn[i, :] = np.cross(Jm[0, :], Jm[1, :]) / np.linalg.norm(np.cross(Jm[0, :], Jm[1, :]))  # normalise the vector

        if np.abs(np.round(Vn[i, 1], 4)) == 1.:  # i.e. Vn is aligned with ey
            V1[i, :] = np.array([0, 0, 1.])  # see 5.110b in Bathe
            V2[i, :] = np.array([1., 0, 0])
        else:
            V1[i, :] = np.cross([0, 1, 0], Vn[i, :]) / np.linalg.norm(np.cross([0, 1, 0], Vn[i, :]))
            V2[i, :] = np.cross(Vn[i, :], V1[i, :]) / np.linalg.norm(np.cross(Vn[i, :], V1[i, :]))

    return Vn, V1, V2

def get_constitutive_tensor(E, nu, k=5/6):
    """Get the constitutive tensor of the given material properties.
    
    Parameters
    ----------
    E : float
        Young's modulus
    nu : float
        poisson ratio
    k : float
        shear correction factor
    """
    D0 = (E / (1 - nu**2)) * np.array([[1, nu, 0, 0, 0, 0],
                                    [nu, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, (1 - nu) / 2, 0, 0],
                                    [0, 0, 0, 0, (k) * (1 - nu) / 2, 0],
                                    [0, 0, 0, 0, 0, (k) * (1 - nu) / 2]])
    
    return D0

def shape_functions(xi, eta):
    """Calculate the shape functions for the given xi and eta coordinates."""
    N = np.array([
        (1 - xi) * (1 - eta) / 4,
        (1 + xi) * (1 - eta) / 4,
        (1 + xi) * (1 + eta) / 4,
        (1 - xi) * (1 + eta) / 4
    ])
    return N

def shape_function_derivatives(xi, eta):
    """Calculate the derivatives of the shape functions for the given xi and eta coordinates."""
    dN_dxi = np.array([
        [-(1 - eta) / 4, -(1 - xi) / 4],
        [(1 - eta) / 4, -(1 + xi) / 4],
        [(1 + eta) / 4, (1 + xi) / 4],
        [-(1 + eta) / 4, (1 - xi) / 4]
    ])

    return dN_dxi

def interpolation_matrix(N1, N2, N3, N4, t, thickness, V1, V2):
    """Construct the interpolation matrix for the given shape functions, thickness
    and vectors V1 and V2 (computed by ``get_normal_vector``).
    
    Parameters
    ----------
    N1 : float
        shape function
    N2 : float
        shape function
    N3 : float
        shape function
    N4 : float
        shape function
    t : float
        coordinate in the direction of the thickness
    thickness : float
        thickness of the element
    V1 : array
        vector orthogonal to the normal vector
    V2 : array
        vector orthogonal to the normal vector
    """
    N = np.zeros((3, 20))
    N[:, :3] = np.diag([N1]*3)
    N[:, 5:8] = np.diag([N2]*3)
    N[:, 10:13] = np.diag([N3]*3)
    N[:, 15:18] = np.diag([N4]*3)

    N[:, 3] = -(t/2) * thickness * N1 * V2[0, :]
    N[:, 4] = (t/2) * thickness * N1 * V1[0, :]
    N[:, 8] = -(t/2) * thickness * N2 * V2[1, :]
    N[:, 9] = (t/2) * thickness * N2 * V1[1, :]
    N[:, 13] = -(t/2) * thickness * N3 * V2[2, :]
    N[:, 14] = (t/2) * thickness * N3 * V1[2, :]
    N[:, 18] = -(t/2) * thickness * N4 * V2[3, :]
    N[:, 19] = (t/2) * thickness * N4 * V1[3, :]
    return N

def construct_Brr(dN1dr, dN2dr, dN3dr, dN4dr, g1, g2, t):
    g1_1, g1_2, g1_3, g1_4 = g1
    g2_1, g2_2, g2_3, g2_4 = g2
    Brr = np.array([
        np.hstack((dN1dr * np.array([1, 0, 0, t * g1_1[0], t * g2_1[0]]),
                dN2dr * np.array([1, 0, 0, t * g1_2[0], t * g2_2[0]]),
                dN3dr * np.array([1, 0, 0, t * g1_3[0], t * g2_3[0]]),
                dN4dr * np.array([1, 0, 0, t * g1_4[0], t * g2_4[0]]))),
        np.hstack((dN1dr * np.array([0, 1, 0, t * g1_1[1], t * g2_1[1]]),
                dN2dr * np.array([0, 1, 0, t * g1_2[1], t * g2_2[1]]),
                dN3dr * np.array([0, 1, 0, t * g1_3[1], t * g2_3[1]]),
                dN4dr * np.array([0, 1, 0, t * g1_4[1], t * g2_4[1]]))),
        np.hstack((dN1dr * np.array([0, 0, 1, t * g1_1[2], t * g2_1[2]]),
                dN2dr * np.array([0, 0, 1, t * g1_2[2], t * g2_2[2]]),
                dN3dr * np.array([0, 0, 1, t * g1_3[2], t * g2_3[2]]),
                dN4dr * np.array([0, 0, 1, t * g1_4[2], t * g2_4[2]])))
    ])
    return Brr

def construct_Bss(dN1ds, dN2ds, dN3ds, dN4ds, g1, g2, t):
    g1_1, g1_2, g1_3, g1_4 = g1
    g2_1, g2_2, g2_3, g2_4 = g2
    Bss = np.array([
        np.hstack((dN1ds * np.array([1, 0, 0, t * g1_1[0], t * g2_1[0]]),
                dN2ds * np.array([1, 0, 0, t * g1_2[0], t * g2_2[0]]),
                dN3ds * np.array([1, 0, 0, t * g1_3[0], t * g2_3[0]]),
                dN4ds * np.array([1, 0, 0, t * g1_4[0], t * g2_4[0]]))),
        np.hstack((dN1ds * np.array([0, 1, 0, t * g1_1[1], t * g2_1[1]]),
                dN2ds * np.array([0, 1, 0, t * g1_2[1], t * g2_2[1]]),
                dN3ds * np.array([0, 1, 0, t * g1_3[1], t * g2_3[1]]),
                dN4ds * np.array([0, 1, 0, t * g1_4[1], t * g2_4[1]]))),
        np.hstack((dN1ds * np.array([0, 0, 1, t * g1_1[2], t * g2_1[2]]),
                dN2ds * np.array([0, 0, 1, t * g1_2[2], t * g2_2[2]]),
                dN3ds * np.array([0, 0, 1, t * g1_3[2], t * g2_3[2]]),
                dN4ds * np.array([0, 0, 1, t * g1_4[2], t * g2_4[2]])))
    ])
    return Bss

def construct_Btt(N1, N2, N3, N4, g1, g2, t):
    g1_1, g1_2, g1_3, g1_4 = g1
    g2_1, g2_2, g2_3, g2_4 = g2
    Btt = np.array([
        np.hstack((N1 * np.array([0, 0, 0, g1_1[0], g2_1[0]]),
                N2 * np.array([0, 0, 0, g1_2[0], g2_2[0]]),
                N3 * np.array([0, 0, 0, g1_3[0], g2_3[0]]),
                N4 * np.array([0, 0, 0, g1_4[0], g2_4[0]]))),
        np.hstack((N1 * np.array([0, 0, 0, g1_1[1], g2_1[1]]),
                N2 * np.array([0, 0, 0, g1_2[1], g2_2[1]]),
                N3 * np.array([0, 0, 0, g1_3[1], g2_3[1]]),
                N4 * np.array([0, 0, 0, g1_4[1], g2_4[1]]))),
        np.hstack((N1 * np.array([0, 0, 0, g1_1[2], g2_1[2]]),
                N2 * np.array([0, 0, 0, g1_2[2], g2_2[2]]),
                N3 * np.array([0, 0, 0, g1_3[2], g2_3[2]]),
                N4 * np.array([0, 0, 0, g1_4[2], g2_4[2]])))
    ])
    return Btt

def construct_rotation_tensor(gr, gs, gt):
    # Calculate the local coordinates
    e3 = gt / np.linalg.norm(gt)
    e1 = np.cross(gs, e3) / np.linalg.norm(np.cross(gs, e3))
    e2 = np.cross(e3, e1)

    # Calculate the contravariant basis
    gR = np.cross(gs, gt) / np.dot(gr, np.cross(gs, gt))
    gS = np.cross(gt, gr) / np.dot(gr, np.cross(gs, gt))
    gT = np.cross(gr, gs) / np.dot(gr, np.cross(gs, gt))

    # Generate the rotation tensor - 5.120 in Bathe
    l1, l2, l3 = np.dot(gR, e1), np.dot(gR, e2), np.dot(gR, e3)
    m1, m2, m3 = np.dot(gS, e1), np.dot(gS, e2), np.dot(gS, e3)
    n1, n2, n3 = np.dot(gT, e1), np.dot(gT, e2), np.dot(gT, e3)

    Q = np.array([
        [l1**2, m1**2, n1**2, l1*m1, m1*n1, n1*l1],
        [l2**2, m2**2, n2**2, l2*m2, m2*n2, n2*l2],
        [l3**2, m3**2, n3**2, l3*m3, m3*n3, n3*l3],
        [2*l1*l2, 2*m1*m2, 2*n1*n2, l1*m2 + l2*m1, m1*n2 + m2*n1, n1*l2 + n2*l1],
        [2*l2*l3, 2*m2*m3, 2*n2*n3, l2*m3 + l3*m2, m2*n3 + m3*n2, n2*l3 + n3*l2],
        [2*l3*l1, 2*m3*m1, 2*n3*n1, l3*m1 + l1*m3, m3*n1 + m1*n3, n3*l1 + n1*l3]
    ])
    return Q

def construct_rotational_matrix(V1, V2, Vn):
    # Rotate stiffness matrix
    T = np.zeros((24, 24))

    for n_index in range(4):
        transf = np.block([[np.eye(3), np.zeros((3, 3))],
                        [np.zeros((3, 3)), np.array([V1[n_index], V2[n_index], Vn[n_index]])]])
        T[n_index * 6: (n_index+1) * 6, n_index * 6:(n_index+1) * 6] = transf

    return T

def lump_mass_matrix(M):
    lumped_masses = M.sum(axis=1)  # Sum of rows
    lumped_mass_matrix = diags_array(lumped_masses, shape=M.shape)  # Create diagonal matrix
    return lumped_mass_matrix

# %%

def MITC4_element(E, nu, rho, thickness, nodes_el):
    """The computation of stiffness and mass matrices for each element.
    
    Using the MITC4 elements. The code is based on the MatLab code and on the
    Finite Element Procedures book by Bathe.
    """
    warnings.warn("MITC4_element is deprecated. Use Shell class instead.", DeprecationWarning)
    # Calculate normal at each node
    Vn, V1, V2 = get_normal_vector(nodes_el)
    
    # see eq 5.114 in Bathe
    g1 = -0.5 * thickness * V2
    g2 = 0.5 * thickness * V1

    # constitutive tensor in local coordinate system
    D0 = get_constitutive_tensor(E, nu)

    # perform full gauss integration (2 by 2 by 2 integration)
    W_gauss = np.array([1, 1])
    points_gauss = np.array([1/np.sqrt(3), -1/np.sqrt(3)])

    #  Compute K and M
    K_el = np.zeros((24, 24))
    M_el = np.zeros((24, 24))
    Volume_el = 0

    for r in points_gauss: # xi
        for s in points_gauss: # eta 
            for t in points_gauss: # zeta

                # interpolation functions
                N1, N2, N3, N4 = shape_functions(r, s)

                # derivative of the interpolation functions
                dN = shape_function_derivatives(r, s)
                dN1dr, dN1ds = dN[0]
                dN2dr, dN2ds = dN[1]
                dN3dr, dN3ds = dN[2]
                dN4dr, dN4ds = dN[3]
                
                # assemble the interpolation matrix (used to get the mass matrix)
                N = interpolation_matrix(N1, N2, N3, N4, t, thickness, V1, V2)

                # Constructing Brr
                Brr = construct_Brr(dN1dr, dN2dr, dN3dr, dN4dr, g1, g2, t)

                # Constructing Bss
                Bss = construct_Bss(dN1ds, dN2ds, dN3ds, dN4ds, g1, g2, t)

                # Constructing Btt
                Btt = construct_Btt(N1, N2, N3, N4, g1, g2, t)

                # Initialize GN with zeros
                GN = np.zeros((3, 4))

                # Fill GN with values
                GN[:2, :] = np.array([
                    [dN1dr, dN2dr, dN3dr, dN4dr],
                    [dN1ds, dN2ds, dN3ds, dN4ds]
                ])

                # Calculate GN_dash
                GN_dash = GN * 0.5 * t * thickness
                GN_dash[2, :] = np.array([N1, N2, N3, N4]) * 0.5 * thickness

                # Calculate the Jacobian
                J = np.dot(GN, nodes_el) + np.dot(GN_dash, Vn)
                detJ = np.linalg.det(J)

                # Covariant bases
                gr = J[0, :]
                gs = J[1, :]
                gt = J[2, :]

                # Calculate the direct strain components
                Err = np.dot(gr, Brr)
                Ess = np.dot(gs, Bss)
                Ett = np.dot(gt, Btt)  # This should be zero
                Ers = 0.5 * (np.dot(gs, Brr) + np.dot(gr, Bss))

                # Coordinates r1 and r2 for points A, B, C, D
                r1 = [0, -1, 0, 1]  # order A B C D
                r2 = [1, 0, -1, 0]

                # Initialize strain results
                Ert_A, Est_B, Ert_C, Est_D = None, None, None, None

                for i in range(4):
                    # Interpolation functions
                    N1, N2, N3, N4 = shape_functions(r1[i], r2[i])

                    # Derivative of the interpolation functions
                    dN = shape_function_derivatives(r1[i], r2[i])
                    dN1dr, dN1ds = dN[0]
                    dN2dr, dN2ds = dN[1]
                    dN3dr, dN3ds = dN[2]
                    dN4dr, dN4ds = dN[3]

                    # Constructing Brr
                    Brr = construct_Brr(dN1dr, dN2dr, dN3dr, dN4dr, g1, g2, t)

                    # Constructing Bss
                    Bss = construct_Bss(dN1ds, dN2ds, dN3ds, dN4ds, g1, g2, t)

                    # Constructing Btt
                    Btt = construct_Btt(N1, N2, N3, N4, g1, g2, t)

                    # Constructing GN_edge and GN_dash_edge
                    GN_edge = np.zeros((3, 4))
                    GN_edge[:2, :] = np.array([
                        [dN1dr, dN2dr, dN3dr, dN4dr],
                        [dN1ds, dN2ds, dN3ds, dN4ds]
                    ])

                    GN_dash_edge = GN_edge * 0.5 * t * thickness
                    GN_dash_edge[2, :] = np.array([N1, N2, N3, N4]) * 0.5 * thickness

                    # Jacobian
                    J_edge = np.dot(GN_edge, nodes_el) + np.dot(GN_dash_edge, Vn)

                    # Covariant bases
                    gr_edge = J_edge[0, :]
                    gs_edge = J_edge[1, :]
                    gt_edge = J_edge[2, :]

                    if i == 0:  # at A
                        Ert_A = 0.5 * (np.dot(gr_edge, Btt) + np.dot(gt_edge, Brr))
                    elif i == 1:  # at B
                        Est_B = 0.5 * (np.dot(gs_edge, Btt) + np.dot(gt_edge, Bss))
                    elif i == 2:  # at C
                        Ert_C = 0.5 * (np.dot(gr_edge, Btt) + np.dot(gt_edge, Brr))
                    else:  # at D
                        Est_D = 0.5 * (np.dot(gs_edge, Btt) + np.dot(gt_edge, Bss))

                # Perform the interpolation for the transverse strains
                Ert = 0.5 * (1 + s) * Ert_A + 0.5 * (1 - s) * Ert_C
                Est = 0.5 * (1 + r) * Est_D + 0.5 * (1 - r) * Est_B

                # Assemble the B vector
                B = np.vstack((Err, Ess, Ett, Ers * 2, Est * 2, Ert * 2))

                # # Generate the rotation tensor - 5.120 in Bathe
                Q = construct_rotation_tensor(gr, gs, gt)

                # Rotate the constitutive tensor
                D = np.dot(np.dot(Q.T, D0), Q)

                # Add contribution of the element matrices
                dofstest = np.setdiff1d(np.arange(24), np.arange(5, 24, 6))
                K_el[np.ix_(dofstest, dofstest)] += np.dot(np.dot(B.T, D), B) * detJ
                M_el[np.ix_(dofstest, dofstest)] += np.dot(np.dot(N.T, rho * np.eye(N.shape[0])), N) * detJ

                # Total volume
                Volume_el += detJ
    return K_el, M_el, {"Volume": Volume_el, "V1": V1, "V2": V2, "Vn": Vn}


def MITC4_global(nodes, elements, E, nu, rho, thickness, verbose=0, mass_lumping=False, 
                 force_nonsingularity=False, mass_threshold=None):
    """Construct the global stiffness matrix and mass matrices.
    
    Parameters
    ----------
    nodes : array
        The nodes of the mesh.
    elements : array
        The elements of the mesh.
    E : float or iterable
        The Young's modulus of the material. If float, all elements are the same.
        If iterable, must be the same length as elements.
    nu : float or iterable
        The Poisson's ratio of the material. If float, all elements are the same.
        If iterable, must be the same length as elements.
    rho : float or iterable
        The density of the material. If float, all elements are the same.
        If iterable, must be the same length as elements.
    thickness : float or iterable
        The thickness of the material. If float, all elements are the same.
        If iterable, must be the same length as elements.
    verbose : bool
        If True, show the progress bar.
    mass_lumping : bool
        If True, perform mass lumping.
    force_nonsingularity : bool
        If True, add a small value to the diagonal of the mass matrix to force nonsingularity.
    mass_threshold : float
        The threshold for the mass matrix. Elements of the matrix below this threshold are set to zero.
    """
    warnings.warn("MITC4_global is deprecated. Use Shell class instead.", DeprecationWarning)
    n_nodes = nodes.shape[0]
    n_el = elements.shape[0]

    dofs_per_node = 6
    nDOF = n_nodes * dofs_per_node

    # check if E is iterable
    if not isinstance(E, Iterable):
        E = np.array([E] * n_el)
    elif len(E) != n_el:
        raise ValueError("E must be the same length as elements.")

    if not isinstance(rho, Iterable):
        rho = np.array([rho] * n_el)
    elif len(rho) != n_el:
        raise ValueError("rho must be the same length as elements.")

    if not isinstance(thickness, Iterable):
        thickness = np.array([thickness] * n_el)
    elif len(thickness) != n_el:
        raise ValueError("nu must be the same length as elements.")
    
    if not isinstance(nu, Iterable):
        nu = np.array([nu] * n_el)
    elif len(nu) != n_el:
        raise ValueError("nu must be the same length as elements.")
    
    n_entries_in_matrix = (elements.shape[1] * dofs_per_node)**2 # 24 * 24 for 4 node, 6 dof per node element
    iIndex = np.zeros(n_el * n_entries_in_matrix, dtype=int)
    jIndex = np.zeros(n_el * n_entries_in_matrix, dtype=int)
    kAll = np.zeros(n_el * n_entries_in_matrix)
    mAll = np.zeros(n_el * n_entries_in_matrix)
    V_All = np.zeros(n_el)  # Initialize the volume vector

    if verbose:
        pbar = lambda x: tqdm(x)
        print("Constructing the global stiffness matrix and mass matrix...")
    else:
        pbar = lambda x: x

    for el, element in enumerate(pbar(elements)):
        nodes_el = nodes[element]
        
        # Get the element stiffness and mass matrix
        K_el, M_el, data_dict = MITC4_element(E[el], nu[el], rho[el], thickness[el], nodes_el)
        
        # Extract normal and orthogonal vectors
        V1 = data_dict["V1"]
        V2 = data_dict["V2"]
        Vn = data_dict["Vn"]

        # Store the volume
        V_All[el] = data_dict["Volume"]

        # --------------------------------------------------------
        # Add the drilling dofs -> rather use the method suggested in bathe book?
        Ketest = np.diag(K_el)
        
        minKetest = np.min(Ketest[Ketest != 0])
        K_el[5::6, 5::6] = (minKetest * 1e-5) * np.eye(4) 
        # I noticed that if the division is by 1000, the mode shape ordering is 
        # not ok (first deformation mode shape is before two of the rigid, rotational mode shapes)
        # --------------------------------------------------------

        # get the dofs for the element
        n1, n2, n3, n4 = element
        dofs = np.concatenate([np.arange(n1 * 6, (n1+1) * 6),
                            np.arange(n2 * 6, (n2+1) * 6),
                            np.arange(n3 * 6, (n3+1) * 6),
                            np.arange(n4 * 6, (n4+1) * 6)])
        
        # Get the rotation matrix for stiffness and mass matrix
        T = construct_rotational_matrix(V1, V2, Vn)

        # Rotate the stiffness and mass matrix
        K_el = np.dot(np.dot(T.T, K_el), T)
        M_el = np.dot(np.dot(T.T, M_el), T)

        # Perform mass lumping
        if mass_lumping:
            M_el = np.diag(np.sum(M_el, axis=1))

        # Add the element matrices to the global matrices
        nEntry = 24 * 24  # Number of entries in the stiffness matrix
        start_idx = el * nEntry
        end_idx = (el+1) * nEntry

        kAll[start_idx:end_idx] = K_el.flatten()
        mAll[start_idx:end_idx] = M_el.flatten()
        iIndex[start_idx:end_idx] = np.tile(dofs, 24)
        jIndex[start_idx:end_idx] = np.repeat(dofs, 24)


    # Create the sparse global stiffness and mass matrices
    # Create COO arrays (easier construction)
    K_coo = coo_array((kAll, (iIndex, jIndex)), shape=(nDOF, nDOF))
    M_coo = coo_array((mAll, (iIndex, jIndex)), shape=(nDOF, nDOF))

    if verbose:
        print("...matrices assembled.")

    # Convert COO arrays to CSR arrays (more efficient computation)
    K = K_coo.tocsr()
    M = M_coo.tocsr()

    if mass_threshold is not None:
        M[M < mass_threshold] = 0

    if force_nonsingularity:
        # add a small value to the diagonal of the mass matrix
        Mtest = np.abs(M.diagonal())
        minMtest = np.min(Mtest[Mtest != 0])
        # nonsingularity_factor = 1e-5
        nonsingularity_factor = 1e-4
        M = M + minMtest * nonsingularity_factor * diags_array(np.ones(M.shape[0]))

    K.eliminate_zeros()
    M.eliminate_zeros()

    return K, M

class Shell:
    def __init__(self, nodes, elements, E, nu, rho, thickness, verbose=0, 
                 mass_lumping=False, force_nonsingularity=False, mass_threshold=None):
        """Initialize the shell model.
        
        Parameters
        ----------
        nodes : array
            The nodes of the mesh.
        elements : array
            The elements of the mesh.
        E : float or iterable
            The Young's modulus of the material. If float, all elements are the same.
            If iterable, must be the same length as elements.
        nu : float or iterable
            The Poisson's ratio of the material. If float, all elements are the same.
            If iterable, must be the same length as elements.
        rho : float or iterable
            The density of the material. If float, all elements are the same.
            If iterable, must be the same length as elements.
        thickness : float or iterable
            The thickness of the material. If float, all elements are the same.
            If iterable, must be the same length as elements.
        verbose : bool
            If True, show the progress bar.
        mass_lumping : bool
            If True, perform mass lumping.
        force_nonsingularity : bool
            If True, add a small value to the diagonal of the mass matrix to force nonsingularity.
        mass_threshold : float
            The threshold for the mass matrix. Elements of the matrix below this threshold are set to zero.
        """
        self.E = E
        self.nu = nu
        self.rho = rho
        self.thickness = thickness
        self.nodes = nodes
        self.elements = elements
        self.verbose = verbose
        self.mass_lumping = mass_lumping
        self.force_nonsingularity = force_nonsingularity
        self.mass_threshold = mass_threshold

        self.n_nodes = self.nodes.shape[0]
        self.n_el = self.elements.shape[0]

        # check if self.E is iterable
        if not isinstance(self.E, Iterable):
            self.E = np.array([self.E] * self.n_el)
        elif len(self.E) != self.n_el:
            raise ValueError("self.E must be the same length as elements.")

        if not isinstance(self.rho, Iterable):
            self.rho = np.array([self.rho] * self.n_el)
        elif len(self.rho) != self.n_el:
            raise ValueError("rho must be the same length as elements.")

        if not isinstance(self.thickness, Iterable):
            self.thickness = np.array([self.thickness] * self.n_el)
        elif len(self.thickness) != self.n_el:
            raise ValueError("nu must be the same length as elements.")
        
        if not isinstance(self.nu, Iterable):
            self.nu = np.array([self.nu] * self.n_el)
        elif len(self.nu) != self.n_el:
            raise ValueError("nu must be the same length as elements.")
        
        # Construct the global stiffness and mass matrices
        self.construct_global_matrices()

    def construct_MITC4_matrices_for_element(self, nodes_el, el):
        """The computation of stiffness and mass matrices for each element.
        
        Using the MITC4 elements. The code is based on the MatLab code and on the
        Finite Element Procedures book by Bathe.
        """
        E = self.E[el]
        nu = self.nu[el]
        rho = self.rho[el]
        thickness = self.thickness[el]

        # Calculate normal at each node
        Vn, V1, V2 = get_normal_vector(nodes_el)
        
        # see eq 5.114 in Bathe
        g1 = -0.5 * thickness * V2
        g2 = 0.5 * thickness * V1

        # constitutive tensor in local coordinate system
        D0 = get_constitutive_tensor(E, nu)

        # perform full gauss integration (2 by 2 by 2 integration)
        W_gauss = np.array([1, 1])
        points_gauss = np.array([1/np.sqrt(3), -1/np.sqrt(3)])

        #  Compute K and M
        K_el = np.zeros((24, 24))
        M_el = np.zeros((24, 24))
        Volume_el = 0

        for r in points_gauss: # xi
            for s in points_gauss: # eta 
                for t in points_gauss: # zeta

                    # interpolation functions
                    N1, N2, N3, N4 = shape_functions(r, s)

                    # derivative of the interpolation functions
                    dN = shape_function_derivatives(r, s)
                    dN1dr, dN1ds = dN[0]
                    dN2dr, dN2ds = dN[1]
                    dN3dr, dN3ds = dN[2]
                    dN4dr, dN4ds = dN[3]
                    
                    # assemble the interpolation matrix (used to get the mass matrix)
                    N = interpolation_matrix(N1, N2, N3, N4, t, thickness, V1, V2)

                    # Constructing Brr
                    Brr = construct_Brr(dN1dr, dN2dr, dN3dr, dN4dr, g1, g2, t)

                    # Constructing Bss
                    Bss = construct_Bss(dN1ds, dN2ds, dN3ds, dN4ds, g1, g2, t)

                    # Constructing Btt
                    Btt = construct_Btt(N1, N2, N3, N4, g1, g2, t)

                    # Initialize GN with zeros
                    GN = np.zeros((3, 4))

                    # Fill GN with values
                    GN[:2, :] = np.array([
                        [dN1dr, dN2dr, dN3dr, dN4dr],
                        [dN1ds, dN2ds, dN3ds, dN4ds]
                    ])

                    # Calculate GN_dash
                    GN_dash = GN * 0.5 * t * thickness
                    GN_dash[2, :] = np.array([N1, N2, N3, N4]) * 0.5 * thickness

                    # Calculate the Jacobian
                    J = np.dot(GN, nodes_el) + np.dot(GN_dash, Vn)
                    detJ = np.linalg.det(J)

                    # Covariant bases
                    gr = J[0, :]
                    gs = J[1, :]
                    gt = J[2, :]

                    # Calculate the direct strain components
                    Err = np.dot(gr, Brr)
                    Ess = np.dot(gs, Bss)
                    Ett = np.dot(gt, Btt)  # This should be zero
                    Ers = 0.5 * (np.dot(gs, Brr) + np.dot(gr, Bss))

                    # Coordinates r1 and r2 for points A, B, C, D
                    r1 = [0, -1, 0, 1]  # order A B C D
                    r2 = [1, 0, -1, 0]

                    # Initialize strain results
                    Ert_A, Est_B, Ert_C, Est_D = None, None, None, None

                    for i in range(4):
                        # Interpolation functions
                        N1, N2, N3, N4 = shape_functions(r1[i], r2[i])

                        # Derivative of the interpolation functions
                        dN = shape_function_derivatives(r1[i], r2[i])
                        dN1dr, dN1ds = dN[0]
                        dN2dr, dN2ds = dN[1]
                        dN3dr, dN3ds = dN[2]
                        dN4dr, dN4ds = dN[3]

                        # Constructing Brr
                        Brr = construct_Brr(dN1dr, dN2dr, dN3dr, dN4dr, g1, g2, t)

                        # Constructing Bss
                        Bss = construct_Bss(dN1ds, dN2ds, dN3ds, dN4ds, g1, g2, t)

                        # Constructing Btt
                        Btt = construct_Btt(N1, N2, N3, N4, g1, g2, t)

                        # Constructing GN_edge and GN_dash_edge
                        GN_edge = np.zeros((3, 4))
                        GN_edge[:2, :] = np.array([
                            [dN1dr, dN2dr, dN3dr, dN4dr],
                            [dN1ds, dN2ds, dN3ds, dN4ds]
                        ])

                        GN_dash_edge = GN_edge * 0.5 * t * thickness
                        GN_dash_edge[2, :] = np.array([N1, N2, N3, N4]) * 0.5 * thickness

                        # Jacobian
                        J_edge = np.dot(GN_edge, nodes_el) + np.dot(GN_dash_edge, Vn)

                        # Covariant bases
                        gr_edge = J_edge[0, :]
                        gs_edge = J_edge[1, :]
                        gt_edge = J_edge[2, :]

                        if i == 0:  # at A
                            Ert_A = 0.5 * (np.dot(gr_edge, Btt) + np.dot(gt_edge, Brr))
                        elif i == 1:  # at B
                            Est_B = 0.5 * (np.dot(gs_edge, Btt) + np.dot(gt_edge, Bss))
                        elif i == 2:  # at C
                            Ert_C = 0.5 * (np.dot(gr_edge, Btt) + np.dot(gt_edge, Brr))
                        else:  # at D
                            Est_D = 0.5 * (np.dot(gs_edge, Btt) + np.dot(gt_edge, Bss))

                    # Perform the interpolation for the transverse strains
                    Ert = 0.5 * (1 + s) * Ert_A + 0.5 * (1 - s) * Ert_C
                    Est = 0.5 * (1 + r) * Est_D + 0.5 * (1 - r) * Est_B

                    # Assemble the B vector
                    B = np.vstack((Err, Ess, Ett, Ers * 2, Est * 2, Ert * 2))

                    # # Generate the rotation tensor - 5.120 in Bathe
                    Q = construct_rotation_tensor(gr, gs, gt)

                    # Rotate the constitutive tensor
                    D = np.dot(np.dot(Q.T, D0), Q)

                    # Add contribution of the element matrices
                    dofstest = np.setdiff1d(np.arange(24), np.arange(5, 24, 6))
                    K_el[np.ix_(dofstest, dofstest)] += np.dot(np.dot(B.T, D), B) * detJ
                    M_el[np.ix_(dofstest, dofstest)] += np.dot(np.dot(N.T, rho * np.eye(N.shape[0])), N) * detJ

                    # Total volume
                    Volume_el += detJ

        return K_el, M_el, {"Volume": Volume_el, "V1": V1, "V2": V2, "Vn": Vn}
    
    def construct_global_matrices(self):
        """Construct the global stiffness matrix and mass matrices."""
        dofs_per_node = 6
        nDOF = self.n_nodes * dofs_per_node
        
        n_entries_in_matrix = (self.elements.shape[1] * dofs_per_node)**2 # 24 * 24 for 4 node, 6 dof per node element
        iIndex = np.zeros(self.n_el * n_entries_in_matrix, dtype=int)
        jIndex = np.zeros(self.n_el * n_entries_in_matrix, dtype=int)
        kAll = np.zeros(self.n_el * n_entries_in_matrix)
        mAll = np.zeros(self.n_el * n_entries_in_matrix)
        V_All = np.zeros(self.n_el)  # Initialize the volume vector

        if self.verbose:
            def pbar(x):
                return tqdm(x)
            print("Constructing the global stiffness matrix and mass matrix...")
        else:
            def pbar(x):
                return x

        for el, element in enumerate(pbar(self.elements)):
            nodes_el = self.nodes[element]
            
            # Get the element stiffness and mass matrix
            # K_el, M_el, data_dict = MITC4_element(self.E[el], self.nu[el], self.rho[el], self.thickness[el], nodes_el)
            K_el, M_el, data_dict = self.construct_MITC4_matrices_for_element(nodes_el, el)
            
            # Extract normal and orthogonal vectors
            V1 = data_dict["V1"]
            V2 = data_dict["V2"]
            Vn = data_dict["Vn"]

            # Store the volume
            V_All[el] = data_dict["Volume"]

            # --------------------------------------------------------
            # Add the drilling dofs -> rather use the method suggested in bathe book?
            Ketest = np.diag(K_el)
            
            minKetest = np.min(Ketest[Ketest != 0])
            K_el[5::6, 5::6] = (minKetest * 1e-5) * np.eye(4) 
            # I noticed that if the division is by 1000, the mode shape ordering is 
            # not ok (first deformation mode shape is before two of the rigid, rotational mode shapes)
            # --------------------------------------------------------

            # get the dofs for the element
            n1, n2, n3, n4 = element
            dofs = np.concatenate([np.arange(n1 * 6, (n1+1) * 6),
                                np.arange(n2 * 6, (n2+1) * 6),
                                np.arange(n3 * 6, (n3+1) * 6),
                                np.arange(n4 * 6, (n4+1) * 6)])
            
            # Get the rotation matrix for stiffness and mass matrix
            T = construct_rotational_matrix(V1, V2, Vn)

            # Rotate the stiffness and mass matrix
            K_el = np.dot(np.dot(T.T, K_el), T)
            M_el = np.dot(np.dot(T.T, M_el), T)

            # Perform mass lumping
            if self.mass_lumping:
                M_el = np.diag(np.sum(M_el, axis=1))

            # Add the element matrices to the global matrices
            nEntry = 24 * 24  # Number of entries in the stiffness matrix
            start_idx = el * nEntry
            end_idx = (el+1) * nEntry

            kAll[start_idx:end_idx] = K_el.flatten()
            mAll[start_idx:end_idx] = M_el.flatten()
            iIndex[start_idx:end_idx] = np.tile(dofs, 24)
            jIndex[start_idx:end_idx] = np.repeat(dofs, 24)


        # Create the sparse global stiffness and mass matrices
        # Create COO arrays (easier construction)
        K_coo = coo_array((kAll, (iIndex, jIndex)), shape=(nDOF, nDOF))
        M_coo = coo_array((mAll, (iIndex, jIndex)), shape=(nDOF, nDOF))

        if self.verbose:
            print("...matrices assembled.")

        # Convert COO arrays to CSR arrays (more efficient computation)
        K = K_coo.tocsr()
        M = M_coo.tocsr()

        if self.mass_threshold is not None:
            M[M < self.mass_threshold] = 0

        if self.force_nonsingularity:
            # add a small value to the diagonal of the mass matrix
            Mtest = np.abs(M.diagonal())
            minMtest = np.min(Mtest[Mtest != 0])
            # nonsingularity_factor = 1e-5
            nonsingularity_factor = 1e-4
            M = M + minMtest * nonsingularity_factor * diags_array(np.ones(M.shape[0]))

        K.eliminate_zeros()
        M.eliminate_zeros()

        self.K = K
        self.M = M