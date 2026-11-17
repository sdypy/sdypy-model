from scipy.sparse.linalg import eigsh
from scipy import sparse
from scipy.linalg import eigh
import warnings

import numpy as np

from ._deprecation import reject_both, require


def solve_eigenvalue(stiffness_matrix=None, mass_matrix=None, n_modes=None,
                     convert_to_hz=False, *, K=None, M=None):
    """
    Solve the eigenvalue problem.

    Parameters
    ----------
    stiffness_matrix : ndarray or sparse
        Stiffness matrix.
    mass_matrix : ndarray or sparse
        Mass matrix.
    n_modes : int or None
        Number of modes to compute. If None, all modes are computed.
    convert_to_hz : bool, optional
        Return natural frequencies in Hz instead of eigenvalues.

    Deprecated parameters
    ---------------------
    K : use stiffness_matrix instead.
    M : use mass_matrix instead.

    Returns
    -------
    eigenvalues : ndarray
        Eigenvalues.
    eigenvectors : ndarray
        Eigenvectors.
    """
    if K is not None:
        reject_both("K", "stiffness_matrix", stiffness_matrix)
        warnings.warn("K is deprecated; use stiffness_matrix instead",
                      DeprecationWarning, stacklevel=2)
        stiffness_matrix = K
    if M is not None:
        reject_both("M", "mass_matrix", mass_matrix)
        warnings.warn("M is deprecated; use mass_matrix instead",
                      DeprecationWarning, stacklevel=2)
        mass_matrix = M
    require("solve_eigenvalue", stiffness_matrix=stiffness_matrix,
            mass_matrix=mass_matrix)
    K = stiffness_matrix
    M = mass_matrix
    if sparse.issparse(K) and sparse.issparse(M):
        if n_modes is None:
            n_modes = K.shape[0] - 1

        eigenvalues, eigenvectors = eigsh(K, M=M, k=n_modes, sigma=0, which='LM')
    else:
        eigenvalues, eigenvectors = eigh(K, M=M)

    if convert_to_hz:
        eigenvalues = np.sqrt(eigenvalues) / (2 * np.pi)

    return eigenvalues, eigenvectors