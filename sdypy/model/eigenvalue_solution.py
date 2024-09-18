from scipy.sparse.linalg import eigsh
from scipy import sparse
from scipy.linalg import eigh
import numpy as np


def solve_eigenvalue(K, M, n_modes=None, convert_to_hz=False):
    """
    Solve the eigenvalue problem.

    Parameters
    ----------
    K : ndarray or sparse
        Stiffness matrix.
    M : ndarray or sparse
        Mass matrix.
    n_modes : int or None
        Number of modes to compute. If None, all modes are computed.

    Returns
    -------
    eigenvalues : ndarray
        Eigenvalues.
    eigenvectors : ndarray
        Eigenvectors.
    """
    if sparse.issparse(K) and sparse.issparse(M):
        if n_modes is None:
            n_modes = K.shape[0] - 1

        eigenvalues, eigenvectors = eigsh(K, M=M, k=n_modes, sigma=0, which='LM')
    else:
        eigenvalues, eigenvectors = eigh(K, M=M)

    if convert_to_hz:
        eigenvalues = np.sqrt(eigenvalues) / (2 * np.pi)

    return eigenvalues, eigenvectors