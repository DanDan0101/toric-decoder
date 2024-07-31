import numpy as np
from scipy.sparse import csc_matrix
from pymatching import Matching

def pcm(L: int) -> csc_matrix:
    """
    Create a parity-check matrix for the 2D toric code. The rows correspond to plaquettes, 
    and the columns correspond to x bonds and y bonds.

    Parameters:
    L (int): The lattice size.

    Returns:
    csc_matrix: L^2 x 2L^2 sparse array with 4L^2 entries, ∈ ℤ/2ℤ.
    """

    row_ind = np.arange(L**2)

    # x bonds, -y
    col_ind_1 = row_ind.copy()

    # x bonds, +y
    col_ind_2 = (row_ind + 1) % L + row_ind - row_ind % L

    # y bonds, -x
    col_ind_3 = row_ind + L**2

    # y bonds, +x
    col_ind_4 = (row_ind + L) % L**2 + L**2

    row_ind = np.tile(row_ind, 4)
    col_ind = np.concatenate([col_ind_1, col_ind_2, col_ind_3, col_ind_4])
    data = np.ones(4 * L**2, dtype = np.uint8)
    return csc_matrix((data, (row_ind, col_ind)), shape = (L**2, 2 * L**2))

def init_mwpm(L: int) -> Matching:
    """
    Initializes a Matching object for minimum-weight perfect-matching.

    Parameters:
    L (int): The lattice size.

    Returns:
    Matching: The initialized Matching object.
    """

    return Matching(pcm(L))