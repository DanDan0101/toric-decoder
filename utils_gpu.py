import cupy as np
from cupyx.scipy.ndimage import correlate
# from numba import jit

# @jit
def roll_2d(a: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    return np.roll(a, shift, axis = (0, 1))

# @jit
def roll_parallel(a: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    return np.roll(a, shift, axis = (1, 2))

# @jit
def laplace_2d(a: np.ndarray) -> np.ndarray:
    """
    Implementation of scipy.ndimage.laplace for 2D arrays,
    with 'wrap' mode, using the Oono-Puri nine-point stencil:
    [[1, 2, 1], [2, -12, 2], [1, 2, 1]] / 4.

    Parameters:
    a (np.ndarray): 2D array to be Laplacian-filtered.

    Returns:
    np.ndarray: Laplacian-filtered 2D array.
    """

    k = np.array([[1, 2, 1], [2, -12, 2], [1, 2, 1]], dtype = np.float32) / 4
    return correlate(a, k, mode = 'wrap')

# @jit
def laplace_parallel(a: np.ndarray) -> np.ndarray:
    """
    Implementation of laplace_2d along axes 1 and 2 for a 3D array.

    Parameters:
    a (np.ndarray): 3D array to be Laplacian-filtered.

    Returns:
    np.ndarray: Laplacian-filtered 3D array.
    """
    assert a.ndim == 3

    k_2d = np.array([[1, 2, 1], [2, -12, 2], [1, 2, 1]], dtype = np.float32) / 4
    z_2d = np.zeros((3, 3), dtype = np.float32)
    k = np.dstack([z_2d, k_2d, z_2d])
    return correlate(a, k, mode = 'wrap')