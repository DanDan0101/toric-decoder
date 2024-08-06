import numpy as np
from numba import njit

@njit
def roll_1d(a: np.ndarray, shift: int) -> np.ndarray:
    """
    Implementation of np.roll for 1D arrays.
    
    Parameters:
    a (np.ndarray): 1D array to be rolled.
    shift (int): Shift amount.

    Returns:
    np.ndarray: Rolled 1D array.
    """

    b = np.empty_like(a)
    shift %= a.shape[0]

    if shift > 0:
        b[:shift] = a[-shift:]
        b[shift:] = a[:-shift]
    
    return b

@njit
def roll_2d(a: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    """
    Implementation of np.roll for 2D arrays.

    Parameters:
    a (np.ndarray): 2D array to be rolled.
    shift (tuple[int, int]): Shifts in the x and y directions.

    Returns:
    np.ndarray: Rolled 2D array.
    """

    b = np.empty_like(a)

    X = a.shape[0]
    Y = a.shape[1]
    shift_x, shift_y = shift
    shift_x %= X
    shift_y %= Y

    if shift_x > 0:
        b[:shift_x, :] = a[-shift_x:, :]
        b[shift_x:, :] = a[:-shift_x, :]

    if shift_y > 0:
        b[:, :shift_y] = a[:, -shift_y:]
        b[:, shift_y:] = a[:, :-shift_y]

    return b

@njit
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

    b = np.zeros_like(a)
    b -= 3 * a
    b += (roll_2d(a, (0, 1)) + roll_2d(a, (0, -1)) + roll_2d(a, (1, 0)) + roll_2d(a, (-1, 0))) / 2
    b += (roll_2d(a, (1, 1)) + roll_2d(a, (-1, -1)) + roll_2d(a, (1, -1)) + roll_2d(a, (-1, 1))) / 4

    return b

@njit
def window(L: int) -> np.ndarray:
    """
    Returns a list of indices for a 3x3 window on a 2D lattice with size L.

    Parameters:
    L (int): Size of the lattice.

    Returns:
    np.ndarray: Window indices.
    """

    # Disgusting hardcoding because outer sum is not supported in numba
    interval = np.array([-1, 0, 1], dtype = np.int32)
    window = np.repeat(interval * L, 3)
    window[::3] -= 1
    window[2::3] += 1
    return window