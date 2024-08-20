import numpy as np
from numba import jit

@jit
def roll_2d(a: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    """
    Implementation of np.roll for 2D arrays.

    Parameters:
    a (np.ndarray): 2D array to be rolled.
    shift (tuple[int, int]): Shifts in the x and y directions.

    Returns:
    np.ndarray: Rolled 2D array.
    """
    assert a.ndim == 2

    b = np.empty_like(a)

    X, Y = a.shape
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

@jit
def roll_parallel(a: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    """
    Implementation of roll_2d along axes 1 and 2 for a 3D array.

    Parameters:
    a (np.ndarray): 3D array to be rolled.
    shift (tuple[int, int]): Shifts along axes 1 and 2.

    Returns:
    np.ndarray: Rolled 3D array.
    """
    assert a.ndim == 3

    b = np.empty_like(a)

    _, X, Y = a.shape
    shift_x, shift_y = shift
    shift_x %= X
    shift_y %= Y

    if shift_x > 0:
        b[:, :shift_x, :] = a[:, -shift_x:, :]
        b[:, shift_x:, :] = a[:, :-shift_x, :]

    if shift_y > 0:
        b[:, :, :shift_y] = a[:, :, -shift_y:]
        b[:, :, shift_y:] = a[:, :, :-shift_y]

    return b

@jit
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
    assert a.ndim == 2

    b = -3 * a
    b += (roll_2d(a, (0, 1)) + roll_2d(a, (0, -1)) + roll_2d(a, (1, 0)) + roll_2d(a, (-1, 0))) / 2
    b += (roll_2d(a, (1, 1)) + roll_2d(a, (-1, -1)) + roll_2d(a, (1, -1)) + roll_2d(a, (-1, 1))) / 4

    return b

@jit
def laplace_parallel(a: np.ndarray) -> np.ndarray:
    """
    Implementation of laplace_2d along axes 1 and 2 for a 3D array.

    Parameters:
    a (np.ndarray): 3D array to be Laplacian-filtered.

    Returns:
    np.ndarray: Laplacian-filtered 3D array.
    """
    assert a.ndim == 3

    b = -3 * a
    b += roll_parallel(a, (0, 1)) / 2
    b += roll_parallel(a, (0, -1)) / 2
    b += roll_parallel(a, (1, 0)) / 2
    b += roll_parallel(a, (-1, 0)) / 2
    b += roll_parallel(a, (1, 1)) / 4
    b += roll_parallel(a, (-1, -1)) / 4
    b += roll_parallel(a, (1, -1)) / 4
    b += roll_parallel(a, (-1, 1)) / 4

    return b